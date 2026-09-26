//! The TurboQuant codec and compressed per-layer caches.

use super::super::{codebooks, lloyd_max, packing, rotation};
use larql_inference::attention::SharedKV;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::{s, Array2};

#[allow(unused_imports)]
use super::*;

/// WHT + Lloyd-Max codec. Stateless — all operations are deterministic
/// functions of the input vector and the pre-computed codebook.
#[derive(Clone)]
pub struct TurboQuant {
    pub bits: u8, // 3 or 4
}

impl TurboQuant {
    pub fn new(bits: u8) -> Self {
        assert!(bits == 3 || bits == 4, "TurboQuant: bits must be 3 or 4");
        Self { bits }
    }

    /// Encode a single vector: normalize → WHT → quantize → pack.
    /// Returns a freshly-allocated `Vec<u8>` — kept for ergonomic API
    /// stability. Hot-path callers use [`encode_vector_into`] with
    /// reusable scratch buffers.
    pub fn encode_vector(&self, x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes_per_vector(x.len()));
        let mut scratch_f32 = vec![0.0f32; x.len()];
        let mut scratch_u8 = Vec::with_capacity(x.len());
        self.encode_vector_into(x, &mut out, &mut scratch_f32, &mut scratch_u8);
        out
    }

    /// Encode into a caller-provided byte buffer using caller-provided
    /// scratch. `scratch_f32` and `scratch_u8` are resized as needed
    /// and may be reused across calls to amortise allocation.
    ///
    /// 2026-05-19 codec hot-path optimisation: hoists the per-call
    /// allocations from [`encode_vector`] (x_hat, WHT output, indices)
    /// into a scratch pair the caller can keep alive across the
    /// compress_matrix loop. Together with [`rotation::wht_inplace`]'s
    /// NEON path this is the recompute_hot win.
    pub fn encode_vector_into(
        &self,
        x: &[f32],
        out: &mut Vec<u8>,
        scratch_f32: &mut Vec<f32>,
        scratch_u8: &mut Vec<u8>,
    ) {
        let d = x.len();
        scratch_f32.resize(d, 0.0);
        scratch_u8.clear();
        scratch_u8.reserve(d);

        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for (i, &v) in x.iter().enumerate() {
                scratch_f32[i] = v * inv;
            }
        } else {
            for v in scratch_f32.iter_mut() {
                *v = 0.0;
            }
        }
        rotation::wht_inplace(scratch_f32);
        // Coordinates of the rotated unit vector have std exactly 1/√d;
        // dividing by that sigma maps them into the N(0, 1) space the
        // unit codebook is trained on.
        let codebook = codebooks::unit_codebook(self.bits);
        let inv_sigma = 1.0 / codebooks::wht_coordinate_sigma(d);
        for &val in scratch_f32.iter() {
            scratch_u8.push(lloyd_max::quantize_scalar(val * inv_sigma, codebook));
        }
        out.extend_from_slice(&norm.to_le_bytes());
        packing::pack_indices(scratch_u8, self.bits, out);
    }

    /// Decode a single vector: unpack → centroids → inverse WHT → rescale.
    /// Returns a freshly-allocated `Vec<f32>` — kept for ergonomic API
    /// stability. Hot-path callers use [`decode_vector_into`].
    pub fn decode_vector(&self, encoded: &[u8], dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; dim];
        let mut scratch_u8 = Vec::with_capacity(dim);
        self.decode_vector_into(encoded, dim, &mut out, &mut scratch_u8);
        out
    }

    /// Decode into a caller-provided f32 buffer using caller-provided
    /// scratch. `out` is resized to `dim`; `scratch_u8` is reused for
    /// the unpacked-index intermediate.
    pub fn decode_vector_into(
        &self,
        encoded: &[u8],
        dim: usize,
        out: &mut Vec<f32>,
        scratch_u8: &mut Vec<u8>,
    ) {
        let norm = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        scratch_u8.clear();
        packing::unpack_indices_into(&encoded[4..], dim, self.bits, scratch_u8);
        let codebook = codebooks::unit_codebook(self.bits);
        out.resize(dim, 0.0);
        for (i, &idx) in scratch_u8.iter().enumerate() {
            out[i] = codebook.centroids[idx as usize];
        }
        rotation::wht_inplace(out);
        // The WHT is linear, so the sigma scaling (undoing encode's
        // ×√d) commutes with it and folds into the norm restore.
        let scale = norm * codebooks::wht_coordinate_sigma(dim);
        for v in out.iter_mut() {
            *v *= scale;
        }
    }

    pub fn bytes_per_vector(&self, dim: usize) -> usize {
        4 + packing::packed_size(dim, self.bits)
    }
}

// ─── Compressed K/V layer ────────────────────────────────────────────────────

pub(in super::super) struct CompressedLayer {
    pub compressed_k: Vec<u8>,
    pub compressed_v: Vec<u8>,
    pub num_vecs: usize,
    pub kv_dim: usize,
    /// Largest power-of-two head dimension detected from kv_dim.
    pub head_dim: usize,
}

impl CompressedLayer {
    pub(in super::super) fn compress(kv: &SharedKV, tq: &TurboQuant) -> Self {
        let (k, v) = kv;
        let num_vecs = k.shape()[0];
        let kv_dim = k.shape()[1];
        let head_dim = detect_head_dim(kv_dim);
        Self {
            compressed_k: compress_matrix(k, tq, head_dim),
            compressed_v: compress_matrix(v, tq, head_dim),
            num_vecs,
            kv_dim,
            head_dim,
        }
    }

    pub(in super::super) fn decompress(&self, tq: &TurboQuant) -> SharedKV {
        let k = decompress_matrix(
            &self.compressed_k,
            self.num_vecs,
            self.kv_dim,
            self.head_dim,
            tq,
        );
        let v = decompress_matrix(
            &self.compressed_v,
            self.num_vecs,
            self.kv_dim,
            self.head_dim,
            tq,
        );
        (k, v)
    }

    /// Append-only ingest of one new K/V row: encode `k_row` / `v_row`
    /// head-by-head and push the packed bytes onto the existing
    /// compressed buffers. Invariant: existing rows' bytes are never
    /// re-encoded — a decompress→re-encode cycle multiplies each stored
    /// norm by the codec's reconstruction-norm ratio (< 1) and
    /// compounds it across decode steps.
    pub(in super::super) fn append_row(
        &mut self,
        k_row: &[f32],
        v_row: &[f32],
        tq: &TurboQuant,
        scratch_f32: &mut Vec<f32>,
        scratch_u8: &mut Vec<u8>,
    ) {
        debug_assert_eq!(k_row.len(), self.kv_dim, "K row width != layer kv_dim");
        debug_assert_eq!(v_row.len(), self.kv_dim, "V row width != layer kv_dim");
        for chunk in k_row.chunks(self.head_dim) {
            tq.encode_vector_into(chunk, &mut self.compressed_k, scratch_f32, scratch_u8);
        }
        for chunk in v_row.chunks(self.head_dim) {
            tq.encode_vector_into(chunk, &mut self.compressed_v, scratch_f32, scratch_u8);
        }
        self.num_vecs += 1;
    }

    /// Drop rows until the layer holds `rows` of them again.
    ///
    /// Byte-exact, and that is not an accident of the codec being good: rows
    /// are appended as whole head-chunks at fixed byte offsets and existing
    /// bytes are never re-encoded (see [`Self::append_row`]), so removing the
    /// tail restores precisely the buffer that preceded it. The compression is
    /// lossy against its *input*, not against what was stored — which is what
    /// lets a K/V-canonical engine rewind at all.
    ///
    /// No-op when `rows` is not smaller than the current count, so a caller
    /// rewinding a layer the failure never reached costs nothing.
    pub(in super::super) fn truncate_rows(&mut self, rows: usize, tq: &TurboQuant) {
        if rows >= self.num_vecs {
            return;
        }
        let heads = self.kv_dim / self.head_dim.max(1);
        let bytes = rows * heads * tq.bytes_per_vector(self.head_dim);
        self.compressed_k.truncate(bytes);
        self.compressed_v.truncate(bytes);
        self.num_vecs = rows;
    }

    pub(in super::super) fn memory_bytes(&self) -> usize {
        self.compressed_k.len() + self.compressed_v.len()
    }
}

pub(in super::super) fn detect_head_dim(kv_dim: usize) -> usize {
    for &hd in &[256usize, 128, 64, 32] {
        if kv_dim.is_multiple_of(hd) {
            return hd;
        }
    }
    kv_dim // fallback: treat whole row as one head
}

/// Fallible companion to [`detect_head_dim`]: the WHT butterfly requires a
/// power-of-two block dim, so a kv_dim with no supported head split (e.g.
/// 80) must surface as a typed error at prefill entry instead of the WHT
/// assert firing mid-prefill.
pub(in super::super) fn resolve_block_dim(kv_dim: usize) -> Result<usize, EngineError> {
    let head_dim = detect_head_dim(kv_dim);
    if head_dim.is_power_of_two() {
        Ok(head_dim)
    } else {
        Err(EngineError::InvariantViolation {
            what: format!(
                "turbo-quant codec requires a power-of-two K/V block dim \
                 (WHT constraint); kv_dim={kv_dim} has no supported head split"
            ),
        })
    }
}

pub(in super::super) fn compress_matrix(
    m: &Array2<f32>,
    tq: &TurboQuant,
    head_dim: usize,
) -> Vec<u8> {
    let rows = m.shape()[0];
    let cols = m.shape()[1];
    let heads_per_row = cols / head_dim;
    let mut buf = Vec::with_capacity(rows * heads_per_row * tq.bytes_per_vector(head_dim));
    // Hot-path scratch reused across every chunk. Eliminates the
    // per-call Vec churn that 2026-05-19 diagnostics flagged as the
    // codec's second-biggest cost (after the WHT butterfly itself).
    let mut scratch_f32 = Vec::with_capacity(head_dim);
    let mut scratch_u8 = Vec::with_capacity(head_dim);
    for row in m.rows() {
        let row_slice = row.as_slice().expect("non-contiguous row");
        for chunk in row_slice.chunks(head_dim) {
            tq.encode_vector_into(chunk, &mut buf, &mut scratch_f32, &mut scratch_u8);
        }
    }
    buf
}

pub(in super::super) fn decompress_matrix(
    bytes: &[u8],
    num_vecs: usize,
    kv_dim: usize,
    head_dim: usize,
    tq: &TurboQuant,
) -> Array2<f32> {
    let heads_per_vec = kv_dim / head_dim;
    let bytes_per_head = tq.bytes_per_vector(head_dim);
    let mut data = vec![0.0f32; num_vecs * kv_dim];
    // The per-vector WHT/codebook decode (`decode_vector_into`) is the per-step
    // bottleneck (a `/usr/bin/sample` profile put ~35% of the decode driver in
    // here, serial). Each vector writes a disjoint `kv_dim`-wide row, so fan it
    // across the spin pool — this keeps the cache COMPRESSED (the engine's
    // point: still decoded every step) but makes the decode parallel instead of
    // single-threaded. Per-chunk scratch (decode needs mutable scratch),
    // amortised over `CHUNK_VECS` vectors so it isn't reallocated per (vec,head).
    const CHUNK_VECS: usize = 8;
    larql_compute::cpu::spin_pool::par_chunks_mut(&mut data, kv_dim * CHUNK_VECS, |ci, chunk| {
        let mut decoded = Vec::with_capacity(head_dim);
        let mut scratch_u8 = Vec::with_capacity(head_dim);
        let base_vec = ci * CHUNK_VECS;
        let vecs_in_chunk = chunk.len() / kv_dim;
        for v in 0..vecs_in_chunk {
            let i = base_vec + v;
            for h in 0..heads_per_vec {
                let offset = (i * heads_per_vec + h) * bytes_per_head;
                tq.decode_vector_into(
                    &bytes[offset..offset + bytes_per_head],
                    head_dim,
                    &mut decoded,
                    &mut scratch_u8,
                );
                let row_start = v * kv_dim + h * head_dim;
                chunk[row_start..row_start + head_dim].copy_from_slice(&decoded);
            }
        }
    });
    Array2::from_shape_vec((num_vecs, kv_dim), data).expect("shape mismatch")
}

pub(in super::super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct TurboQuantEngine {
    pub(in super::super) tq: TurboQuant,
    pub(in super::super) backend: Box<dyn EngineBackend>,
    pub(in super::super) layers: Vec<CompressedLayer>,
    pub(in super::super) abs_position: usize,
    pub(in super::super) profiling: bool,
    pub(in super::super) profile: crate::profiler::EngineProfiler,
    /// W1-GPU: handle into the backend's internal K/V cache, populated
    /// when prefill routes through `coarse_prefill_with_state`. `None`
    /// means the engine took the legacy per-layer walk path.
    pub(in super::super) kv_handle: Option<larql_inference::KvHandle>,
    /// Engine-owned f32 dequant scratch for the per-layer fallback (see
    /// `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(in super::super) dequant_scratch: larql_inference::DequantScratch,
}

impl TurboQuantEngine {
    /// Decode reads each layer's compressed cache; one missing is a
    /// decode before (or across a mismatched) prefill, refused by name
    /// instead of indexing past the cache.
    pub(super) fn require_prefilled(&self, num_layers: usize) -> Result<(), EngineError> {
        if self.layers.len() < num_layers {
            return Err(EngineError::InvariantViolation {
                what: format!(
                    "turbo-quant decode before prefill: {} of {num_layers} layer caches present",
                    self.layers.len()
                ),
            });
        }
        Ok(())
    }

    pub fn new(bits: u8) -> Self {
        Self::with_backend(bits, cpu_engine_backend())
    }

    pub fn with_backend(bits: u8, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            tq: TurboQuant::new(bits),
            backend,
            layers: Vec::new(),
            abs_position: 0,
            profiling: false,
            profile: crate::profiler::EngineProfiler::default(),
            kv_handle: None,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// First-use validation of the codec's power-of-two block-dim
    /// requirement across all layers. Called from every prefill entry
    /// point; decode paths inherit the guarantee from a successful
    /// prefill.
    pub(super) fn validate_block_dims(&self, weights: &ModelWeights) -> Result<(), EngineError> {
        let arch = &*weights.arch;
        for layer in 0..weights.num_layers {
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            resolve_block_dim(kv_dim)?;
        }
        Ok(())
    }
}
