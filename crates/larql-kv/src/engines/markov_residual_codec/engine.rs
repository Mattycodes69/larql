//! `MarkovResidualCodecEngine` — `KvEngine` implementation.

use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::{s, Array2};

// ── W8.2 helpers (mirror of crate::engines::markov_residual::engine) ─────
// Same shape as markov_residual; pre-allocated doubling-capacity buffers
// for `stored` / `hot_kv` so the dispatch hot path appends in-place
// rather than allocating a fresh Array2 per token (which the flamegraph
// surfaced as 58% of decode CPU pre-W8.2).

fn window_capacity(prompt_len: usize, window_size: Option<usize>) -> usize {
    match window_size {
        Some(w) => prompt_len.max(w),
        None => (prompt_len * 2).max(64),
    }
}

fn grow_capacity_2d(src: &Array2<f32>, len: usize, cap: usize) -> Array2<f32> {
    debug_assert_eq!(src.shape()[0], len, "src shape disagrees with len");
    debug_assert!(cap >= len, "cap {cap} smaller than len {len}");
    let cols = src.shape()[1];
    let mut buf = Array2::<f32>::zeros((cap, cols));
    if len > 0 {
        buf.slice_mut(s![..len, ..]).assign(src);
    }
    buf
}

fn append_row(buf: &mut Array2<f32>, row: &Array2<f32>, len: usize) {
    let cap = buf.shape()[0];
    if len == cap {
        let cols = buf.shape()[1];
        let new_cap = (cap * 2).max(8);
        let mut new_buf = Array2::<f32>::zeros((new_cap, cols));
        new_buf
            .slice_mut(s![..len, ..])
            .assign(&buf.slice(s![..len, ..]));
        *buf = new_buf;
    }
    buf.slice_mut(s![len..len + 1, ..]).assign(row);
}

use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::compute::{rs_decode_step_codec, rs_prefill_codec};
use crate::engines::markov_residual_codec::store::RsStoreCodec;
use crate::engines::markov_residual_codec::walk::{
    rs_decode_step_codec_walk, rs_prefill_codec_walk,
};
use crate::profiler::EngineProfiler;
use crate::{DecodeStageSummary, EngineInfo, KvEngine};

/// `MarkovResidualCodecEngine` — `MarkovResidualEngine` with a codec-encoded
/// cold tier.
pub struct MarkovResidualCodecEngine {
    window_size: Option<usize>,
    codec: ColdResidualCodec,
    store: Option<RsStoreCodec>,
    backend: Box<dyn EngineBackend>,
    profiling: bool,
    profile: EngineProfiler,
    /// W1-GPU: see `MarkovResidualEngine::kv_handle`.
    kv_handle: Option<larql_inference::KvHandle>,
    abs_position: usize,
}

impl MarkovResidualCodecEngine {
    /// Construct with the default CPU backend.
    pub fn new(window_size: Option<usize>, codec: ColdResidualCodec) -> Self {
        Self::with_backend(window_size, codec, cpu_engine_backend())
    }

    /// Construct with an explicit compute backend.
    pub fn with_backend(
        window_size: Option<usize>,
        codec: ColdResidualCodec,
        backend: Box<dyn EngineBackend>,
    ) -> Self {
        Self {
            window_size,
            codec,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
            kv_handle: None,
            abs_position: 0,
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    pub fn codec(&self) -> ColdResidualCodec {
        self.codec
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
    }

    /// W1-GPU: mirrors `MarkovResidualEngine::try_prefill_via_dispatch`.
    /// On the codec engine the prefill payload is identical (stored +
    /// hot_kv from state.h_in / k_new / v_new). The cold tier
    /// (`cold_encoded`) is codec-encoded; on overflow we still
    /// invalidate `cold_kv` because codec round-trip is lossy.
    fn try_prefill_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual_codec::store::EncodedColdLayer;
        use larql_inference::PerLayerDecodeState;
        if !larql_inference::vindex::supports_cached_decode(weights)
            || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
        {
            return None;
        }
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let (hidden, handle) = self.backend.as_ref().coarse_prefill_with_state(
            weights,
            token_ids,
            Some(index),
            Some(&mut state),
        )?;
        if !state.is_complete_for(num_layers) {
            return None;
        }
        let hidden_size = weights.hidden_size;
        // W8.2: pre-allocate doubling-capacity stored / hot_kv buffers.
        // See `markov_residual::engine` for the rationale.
        let prompt_len = token_ids.len();
        let initial_cap = window_capacity(prompt_len, self.window_size);
        // W10 Phase A: consume each layer's handle into an owned
        // Array2; CpuStateHandle moves without a copy.
        let stored: Vec<Array2<f32>> = state
            .h_in_per_layer
            .into_iter()
            .map(|h| grow_capacity_2d(&h.into_array(), prompt_len, initial_cap))
            .collect();
        let hot_kv: Vec<larql_inference::attention::SharedKV> = state
            .k_new_per_layer
            .into_iter()
            .zip(state.v_new_per_layer)
            .map(|(k, v)| {
                (
                    grow_capacity_2d(&k.into_array(), prompt_len, initial_cap),
                    grow_capacity_2d(&v.into_array(), prompt_len, initial_cap),
                )
            })
            .collect();
        // W10 Phase B: when `LARQL_W10_HONLY=1`, drop the hot_kv
        // shadow. Metal's kv cache becomes the K/V source of truth;
        // markov_residual_codec treats hot K/V as derivative (the
        // codec-encoded residuals are canonical, the cached K/V is
        // derivable). See state-policy.md §2.2.
        //
        // W10 Phase C: when window_size is also None, no cold-tier
        // eviction triggers and `rs.stored` is dead weight. Drop it;
        // decode steps request None mask, skipping h_in capture too.
        let drop_hot_kv_shadow = std::env::var("LARQL_W10_HONLY")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(false);
        let drop_stored_shadow = drop_hot_kv_shadow && self.window_size.is_none();
        let stored = if drop_stored_shadow {
            (0..num_layers)
                .map(|_| ndarray::Array2::<f32>::zeros((0, hidden_size)))
                .collect()
        } else {
            stored
        };
        let mut rs = RsStoreCodec {
            stored,
            cold_encoded: None,
            cold_kv: None,
            hot_kv: if drop_hot_kv_shadow {
                None
            } else {
                Some(hot_kv)
            },
            cold_abs_start: 0,
            next_position: prompt_len,
            max_window: self.window_size,
            codec: self.codec,
            hot_len: if drop_stored_shadow { 0 } else { prompt_len },
        };
        // Clip on prefill — overflow encoded into the bf16 cold tier.
        let mut overflow_per_layer: Vec<ndarray::Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
            for overflow in overflow_per_layer.iter() {
                let mut enc = EncodedColdLayer::empty(hidden_size);
                enc.append(self.codec, overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            // Codec is lossy → cold_kv must be recomputed against the
            // decoded bytes on next decode. Leave as None.
            rs.cold_abs_start = 0;
        }
        self.store = Some(rs);
        self.kv_handle = Some(handle);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    /// W1-GPU: codec decode through the dispatch surface. Same shape
    /// as `MarkovResidualEngine::decode_step_via_dispatch` but the
    /// overflow path encodes into `cold_encoded` (bf16) and clears
    /// `cold_kv` so the next step recomputes against the decoded bytes.
    fn decode_step_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        // W10 instrumentation: decode_total wraps the whole step so
        // stage_summary returns Some on the dispatch hot path (mirrors
        // MarkovResidualEngine).
        let t_total = std::time::Instant::now();
        use crate::engines::markov_residual_codec::store::EncodedColdLayer;
        use larql_inference::PerLayerDecodeState;
        use ndarray::Array2;
        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let handle = self.kv_handle.as_mut()?;
        // W10 Phase B/C: same mask selection as MarkovResidualEngine.
        // None when both shadows dropped (windowless), HOnly when only
        // hot_kv dropped, Full otherwise.
        let env_on = std::env::var("LARQL_W10_HONLY")
            .ok()
            .map(|v| v == "1")
            .unwrap_or(false);
        let drop_hot_kv = self
            .store
            .as_ref()
            .map(|s| s.hot_kv.is_none())
            .unwrap_or(false)
            && env_on;
        let drop_stored = self
            .store
            .as_ref()
            .map(|s| s.stored.first().map(|a| a.shape()[0] == 0).unwrap_or(false))
            .unwrap_or(false)
            && env_on;
        let mask = if drop_stored && drop_hot_kv {
            larql_compute::StateDumpMask::None
        } else if drop_hot_kv {
            larql_compute::StateDumpMask::HOnly
        } else {
            larql_compute::StateDumpMask::Full
        };
        // W10 instrumentation: state_capture wraps the backend call.
        let t_capture = std::time::Instant::now();
        let hidden = self.backend.as_ref().coarse_decode_step_with_state_masked(
            weights,
            token_id,
            Some(index),
            handle,
            self.abs_position,
            Some(&mut state),
            mask,
        )?;
        if self.profiling {
            self.profile.state_capture.record(t_capture);
        }
        if !state.is_complete_under(num_layers, mask) {
            self.kv_handle = None;
            return None;
        }
        let mut rs = self.store.take()?;
        // W8.2: append in-place into pre-allocated buffers; doubling
        // happens inside `append_row` only on cap overflow.
        // W10 Phase A: consume handles via into_array() — zero-copy
        // move on the CPU happy path.
        // W10 Phase B: under HOnly the K/V handle vecs are empty;
        // only h_in is consumed. Under None all are empty.
        let len = rs.hot_len;
        let h_handles = std::mem::take(&mut state.h_in_per_layer);
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        let did_append = !matches!(mask, larql_compute::StateDumpMask::None);
        if matches!(mask, larql_compute::StateDumpMask::None) {
            // hot_len deliberately stays 0; see markov_residual.
            drop((h_handles, k_handles, v_handles));
        } else if matches!(mask, larql_compute::StateDumpMask::HOnly) {
            drop((k_handles, v_handles));
            for (layer, h) in h_handles.into_iter().enumerate() {
                let t_mat = std::time::Instant::now();
                let h_arr = h.into_array();
                if self.profiling {
                    self.profile.state_materialise.record(t_mat);
                }
                let t_app = std::time::Instant::now();
                append_row(&mut rs.stored[layer], &h_arr, len);
                if self.profiling {
                    self.profile.state_append.record(t_app);
                }
            }
        } else {
            for (layer, ((h, k), v)) in h_handles
                .into_iter()
                .zip(k_handles)
                .zip(v_handles)
                .enumerate()
            {
                let t_mat = std::time::Instant::now();
                let h_arr = h.into_array();
                let kv_arrs = if rs.hot_kv.is_some() {
                    Some((k.into_array(), v.into_array()))
                } else {
                    None
                };
                if self.profiling {
                    self.profile.state_materialise.record(t_mat);
                }
                let t_app = std::time::Instant::now();
                append_row(&mut rs.stored[layer], &h_arr, len);
                if let Some(hot_kv) = rs.hot_kv.as_mut() {
                    if let Some((k_arr, v_arr)) = kv_arrs {
                        append_row(&mut hot_kv[layer].0, &k_arr, len);
                        append_row(&mut hot_kv[layer].1, &v_arr, len);
                    }
                }
                if self.profiling {
                    self.profile.state_append.record(t_app);
                }
            }
        }
        if did_append {
            rs.hot_len = len + 1;
        }
        // Clip + bf16-encode the overflow.
        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            match rs.cold_encoded.as_mut() {
                Some(layers) => {
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        layers[layer].append(rs.codec, overflow);
                    }
                }
                None => {
                    let mut layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
                    for overflow in overflow_per_layer.iter() {
                        let mut enc = EncodedColdLayer::empty(hidden_size);
                        enc.append(rs.codec, overflow);
                        layers.push(enc);
                    }
                    rs.cold_encoded = Some(layers);
                }
            }
            // Lossy codec → invalidate cold_kv.
            rs.cold_kv = None;
        }
        self.store = Some(rs);
        self.abs_position += 1;
        if self.profiling {
            self.profile.decode_total.record(t_total);
        }
        Some(hidden)
    }
}

impl KvEngine for MarkovResidualCodecEngine {
    fn name(&self) -> &str {
        "markov-rs-codec"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w},codec={}", self.codec.label()),
            None => format!("window=full,codec={}", self.codec.label()),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs-codec".into(),
            description: format!(
                "residual-stream KV replacement with {} cold codec (mem={:.1}MB)",
                self.codec.label(),
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let result = rs_prefill_codec(
            weights,
            token_ids,
            self.window_size,
            self.codec,
            self.backend.as_ref(),
        );
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let rs = self.store.take()?;
        let (hidden, new_rs) = rs_decode_step_codec(weights, token_id, rs, self.backend.as_ref())?;
        self.store = Some(new_rs);
        Some(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.total_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn prefill_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        // W1-GPU path: try the dispatch route first (see
        // `MarkovResidualEngine::try_prefill_via_dispatch` for the design
        // notes). Same shape: prefill captures per-layer h_in / K_new /
        // V_new in one backend call; engine reads the dump.
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Some(hidden);
        }
        ensure_attn_tensors_dequantised(weights, index);
        let result = rs_prefill_codec_walk(
            weights,
            index,
            token_ids,
            self.window_size,
            self.codec,
            backend,
        );
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        self.kv_handle = None;
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        if self.kv_handle.is_some() {
            return self.decode_step_via_dispatch(weights, index, token_id);
        }
        ensure_attn_tensors_dequantised(weights, index);
        let rs = self.store.take()?;
        let prof = self.profiling.then_some(&mut self.profile);
        let (hidden, new_rs) =
            rs_decode_step_codec_walk(weights, index, token_id, rs, backend, prof)?;
        self.store = Some(new_rs);
        self.abs_position += 1;
        Some(hidden)
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("markov-rs-codec", self.backend.name()))
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────
    //
    // Same pattern as `MarkovResidualEngine::*_via_executor`. The codec
    // cold tier (bf16-encoded) is engine state; the per-layer
    // attention+FFN compute is delegated to the executor. The caller's
    // FFN backend is honored.

    fn prefill_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual::recompute_kv;
        use crate::engines::markov_residual_codec::store::EncodedColdLayer;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::Array2;

        // Per spec §3.4: this engine's state policy (codec cold tier)
        // requires per-layer dispatch. Transparent degrade on fused
        // executor until Phase 3's refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }

        ensure_attn_tensors_dequantised(weights, index);

        let backend = executor.backend();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let hidden_size = weights.hidden_size;
        let mut h = embed_tokens_pub(weights, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            stored.push(h.clone());
            let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
            h = h_out;
        }

        let mut rs = RsStoreCodec {
            hot_len: stored.first().map_or(0, |s| s.shape()[0]),
            stored,
            cold_encoded: None,
            cold_kv: None,
            // Executor path doesn't yet capture K/V; falls back to
            // recompute-from-residuals (W2 follow-up).
            hot_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
            codec: self.codec,
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
            let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                // Round-trip through the codec so cold K/V is computed
                // from the bf16-reconstructed residuals (matches what
                // future decode steps will see).
                let mut tmp = EncodedColdLayer::empty(hidden_size);
                tmp.append(self.codec, overflow);
                let decoded = tmp.decode(self.codec);
                let (k, v) = recompute_kv(weights, &decoded, layer, 0, backend, Some(index))
                    .expect("cold K/V pre-computation failed");
                cold_kv.push((k, v));
                let mut enc = EncodedColdLayer::empty(hidden_size);
                enc.append(self.codec, overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }

        let hidden = {
            use ndarray::s;
            let last = h.shape()[0] - 1;
            h.slice(s![last..=last, ..]).to_owned()
        };
        self.store = Some(rs);
        Some(hidden)
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        use crate::engines::markov_residual::recompute_kv;
        use crate::engines::markov_residual_codec::store::EncodedColdLayer;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::{s, Array2};

        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }

        ensure_attn_tensors_dequantised(weights, index);

        let backend = executor.backend();
        let rs = self.store.take()?;
        let num_layers = weights.num_layers;
        let abs_position = rs.next_position;
        let mut h_new = embed_tokens_pub(weights, &[token_id]);
        let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let h_hot = &rs.stored[layer];
            let s_hot = h_hot.shape()[0];
            let hot_abs_start = abs_position.saturating_sub(s_hot);

            let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold, v_cold) = &cold_kv[layer];
                let (k_hot, v_hot) =
                    recompute_kv(weights, h_hot, layer, hot_abs_start, backend, Some(index))?;
                let c = k_cold.shape()[0];
                let kv_dim = k_cold.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined.slice_mut(s![..c, ..]).assign(k_cold);
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined.slice_mut(s![..c, ..]).assign(v_cold);
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                (k_combined, v_combined)
            } else {
                let (h_full, full_abs_start) = match &rs.cold_encoded {
                    Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                        let decoded = cold_layers[layer].decode(rs.codec);
                        let hidden = h_hot.shape()[1];
                        let mut combined =
                            Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                        combined
                            .slice_mut(s![..decoded.shape()[0], ..])
                            .assign(&decoded);
                        combined
                            .slice_mut(s![decoded.shape()[0].., ..])
                            .assign(h_hot);
                        (combined, rs.cold_abs_start)
                    }
                    _ => (h_hot.clone(), hot_abs_start),
                };
                recompute_kv(
                    weights,
                    &h_full,
                    layer,
                    full_abs_start,
                    backend,
                    Some(index),
                )?
            };

            new_stored.push(h_new.clone());
            let (h_out, _new_kv) =
                executor.run_decode_layer(weights, layer, &h_new, &prior_kv, abs_position, ffn)?;
            h_new = h_out;
        }

        // Append new row + clip overflow into encoded cold tier.
        let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old = stored.shape()[0];
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
            combined.slice_mut(s![..s_old, ..]).assign(stored);
            combined.slice_mut(s![s_old.., ..]).assign(new_row);
            updated_stored.push(combined);
        }

        let mut updated_rs = RsStoreCodec {
            hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
            stored: updated_stored,
            cold_encoded: rs.cold_encoded,
            cold_kv: rs.cold_kv,
            hot_kv: rs.hot_kv,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
            codec: rs.codec,
        };

        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(updated_rs.clip_layer_overflow(layer));
        }
        updated_rs.finalise_hot_len_after_clip();
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            match updated_rs.cold_encoded.as_mut() {
                Some(layers) => {
                    for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                        layers[layer].append(updated_rs.codec, overflow);
                    }
                }
                None => {
                    let hidden = weights.hidden_size;
                    let mut layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
                    for overflow in overflow_per_layer.iter() {
                        let mut enc = EncodedColdLayer::empty(hidden);
                        enc.append(updated_rs.codec, overflow);
                        layers.push(enc);
                    }
                    updated_rs.cold_encoded = Some(layers);
                }
            }
            updated_rs.cold_kv = None;
        }

        let last = h_new.shape()[0] - 1;
        let out = h_new.slice(s![last..=last, ..]).to_owned();
        self.store = Some(updated_rs);
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::markov_residual::MarkovResidualEngine;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn engine_name_is_markov_rs_codec() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.name(), "markov-rs-codec");
    }

    #[test]
    fn engine_info_reports_codec_and_window() {
        let eng = MarkovResidualCodecEngine::new(Some(128), ColdResidualCodec::Bf16);
        let info = eng.info();
        assert!(info.config.contains("window=128"));
        assert!(info.config.contains("codec=bf16"));
        assert!(info.description.contains("bf16"));
    }

    #[test]
    fn engine_info_unbounded_window() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let info = eng.info();
        assert!(info.config.contains("window=full"));
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn codec_accessor_returns_configured_codec() {
        let eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert_eq!(eng.codec(), ColdResidualCodec::Bf16);
    }

    // ── Prefill / decode ──────────────────────────────────────────────────────

    #[test]
    fn prefill_populates_store_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = eng.prefill(&weights, &ffn, &[0u32, 1, 2]).expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(eng.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = eng.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        assert!(eng.decode_step(&weights, &ffn, 0).is_none());
    }

    #[test]
    fn multiple_decode_steps_produce_consistent_shapes() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        for step in 0..3 {
            let h = eng
                .decode_step(&weights, &ffn, step as u32)
                .expect("decode");
            assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
        }
    }

    // ── Cold tier ─────────────────────────────────────────────────────────────

    #[test]
    fn windowed_prefill_creates_codec_encoded_cold_tier() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(eng.window_tokens() <= 2);
        assert!(
            eng.cold_bytes() > 0,
            "cold tier should be non-empty after overflow"
        );
    }

    #[test]
    fn encoded_cold_payload_is_half_of_f32_equivalent() {
        // Memory contract: bf16 cold payload is exactly 50% the size of an
        // f32 residual tier for the same positions. cold_bytes also bundles
        // cold_kv (which is K/V tensors, not residuals) — we measure the
        // payload directly via the store.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(Some(1), ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4])
            .expect("prefill 5 tokens");
        let store = eng.store.as_ref().expect("store populated after prefill");
        let n_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let cold_positions = 4; // 5 tokens, window=1
        let f32_equivalent_payload = cold_positions * n_layers * hidden * 4;
        let payload: usize = store
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let expected_bf16_payload = cold_positions * n_layers * hidden * 2;
        assert_eq!(
            payload, expected_bf16_payload,
            "bf16 payload should be exactly 2 bytes per element × {cold_positions} × {n_layers} × {hidden}"
        );
        assert_eq!(
            payload * 2,
            f32_equivalent_payload,
            "bf16 cold payload should be exactly half of f32-equivalent"
        );
    }

    #[test]
    fn memory_grows_with_each_decode_step() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        eng.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let m0 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 1).expect("decode 1");
        let m1 = eng.memory_bytes();
        eng.decode_step(&weights, &ffn, 2).expect("decode 2");
        let m2 = eng.memory_bytes();
        assert!(m1 > m0);
        assert!(m2 > m1);
    }

    // ── Bf16 codec contract: bounded KL vs MarkovResidualEngine ───────────────

    #[test]
    fn bf16_output_is_close_to_markov_residual_baseline() {
        // The contract is "bounded KL", not bit-identity. Bf16 introduces
        // round-off on cold residuals; with the test fixture this stays
        // within a small per-element bound.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut baseline = MarkovResidualEngine::new(Some(2));
        let mut codec_eng = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        baseline
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("baseline prefill");
        codec_eng
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("codec prefill");
        let h_b = baseline.decode_step(&weights, &ffn, 4).expect("baseline");
        let h_c = codec_eng.decode_step(&weights, &ffn, 4).expect("codec");
        assert_eq!(h_b.shape(), h_c.shape());
        // Bf16 cold tier should leave the live forward pass within bf16
        // precision on average.
        let max_abs: f32 = h_b
            .iter()
            .zip(h_c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_baseline_abs: f32 = h_b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // 5% relative + small absolute tolerance is generous for a test
        // fixture; production calibration would tighten this.
        assert!(
            max_abs < max_baseline_abs * 0.05 + 1e-2,
            "max_abs={max_abs} exceeded tolerance (baseline max_abs={max_baseline_abs})"
        );
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // On a CPU backend, `quant_prefill_metal` (= `fused_prefill`) returns
    // `None` for the synthetic vindex (no interleaved-Q4K FFN bytes), so
    // the engine falls through to `rs_prefill_codec_walk`. Same pattern
    // `MarkovResidualEngine::prefill_quant_cpu_fallback_runs_walk_path`
    // uses to exercise its CPU walk path.

    #[test]
    fn prefill_quant_cpu_fallback_runs_walk_path() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_cpu_fallback_extends_store() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "store should grow after decode_step_quant"
        );
    }

    #[test]
    fn prefill_quant_with_window_populates_encoded_cold_tier() {
        // Drive the walk path with a window small enough to force overflow
        // into the codec-encoded cold tier (lines 149-152 of engine.rs).
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill_quant with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(
            engine.cold_bytes() > 0,
            "windowed prefill_quant should populate the bf16 cold tier"
        );
    }

    #[test]
    fn decode_step_quant_without_prefill_returns_none() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        // No prefill → store is None → decode_step_quant takes the None
        // branch on `self.store.take()` and returns None.
        assert!(engine
            .decode_step_quant(&mut weights, &ffn, &index, 0, &*backend)
            .is_none());
    }

    #[test]
    fn unbounded_codec_matches_markov_residual_when_no_overflow() {
        // With window=None and prompt small enough to never overflow, the
        // cold codec is never applied. Output should match
        // MarkovResidualEngine bit-for-bit.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut baseline = MarkovResidualEngine::new(None);
        let mut codec_eng = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        baseline
            .prefill(&weights, &ffn, &[0u32, 1])
            .expect("baseline");
        codec_eng
            .prefill(&weights, &ffn, &[0u32, 1])
            .expect("codec");
        let h_b = baseline.decode_step(&weights, &ffn, 2).expect("baseline");
        let h_c = codec_eng.decode_step(&weights, &ffn, 2).expect("codec");
        assert_eq!(h_b, h_c);
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────

    /// Same `CountingFfn` pattern as the markov_residual migration —
    /// proves the codec engine's executor path dispatches FFN through
    /// the caller's backend.
    struct CountingFfn {
        calls: std::sync::atomic::AtomicUsize,
        hidden: usize,
    }
    impl larql_inference::ffn::FfnBackend for CountingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ndarray::Array2::zeros((x.shape()[0], self.hidden))
        }
        fn forward_with_activation(
            &self,
            layer: usize,
            x: &ndarray::Array2<f32>,
        ) -> (ndarray::Array2<f32>, ndarray::Array2<f32>) {
            let out = self.forward(layer, x);
            (out.clone(), out)
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[test]
    fn prefill_quant_via_executor_runs_and_honors_ffn() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(
            ffn.calls.load(std::sync::atomic::Ordering::SeqCst),
            weights.num_layers,
            "codec engine should dispatch FFN through the supplied backend"
        );
    }

    #[test]
    fn decode_step_quant_via_executor_extends_store() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    #[test]
    fn executor_path_populates_codec_cold_tier_under_window() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        // window=2, prefill 4 tokens → overflow → cold tier populates
        // through the codec (bf16).
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill with overflow");
        assert!(engine.window_tokens() <= 2);
        assert!(
            engine.cold_bytes() > 0,
            "executor-driven prefill should populate the bf16 cold tier under window cap"
        );
    }

    /// W2 fast path for the codec engine: both cold_kv AND hot_kv
    /// cached. Drives the triple-condition branch in
    /// `rs_decode_step_codec_walk` that memcpy-concatenates the
    /// cached cold tier with the cached hot tier.
    #[test]
    fn decode_step_quant_w2_codec_cached_hot_and_cold_steady_state() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill with overflow");
        assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
        assert!(engine.store.as_ref().unwrap().cold_kv.is_some());
        for tok in 4u32..7 {
            // First decode goes through cached cold+hot, then overflow
            // invalidates cold_kv (codec is lossy), so next decode
            // takes the recompute-via-cold_encoded path. Second decode
            // then has cold_kv populated again (lazy rebuild via
            // recompute) — exercises both sides of the codec's
            // post-overflow flow.
            let _ = engine
                .decode_step_quant(&mut weights, &ffn, &index, tok, &*backend)
                .expect("decode");
        }
    }

    /// Drive the codec engine's fallback when hot_kv is None
    /// (pre-W2 / via_executor path). Covers the cached-cold-only
    /// arm with manual hot_kv invalidation.
    #[test]
    fn decode_step_quant_w2_codec_falls_back_when_hot_kv_dropped() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill");
        engine.store.as_mut().unwrap().hot_kv = None;
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 4, &*backend)
            .expect("decode via fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive `rs_decode_step_codec_walk`'s `Some(profiler)` arms —
    /// stage_summary returns Some only after with_profiling(true) AND
    /// at least one decode step on the Q4K path.
    #[test]
    fn decode_step_codec_walk_with_profiling_populates_summary() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut eng =
            MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16).with_profiling(true);
        eng.prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill");
        eng.decode_step_quant(&mut weights, &ffn, &index, 4, &*backend)
            .expect("decode 1");
        eng.decode_step_quant(&mut weights, &ffn, &index, 5, &*backend)
            .expect("decode 2");
        let summary = eng
            .stage_summary()
            .expect("codec walk profiler should populate summary");
        assert_eq!(summary.engine, "markov-rs-codec");
        assert!(summary.steps >= 2);
        assert!(summary.avg_attention_us > 0.0);
        assert!(summary.avg_ffn_us > 0.0);
    }

    /// Decode through the executor with cold_kv pre-computed by the
    /// windowed prefill (lines ~321-333 of engine.rs).
    #[test]
    fn decode_via_executor_uses_cold_kv_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill overflow");
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 4)
            .expect("decode via cold_kv");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive the cold_encoded recompute branch (lines ~336-348):
    /// after the first decode overflows and clears cold_kv, the next
    /// decode recomputes K/V from the bf16-encoded cold residuals.
    #[test]
    fn decode_via_executor_hits_cold_encoded_branch() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill");
        engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 4)
            .expect("first decode clears cold_kv");
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 5)
            .expect("decode via cold_encoded recompute");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Fused-executor fallback: lines 223-224 / 303-304 dispatch back
    /// through `prefill_quant` / `decode_step_quant`.
    struct FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
            larql_inference::layer_executor::ExecutorDispatchKind::Fused
        }
        fn name(&self) -> &str {
            "fused-stub"
        }
    }

    #[test]
    fn fused_executor_falls_back_to_legacy_quant_path() {
        use larql_inference::ffn::NullFfn;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let exec = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant_via_executor(&mut weights, &exec, &ffn, &index, &[0u32, 1])
            .expect("fused fallback prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        let h2 = engine
            .decode_step_quant_via_executor(&mut weights, &exec, &ffn, &index, 2)
            .expect("fused fallback decode");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
    }

    // ── Q4K dispatch path (try_prefill_via_dispatch + decode_step_via_dispatch) ──
    //
    // Mirrors the markov_residual sibling — CpuBackend's
    // `coarse_prefill_with_state` / `coarse_decode_step_with_state_masked`
    // fire on Q4K-backed vindexes, taking the engine down the W1-GPU
    // dispatch path on CPU. The cold tier here is codec-encoded
    // (bf16 round-trip) rather than raw f32.

    #[test]
    fn prefill_via_dispatch_with_q4k_vindex_populates_store_and_handle() {
        if std::env::var("LARQL_W10_HONLY").as_deref() == Ok("1") {
            // W10 opt-in drops the shadow; this test pins Full-mode
            // population. Add a separate test for the None-mask path.
            return;
        }
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        let h = engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("dispatch-path prefill on Q4K vindex");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.kv_handle.is_some());
        assert!(engine.memory_bytes() > 0);
        let store = engine.store.as_ref().expect("store populated");
        assert_eq!(store.next_position, 3);
        assert!(store.hot_kv.is_some());
        // No window → no overflow → codec-encoded cold tier stays None.
        assert!(store.cold_encoded.is_none());
        assert!(store.cold_kv.is_none());
    }

    #[test]
    fn decode_via_dispatch_grows_buffers_in_place() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill dispatch");
        let mem_after_prefill = engine.memory_bytes();
        for tok in 2..6u32 {
            let h = engine
                .decode_step_quant(&mut weights, &ffn, &index, tok, &*backend)
                .expect("dispatch decode step");
            assert_eq!(h.shape(), &[1, weights.hidden_size]);
        }
        assert!(engine.memory_bytes() >= mem_after_prefill);
        assert!(engine.kv_handle.is_some());
    }

    #[test]
    fn dispatch_decode_with_profiling_runs_w10_state_capture_branches() {
        // The codec dispatch path bumps state_capture / state_materialise /
        // state_append but not decode_total (decode_total is the walk-path
        // accumulator). `stage_summary` gates on decode_total > 0, so it
        // returns None on the pure-dispatch path — this test only asserts
        // that the profiling branches don't panic and produce finite
        // logits.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine =
            MarkovResidualCodecEngine::new(None, ColdResidualCodec::Bf16).with_profiling(true);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode 1");
        assert!(h.iter().all(|v| v.is_finite()));
        let h2 = engine
            .decode_step_quant(&mut weights, &ffn, &index, 3, &*backend)
            .expect("decode 2");
        assert!(h2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dispatch_prefill_with_window_evicts_to_codec_cold_tier() {
        // Window < prompt_len triggers the on-prefill eviction branch
        // inside try_prefill_via_dispatch. Evicted residuals get
        // bf16-encoded into cold_encoded; cold_kv is intentionally left
        // None — the codec is lossy, so next-step K/V must be recomputed
        // from the decoded payload.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
            .expect("prefill with window");
        let store = engine.store.as_ref().expect("store");
        assert!(store.cold_encoded.is_some());
        // Codec engine deliberately keeps cold_kv = None on overflow
        // (decoded bf16 ≠ original f32, so any raw K/V would diverge).
        assert!(store.cold_kv.is_none());
        assert!(engine.window_tokens() <= 2);
        assert!(engine.cold_bytes() > 0);
        let cold = store.cold_encoded.as_ref().unwrap();
        assert_eq!(cold[0].n_positions, 2);
    }

    #[test]
    fn dispatch_decode_overflow_appends_to_codec_cold_tier() {
        // Each decode step beyond the window evicts one residual row
        // into the cold tier. Exercises the codec-side post-decode
        // overflow handler: encode + append to cold_encoded payload,
        // optionally extend cold_kv.
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = MarkovResidualCodecEngine::new(Some(2), ColdResidualCodec::Bf16);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill at window cap");
        for tok in 3..6u32 {
            engine
                .decode_step_quant(&mut weights, &ffn, &index, tok, &*backend)
                .expect("dispatch decode with codec overflow");
        }
        let store = engine.store.as_ref().expect("store");
        assert!(store.cold_encoded.is_some());
        let cold = store.cold_encoded.as_ref().unwrap();
        // Prefill evicted 1 row; each of 3 decode steps evicted 1 more.
        assert!(cold[0].n_positions >= 3);
        assert!(engine.window_tokens() <= 2);
    }
}
