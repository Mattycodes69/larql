//! `KvDispatch` implementation for `crate::CpuBackend`.
//!
//! Lives here (not in `larql-compute`) so the bodies can call into the
//! inference-side forward-pass functions (`run_attention_*`, `run_ffn`,
//! `forward_from_layer`). Orphan rules: the [`KvDispatch`] trait is
//! local to this crate, so implementing it for a foreign type
//! (`CpuBackend`) is allowed.
//!
//! See `docs/specs/compute-backend-redesign.md` §10.2 for the trait-
//! location rationale.
//!
//! ## Implementation strategy
//!
//! - `KvHandle` wraps **a single layer's** K and V tensors. Engines
//!   that need multi-layer caches hold a `Vec<KvHandle>` (one per
//!   layer). This matches the trait's per-layer API
//!   (`alloc_kv_buffer(layer, ...)`).
//! - `ResidualHandle` is a thin wrap around `Array2<f32>` — CPU has no
//!   device memory to manage.
//! - `attention_step` / `attention_prefill` delegate to the existing
//!   `run_attention_*` functions.
//! - `forward_from_layer` delegates to
//!   `crate::forward::forward_from_layer`.
//! - Engine-specific intents (`recompute_kv_from_residuals`,
//!   `compressed_kv_append`) stay at the trait defaults until Step 3
//!   migrates the engines that need them.

use crate::CpuBackend;
use ndarray::Array2;

use super::{KvDispatch, KvHandle, KvHandleInner, ResidualHandle, ResidualHandleInner};
use crate::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use larql_models::ModelWeights;

// ─── CpuKvHandle ────────────────────────────────────────────────────────────

/// Single-layer K/V cache held in host memory. Wraps the existing
/// `SharedKV = (K, V)` shape — `K` and `V` are owned `Array2<f32>`
/// growing by one row per `append_kv` call.
pub struct CpuKvHandle {
    /// Layer index this handle was minted for. Carried for debugging
    /// / future trait surface; not consulted by the current append /
    /// attend paths (the trait already takes `layer` per call).
    #[allow(dead_code)]
    layer: usize,
    kv_dim: usize,
    /// `None` before the first `append_kv` / `attention_prefill`.
    state: Option<SharedKV>,
}

impl CpuKvHandle {
    fn new(layer: usize, kv_dim: usize) -> Self {
        Self {
            layer,
            kv_dim,
            state: None,
        }
    }

    /// Replace the internal state — used by backend impls that
    /// populate the handle from the prefill path (which returns a
    /// fresh `SharedKV` rather than appending incrementally).
    fn replace_state(&mut self, kv: SharedKV) {
        self.state = Some(kv);
    }

    fn as_shared_kv(&self) -> Option<&SharedKV> {
        self.state.as_ref()
    }
}

impl KvHandleInner for CpuKvHandle {
    fn cached_len(&self) -> usize {
        self.state.as_ref().map_or(0, |(k, _)| k.shape()[0])
    }

    fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Downcast helper — backend implementations use this to retrieve the
/// concrete handle type from an opaque `KvHandle`. Panics if the
/// handle was allocated by a different backend.
fn cpu_handle(h: &KvHandle) -> &CpuKvHandle {
    h.as_inner()
        .as_any()
        .downcast_ref::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                h.backend_name()
            )
        })
}

fn cpu_handle_mut(h: &mut KvHandle) -> &mut CpuKvHandle {
    let name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuKvHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign handle (backend={name}); \
                 handles must be allocated by the same backend that consumes them"
            )
        })
}

// ─── CpuResidualHandle ──────────────────────────────────────────────────────

/// Host-resident residual upload. CPU has no device memory to manage,
/// so this is just a flat `Vec<f32>` wrapper. Storing flat matches
/// what `forward_from_layer` consumes (`&[f32]` interpreted as
/// `[seq_len, hidden]` row-major).
pub struct CpuResidualHandle {
    flat: Vec<f32>,
    shape: (usize, usize),
}

impl ResidualHandleInner for CpuResidualHandle {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn backend_name(&self) -> &'static str {
        "cpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn cpu_residual(r: &ResidualHandle) -> &CpuResidualHandle {
    r.as_inner()
        .as_any()
        .downcast_ref::<CpuResidualHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::KvDispatch received a foreign residual handle (backend={}); \
                 handles must be allocated by the same backend that consumes them",
                r.backend_name()
            )
        })
}

// ─── CpuQ4kCacheHandle — Q4K cached-decode handle ──────────────────────────
//
// Wraps the production `CpuKvCache` (per-layer K/V) so it can flow through
// the dispatch trait's `KvHandle` shape. Cache populated by
// `cached_prefill_q4k`; consumed by `cached_decode_step_q4k`.
//
// One handle per engine (not per layer), unlike the legacy `CpuKvHandle`
// (one per layer for the f32 per-layer dispatch path). The two shapes
// coexist because they serve different dispatch granularities.

pub struct CpuQ4kCacheHandle {
    cache: crate::kquant_forward::CpuKvCache,
}

impl KvHandleInner for CpuQ4kCacheHandle {
    fn cached_len(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[0])
            .next()
            .unwrap_or(0)
    }

    fn kv_dim(&self) -> usize {
        self.cache
            .iter()
            .filter_map(|o| o.as_ref())
            .map(|(k, _)| k.shape()[1])
            .next()
            .unwrap_or(0)
    }

    fn backend_name(&self) -> &'static str {
        "cpu-q4k"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn cpu_q4k_cache_mut(h: &mut KvHandle) -> &mut CpuQ4kCacheHandle {
    let backend_name = h.backend_name();
    h.as_inner_mut()
        .as_any_mut()
        .downcast_mut::<CpuQ4kCacheHandle>()
        .unwrap_or_else(|| {
            panic!(
                "CpuBackend::cached_decode_step_q4k received a foreign handle \
                 (backend={backend_name}); handles must be allocated by the same \
                 backend that consumes them"
            )
        })
}

// ─── KvDispatch impl ────────────────────────────────────────────────────────

impl KvDispatch for CpuBackend {
    fn alloc_kv_buffer(&self, layer: usize, _max_tokens: usize, kv_dim: usize) -> KvHandle {
        // `max_tokens` is informational on CPU — we grow the buffer on
        // append rather than pre-allocate. GPU backends will pre-allocate.
        KvHandle::new(CpuKvHandle::new(layer, kv_dim))
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], _abs_position: usize) {
        // `abs_position` is informational on CPU — the K/V buffer is
        // ordered by insertion, and RoPE rotations are applied by the
        // caller (or by attention_step's underlying function).
        let h = cpu_handle_mut(handle);
        debug_assert_eq!(k_row.len(), h.kv_dim);
        debug_assert_eq!(v_row.len(), h.kv_dim);

        let new_k_row = Array2::from_shape_vec((1, k_row.len()), k_row.to_vec())
            .expect("k_row length doesn't match handle's kv_dim");
        let new_v_row = Array2::from_shape_vec((1, v_row.len()), v_row.to_vec())
            .expect("v_row length doesn't match handle's kv_dim");

        h.state = Some(match h.state.take() {
            Some((mut k, mut v)) => {
                k.append(ndarray::Axis(0), new_k_row.view()).unwrap();
                v.append(ndarray::Axis(0), new_v_row.view()).unwrap();
                (k, v)
            }
            None => (new_k_row, new_v_row),
        });
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        let h = cpu_handle_mut(handle);
        if let Some((k, v)) = h.state.as_mut() {
            let rows = k.shape()[0];
            if rows > window_size {
                let start = rows - window_size;
                let k_slice = k.slice(ndarray::s![start..rows, ..]).to_owned();
                let v_slice = v.slice(ndarray::s![start..rows, ..]).to_owned();
                *k = k_slice;
                *v = v_slice;
            }
        }
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        let h = cpu_handle(handle);
        h.state.as_ref().map(|(k, v)| (k.clone(), v.clone()))
    }

    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        _index: Option<&dyn crate::KvIndex>,
    ) -> Option<Array2<f32>> {
        // CpuBackend reads f32 attention tensors out of `weights.tensors`.
        // When the caller has a Q4K `VectorIndex`, it's expected to have
        // already populated `weights.tensors` via
        // `crate::kquant_forward::ensure_attn_tensors_dequantised` before
        // dispatching here. Until phase-3 CPU Q4K matvec kernels land,
        // the `index` parameter is accepted for trait-shape compatibility
        // but not consumed.
        let h = cpu_handle_mut(kv);
        let prior_kv = h.as_shared_kv().cloned();
        let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
            weights,
            query,
            layer,
            prior_kv.as_ref(),
            abs_position,
            Some(self),
        )?;
        h.replace_state(new_kv);
        Some(h_post_attn)
    }

    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        _window: Option<usize>,
        _index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        // See `attention_step` doc for the `_index` convention.
        let (h_post_attn, k_rope, v) =
            run_attention_with_kv_backend(weights, tokens_embedded, layer, Some(self))?;
        let kv_dim = k_rope.shape()[1];
        let mut handle = CpuKvHandle::new(layer, kv_dim);
        handle.replace_state((k_rope, v));
        Some((h_post_attn, KvHandle::new(handle)))
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let s = residual.shape();
        let (rows, cols) = (s[0], s[1]);
        let flat = residual
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| residual.iter().copied().collect());
        Some(ResidualHandle::new(CpuResidualHandle {
            flat,
            shape: (rows, cols),
        }))
    }

    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let r = cpu_residual(residuals);
        let raw =
            crate::forward::forward_from_layer(weights, token_ids, &r.flat, start_layer, None);
        // The returned `RawForward` has `h_pre_norm` shape [seq_len, hidden];
        // engines want the last position's hidden as [1, hidden].
        let h = raw.h_pre_norm;
        let last = h.shape()[0] - 1;
        Some(h.slice(ndarray::s![last..=last, ..]).to_owned())
    }

    // `recompute_kv_from_residuals`, `compressed_kv_append`,
    // `attention_step_windowed`, and `residual_norm_store` use the
    // trait defaults (decomposition / unimplemented). Step 3 engine
    // migration adds overrides when the engines that consume them
    // actually need a CPU body.

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through the production cached-decode pipeline. Backend
    // inspects `index` (when present) and `weights` to pick the right
    // kernel — Q4K matvec today, future quant formats slot in without
    // changing the trait surface or the engine call sites.

    fn coarse_prefill(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        self.coarse_prefill_with_state(weights, token_ids, index, None)
    }

    fn coarse_prefill_with_state(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn crate::KvIndex>,
        state: Option<&mut crate::PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        if token_ids.is_empty() {
            return None;
        }
        let index = index?;
        if !crate::kquant_forward::supports_cached_decode(weights) {
            return None;
        }
        let (h_full, cache, _timings) = crate::kquant_forward::predict_kquant_prefill_with_state(
            weights, token_ids, index, state,
        );
        let last = h_full.shape()[0] - 1;
        let h = h_full.slice(ndarray::s![last..=last, ..]).to_owned();
        let handle = KvHandle::new(CpuQ4kCacheHandle { cache });
        Some((h, handle))
    }

    fn coarse_decode_step(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        let inner = cpu_q4k_cache_mut(handle);
        // Prefer direct-matvec (no per-layer dequant) when supported.
        if crate::kquant_forward::supports_direct_matvec_decode(weights, index) {
            crate::kquant_forward::predict_kquant_decode_step_direct(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
            )
        } else {
            crate::kquant_forward::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }

    /// CPU per-layer decode with optional state capture (W1-GPU step 3).
    /// Threads `Option<&mut PerLayerDecodeState>` into the same direct-
    /// matvec walk; when `Some`, each layer's `h_in` / `k_new` / `v_new`
    /// is captured at zero re-compute cost (the values already flow
    /// through the per-layer loop). Falls back to the plain
    /// `coarse_decode_step` for the non-direct-matvec path — that
    /// path doesn't expose per-layer state today (would need a
    /// `predict_kquant_decode_step_with_state` sibling; deferred until
    /// an engine asks for it on the indirect path).
    fn coarse_decode_step_with_state(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&dyn crate::KvIndex>,
        handle: &mut KvHandle,
        abs_position: usize,
        state: Option<&mut crate::PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        let index = index?;
        let inner = cpu_q4k_cache_mut(handle);
        if crate::kquant_forward::supports_direct_matvec_decode(weights, index) {
            crate::kquant_forward::predict_kquant_decode_step_direct_with_state(
                weights,
                token_id,
                index,
                self,
                &mut inner.cache,
                abs_position,
                state,
            )
        } else {
            // Indirect-matvec path; no state capture wired yet.
            // Drop the state arg and run the standard decode.
            let _ = state;
            crate::kquant_forward::predict_kquant_decode_step(
                weights,
                token_id,
                index,
                &mut inner.cache,
                abs_position,
            )
            .map(|(h, _)| h)
        }
    }
}
