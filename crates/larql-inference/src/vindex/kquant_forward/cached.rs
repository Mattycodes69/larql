//! KV-cached CPU Q4_K decode — the `VectorIndex`-typed adapter over the
//! substrate forward in [`larql_compute::kquant_forward`].
//!
//! Every function here coerces `&VectorIndex` to `&dyn larql_compute::KvIndex`
//! and delegates: larql-compute owns the single implementation of the Q4_K CPU
//! prefill/decode (ADR-0022), including the q4k-direct FFN prefill. This module
//! exists only to keep a stable, `VectorIndex`-typed API and the local
//! `CachedTimings` type for callers that hold a concrete `VectorIndex`.
//!
//! The unique, non-delegated Q4_K paths — `predict_kquant_hidden`, the OV/RD
//! interventions, Metal, MoE, and streaming generation — live in the sibling
//! modules of `vindex::kquant_forward`, not here.

#![allow(clippy::type_complexity)]

use larql_compute::ComputeBackend;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

/// Per-layer K/V captured during prefill. One entry per layer; matches
/// the [`crate::attention::decode::KvCache`] convention so future work
/// can swap in window clipping or surgery without churn here.
pub type CpuKvCache = Vec<Option<(Array2<f32>, Array2<f32>)>>;

/// Timing instrumentation for the cached CPU Q4K path. Times are
/// summed across all layers in a single call (prefill = one call;
/// decode = one call per generated token).
#[derive(Debug, Default, Clone, Copy)]
pub struct CachedTimings {
    pub dequant_ms: f64,
}

impl CachedTimings {
    fn merge(&mut self, other: CachedTimings) {
        self.dequant_ms += other.dequant_ms;
    }
}

/// True if the cached decode loop can handle this model. False for
/// hybrid-MoE (router/expert path runs through `run_moe_layer_cpu`)
/// and for architectures with cross-layer KV sharing (the decode-step
/// attention helper only knows the "this layer has its own K/V" case
/// today).
pub fn supports_cached_decode(weights: &ModelWeights) -> bool {
    larql_compute::kquant_forward::supports_cached_decode(weights)
}

/// Prefill: run the full prompt through every layer once, capturing
/// each layer's post-RoPE K and final V into the returned cache.
/// Returns the `[seq_len, hidden]` hidden state and the populated
/// cache. Caller takes the last row for lm_head.
pub fn predict_kquant_prefill(
    weights: &mut ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    predict_kquant_prefill_with_state(weights, token_ids, index, None)
}

/// Prefill with optional per-layer state capture (W1-GPU step 3
/// sibling of [`predict_kquant_decode_step_direct_with_state`]). When
/// `state` is `Some`, populates per-layer `h_in` ([seq_len, hidden]),
/// `k_new` ([seq_len, kv_dim]), `v_new` ([seq_len, kv_dim]) for every
/// position in the prompt — engines (markov_residual,
/// windowed_checkpoint, turbo_quant) use this to seed their state policy
/// from a single prefill pass without a follow-up CPU re-walk. When
/// `state` is `None`, bit-identical to [`predict_kquant_prefill`].
pub fn predict_kquant_prefill_with_state(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    state: Option<&mut crate::PerLayerDecodeState>,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    // Delegate to the substrate copy in larql-compute — the single source of
    // truth for the Q4_K CPU forward (ADR-0022). `VectorIndex` satisfies
    // `larql_compute::KvIndex`; the only impedance is `CachedTimings`, a
    // same-shape struct defined in each crate.
    let (h, cache, timings) = larql_compute::kquant_forward::predict_kquant_prefill_with_state(
        weights,
        token_ids,
        index as &dyn larql_compute::KvIndex,
        state,
    );
    (
        h,
        cache,
        CachedTimings {
            dequant_ms: timings.dequant_ms,
        },
    )
}

/// Decode step: run a single new token through every layer using the
/// prefill cache. Each layer's cache entry is appended to in place.
/// Returns the new `[1, hidden]` hidden state for lm_head.
///
/// `abs_position` is the absolute RoPE position of the new token —
/// `prompt_len + steps_already_decoded`. The caller maintains this
/// counter (typical: `prompt_len + step_index` starting at 0).
pub fn predict_kquant_decode_step(
    weights: &ModelWeights,
    token_id: u32,
    index: &VectorIndex,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<(Array2<f32>, CachedTimings)> {
    // Delegate to the substrate copy in larql-compute (ADR-0022); bridge the
    // per-crate `CachedTimings`.
    larql_compute::kquant_forward::predict_kquant_decode_step(
        weights,
        token_id,
        index as &dyn larql_compute::KvIndex,
        cache,
        abs_position,
    )
    .map(|(h, timings)| {
        (
            h,
            CachedTimings {
                dequant_ms: timings.dequant_ms,
            },
        )
    })
}

impl CachedTimings {
    /// Merge another timing block into self. Useful for accumulating
    /// per-step decode timings across a generation loop.
    pub fn add(&mut self, other: CachedTimings) {
        self.merge(other);
    }
}

// ── Phase 2: dequant-free decode step ───────────────────────────────────
//
// `predict_kquant_decode_step` (above) still pays the per-step Q4_K/Q6_K →
// f32 dequant cost via `insert_q4k_layer_tensors`. Profiling showed
// dequant is ~93% of CPU forward time even with the KV cache wired —
// gemm and attention are a small slice. This module routes Q/K/V/O and
// gate/up/down projections straight through `backend.quant_matvec`
// (CPU `q4k_matvec_into` / `q6k_matvec_into`), skipping the dequant
// staging entirely.

/// True when the whole model can run on the direct-matvec decode path.
/// Same gating as [`supports_cached_decode`] plus a per-layer format
/// check. Used by the bench labeler and as the cpu.rs routing key.
pub fn supports_direct_matvec_decode(weights: &ModelWeights, index: &VectorIndex) -> bool {
    larql_compute::kquant_forward::supports_direct_matvec_decode(
        weights,
        index as &dyn larql_compute::KvIndex,
    )
}

/// Fused Q4_K prefill via the compute backend (Metal fast path).
/// Delegates to the larql-compute substrate copy (ADR-0022).
pub fn fused_prefill(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::fused_prefill(
        weights,
        index as &dyn larql_compute::KvIndex,
        token_ids,
        backend,
    )
}

/// Fused Q4_K decode step via the compute backend. Delegates to larql-compute.
pub fn fused_decode_step(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::fused_decode_step(
        weights,
        index as &dyn larql_compute::KvIndex,
        token_id,
        backend,
    )
}

/// Fused Q4_K decode step with a per-layer state dump. Delegates to larql-compute.
pub fn fused_decode_step_with_state(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_id: u32,
    backend: &dyn ComputeBackend,
    state: &mut larql_compute::DecodeStateDump,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::fused_decode_step_with_state(
        weights,
        index as &dyn larql_compute::KvIndex,
        token_id,
        backend,
        state,
    )
}

/// Dequant-free attention decode step (Q4_K/Q6_K x Q8_K direct matvec).
/// Delegates to the larql-compute substrate copy (ADR-0022).
#[allow(clippy::type_complexity)]
pub fn attention_decode_step_native(
    weights: &ModelWeights,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&(Array2<f32>, Array2<f32>)>,
    abs_position: usize,
) -> Option<(Array2<f32>, (Array2<f32>, Array2<f32>))> {
    larql_compute::kquant_forward::attention_decode_step_native(
        weights,
        index as &dyn larql_compute::KvIndex,
        backend,
        h_new,
        layer,
        kv_entry,
        abs_position,
    )
}

/// Dequant-free FFN decode step (gate/up/down via direct Q4_K matvec).
/// Delegates to the larql-compute substrate copy (ADR-0022).
pub fn ffn_decode_step_native(
    weights: &ModelWeights,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    larql_compute::kquant_forward::ffn_decode_step_native(
        weights,
        index as &dyn larql_compute::KvIndex,
        backend,
        h_post_attn,
        layer,
    )
}

/// Dequant-free decode step. Same shape contract as
/// [`predict_kquant_decode_step`] but routes every projection through
/// `backend.quant_matvec` instead of the per-layer
/// `insert_q4k_layer_tensors` → dense f32 staging dance. Returns `None`
/// if any layer has a format the direct-matvec path doesn't handle
/// (caller falls back to [`predict_kquant_decode_step`]).
pub fn predict_kquant_decode_step_direct(
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<Array2<f32>> {
    predict_kquant_decode_step_direct_with_state(
        weights,
        token_id,
        index,
        backend,
        cache,
        abs_position,
        None,
    )
}

/// Decode step with optional per-layer state capture (`Some(state)`
/// populates `h_in` / `k_new` / `v_new` per layer at near-zero cost
/// since this CPU path already walks the layers serially). Engines
/// that need per-layer state — `markov_residual` for residual storage,
/// `markov_residual_codec` ditto, `turbo_quant` for per-layer K/V
/// compression — call through here via `KvDispatch::
/// coarse_decode_step_with_state`. When `state` is `None` this is
/// bit-identical to [`predict_kquant_decode_step_direct`].
pub fn predict_kquant_decode_step_direct_with_state(
    weights: &mut ModelWeights,
    token_id: u32,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
    state: Option<&mut crate::PerLayerDecodeState>,
) -> Option<Array2<f32>> {
    // Delegate to the substrate copy in larql-compute (ADR-0022).
    larql_compute::kquant_forward::predict_kquant_decode_step_direct_with_state(
        weights,
        token_id,
        index as &dyn larql_compute::KvIndex,
        backend,
        cache,
        abs_position,
        state,
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod branch_tests;
