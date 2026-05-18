//! `AsyncComputeBackend` implementation for `crate::MetalBackend`
//! — Step A3 scaffolding.
//!
//! **Behaviour:** every async method delegates to
//! [`larql_compute::CpuBackend`]'s [`AsyncComputeBackend`] impl. Handles
//! are CPU-resident; the in-flight command buffer is conceptual only.
//! No real GPU compute, no deferred dispatch — the goal of this step is
//! to exercise the trait shape against actual `MetalBackend` ownership
//! patterns so engines can migrate to async dispatch safely on both
//! backends in Step A5.
//!
//! Tok/s impact: catastrophically worse than the current Metal fused
//! `decode_token` path (every call has CpuBackend's cost). Acceptance
//! criterion is correctness, not speed. Real deferred dispatch — one
//! `MTLCommandBuffer` per session, commit at engine checkpoints — lands
//! in Step A4. Per-engine specialised shaders land in Step A6.
//!
//! Feature-gated behind `metal` (same as `crate::MetalBackend`).

use ndarray::Array2;

use crate::MetalBackend;
use larql_compute::async_compute_backend::{
    AsyncComputeBackend, AttentionHandle, ResidualUploadHandle,
};
use larql_compute::ffn::FfnBackend;
use larql_compute::kv_dispatch::{KvHandle, ResidualHandle};
use larql_compute::CpuBackend;
use larql_models::ModelWeights;

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

impl AsyncComputeBackend for MetalBackend {
    fn attention_step_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> AttentionHandle {
        // Handles are CPU-resident at Step A3. When Step A4's deferred
        // dispatch lands, this records the intent into an in-flight
        // `MTLCommandBuffer` and returns a `MetalAttentionHandle`.
        CPU.attention_step_async(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> AttentionHandle {
        CPU.attention_step_windowed_async(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill_async(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> (AttentionHandle, KvHandle) {
        CPU.attention_prefill_async(weights, tokens_embedded, layer, window, index)
    }

    fn upload_boundary_residual_async(
        &self,
        residual: &Array2<f32>,
    ) -> (ResidualUploadHandle, ResidualHandle) {
        // CPU-resident upload at Step A3. When Step A6 lands the
        // pipelined boundary-upload kernel (Apollo's win), this returns
        // a `MetalResidualHandle` whose upload fuses with the next
        // attention encode in the same command buffer.
        CPU.upload_boundary_residual_async(residual)
    }

    fn forward_from_layer_async(
        &self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> AttentionHandle {
        CPU.forward_from_layer_async(weights, ffn, start_layer, residuals, token_ids)
    }
}

// `recompute_kv_from_residuals_async` stays at the trait default
// (`unimplemented!()`). MarkovResidual is the only engine that needs
// it; the real Metal K/V-recompute kernel lands in Step A6 alongside
// that engine's migration. CpuBackend's sync `KvDispatch` doesn't
// implement it either, so a CPU-delegating Metal scaffold would just
// surface the same `unimplemented!()`.
