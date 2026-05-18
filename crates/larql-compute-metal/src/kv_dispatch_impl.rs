//! `KvDispatch` implementation for `crate::MetalBackend` — Step 4
//! scaffolding.
//!
//! **Behaviour:** every method delegates to
//! [`larql_compute::CpuBackend`]'s [`KvDispatch`] impl. K/V handles are
//! CPU-resident (host memory). No real GPU compute — the goal of this
//! step is to exercise the trait shape against actual Metal types so
//! engines can migrate to dispatch-through-trait safely on both
//! backends (Step 3c).
//!
//! Tok/s impact: catastrophically worse than the current Metal path
//! (every call has the same cost as CpuBackend). Acceptance criterion
//! is correctness, not speed. Real Metal kernels land in Step 5; this
//! file is the place where they bind.
//!
//! Feature-gated behind `metal` (same as `crate::MetalBackend`).

use ndarray::Array2;

use crate::MetalBackend;
use larql_compute::kv_dispatch::{
    CompressionCodec, KvDispatch, KvHandle, KvHandleInner, ResidualHandle,
};
use larql_compute::CpuBackend;
use larql_models::ModelWeights;

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

impl KvDispatch for MetalBackend {
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        // Handles are CPU-resident at Step 4. When real Metal kernels land
        // (Step 5), this returns a `MetalKvHandle` wrapping an
        // `MTLBuffer` instead.
        CPU.alloc_kv_buffer(layer, max_tokens, kv_dim)
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        CPU.append_kv(handle, k_row, v_row, abs_position);
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        CPU.clip_kv(handle, window_size);
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        CPU.read_kv_to_host(handle)
    }

    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<Array2<f32>> {
        // A3 scaffold delegates to CPU. A4/A6 will introduce a Q4K-native
        // Metal path when `index` is `Some` and Q4K data is available.
        CPU.attention_step(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<Array2<f32>> {
        CPU.attention_step_windowed(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.attention_prefill(weights, tokens_embedded, layer, window, index)
    }

    fn recompute_kv_from_residuals(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        CPU.recompute_kv_from_residuals(weights, residuals, layer)
    }

    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        CPU.compressed_kv_append(handle, k, v, codec);
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        // CPU-resident upload. When Step 5 lands the pipelined boundary
        // upload kernel, this returns a `MetalResidualHandle` instead.
        CPU.upload_boundary_residual(residual)
    }

    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        CPU.forward_from_layer(weights, start_layer, residuals, token_ids)
    }

    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        CPU.residual_norm_store(x, residual, norm_weights)
    }

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through Metal's fused `prefill_kquant` / `decode_token` kernels
    // — the production Metal hot path that powers `larql bench` at
    // ~87–100 tok/s on Gemma 3 4B Q4K. K/V cache state lives inside
    // `MetalBackend`'s internal `kv_cache` mutex; the returned
    // `KvHandle` is a sentinel since the engine doesn't manage the
    // state directly.

    // ── Coarse fused intents (Q4_K path) ──────────────────────────────────
    //
    // Route through compute's `kquant_forward::fused_*` helpers
    // (ADR-0022 Step 7). The helpers internally call
    // `backend.prefill_kquant` / `backend.decode_token_with_state_dump`
    // — both DecodeBackend methods that Metal overrides with real
    // fused kernels. K/V cache state lives inside `MetalBackend`'s
    // internal mutex; the returned `KvHandle` is a sentinel.

    fn coarse_prefill(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let index = index?;
        let hidden = larql_compute::kquant_forward::fused_prefill(weights, index, token_ids, self)?;
        Some((hidden, KvHandle::new(MetalCoarseHandle)))
    }

    fn coarse_prefill_with_state(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&dyn larql_compute::KvIndex>,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let index = index?;
        let Some(state) = state else {
            return self.coarse_prefill(weights, token_ids, Some(index));
        };
        if token_ids.is_empty() {
            return None;
        }
        // Iterative Metal prefill: run `fused_decode_step_with_state`
        // per prefill token, accumulating per-layer state into the
        // pre-allocated `PerLayerDecodeState` Array2s. Replaces the
        // ~2.7s CPU walk (`predict_kquant_prefill_with_state`) with
        // ~12 ms × seq_len of pure Metal dispatch — 40× speedup on
        // 5-token prompts. The Metal KV cache is populated by the
        // decode kernel itself as a side effect, so no separate
        // `fused_prefill` is needed.

        use larql_compute::DecodeBackend as _;
        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let seq_len = token_ids.len();
        let arch = &*weights.arch;

        // Pre-allocate per-layer Array2s sized for the full prefill.
        // Engine-facing contract: each layer's entry is
        // `[seq_len, hidden]` (or `[seq_len, kv_dim]`).
        state.h_in_per_layer = (0..num_layers)
            .map(|_| Array2::<f32>::zeros((seq_len, hidden_size)))
            .collect();
        let kv_dims: Vec<usize> = (0..num_layers)
            .map(|l| arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l))
            .collect();
        state.k_new_per_layer = (0..num_layers)
            .map(|l| Array2::<f32>::zeros((seq_len, kv_dims[l])))
            .collect();
        state.v_new_per_layer = (0..num_layers)
            .map(|l| Array2::<f32>::zeros((seq_len, kv_dims[l])))
            .collect();

        // Reset + preallocate the Metal KV cache once before the loop.
        self.reset_kv_cache();
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        self.preallocate_kv_cache_per_layer(
            &kv_shapes,
            larql_compute::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ,
        );

        let mut last_hidden: Option<Array2<f32>> = None;
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let mut dump = larql_compute::DecodeStateDump::with_capacity(num_layers);
            let h_arr = larql_compute::kquant_forward::fused_decode_step_with_state(
                weights, index, token_id, self, &mut dump,
            )?;

            // Bridge dump → engine state: write captured per-layer
            // (h_in, k_new, v_new) into pre-allocated row `pos`.
            // Range loop is clearer than enumerate() here because we
            // index five parallel collections by `layer`.
            #[allow(clippy::needless_range_loop)]
            for layer in 0..num_layers {
                let h_layer = std::mem::take(&mut dump.h_in_per_layer[layer]);
                let k_layer = std::mem::take(&mut dump.k_new_per_layer[layer]);
                let v_layer = std::mem::take(&mut dump.v_new_per_layer[layer]);
                if h_layer.len() != hidden_size
                    || k_layer.len() != kv_dims[layer]
                    || v_layer.len() != kv_dims[layer]
                {
                    // Kernel didn't populate this layer (defensive guard).
                    // Caller's `is_complete_for` check will catch it.
                    return None;
                }
                let mut h_row = state.h_in_per_layer[layer].row_mut(pos);
                for (j, v) in h_layer.iter().enumerate() {
                    h_row[j] = *v;
                }
                let mut k_row = state.k_new_per_layer[layer].row_mut(pos);
                for (j, v) in k_layer.iter().enumerate() {
                    k_row[j] = *v;
                }
                let mut v_row = state.v_new_per_layer[layer].row_mut(pos);
                for (j, v) in v_layer.iter().enumerate() {
                    v_row[j] = *v;
                }
            }

            if pos == seq_len - 1 {
                last_hidden = Some(h_arr);
            }
        }

        Some((last_hidden?, KvHandle::new(MetalCoarseHandle)))
    }

    fn coarse_decode_step(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        _handle: &mut KvHandle,
        _abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        larql_compute::kquant_forward::fused_decode_step(weights, index, token_id, self)
    }

    fn coarse_decode_step_with_state(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&dyn larql_compute::KvIndex>,
        _handle: &mut KvHandle,
        _abs_position: usize,
        state: Option<&mut larql_compute::PerLayerDecodeState>,
    ) -> Option<Array2<f32>> {
        let index = index?;
        let Some(state) = state else {
            return larql_compute::kquant_forward::fused_decode_step(
                weights, index, token_id, self,
            );
        };
        // Bridge engine-facing `PerLayerDecodeState` and substrate
        // `DecodeStateDump`. Metal's `decode_token_with_state_dump`
        // populates the dump; we then unflatten per-layer entries
        // back into `Array2<f32>` for the engine.
        let mut dump = larql_compute::DecodeStateDump::with_capacity(weights.num_layers);
        let hidden = larql_compute::kquant_forward::fused_decode_step_with_state(
            weights, index, token_id, self, &mut dump,
        )?;
        let num_layers = weights.num_layers;
        if dump.h_in_per_layer.len() != num_layers
            || dump.k_new_per_layer.len() != num_layers
            || dump.v_new_per_layer.len() != num_layers
        {
            // Kernel didn't populate per-layer entries (defensive guard).
            return Some(hidden);
        }
        let hidden_size = weights.hidden_size;
        for layer in 0..num_layers {
            let h_vec = std::mem::take(&mut dump.h_in_per_layer[layer]);
            let k_vec = std::mem::take(&mut dump.k_new_per_layer[layer]);
            let v_vec = std::mem::take(&mut dump.v_new_per_layer[layer]);
            let kv_dim = k_vec.len();
            state
                .h_in_per_layer
                .push(Array2::from_shape_vec((1, hidden_size), h_vec).ok()?);
            state
                .k_new_per_layer
                .push(Array2::from_shape_vec((1, kv_dim), k_vec).ok()?);
            state
                .v_new_per_layer
                .push(Array2::from_shape_vec((1, kv_dim), v_vec).ok()?);
        }
        Some(hidden)
    }
}

/// Sentinel `KvHandleInner` for `MetalBackend::coarse_prefill` — the
/// actual K/V state lives in `MetalBackend`'s internal `kv_cache`
/// mutex, populated by the fused `prefill_kquant` / `decode_token` kernels.
/// The handle exists to satisfy the trait shape; engines must treat it
/// opaquely.
pub struct MetalCoarseHandle;

impl KvHandleInner for MetalCoarseHandle {
    fn cached_len(&self) -> usize {
        // Backend-side state; not exposed through the handle. Engines
        // that need the cache length should query the backend directly.
        0
    }
    fn kv_dim(&self) -> usize {
        0
    }
    fn backend_name(&self) -> &'static str {
        "metal-coarse"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// `KvHandleInner` and `ResidualHandleInner` placeholders for the
// per-layer dispatch path are not needed at Step 4 — we reuse
// `CpuKvHandle` and `CpuResidualHandle` from the CPU module since
// handles are host-resident. Step 5 will introduce `MetalKvHandle`
// (wrapping `MTLBuffer`) once real per-layer Metal compute lands.
