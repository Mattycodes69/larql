//! TurboQuantEngine — WHT + Lloyd-Max K/V cache compression.
//!
//! Algorithm (ICLR 2026 style):
//!   1. Normalize vector → unit norm (store scalar)
//!   2. Walsh-Hadamard rotation (spreads coordinates to Beta distribution)
//!   3. Lloyd-Max scalar quantization (3 or 4 bits per coordinate)
//!   4. Bit-pack indices
//!   5. Decode: unpack → centroids → inverse WHT → rescale
//!
//! The `TurboQuantEngine` wraps this codec around the K/V cache.
//! Every decode path (W1-GPU dispatch, CPU walk, executor-routed,
//! legacy `decode_step`) is **append-only**: each step decompresses
//! the prior cache for attention (O(N)), then encodes ONLY the newly
//! produced K/V row head-by-head and appends its packed bytes
//! (`CompressedLayer::append_row`). Old rows' bytes are never
//! re-encoded — the codec's reconstruction norm is slightly below the
//! stored norm, so a decompress→re-encode cycle would multiply every
//! stored norm by that ratio and compound it across steps.
//!
//! Codec contract is the same in all paths: WHT + Lloyd-Max 3/4-bit
//! per scalar, bit-pack indices; per-row round-trip cos ≈ 0.9954 at
//! 4-bit / ≈ 0.980 at 3-bit on isotropic unit vectors (Gaussian
//! simulation, 2026-07-30).

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::{EngineInfo, KvEngine};
use larql_inference::attention::run_attention_with_kv_backend;
use larql_inference::ffn::{BackendFfn, FfnBackend};
use larql_inference::forward::embed_tokens_pub;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;

mod compressed;
mod quant_cpu;
pub use compressed::*;

// ─── TurboQuant codec ────────────────────────────────────────────────────────

// W1-GPU dispatch methods (`try_prefill_via_dispatch` /
// `decode_step_via_dispatch`) live in [`super::dispatch`] as an
// additional `impl TurboQuantEngine` block. They mutate the
// `pub(super)` fields above.

impl TurboQuantEngine {
    /// Shared body for `decode_step` / `decode_step_resident`.
    ///
    /// **Transactional.** Unlike the residual-canonical engines, this one's
    /// canonical state *is* the K/V: each layer's compressed cache grows
    /// before the FFN gets its chance to refuse, so a step that does not
    /// finish must undo those appends rather than leave a cache holding a
    /// token that produced no output. [`CompressedLayer::truncate_rows`] does
    /// that byte-exactly, and `abs_position` advances only on success — so a
    /// caller who fixes the cause can drive the same token again.
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        // Recorded per layer rather than assumed uniform: nothing promises
        // every layer caches the same number of rows, and a wrong assumption
        // would rewind to a length no layer ever had.
        let entry_rows: Vec<usize> = self.layers.iter().map(|l| l.num_vecs).collect();
        match self.decode_step_appending(weights, ffn, token_id, index) {
            Ok(hidden) => Ok(hidden),
            Err(failure) => {
                for (layer, &rows) in self.layers.iter_mut().zip(&entry_rows) {
                    layer.truncate_rows(rows, &self.tq);
                }
                Err(failure)
            }
        }
    }

    /// The body of a decode step, which appends to `self.layers` as it goes.
    ///
    /// Split out so the rewind above can wrap every exit rather than every
    /// `?` having to remember it.
    fn decode_step_appending(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let mut h = embed_tokens_pub(weights, &[token_id]);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);
        // Codec scratch reused across layers.
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();

        self.require_prefilled(num_layers)?;
        for layer in 0..num_layers {
            // Decompress full prior K/V for attention.
            let prior_kv = self.layers[layer].decompress(&self.tq);

            // Decode step returns updated K/V (prior + new token).
            let (h_post_attn, updated_kv) =
                larql_inference::attention::run_attention_block_decode_step_auto(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h,
                    layer,
                    Some(&prior_kv),
                    abs_position,
                    Some(self.backend.as_ref()),
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "run_attention_block_decode_step_backend returned None".into(),
                })?;

            // Append-only codec path: encode just the new row head-by-
            // head and push onto the existing compressed buffer.
            let layer_slot = &mut self.layers[layer];
            let new_rows = updated_kv.0.shape()[0];
            debug_assert_eq!(new_rows, layer_slot.num_vecs + 1, "decode adds one row");
            let k_last = updated_kv.0.row(new_rows - 1).to_owned();
            let v_last = updated_kv.1.row(new_rows - 1).to_owned();
            layer_slot.append_row(
                k_last.as_slice().expect("k row contig"),
                v_last.as_slice().expect("v row contig"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );

            let bffn = BackendFfn {
                weights,
                backend: self.backend.as_ref(),
            };
            h = crate::engines::layer_ffn_or_moe(
                weights,
                &h_post_attn,
                layer,
                &bffn,
                Some(ffn),
                ple_inputs.get(layer),
            )
            .map_err(EngineError::Execution)?;
        }

        self.abs_position += 1;
        Ok(last_row(&h))
    }
}

impl KvEngine for TurboQuantEngine {
    fn name(&self) -> &str {
        "turbo-quant"
    }

    fn info(&self) -> EngineInfo {
        let mem: usize = self.layers.iter().map(|l| l.memory_bytes()).sum();
        EngineInfo {
            name: "turbo-quant".into(),
            description: format!(
                "{}-bit WHT+Lloyd-Max K/V compression (mem={:.1}MB)",
                self.tq.bits,
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config: format!("bits={}", self.tq.bits),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.validate_block_dims(weights)?;
        let num_layers = weights.num_layers;
        let be = Some(self.backend.as_compute());
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
        // Built into a local, not into `self.layers`: a prefill that refuses
        // partway must leave the engine holding whatever cache it already had
        // rather than a truncated one for a prompt it never finished.
        let mut layers: Vec<CompressedLayer> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let (h_post_attn, k, v) = run_attention_with_kv_backend(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &h,
                layer,
                be,
                None,
            )
            .ok_or_else(|| EngineError::BackendFailure {
                details: "run_attention_with_kv_backend returned None".into(),
            })?;
            layers.push(CompressedLayer::compress(&(k, v), &self.tq));

            let bffn = BackendFfn {
                weights,
                backend: self.backend.as_ref(),
            };
            h = crate::engines::layer_ffn_or_moe(
                weights,
                &h_post_attn,
                layer,
                &bffn,
                Some(ffn),
                ple_inputs.get(layer),
            )
            .map_err(EngineError::Execution)?;
        }

        self.layers = layers;
        self.abs_position = token_ids.len();
        Ok(last_row(&h))
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, None)
    }

    /// Resident-path decode: threads `index` to the attention step's
    /// Q4K-direct route (the non-standard-engine structural-gap fix).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, Some(index))
    }

    fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    fn stage_summary(&self) -> Option<crate::DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("turbo-quant", self.backend.name()))
    }

    /// Quant path: always run the per-layer compression cycle (capture
    /// K/V per layer, WHT+Lloyd-Max encode, decompress prior, etc.).
    /// W1-GPU: when the engine's backend supports `coarse_prefill_with_state`,
    /// route through the dispatch path — backend computes K/V on GPU,
    /// engine compresses the per-layer captured state into
    /// `CompressedLayer` entries. Falls back to the legacy CPU walk
    /// (`prefill_quant_cpu`) for backends without state-capture support.
    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.validate_block_dims(weights)?;
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        self.kv_handle = None;
        let out = self
            .prefill_quant_cpu(weights, index, token_ids, backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "prefill_quant_cpu returned None".into(),
            })?;
        self.abs_position = token_ids.len();
        Ok(out)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if self.kv_handle.is_some() {
            return self
                .decode_step_via_dispatch(weights, index, token_id)
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "decode_step_via_dispatch returned None".into(),
                });
        }
        self.decode_step_quant_cpu(weights, index, token_id, backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "decode_step_quant_cpu returned None".into(),
            })
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // The legacy `prefill_quant_cpu` / `decode_step_quant_cpu` paths construct
    // their own `WalkFfn` and ignore the FFN parameter. The methods below
    // drive the per-layer loop through a caller-supplied `LayerExecutor` and
    // honor the FFN dispatcher — required for `larql bench --ffn
    // http://shard:8080` to route through the remote shard.
    //
    // Compression policy (WHT + Lloyd-Max per layer) is engine state and
    // stays here; only the per-layer compute is delegated.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.validate_block_dims(weights)?;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
        self.layers.clear();

        for layer in 0..num_layers {
            let (h_out, kv) = executor
                .run_prefill_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_prefill_layer returned None".into(),
                })?;
            self.layers.push(CompressedLayer::compress(&kv, &self.tq));
            // `LayerExecutor::run_*_layer` returns attention + bare FFN only
            // (`LocalWalkExecutor`, the sole production impl, ends at
            // `run_ffn`); the PLE + layer_scalar tail is the driving loop's
            // responsibility, mirroring the legacy `kv_prefill_run` sequence.
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        self.abs_position = token_ids.len();
        Ok(last_row(&h))
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let mut h = embed_tokens_pub(weights, &[token_id]);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);
        // Codec scratch reused across layers.
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();

        self.require_prefilled(num_layers)?;
        for layer in 0..num_layers {
            let prior_kv = self.layers[layer].decompress(&self.tq);
            let (h_out, updated_kv) = executor
                .run_decode_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    &prior_kv,
                    abs_position,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_decode_layer returned None".into(),
                })?;
            // Append-only codec path (same structure as `decode_step_impl`
            // and `decode_step_quant_cpu`): only the LAST row of the
            // executor's updated K/V is new — the prior rows came from
            // decompressing our own cache, and re-encoding decompressed
            // rows compounds the codec's norm shrink every step.
            let layer_slot = &mut self.layers[layer];
            let new_rows = updated_kv.0.shape()[0];
            debug_assert_eq!(new_rows, layer_slot.num_vecs + 1, "decode adds one row");
            let k_last = updated_kv.0.row(new_rows - 1).to_owned();
            let v_last = updated_kv.1.row(new_rows - 1).to_owned();
            layer_slot.append_row(
                k_last.as_slice().expect("k row contig"),
                v_last.as_slice().expect("v row contig"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );
            // Executor returns bare post-FFN hidden; PLE + layer_scalar tail
            // is the driving loop's responsibility (see prefill loop above).
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        self.abs_position += 1;
        Ok(last_row(&h))
    }
}

// ── CPU quant-path helper methods (not part of the KvEngine trait) ───────────

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;

// ─── Integration tests with synthetic weights ─────────────────────────────────

#[cfg(test)]
mod integration_tests;
