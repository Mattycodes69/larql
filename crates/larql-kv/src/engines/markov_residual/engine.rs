//! MarkovResidualEngine — KvEngine implementation.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::prefill::rs_prefill;
use super::step::{rs_decode_step, rs_decode_step_profiled};
use super::store::RsStore;
use super::walk::{ensure_attn_tensors_dequantised, rs_decode_step_walk, rs_prefill_walk};
use crate::profiler::EngineProfiler;
use crate::{DecodeStageSummary, EngineInfo, KvEngine};
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};

pub struct MarkovResidualEngine {
    pub(super) window_size: Option<usize>,
    pub(super) store: Option<RsStore>,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) profiling: bool,
    pub(super) profile: EngineProfiler,
    /// W1-GPU: handle into the backend's internal K/V cache, populated
    /// when `prefill_quant` routes through `coarse_prefill_with_state`.
    /// `None` means the engine took the legacy per-layer walk path.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    /// Position counter used by `coarse_decode_step_with_state` for RoPE.
    /// Tracks `prompt_len + steps_already_decoded`.
    pub(super) abs_position: usize,
    /// Engine-owned f32 dequant scratch for the per-layer walk fallback —
    /// `prefill_quant`/`decode_step_quant` populate it; the walk resolves
    /// attention through a `WeightsView::with_scratch` over it. Empty on the
    /// W1-GPU/coarse path. Keeps `weights` immutable (no `weights.tensors`
    /// mutation).
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

impl MarkovResidualEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            store: None,
            backend,
            profiling: false,
            profile: EngineProfiler::default(),
            kv_handle: None,
            abs_position: 0,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Residual store + any K/V held on this engine's behalf elsewhere.
    ///
    /// On the W1-GPU path the store stays empty and the K/V lives outside
    /// the engine, so the store alone reports 0 — which reads as "costs
    /// nothing" rather than "measured somewhere else". It can be in
    /// either of two places depending on backend, and they are mutually
    /// exclusive: inside the `kv_handle` (CPU's whole-model Q4K cache) or
    /// inside the backend itself (Metal, whose handle is a sentinel and
    /// whose `backend_resident_kv_bytes` is the only way to see it).
    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }
}

// W1-GPU dispatch methods (`try_prefill_via_dispatch` /
// `decode_step_via_dispatch`) live in [`super::dispatch`] as an
// additional `impl MarkovResidualEngine` block. They mutate the
// `pub(super)` fields above.

/// §4 architecture-precondition gate
/// (`crates/larql-inference/docs/specs/markov-residual-engine.md`): the
/// exactness contract holds only when per-layer K/V is a pure function
/// of `(residual, weights, absolute position)` — stateless norms,
/// RoPE-only position encoding, direct K/V projection. Constructors
/// don't take weights, so the check runs at the earliest point the
/// engine sees the model (every prefill entry) and fails closed with a
/// typed error. Queries the structural
/// `ModelArchitecture::kv_recomputable_from_residuals` predicate — no
/// architecture-name matching. Shared with the codec twin.
pub(crate) fn check_residual_recompute_preconditions(
    engine_name: &str,
    weights: &ModelWeights,
) -> Result<(), EngineError> {
    if weights.arch.kv_recomputable_from_residuals() {
        Ok(())
    } else {
        Err(EngineError::InvariantViolation {
            what: format!(
                "{engine_name}: architecture '{}' violates the residual-recompute \
                 preconditions (markov-residual-engine.md §4); refusing to run non-exact",
                weights.arch.family()
            ),
        })
    }
}

impl MarkovResidualEngine {
    /// Shared body for `decode_step` / `decode_step_resident` — `index`
    /// reaches the attention step's Q4K-direct route when present.
    /// The store is borrowed, never taken. A failed step therefore costs the
    /// engine nothing but its droppable `hot_kv` derivative — see
    /// [`super::step`]'s failure invariant — so a refusal is reported as
    /// itself (`EngineError::Execution`, retryable) rather than as a dead
    /// engine wearing the words "called before prefill".
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let Self {
            store,
            backend,
            profile,
            profiling,
            ..
        } = self;
        let rs = store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (store missing)".into(),
            })?;
        if *profiling {
            rs_decode_step_profiled(
                larql_inference::WeightsView::dense(weights),
                token_id,
                rs,
                backend.as_ref(),
                profile,
                Some(ffn),
                index,
            )
        } else {
            rs_decode_step(
                larql_inference::WeightsView::dense(weights),
                token_id,
                rs,
                backend.as_ref(),
                Some(ffn),
                index,
            )
        }
    }
}

impl KvEngine for MarkovResidualEngine {
    fn name(&self) -> &str {
        "markov-rs"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "markov-rs".into(),
            description: format!(
                "residual-stream KV replacement — K/V recomputed from stored residuals (mem={:.1}MB)",
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // `?` before the assignment: a refused prefill must not replace the
        // store an earlier one built. See `prefill`'s transactional contract.
        let result = rs_prefill(
            larql_inference::WeightsView::dense(weights),
            token_ids,
            self.window_size,
            self.backend.as_ref(),
            Some(ffn),
        )?;
        self.store = Some(result.store);
        Ok(result.hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, None)
    }

    /// Resident-path decode: threads `index` down to the walk's attention
    /// step so the Q4K-direct projections (`LARQL_Q4K_DIRECT_ATTN` family)
    /// can fire — the structural gap that left non-standard engines on the
    /// f32 path after the standard engine got the fast attention.
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
        self.total_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `kv_handle` is stashed only by the W1-GPU coarse prefill, and
        // cleared when that path is abandoned; `store` exists after any
        // successful prefill. Neither set = no prefill yet.
        match (self.kv_handle.is_some(), self.store.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    fn stage_summary(&self) -> Option<DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("markov-rs", self.backend.name()))
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // W1-GPU path: route through KvDispatch's coarse_prefill_with_state
        // when the engine's stored EngineBackend supports it. State capture
        // gives us per-layer h_in (= the residual we'd store) and per-layer
        // K/V (= the hot K/V tier from W2) in a single backend call —
        // backend can run on GPU; engine's state policy reads the dump.
        // Legacy per-layer walk remains as the fallback so unmigrated
        // backends keep working.
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let result = rs_prefill_walk(view, index, token_ids, self.window_size, backend);
        let hidden = result.hidden.clone();
        self.store = Some(result.store);
        self.kv_handle = None; // ensure dispatch path is not used for subsequent decode
        self.abs_position = token_ids.len();
        Ok(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        // W1-GPU path: if prefill went through coarse_prefill_with_state
        // and stashed `kv_handle`, continue on that path. State capture
        // gives us per-layer h_in / K_new / V_new to update engine state.
        if self.kv_handle.is_some() {
            return self
                .decode_step_via_dispatch(weights, index, token_id)
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "decode_step_via_dispatch returned None".into(),
                });
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .take()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_quant called before prefill (store missing)".into(),
            })?;
        let prof = self.profiling.then_some(&mut self.profile);
        let (hidden, new_rs) = rs_decode_step_walk(view, index, token_id, rs, backend, prof)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "rs_decode_step_walk returned None".into(),
            })?;
        self.store = Some(new_rs);
        self.abs_position += 1;
        Ok(hidden)
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // The methods below override the trait defaults to drive the layer
    // loop through a caller-supplied `LayerExecutor` and honor the
    // caller-supplied `FfnBackend`. Old `prefill_quant` /
    // `decode_step_quant` stay above for backward compat; they construct
    // their own WalkFfn and ignore the FFN parameter. The new methods
    // are what remote-FFN deployments and per-layer codec engines must
    // call to get the engine's contract.

    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::Array2;
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Engines whose state policy requires per-layer dispatch (this
        // one) must refuse fused executors at construction. Until the
        // `requires_per_layer_dispatch()` trait hook lands (Phase 3),
        // degrade transparently to the legacy fused-or-walk path.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }

        // Q4K attn weights need dequant once before the per-layer
        // executor can drive f32 attention against them.
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);

        let backend = executor.backend();
        let num_layers = weights.num_layers;
        let seq_len = token_ids.len();
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs =
            larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
        let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            stored.push(h.clone());
            // Executor drives attention + FFN; engine doesn't care which
            // backend or whether FFN is local/remote. Engine discards
            // the layer's K/V — residual-stream contract recomputes K/V
            // per decode step from the stored residuals.
            let (h_out, _kv) = executor
                .run_prefill_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_prefill_layer returned None".into(),
                })?;
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

        // State management identical to `rs_prefill_walk`: build the
        // store, clip overflow into cold tier, precompute cold K/V via
        // `recompute_kv` (engine policy — the executor doesn't own this).
        let mut rs = RsStore {
            hot_len: stored.first().map_or(0, |s| s.shape()[0]),
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_len: 0,
            // Executor path doesn't yet capture K/V from the executor's
            // `run_prefill_layer` return; falls back to recompute-on-decode
            // for now (W2 follow-up: thread the captured K/V through
            // `LayerExecutor::run_prefill_layer`'s return tuple).
            hot_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: self.window_size,
        };
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        rs.finalise_hot_len_after_clip();
        if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
            let cold_kv: Vec<SharedKV> = (0..num_layers)
                .map(|layer| {
                    recompute_kv(
                        larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                        &cold[layer],
                        layer,
                        0,
                        backend,
                        Some(index),
                    )
                    .expect("cold K/V pre-computation failed")
                })
                .collect();
            // 2026-05-19 audit fix: doubling-capacity append.
            rs.append_cold_overflow(cold, Some(cold_kv));
            rs.cold_abs_start = 0;
        }

        let hidden = {
            use ndarray::s;
            let last = h.shape()[0] - 1;
            h.slice(s![last..=last, ..]).to_owned()
        };
        self.store = Some(rs);
        Ok(hidden)
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use crate::engines::markov_residual::recompute_kv;
        use larql_inference::attention::SharedKV;
        use larql_inference::forward::embed_tokens_pub;
        use larql_inference::layer_executor::ExecutorDispatchKind;
        use ndarray::{s, Array2};

        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }

        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);

        let backend = executor.backend();
        let rs = self
            .store
            .take()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_quant_via_executor called before prefill (store missing)".into(),
            })?;
        let num_layers = weights.num_layers;
        let abs_position = rs.next_position;
        let mut h_new = embed_tokens_pub(weights, &[token_id]);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = larql_inference::forward::ple::precompute_per_layer_inputs(
            weights,
            &h_new,
            &[token_id],
        );
        let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            // Logical lengths only — `stored` and the cold buffers are
            // doubling-capacity (`hot_len` / `cold_len` are the row
            // counts, see RsStore docs); the zero rows past the logical
            // length must never reach attention.
            let s_hot = rs.hot_len;
            let h_hot = rs.stored[layer].slice(s![..s_hot, ..]);
            let hot_abs_start = abs_position.saturating_sub(s_hot);

            // Engine assembles the K/V to attend against from its store.
            // The executor doesn't own state — it just receives prior_kv
            // and runs the layer.
            let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
                let (k_cold_buf, v_cold_buf) = &cold_kv[layer];
                let h_hot_owned = h_hot.to_owned();
                let (k_hot, v_hot) = recompute_kv(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h_hot_owned,
                    layer,
                    hot_abs_start,
                    backend,
                    Some(index),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "recompute_kv (hot) returned None".into(),
                })?;
                let c = rs.cold_len;
                let kv_dim = k_cold_buf.shape()[1];
                let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                k_combined
                    .slice_mut(s![..c, ..])
                    .assign(&k_cold_buf.slice(s![..c, ..]));
                k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
                let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
                v_combined
                    .slice_mut(s![..c, ..])
                    .assign(&v_cold_buf.slice(s![..c, ..]));
                v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
                (k_combined, v_combined)
            } else {
                let s_cold = if rs.cold_residuals.is_some() {
                    rs.cold_len
                } else {
                    0
                };
                let (h_full, full_abs_start) = match &rs.cold_residuals {
                    Some(cold) if s_cold > 0 => {
                        let h_cold = cold[layer].slice(s![..s_cold, ..]);
                        let hidden = h_hot.shape()[1];
                        let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                        combined.slice_mut(s![..s_cold, ..]).assign(&h_cold);
                        combined.slice_mut(s![s_cold.., ..]).assign(&h_hot);
                        (combined, rs.cold_abs_start)
                    }
                    _ => (h_hot.to_owned(), hot_abs_start),
                };
                recompute_kv(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h_full,
                    layer,
                    full_abs_start,
                    backend,
                    Some(index),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "recompute_kv (full) returned None".into(),
                })?
            };

            new_stored.push(h_new.clone());
            // Run the layer through the executor.
            let (h_out, _new_kv) = executor
                .run_decode_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h_new,
                    &prior_kv,
                    abs_position,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_decode_layer returned None".into(),
                })?;
            // Executor returns bare post-FFN hidden; PLE + layer_scalar tail
            // is the driving loop's responsibility (see prefill loop above).
            h_new = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        // Append new row to store, clip overflow into cold. Note: this
        // is the executor (non-dispatch) decode path, which doesn't go
        // through the W8.2 hot-path optimisation — it still allocates
        // a fresh Array2 per step. The CPU/executor path is a fallback;
        // the dispatch hot path in `decode_step_via_dispatch` is the
        // one that matters for tok/s.
        let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
            let s_old_logical = rs.hot_len; // logical row count
            let hidden_dim = stored.shape()[1];
            let mut combined = Array2::<f32>::zeros((s_old_logical + 1, hidden_dim));
            if s_old_logical > 0 {
                combined
                    .slice_mut(s![..s_old_logical, ..])
                    .assign(&stored.slice(s![..s_old_logical, ..]));
            }
            combined.slice_mut(s![s_old_logical.., ..]).assign(new_row);
            updated_stored.push(combined);
        }

        let mut updated_rs = RsStore {
            hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
            stored: updated_stored,
            cold_residuals: rs.cold_residuals,
            cold_kv: rs.cold_kv,
            cold_len: rs.cold_len,
            // `run_decode_layer`'s returned K/V row is not appended here,
            // so a carried-through hot_kv would be one row short of the
            // new `hot_len` — stale reads on the next walk step, or an
            // out-of-bounds clip below. Drop it; the next step recomputes
            // from `stored`.
            hot_kv: None,
            cold_abs_start: rs.cold_abs_start,
            next_position: abs_position + 1,
            max_window: rs.max_window,
        };

        let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            updated_rs.clip_layer(layer, &mut overflow);
        }
        updated_rs.finalise_hot_len_after_clip();
        // 2026-05-19 audit fix: doubling-capacity append. Via-executor
        // path doesn't carry evicted_hot_kv, so the helper invalidates
        // cold_kv (matches the prior `updated_rs.cold_kv = None` behaviour).
        updated_rs.append_cold_overflow(overflow, None);

        let last = h_new.shape()[0] - 1;
        let out = h_new.slice(s![last..=last, ..]).to_owned();
        self.store = Some(updated_rs);
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
