//! `MarkovResidualCodecEngine` — `KvEngine` implementation.
//!
//! Implementation is split across sibling modules (mirrors the layout
//! used by `boundary_per_layer`):
//!
//! - this file: struct + construction + `KvEngine` trait glue
//! - [`super::walk`] — CPU dense walk path
//!   (`rs_prefill_codec_walk` / `rs_decode_step_codec_walk`)
//! - [`super::prefill`] / [`super::step`] — Q4K-native walk path
//!   (`rs_prefill_codec` / `rs_decode_step_codec`)
//! - [`super::dispatch`] — W1-GPU dispatch fast path with W10 mask
//!   cascade
//! - [`super::executor`] — `LayerExecutor`-driven path
//! - [`super::helpers`] — W8.2 doubling-capacity buffer helpers

use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::markov_residual::engine::check_residual_recompute_preconditions;
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::prefill::rs_prefill_codec;
use crate::engines::markov_residual_codec::step::rs_decode_step_codec;
use crate::engines::markov_residual_codec::store::RsStoreCodec;
use crate::engines::markov_residual_codec::walk::{
    rs_decode_step_codec_walk, rs_prefill_codec_walk,
};
use crate::profiler::EngineProfiler;
use crate::{DecodeStageSummary, EngineInfo, KvEngine};

/// `MarkovResidualCodecEngine` — `MarkovResidualEngine` with a codec-encoded
/// cold tier.
pub struct MarkovResidualCodecEngine {
    pub(super) window_size: Option<usize>,
    pub(super) codec: ColdResidualCodec,
    pub(super) store: Option<RsStoreCodec>,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) profiling: bool,
    pub(super) profile: EngineProfiler,
    /// W1-GPU: see `MarkovResidualEngine::kv_handle`.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    pub(super) abs_position: usize,
    /// Engine-owned f32 dequant scratch for the per-layer codec-walk fallback
    /// (see `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
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
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    pub fn codec(&self) -> ColdResidualCodec {
        self.codec
    }

    /// Residual store + backend-resident K/V — see
    /// [`super::super::markov_residual::MarkovResidualEngine::total_memory_bytes`]
    /// for why the backend term is needed on the W1-GPU path.
    pub fn total_memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }
}

// The W1-GPU dispatch methods (`try_prefill_via_dispatch` /
// `decode_step_via_dispatch`) and executor-driven helpers
// (`prefill_via_executor_impl` / `decode_step_via_executor_impl`)
// live as additional `impl MarkovResidualCodecEngine` blocks in
// sibling files [`super::dispatch`] and [`super::executor`]. They
// mutate `store` / `kv_handle` / `abs_position` / `profile` (all
// `pub(super)`).

impl MarkovResidualCodecEngine {
    /// Shared body for `decode_step` / `decode_step_resident`.
    ///
    /// The store is borrowed, never taken: a failed step leaves canonical
    /// state exactly as it was (see [`super::step`]'s failure invariant), so a
    /// refusal reports itself and the engine stays usable.
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let backend = self.backend.as_ref();
        let rs = self
            .store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (store missing)".into(),
            })?;
        rs_decode_step_codec(
            larql_inference::WeightsView::dense(weights),
            token_id,
            rs,
            backend,
            Some(ffn),
            index,
        )
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
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // `?` before the assignment: a refused prefill must not replace the
        // store an earlier one built.
        let result = rs_prefill_codec(
            larql_inference::WeightsView::dense(weights),
            token_ids,
            self.window_size,
            self.codec,
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
        // Same rule as `markov_residual`: `kv_handle` marks the W1-GPU
        // coarse path, `store` marks "prefill happened at all".
        match (self.kv_handle.is_some(), self.store.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // W1-GPU path: try the dispatch route first (see
        // `MarkovResidualEngine::try_prefill_via_dispatch` for the design
        // notes). Same shape: prefill captures per-layer h_in / K_new /
        // V_new in one backend call; engine reads the dump.
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let result = rs_prefill_codec_walk(
            view,
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
        Ok(hidden)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
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
        let (hidden, new_rs) = rs_decode_step_codec_walk(view, index, token_id, rs, backend, prof)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "rs_decode_step_codec_walk returned None".into(),
            })?;
        self.store = Some(new_rs);
        self.abs_position += 1;
        Ok(hidden)
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
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        check_residual_recompute_preconditions(self.name(), weights)?;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Per spec §3.4: this engine's state policy (codec cold tier)
        // requires per-layer dispatch. Transparent degrade on fused
        // executor until Phase 3's refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        self.prefill_via_executor_impl(weights, executor, ffn, index, token_ids)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "prefill_via_executor_impl returned None".into(),
            })
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        self.decode_step_via_executor_impl(weights, executor, ffn, index, token_id)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "decode_step_via_executor_impl returned None".into(),
            })
    }
}

#[cfg(test)]
mod tests;
