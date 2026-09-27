//! `BoundaryPerLayerEngine` — `KvEngine` implementation with per-layer
//! codec policy on the cold tier.
//!
//! The engine refuses to construct without a matching calibration record
//! (per spec §4.7 + §4.9). v0.1 supports `Bf16` per layer only; other
//! codec choices are rejected at policy construction (per
//! [`super::policy::PolicyError`]).
//!
//! Implementation is split across sibling modules:
//!
//! - this file: struct + construction + `KvEngine` trait glue
//! - [`super::walk`] — CPU dense walk path (`run_prefill`/`run_decode`)
//! - [`super::dispatch`] — W1-GPU dispatch fast path
//!   (`try_prefill_via_dispatch`/`decode_step_via_dispatch`)
//! - [`super::executor`] — `LayerExecutor`-driven path
//!   (`prefill_via_executor`/`decode_step_via_executor`)
//! - [`super::cold_tier`] — cold-tier maintenance
//!   (`extend_cold_kv_with_overflow` + small helpers)

use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::boundary_per_layer::calibration::{
    BoundaryCalibrationRecord, BoundaryCalibrationStore, CalibrationError,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::RsStorePerLayer;
use crate::engines::boundary_per_layer::{dispatch, executor, walk};
use crate::{EngineInfo, KvEngine};

/// Errors during engine construction (preconditions per spec §4.6).
#[derive(Debug, thiserror::Error)]
pub enum EngineConstructionError {
    #[error("policy targets {policy_layers} layers but model has {model_layers}")]
    LayerCountMismatch {
        policy_layers: usize,
        model_layers: usize,
    },
    #[error(transparent)]
    Calibration(#[from] CalibrationError),
}

/// Policy revision label for an engine that adopts the model's depth.
const ADOPTED_POLICY_REVISION: &str = "model-depth";

/// `BoundaryPerLayerEngine` — per-layer codec policy on the cold tier.
pub struct BoundaryPerLayerEngine {
    pub(super) window_size: Option<usize>,
    pub(super) policy: BoundaryLayerPolicy,
    pub(super) record: BoundaryCalibrationRecord,
    pub(super) store: Option<RsStorePerLayer>,
    /// W1-GPU dispatch handle. `Some` when the prefill went through
    /// `dispatch::try_prefill_via_dispatch` and decode is using the
    /// kernel-fused fast path. `None` when the engine fell back to the
    /// dense walk (e.g. backend lacks cached_decode support).
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    pub(super) backend: Box<dyn EngineBackend>,
    /// Engine-owned f32 dequant scratch for the per-layer walk fallback
    /// (see `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
    /// The policy was built without a declared depth and takes the served
    /// model's at prefill (uniform bf16 only).
    pub(super) adopt_model_depth: bool,
}

impl BoundaryPerLayerEngine {
    /// Construct with policy validation against the supplied calibration
    /// store. Returns `Err` when:
    ///
    /// - The policy's layer count does not match `num_model_layers` (§4.6).
    /// - No calibration record exists for the policy's fingerprint (§4.7,
    ///   §8.3).
    pub fn new(
        window_size: Option<usize>,
        policy: BoundaryLayerPolicy,
        num_model_layers: usize,
        calibration: &dyn BoundaryCalibrationStore,
    ) -> Result<Self, EngineConstructionError> {
        Self::with_backend(
            window_size,
            policy,
            num_model_layers,
            calibration,
            cpu_engine_backend(),
        )
    }

    /// Convenience constructor for the v0.1 cold-start case: any
    /// uniform-bf16 policy inherits `MarkovResidualCodecEngine`'s
    /// trivial bf16 calibration record (KL ≤ 0.01 nats — the
    /// spec's §4.7 "uncalibrated but trivially safe" record).
    ///
    /// Use this when you don't have a calibration store handy (e.g.
    /// a freshly-downloaded model). For non-bf16 policies the
    /// engine still requires an explicit calibration via [`new`] —
    /// non-trivial codecs need a measured KL bound to be safe.
    /// Equivalent to what `EngineKind::BoundaryPerLayer.build()`
    /// does internally.
    pub fn new_with_default_calibration(
        window_size: Option<usize>,
        num_model_layers: usize,
    ) -> Result<Self, EngineConstructionError> {
        let policy = BoundaryLayerPolicy::bf16_uniform("default", num_model_layers);
        let cal = crate::engines::boundary_per_layer::calibration::InMemoryCalibrationStore::new();
        cal.put(BoundaryCalibrationRecord::bf16_uniform_default(
            policy.fingerprint(),
        ))?;
        Self::new(window_size, policy, num_model_layers, &cal)
    }

    pub fn with_backend(
        window_size: Option<usize>,
        policy: BoundaryLayerPolicy,
        num_model_layers: usize,
        calibration: &dyn BoundaryCalibrationStore,
        backend: Box<dyn EngineBackend>,
    ) -> Result<Self, EngineConstructionError> {
        if policy.num_layers() != num_model_layers {
            return Err(EngineConstructionError::LayerCountMismatch {
                policy_layers: policy.num_layers(),
                model_layers: num_model_layers,
            });
        }
        let record = calibration.get(&policy.fingerprint())?;
        Ok(Self {
            window_size,
            policy,
            record,
            store: None,
            kv_handle: None,
            backend,
            dequant_scratch: larql_inference::DequantScratch::new(),
            adopt_model_depth: false,
        })
    }

    /// A uniform-bf16 engine with no declared depth: the policy is sized
    /// to the served model at each prefill, so no model's layer count is
    /// assumed. A spec that names `layers=N` uses [`Self::with_backend`]
    /// instead and has the mismatch refused.
    pub fn adopting_model_depth(
        window_size: Option<usize>,
        backend: Box<dyn EngineBackend>,
    ) -> Self {
        let policy = BoundaryLayerPolicy::bf16_uniform(ADOPTED_POLICY_REVISION, 0);
        let record = BoundaryCalibrationRecord::bf16_uniform_default(policy.fingerprint());
        Self {
            window_size,
            policy,
            record,
            store: None,
            kv_handle: None,
            backend,
            dequant_scratch: larql_inference::DequantScratch::new(),
            adopt_model_depth: true,
        }
    }

    /// Size an adopting engine's uniform policy to `num_layers`.
    fn fit_policy_to(&mut self, num_layers: usize) {
        if self.adopt_model_depth && self.policy.num_layers() != num_layers {
            self.policy = BoundaryLayerPolicy::bf16_uniform(ADOPTED_POLICY_REVISION, num_layers);
            self.record =
                BoundaryCalibrationRecord::bf16_uniform_default(self.policy.fingerprint());
        }
    }

    pub fn policy(&self) -> &BoundaryLayerPolicy {
        &self.policy
    }

    pub fn calibration_record(&self) -> &BoundaryCalibrationRecord {
        &self.record
    }
}

impl BoundaryPerLayerEngine {
    /// Shared body for `decode_step` / `decode_step_resident`.
    ///
    /// The store is mutated in place (never `take`n): a failing step must
    /// leave `self.store` populated so a retry or the dense-walk fallback
    /// can still run — one transient backend failure must not brick the
    /// session. `walk::run_decode` guarantees canonical state is untouched
    /// on failure (see its docs).
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (store missing)".into(),
            })?;
        walk::run_decode(
            view,
            ffn,
            self.backend.as_ref(),
            &self.policy,
            rs,
            token_id,
            index,
        )
    }
}

impl KvEngine for BoundaryPerLayerEngine {
    fn name(&self) -> &str {
        "boundary-per-layer"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w},layers={}", self.policy.num_layers()),
            None => format!("window=full,layers={}", self.policy.num_layers()),
        };
        let mem = self.store.as_ref().map_or(0, |s| s.memory_bytes());
        EngineInfo {
            name: "boundary-per-layer".into(),
            description: format!(
                "per-layer codec policy on cold tier (kl_bound={:.3} nats, mem={:.1}MB)",
                self.record.kl_bound_nats,
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
        self.fit_policy_to(weights.num_layers);
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Engine reuse: a stale dispatch handle from a previous prompt's
        // `prefill_quant` must never survive a re-prefill, or the next
        // `decode_step_quant` would dispatch against the OLD prompt's KV.
        self.kv_handle = None;
        let (hidden, store) = walk::run_prefill(
            larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
            ffn,
            self.backend.as_ref(),
            &self.policy,
            self.window_size,
            token_ids,
        )?;
        self.store = Some(store);
        Ok(hidden)
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
        // Store + whichever off-engine home holds the K/V on the coarse
        // path: the handle (CPU whole-model cache) or the backend itself
        // (Metal sentinel handle). Mutually exclusive, so summing is safe.
        self.store.as_ref().map_or(0, |s| s.memory_bytes())
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.window_tokens())
    }

    fn cold_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.cold_bytes())
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `kv_handle` = the W1-GPU fused path; cleared on fallback to
        // the dense walk. `store` marks that a prefill has happened.
        match (self.kv_handle.is_some(), self.store.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    // ── Q4K path ─────────────────────────────────────────────────────────
    //
    // Try W1-GPU dispatch first; fall back to dense walk with attn
    // tensors dequantised when the backend / vindex doesn't support
    // direct-matvec decode.

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        _backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        self.fit_policy_to(weights.num_layers);
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        if let Some((hidden, store, handle)) = dispatch::try_prefill_via_dispatch(
            weights,
            self.backend.as_ref(),
            &self.policy,
            self.window_size,
            index,
            token_ids,
            &mut self.dequant_scratch,
        ) {
            self.store = Some(store);
            self.kv_handle = Some(handle);
            return Ok(hidden);
        }
        // Fall back to dense f32 walk (compact vindexes / CPU backend).
        self.kv_handle = None;
        larql_inference::vindex::dequant::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        self.prefill(weights, ffn, token_ids)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        _backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        // If prefill went through dispatch, decode does too.
        if self.kv_handle.is_some() {
            let mut handle =
                self.kv_handle
                    .take()
                    .ok_or_else(|| EngineError::InvariantViolation {
                        what: "decode_step_quant kv_handle vanished mid-call".into(),
                    })?;
            // The store is mutated in place (never `take`n): on a dispatch
            // failure it must survive so the dense-walk fallback below
            // stays reachable on the next call — one transient failure
            // must not brick the session.
            let rs = self
                .store
                .as_mut()
                .ok_or_else(|| EngineError::InvariantViolation {
                    what: "decode_step_quant called before prefill (store missing)".into(),
                })?;
            let result = dispatch::decode_step_via_dispatch(
                weights,
                self.backend.as_ref(),
                &self.policy,
                &mut handle,
                rs,
                index,
                token_id,
                &mut self.dequant_scratch,
            );
            match result {
                Some(hidden) => {
                    self.kv_handle = Some(handle);
                    return Ok(hidden);
                }
                None => {
                    // State-dump failure — the handle is no longer trusted,
                    // so clear it (the next decode takes the dense walk over
                    // the preserved store); surface as BackendFailure so the
                    // harness can route accordingly.
                    self.kv_handle = None;
                    return Err(EngineError::BackendFailure {
                        details: "state-dump payload incomplete".into(),
                    });
                }
            }
        }
        larql_inference::vindex::dequant::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        self.decode_step(weights, ffn, token_id)
    }

    // ── Phase 2 migration: executor-driven path ──────────────────────────
    //
    // Per-layer codec policy requires per-layer dispatch. The executor
    // path drives the layer loop through a caller-supplied executor +
    // honours the caller's FFN backend.

    fn prefill_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.fit_policy_to(weights.num_layers);
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Engine reuse: clear any stale dispatch handle (see `prefill`).
        self.kv_handle = None;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            // State policy can't fire under fused dispatch; degrade.
            return self.prefill(weights, ffn, token_ids);
        }
        let (hidden, store) = executor::run_prefill(
            larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
            executor,
            ffn,
            &self.policy,
            self.window_size,
            token_ids,
        )
        .ok_or_else(|| EngineError::BackendFailure {
            details: "executor::run_prefill returned None".into(),
        })?;
        self.store = Some(store);
        Ok(hidden)
    }

    fn decode_step_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step(weights, ffn, token_id);
        }
        // Store mutated in place (never `take`n): a transient executor
        // failure must leave `self.store` populated so a retry or the
        // dense-walk fallback still works (see `executor::run_decode`'s
        // failure invariant).
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let rs = self
            .store
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step_via_executor called before prefill (store missing)".into(),
            })?;
        executor::run_decode(view, executor, ffn, &self.policy, rs, token_id).ok_or_else(|| {
            EngineError::BackendFailure {
                details: "executor::run_decode returned None".into(),
            }
        })
    }
}

#[cfg(test)]
mod tests;
