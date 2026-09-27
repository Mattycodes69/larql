//! `WindowedCheckpointEngine` — window-based KV cache with boundary-checkpoint replay.
//!
//! Window lifecycle:
//!   1. `process(tokens)` — extends the active window's K,V via
//!      `rs_extend_from_checkpoint`. Auto-closes when the window fills.
//!   2. `close_window()` — saves last-position K,V to `CheckpointStore`,
//!      appends token IDs to `TokenArchive`, resets active window.
//!   3. `replay_window(id)` — reconstructs a window's full K,V by replaying
//!      archived tokens from the prior checkpoint.
//!   4. `stats()` — total bytes, windows, compression ratio vs full KV.
//!
//! Memory at 370K tokens (Gemma 3 4B, W=512):
//!   Checkpoints ≈ 278 KB/window × N_windows
//!   Token archive = 4 bytes/token
//!   Total ≈ 30 MB  vs  25.8 GB for Standard KV  (≈2,000×)

use larql_compute::ComputeBackend;
use larql_inference::EngineBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;
use serde::Serialize;

use super::checkpoint_store::CheckpointStore;
use super::token_archive::TokenArchive;
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::{EngineInfo, KvEngine};
use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;

mod inherent;

// ─── EngineStats ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    pub total_tokens: usize,
    pub archived_windows: usize,
    pub current_window_id: usize,
    pub current_window_tokens: usize,
    pub checkpoint_bytes: usize,
    pub archive_bytes: usize,
    pub total_boundary_bytes: usize,
    pub equivalent_kv_bytes: usize,
    pub compression_ratio: f64,
}

impl EngineStats {
    pub fn summary(&self) -> String {
        format!(
            "{} windows / {} tokens — {:.0}× compression vs full KV",
            self.archived_windows, self.total_tokens, self.compression_ratio,
        )
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct WindowedCheckpointEngine {
    pub window_size: usize,
    pub checkpoints: CheckpointStore,
    pub archive: TokenArchive,

    pub(super) current_window_id: usize,
    pub(super) current_window_tokens: Vec<u32>,
    /// Per-layer K/V for the current (partial) window.
    ///
    /// Two layouts coexist:
    /// - **Pre-allocated** (dispatch hot path): `Array2` is shaped
    ///   `[window_size, kv_dim]` with only the first
    ///   `current_window_kv_len` rows valid; the rest are zeros. Used by
    ///   `try_prefill_via_dispatch` / `decode_step_via_dispatch` so the
    ///   per-step append is one `slice_mut().assign(row)`, not a fresh
    ///   `Array2::zeros((n+1, kv_dim)) + slice-copy`.
    /// - **Narrow** (CPU walk path): `Array2` is shaped `[n, kv_dim]`,
    ///   matching the arrays returned by `rs_extend_from_checkpoint_*`.
    ///   `current_window_kv_len` equals `n` here, so readers can treat
    ///   the two layouts uniformly via the counter.
    ///
    /// Readers that need the logical length **must** use
    /// `current_window_kv_len`, not `k.shape()[0]`.
    pub(super) current_window_kv: Option<Vec<SharedKV>>,
    /// Logical row count for `current_window_kv`. See field doc above.
    pub(super) current_window_kv_len: usize,
    pub(super) abs_offset: usize,
    /// Hidden state at the last processed token; set by `process()`.
    pub(super) last_hidden: Option<Array2<f32>>,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) profiling: bool,
    pub(super) profile: crate::profiler::EngineProfiler,
    /// W1-GPU: handle into the backend's K/V cache, populated when
    /// prefill routes through `coarse_prefill_with_state`. `None` =
    /// legacy CPU walk path.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    /// Engine-owned f32 dequant scratch for the per-layer walk fallback
    /// (see `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

/// Smallest legal window. `window_size == 0` would make the fill check
/// in `process()` (`current_window_tokens.len() >= window_size`) true
/// forever with nothing consumed — an infinite close loop — so it is
/// rejected at construction. `window_size == 1` is legal.
const MIN_WINDOW_SIZE: usize = 1;

impl KvEngine for WindowedCheckpointEngine {
    fn name(&self) -> &str {
        "windowed-checkpoint"
    }

    fn info(&self) -> EngineInfo {
        let mem =
            self.checkpoints.total_bytes() + self.archive.total_bytes() + self.current_kv_bytes();
        EngineInfo {
            name: "windowed-checkpoint".into(),
            description: format!(
                "window-boundary KV checkpoints + token replay \
                 (windows={}, tokens={}, mem={:.1}MB)",
                self.archive.len(),
                self.archive.total_tokens() + self.current_window_tokens.len(),
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config: format!("window={}", self.window_size),
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
        self.process(weights, token_ids, Some(ffn))?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after prefill".into(),
            })
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.process(weights, &[token_id], Some(ffn))?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step".into(),
            })
    }

    /// Resident-path decode: threads `index` to the per-token attention
    /// steps' Q4K-direct route (the non-standard-engine structural-gap fix).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.process_with_index(weights, &[token_id], Some(ffn), Some(index))?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step".into(),
            })
    }

    fn memory_bytes(&self) -> usize {
        // The last term covers the coarse dispatch path, where the live
        // window's K/V is held by the backend rather than in
        // `current_window_kv` — without it the hot tier reads as 0.0MB
        // and the ratio against a standard cache becomes unbounded.
        self.checkpoints.total_bytes()
            + self.archive.total_bytes()
            + self.current_kv_bytes()
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.current_window_tokens.len()
    }

    fn cold_bytes(&self) -> usize {
        self.checkpoints.total_bytes() + self.archive.total_bytes()
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `kv_handle` marks the coarse W1-GPU path; `current_window_kv`
        // is the per-layer shadow this engine keeps when it walks
        // layer-by-layer. Neither = nothing prefilled yet.
        match (self.kv_handle.is_some(), self.current_window_kv.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    fn stage_summary(&self) -> Option<crate::DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(
            self.profile
                .summary("windowed-checkpoint", self.backend.name()),
        )
    }

    /// Quant prefill — runs the windowed-checkpoint extension regardless
    /// of backend or vindex format. W1-GPU: tries `coarse_prefill_with_state`
    /// first; falls back to the legacy CPU per-layer walk when state
    /// capture isn't available. The engine's window-checkpoint
    /// contract is preserved either way: `current_window_kv` is built
    /// from captured per-layer state (W1-GPU) or computed via walk.
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
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        self.kv_handle = None;
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_quant(weights, index, token_ids, backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_quant returned None during prefill_quant".into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after prefill_quant".into(),
            })
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
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_quant(weights, index, &[token_id], backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_quant returned None during decode_step_quant".into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step_quant".into(),
            })
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // Drive the per-token layer loop through a caller-supplied `LayerExecutor`
    // and honor the caller-supplied `FfnBackend`. The legacy `*_quant` methods
    // construct their own `WalkFfn` and ignore the FFN parameter; remote-FFN
    // deployments (`larql bench --ffn http://shard:8080`) need this path so
    // the engine actually dispatches through the supplied backend.
    //
    // Window-close semantics (checkpoint + archive at window boundaries) are
    // identical to `process_quant` / `extend_current_quant` — the executor only
    // owns per-layer compute; window state is engine state.
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
        // Spec §3.4: this engine's state policy (windowed checkpoints) is
        // expressible against per-layer dispatch only. Transparent degrade
        // on fused executors until the Phase 3 refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_via_executor(weights, executor, ffn, token_ids)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_via_executor returned None during prefill_quant_via_executor"
                    .into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after prefill_quant_via_executor".into(),
            })
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
        self.process_via_executor(weights, executor, ffn, &[token_id])
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_via_executor returned None during decode_step_quant_via_executor"
                    .into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step_quant_via_executor".into(),
            })
    }
}

// ── Executor-driven window extension ─────────────────────────────────────────

impl WindowedCheckpointEngine {
    /// Executor-aware analogue of `process_quant`: feeds tokens into the
    /// current window, auto-closes on fill, drives per-layer compute
    /// through `executor` instead of constructing a local `WalkFfn`.
    fn process_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        tokens: &[u32],
    ) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current_via_executor(weights, executor, ffn, chunk)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    fn extend_current_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        chunk: &[u32],
    ) -> Option<()> {
        use larql_inference::forward::embed_tokens_pub;
        if chunk.is_empty() {
            return Some(());
        }

        let mut kv_cache: Vec<SharedKV> = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                super::extend::empty_prior(weights)
            }
        } else {
            // Mid-window the shadow MUST exist — see extend_current_quant.
            self.current_window_kv.take()?
        };

        let num_layers = weights.num_layers;
        if kv_cache.len() != num_layers {
            return None;
        }
        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let mut last_hidden: Option<Array2<f32>> = None;

        for (i, &token_id) in chunk.iter().enumerate() {
            let abs_position = abs_start + i;
            let mut h = embed_tokens_pub(weights, &[token_id]);
            // PLE inputs are per-token — this loop embeds one token at a
            // time, matching the legacy `kv_decode_step_run` recipe exactly.
            let ple_inputs = larql_inference::forward::ple::precompute_per_layer_inputs(
                weights,
                &h,
                &[token_id],
            );

            for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
                let (h_out, new_kv) = executor.run_decode_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    kv_slot,
                    abs_position,
                    ffn,
                )?;
                // `LayerExecutor::run_decode_layer` returns attention + bare
                // FFN only (`LocalWalkExecutor`, the sole production impl,
                // ends at `run_ffn`); the PLE + layer_scalar tail is the
                // driving loop's responsibility, mirroring the legacy
                // `kv_decode_step_run` sequence.
                h = crate::engines::apply_ple_and_layer_scalar(
                    weights,
                    &h_out,
                    layer,
                    ple_inputs.get(layer),
                );
                *kv_slot = new_kv;
            }
            last_hidden = Some(h);
        }

        self.last_hidden = last_hidden;
        // CPU walk path via executor: kv_cache is narrow arrays.
        self.current_window_kv_len = kv_cache.first().map_or(0, |(k, _)| k.shape()[0]);
        self.current_window_kv = Some(kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
