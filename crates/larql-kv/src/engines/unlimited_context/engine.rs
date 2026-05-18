//! `UnlimitedContextEngine` — window-based KV cache with boundary-checkpoint replay.
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
use larql_inference::{cpu_engine_backend, EngineBackend};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};
use serde::Serialize;

use super::checkpoint_store::CheckpointStore;
use super::extend::{
    empty_prior, rs_extend_from_checkpoint_backend, rs_extend_from_checkpoint_quant,
};
use super::token_archive::TokenArchive;
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use crate::{EngineInfo, KvEngine};
use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::model::ModelWeights;

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

pub struct UnlimitedContextEngine {
    pub window_size: usize,
    pub checkpoints: CheckpointStore,
    pub archive: TokenArchive,

    current_window_id: usize,
    current_window_tokens: Vec<u32>,
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
    current_window_kv: Option<Vec<SharedKV>>,
    /// Logical row count for `current_window_kv`. See field doc above.
    current_window_kv_len: usize,
    abs_offset: usize,
    /// Hidden state at the last processed token; set by `process()`.
    last_hidden: Option<Array2<f32>>,
    backend: Box<dyn EngineBackend>,
    profiling: bool,
    profile: crate::profiler::EngineProfiler,
    /// W1-GPU: handle into the backend's K/V cache, populated when
    /// prefill routes through `coarse_prefill_with_state`. `None` =
    /// legacy CPU walk path.
    kv_handle: Option<larql_inference::KvHandle>,
}

impl UnlimitedContextEngine {
    pub fn new(window_size: usize) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: usize, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            checkpoints: CheckpointStore::new(),
            archive: TokenArchive::new(),
            current_window_id: 0,
            current_window_tokens: Vec::new(),
            current_window_kv: None,
            current_window_kv_len: 0,
            abs_offset: 0,
            last_hidden: None,
            backend,
            profiling: false,
            profile: crate::profiler::EngineProfiler::default(),
            kv_handle: None,
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Feed tokens into the engine. Windows auto-close when they fill.
    pub fn process(&mut self, weights: &ModelWeights, tokens: &[u32]) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current(weights, chunk)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    /// Close any partial current window. Call before replay if the window hasn't filled.
    pub fn flush(&mut self) {
        if !self.current_window_tokens.is_empty() {
            self.close_window();
        }
    }

    /// Reconstruct a window's full K,V by replaying its archived tokens from
    /// the prior window's boundary checkpoint.
    pub fn replay_window(
        &self,
        weights: &ModelWeights,
        window_id: usize,
    ) -> Option<(Vec<SharedKV>, usize)> {
        let (tokens, abs_offset) = self.archive.retrieve(window_id)?;

        let prior = if window_id > 0 && self.checkpoints.contains(window_id - 1) {
            let (ckpt, _) = self.checkpoints.load(window_id - 1)?;
            ckpt
        } else {
            empty_prior(weights)
        };

        let out = rs_extend_from_checkpoint_backend(
            weights,
            tokens,
            prior,
            abs_offset,
            self.backend.as_ref(),
        )?;
        let abs_end = abs_offset + tokens.len() - 1;
        Some((out.kv_cache, abs_end))
    }

    /// Total storage and context statistics.
    pub fn stats(&self, weights: &ModelWeights) -> EngineStats {
        let arch = &*weights.arch;
        let num_layers = weights.num_layers;
        let kv_dim_sum: usize = (0..num_layers)
            .map(|l| arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l))
            .sum();

        let total_archived = self.archive.total_tokens();
        let current = self.current_window_tokens.len();
        let total_tokens = total_archived + current;

        let equivalent_kv_bytes = total_tokens * kv_dim_sum * 2 * 2;
        let checkpoint_bytes = self.checkpoints.total_bytes();
        let archive_bytes = self.archive.total_bytes();
        let total_boundary_bytes = checkpoint_bytes + archive_bytes;
        let compression_ratio = if total_boundary_bytes == 0 {
            0.0
        } else {
            equivalent_kv_bytes as f64 / total_boundary_bytes as f64
        };

        EngineStats {
            total_tokens,
            archived_windows: self.archive.len(),
            current_window_id: self.current_window_id,
            current_window_tokens: current,
            checkpoint_bytes,
            archive_bytes,
            total_boundary_bytes,
            equivalent_kv_bytes,
            compression_ratio,
        }
    }

    /// Quant-aware equivalent of `process()` — uses
    /// `rs_extend_from_checkpoint_quant` (WalkFfn for FFN; dispatches on
    /// the vindex's format) instead of the f32-backed
    /// `rs_extend_from_checkpoint_backend`.
    fn process_quant(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        tokens: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current_quant(weights, index, chunk, backend)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    fn extend_current_quant(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        chunk: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<()> {
        if chunk.is_empty() {
            return Some(());
        }

        let prior = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                empty_prior(weights)
            }
        } else {
            self.current_window_kv
                .take()
                .unwrap_or_else(|| empty_prior(weights))
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let prof = self.profiling.then_some(&mut self.profile);
        let out = rs_extend_from_checkpoint_quant(
            weights, index, chunk, prior, abs_start, backend, prof,
        )?;

        self.last_hidden = Some(out.last_hidden);
        // CPU walk path returns narrow `[n, kv_dim]` arrays — counter
        // equals shape[0] here. Hot path (`decode_step_via_dispatch`)
        // will re-normalise to pre-allocated `[window_size, kv_dim]`
        // on the next prefill if needed; mixed-mode within a single
        // window isn't supported (and isn't reachable today since
        // `kv_handle` gates the two paths).
        self.current_window_kv_len = out.kv_cache.first().map_or(0, |(k, _)| k.shape()[0]);
        self.current_window_kv = Some(out.kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }

    fn current_kv_bytes(&self) -> usize {
        // W8: count only the logically valid rows. Buffers may be
        // pre-allocated `[window_size, kv_dim]` so `k.len()` overstates
        // by `(window_size - current_window_kv_len) * kv_dim`.
        let rows = self.current_window_kv_len;
        if rows == 0 {
            return 0;
        }
        self.current_window_kv.as_ref().map_or(0, |kv| {
            kv.iter()
                .map(|(k, v)| (k.shape()[1] + v.shape()[1]) * rows * 4)
                .sum()
        })
    }

    fn extend_current(&mut self, weights: &ModelWeights, chunk: &[u32]) -> Option<()> {
        if chunk.is_empty() {
            return Some(());
        }

        let prior = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                empty_prior(weights)
            }
        } else {
            self.current_window_kv
                .take()
                .unwrap_or_else(|| empty_prior(weights))
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let out = rs_extend_from_checkpoint_backend(
            weights,
            chunk,
            prior,
            abs_start,
            self.backend.as_ref(),
        )?;

        self.last_hidden = Some(out.last_hidden);
        // CPU walk path: see comment on extend_current_quant — narrow
        // arrays, counter == shape[0].
        self.current_window_kv_len = out.kv_cache.first().map_or(0, |(k, _)| k.shape()[0]);
        self.current_window_kv = Some(out.kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }

    fn close_window(&mut self) {
        let kv = match self.current_window_kv.take() {
            Some(kv) => kv,
            None => return,
        };
        // W8: use the logical length counter; buffers may be
        // pre-allocated `[window_size, kv_dim]` with only the first
        // `current_window_kv_len` rows valid.
        let n = self.current_window_kv_len;
        let last_kv: Vec<SharedKV> = if n == 0 {
            Vec::new()
        } else {
            kv.iter()
                .map(|(k, v)| {
                    let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
                    let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
                    (last_k, last_v)
                })
                .collect()
        };
        self.current_window_kv_len = 0;

        let window_len = self.current_window_tokens.len();
        let abs_end = self.abs_offset + window_len - 1;

        self.checkpoints
            .save(self.current_window_id, last_kv, abs_end);
        self.archive.archive(
            self.current_window_id,
            std::mem::take(&mut self.current_window_tokens),
            self.abs_offset,
        );
        self.abs_offset += window_len;
        self.current_window_id += 1;
    }

    /// W1-GPU step 4: prefill via `coarse_prefill_with_state`.
    /// Captured per-layer K/V populates `current_window_kv` directly.
    /// Window-overflow handling (auto-close, archive, checkpoint) is
    /// still driven by the engine's token-list tracking — the
    /// `process_via_dispatch` helper threads tokens through
    /// `decode_step_via_dispatch` per-token to keep window state
    /// consistent.
    fn try_prefill_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
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
        // Build current_window_kv from captured prefill K/V into
        // **pre-allocated** [window_size, kv_dim] buffers so subsequent
        // decode steps can append a single row in-place rather than
        // allocating a fresh Array2 each call (W8 — 58% of post-W7
        // decode CPU was `__bzero` + `zip_mut_with_same_shape` +
        // `madvise` from the `Array2::zeros((n+1, kv_dim))` pattern).
        // Prefill is one-shot, so the alloc cost here is amortised
        // across the whole window.
        let prompt_len = token_ids.len();
        let window_cap = self.window_size.max(prompt_len);
        let kv: Vec<SharedKV> = state
            .k_new_per_layer
            .into_iter()
            .zip(state.v_new_per_layer)
            .map(|(k_src, v_src)| {
                let kv_dim = k_src.shape()[1];
                let mut k_buf = Array2::<f32>::zeros((window_cap, kv_dim));
                let mut v_buf = Array2::<f32>::zeros((window_cap, kv_dim));
                if prompt_len > 0 {
                    k_buf.slice_mut(s![..prompt_len, ..]).assign(&k_src);
                    v_buf.slice_mut(s![..prompt_len, ..]).assign(&v_src);
                }
                (k_buf, v_buf)
            })
            .collect();
        self.current_window_kv = Some(kv);
        self.current_window_kv_len = prompt_len;
        self.current_window_tokens = token_ids.to_vec();
        self.last_hidden = Some(hidden.clone());
        self.kv_handle = Some(handle);
        Some(hidden)
    }

    /// W1-GPU step 4: decode through dispatch. State capture gives
    /// us the new K/V row per layer; we append to `current_window_kv`
    /// and emit a window checkpoint when token count crosses
    /// `window_size`. The legacy `close_window` flow uses the
    /// engine-stored K/V for checkpoint emission, so this dispatch
    /// path preserves the engine's contract.
    fn decode_step_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        use larql_inference::PerLayerDecodeState;
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let abs_position = self.abs_offset + self.current_window_tokens.len();
        let handle = self.kv_handle.as_mut()?;
        let hidden = self.backend.as_ref().coarse_decode_step_with_state(
            weights,
            token_id,
            Some(index),
            handle,
            abs_position,
            Some(&mut state),
        )?;
        if !state.is_complete_for(num_layers) {
            self.kv_handle = None;
            return None;
        }
        // W8: append one K/V row per layer **in place** into the
        // pre-allocated `[window_size, kv_dim]` buffers seeded by
        // `try_prefill_via_dispatch`. The previous `Array2::zeros((n+1,
        // kv_dim)) + slice-copy` pattern was 58% of post-W7 decode
        // CPU (`__bzero` + `zip_mut_with_same_shape` + `madvise`); this
        // path does a single `slice_mut(s![pos..pos+1, ..]).assign(row)`
        // per layer per side instead.
        let pos = self.current_window_kv_len;
        let window_kv = self
            .current_window_kv
            .as_mut()
            .expect("dispatch decode without prefill — kv_handle invariant violated");
        debug_assert!(
            pos < window_kv[0].0.shape()[0],
            "current_window_kv_len {pos} >= buffer capacity {} — \
             window auto-close should have fired before this",
            window_kv[0].0.shape()[0]
        );
        for (slot, (k_new_row, v_new_row)) in window_kv
            .iter_mut()
            .zip(state.k_new_per_layer.iter().zip(state.v_new_per_layer.iter()))
            .take(num_layers)
        {
            slot.0.slice_mut(s![pos..pos + 1, ..]).assign(k_new_row);
            slot.1.slice_mut(s![pos..pos + 1, ..]).assign(v_new_row);
        }
        self.current_window_kv_len = pos + 1;
        self.current_window_tokens.push(token_id);
        self.last_hidden = Some(hidden.clone());

        // Window auto-close: same trigger as the legacy process loop.
        if self.current_window_tokens.len() >= self.window_size {
            self.close_window();
        }
        Some(hidden)
    }
}

impl KvEngine for UnlimitedContextEngine {
    fn name(&self) -> &str {
        "unlimited-context"
    }

    fn info(&self) -> EngineInfo {
        let mem =
            self.checkpoints.total_bytes() + self.archive.total_bytes() + self.current_kv_bytes();
        EngineInfo {
            name: "unlimited-context".into(),
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
        _ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        self.process(weights, token_ids)?;
        self.last_hidden.clone()
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        self.process(weights, &[token_id])?;
        self.last_hidden.clone()
    }

    fn memory_bytes(&self) -> usize {
        self.checkpoints.total_bytes() + self.archive.total_bytes() + self.current_kv_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.current_window_tokens.len()
    }

    fn cold_bytes(&self) -> usize {
        self.checkpoints.total_bytes() + self.archive.total_bytes()
    }

    fn stage_summary(&self) -> Option<crate::DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(
            self.profile
                .summary("unlimited-context", self.backend.name()),
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
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Some(hidden);
        }
        self.kv_handle = None;
        ensure_attn_tensors_dequantised(weights, index);
        self.process_quant(weights, index, token_ids, backend)?;
        self.last_hidden.clone()
    }

    fn decode_step_quant(
        &mut self,
        weights: &mut ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        if self.kv_handle.is_some() {
            return self.decode_step_via_dispatch(weights, index, token_id);
        }
        ensure_attn_tensors_dequantised(weights, index);
        self.process_quant(weights, index, &[token_id], backend)?;
        self.last_hidden.clone()
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
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        // Spec §3.4: this engine's state policy (windowed checkpoints) is
        // expressible against per-layer dispatch only. Transparent degrade
        // on fused executors until the Phase 3 refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        ensure_attn_tensors_dequantised(weights, index);
        self.process_via_executor(weights, executor, ffn, token_ids)?;
        self.last_hidden.clone()
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &mut ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        ensure_attn_tensors_dequantised(weights, index);
        self.process_via_executor(weights, executor, ffn, &[token_id])?;
        self.last_hidden.clone()
    }
}

// ── Executor-driven window extension ─────────────────────────────────────────

impl UnlimitedContextEngine {
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
            self.current_window_kv
                .take()
                .unwrap_or_else(|| super::extend::empty_prior(weights))
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

            for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
                let (h_out, new_kv) =
                    executor.run_decode_layer(weights, layer, &h, kv_slot, abs_position, ffn)?;
                h = h_out;
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
mod tests {
    use super::*;

    #[test]
    fn new_engine_is_empty() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_size, 512);
        assert_eq!(eng.archive.len(), 0);
        assert_eq!(eng.checkpoints.len(), 0);
        assert_eq!(eng.current_window_id, 0);
        assert_eq!(eng.memory_bytes(), 0);
    }

    #[test]
    fn engine_info_backend_is_cpu() {
        let eng = UnlimitedContextEngine::new(256);
        let info = eng.info();
        assert_eq!(info.name, "unlimited-context");
        assert!(
            info.backend.starts_with("cpu"),
            "expected cpu backend, got {:?}",
            info.backend
        );
        assert_eq!(info.config, "window=256");
        assert!(info.summary().contains("unlimited-context"));
        assert!(info.summary().contains("cpu"));
    }

    #[test]
    fn engine_info_config_contains_window_size() {
        let eng = UnlimitedContextEngine::new(1024);
        assert!(eng.info().config.contains("1024"));
    }

    #[test]
    fn window_tokens_and_cold_bytes_start_zero() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    // ── prefill / decode cycle ─────────────────────────────────────────────────

    #[test]
    fn prefill_returns_hidden_state() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill failed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "hidden state should be finite"
        );
    }

    #[test]
    fn decode_step_returns_hidden_state() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 1).expect("decode_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn window_auto_closes_when_full() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let window_size = 3usize;
        let mut engine = UnlimitedContextEngine::new(window_size);

        // Feed exactly window_size tokens → triggers close
        for tok in 0..window_size as u32 {
            engine.process(&weights, &[tok]).expect("process failed");
        }
        assert_eq!(engine.archive.len(), 1, "one window should be archived");
        assert_eq!(
            engine.current_window_tokens.len(),
            0,
            "current window should be empty"
        );
        assert_eq!(
            engine.checkpoints.len(),
            1,
            "one checkpoint should be saved"
        );
    }

    #[test]
    fn two_full_windows_archives_two() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(2);

        // 4 tokens = 2 complete windows
        for tok in 0u32..4 {
            engine.process(&weights, &[tok]).expect("process");
        }
        assert_eq!(engine.archive.len(), 2);
        assert_eq!(engine.checkpoints.len(), 2);
    }

    #[test]
    fn partial_window_after_process() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(4);

        // 3 tokens < window_size=4 → no close
        engine.process(&weights, &[0u32, 1, 2]).expect("process");
        assert_eq!(engine.archive.len(), 0, "no window closed yet");
        assert_eq!(engine.window_tokens(), 3);
    }

    #[test]
    fn flush_closes_partial_window() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(4);
        engine.process(&weights, &[0u32, 1]).expect("process");
        assert_eq!(engine.archive.len(), 0);
        engine.flush();
        assert_eq!(engine.archive.len(), 1, "flush should close partial window");
    }

    #[test]
    fn cold_bytes_grow_after_window_close() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = UnlimitedContextEngine::new(2);
        assert_eq!(engine.cold_bytes(), 0);
        engine.process(&weights, &[0u32, 1]).expect("process"); // closes window
        assert!(
            engine.cold_bytes() > 0,
            "cold tier should grow after window close"
        );
    }

    #[test]
    fn memory_bytes_nonzero_after_prefill() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        assert_eq!(engine.memory_bytes(), 0);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn logits_from_unlimited_context_are_finite() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::forward::hidden_to_raw_logits;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let logits = hidden_to_raw_logits(&weights, &h);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits should be finite"
        );
    }

    // ── Q4K paths via Q4K fixture ─────────────────────────────────────────
    //
    // `prefill_quant` first tries `fused_prefill` (Metal fast path); on
    // CPU that returns None (no fused decode kernel), so we fall through
    // to the dequant + cached-decode path. The Q4K fixture has the attn
    // Q4K slices the dequant step needs.

    #[test]
    fn prefill_quant_cpu_runs_via_dequant_path() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_quant_cpu_extends_state() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let h = engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_quant_without_prefill_returns_none() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        // No prefill → decode falls through fast-path checks and returns None
        // (or some empty hidden) without panicking.
        let _ = engine.decode_step_quant(&mut weights, &ffn, &index, 0, &*backend);
    }

    // ── Public utility methods (stats, replay_window, summary) ────────────

    #[test]
    fn engine_stats_summary_includes_archived_and_compression() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        let stats = engine.stats(&weights);
        assert!(stats.total_tokens >= 3);
        // EngineStats::summary builds a one-line string that includes
        // window count and token count.
        let s = stats.summary();
        assert!(s.contains("windows"));
        assert!(s.contains("tokens"));
    }

    #[test]
    fn engine_stats_with_empty_engine_handles_zero_division() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = UnlimitedContextEngine::new(512);
        let stats = engine.stats(&weights);
        // No prefill → all counters zero, compression ratio short-circuits
        // to 0.0 (no division by zero).
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.archived_windows, 0);
        assert!(
            stats.compression_ratio == 0.0,
            "compression should be 0 when no boundary bytes archived"
        );
        // Summary still produces a string for the empty case.
        let _ = stats.summary();
    }

    #[test]
    fn replay_window_returns_none_for_missing_window() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = UnlimitedContextEngine::new(512);
        // No windows archived → any window_id returns None at the
        // `self.archive.retrieve(window_id)?` line.
        assert!(engine.replay_window(&weights, 0).is_none());
        assert!(engine.replay_window(&weights, 99).is_none());
    }

    #[test]
    fn replay_window_succeeds_after_window_overflow() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // window=2; prefill 4 tokens → archives at least 1 window.
        let mut engine = UnlimitedContextEngine::new(2);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        let stats = engine.stats(&weights);
        assert!(
            stats.archived_windows >= 1,
            "expected at least 1 archived window after overflow, got {}",
            stats.archived_windows
        );
        // Replay the first archived window — exercises the
        // `rs_extend_from_checkpoint_backend` path (lines 132-138).
        let replay = engine.replay_window(&weights, 0);
        assert!(replay.is_some(), "replay_window(0) should succeed");
        let (kv, abs_end) = replay.unwrap();
        assert!(!kv.is_empty(), "replayed K/V cache should be non-empty");
        assert!(
            abs_end < 4,
            "abs_end {abs_end} should be within the prefill"
        );
    }

    // ── Phase 2: executor-driven path ─────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_runs_through_local_walk() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        let h = engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("executor prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_via_executor_extends_state() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let h = engine
            .decode_step_quant_via_executor(&mut weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive `rs_extend_from_checkpoint_quant`'s `Some(profiler)` arms
    /// — covers the per-stage `if timing { ... }` blocks and the
    /// profiler accumulator at the end of the function.
    #[test]
    fn process_quant_with_profiling_populates_summary() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = UnlimitedContextEngine::new(512).with_profiling(true);
        engine
            .prefill_quant(&mut weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        engine
            .decode_step_quant(&mut weights, &ffn, &index, 2, &*backend)
            .expect("decode");
        let summary = engine
            .stage_summary()
            .expect("unlimited_context profiler should populate summary");
        assert_eq!(summary.engine, "unlimited-context");
        assert!(summary.steps >= 1);
        assert!(summary.avg_attention_us > 0.0);
        assert!(summary.avg_ffn_us > 0.0);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    /// Counting FFN that records every `forward` call. Proves the executor
    /// path actually dispatches through the caller's `FfnBackend` instead
    /// of constructing a local `WalkFfn` (the legacy coupling the migration
    /// removes).
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
    fn executor_path_honors_ffn_parameter() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = UnlimitedContextEngine::new(512);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");

        let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
        // 3 tokens × num_layers — one FFN dispatch per (token, layer)
        // because the engine's per-token loop runs every layer through
        // `run_decode_layer`, which in turn invokes the caller's FFN.
        let expected = 3 * weights.num_layers;
        assert_eq!(
            call_count, expected,
            "executor path should dispatch FFN through the supplied backend \
             once per (token, layer); got {call_count} for {expected} \
             expected — engine is likely constructing its own FFN internally",
        );
    }

    #[test]
    fn prefill_quant_via_executor_with_small_window_archives() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let mut weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        // window=2, 4 tokens → triggers two window-close cycles via
        // `process_via_executor`. Exercises the prior-checkpoint-load
        // branch in `extend_current_via_executor`.
        let mut engine = UnlimitedContextEngine::new(2);
        engine
            .prefill_quant_via_executor(&mut weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens through executor");
        let stats = engine.stats(&weights);
        assert!(
            stats.archived_windows >= 1,
            "expected at least 1 archived window, got {}",
            stats.archived_windows
        );
    }
}
