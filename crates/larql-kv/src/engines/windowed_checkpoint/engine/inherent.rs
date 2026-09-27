//! Windowed-checkpoint engine construction and checkpointing.

use super::super::checkpoint_store::CheckpointStore;
use super::super::extend::{
    empty_prior, rs_extend_from_checkpoint_backend, rs_extend_from_checkpoint_quant,
    rs_extend_inplace, truncate_kv_rows,
};
use super::super::token_archive::TokenArchive;
use larql_compute::ComputeBackend;
use larql_inference::attention::SharedKV;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use larql_vindex::VectorIndex;
use ndarray::Array2;

#[allow(unused_imports)]
use super::*;

impl WindowedCheckpointEngine {
    /// # Panics
    ///
    /// Panics if `window_size < MIN_WINDOW_SIZE` (i.e. zero).
    pub fn new(window_size: usize) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    /// # Panics
    ///
    /// Panics if `window_size < MIN_WINDOW_SIZE` (i.e. zero).
    pub fn with_backend(window_size: usize, backend: Box<dyn EngineBackend>) -> Self {
        assert!(
            window_size >= MIN_WINDOW_SIZE,
            "WindowedCheckpointEngine window_size must be >= {MIN_WINDOW_SIZE}, got {window_size}"
        );
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
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Feed tokens into the engine. Windows auto-close when they fill.
    pub fn process(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    ) -> Result<(), EngineError> {
        self.process_with_index(weights, tokens, moe_ffn, None)
    }

    /// `process` with an optional vindex threaded to the per-token attention
    /// steps (Q4K-direct route under `LARQL_Q4K_DIRECT_ATTN` — the
    /// non-standard-engine structural-gap fix).
    pub fn process_with_index(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<(), EngineError> {
        let mut remaining = tokens;
        // Closing a window archives its tokens and saves its checkpoint, and
        // neither is undoable. `extend_current` rewinds the *current* window
        // exactly, so a failure is retryable right up until the first close —
        // after that the engine holds a stream it cannot complete, and must
        // say so rather than let a caller retry into a duplicated window.
        let mut closed_a_window = false;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            if let Err(failure) = self.extend_current(weights, chunk, moe_ffn, index) {
                return Err(if closed_a_window {
                    failure.invalidating_engine_state()
                } else {
                    failure
                });
            }
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
                closed_a_window = true;
            }
        }
        Ok(())
    }

    /// Close any partial current window. Call before replay if the window hasn't filled.
    pub fn flush(&mut self) {
        if !self.current_window_tokens.is_empty() {
            self.close_window();
        }
    }

    /// Reconstruct a window's full K,V by replaying its archived tokens from
    /// the prior window's boundary checkpoint.
    ///
    /// For hybrid-MoE models, pass the FFN hook + vindex so the replay
    /// dispatches experts exactly like the live-window path
    /// ([`extend_current`](Self::extend_current)); pass `None`/`None` for dense
    /// models. (Previously this always passed `None` → dense FFN, which would
    /// have produced wrong K/V for an evicted MoE window — the C1 follow-up.)
    pub fn replay_window(
        &self,
        weights: &ModelWeights,
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
        index: Option<&larql_vindex::VectorIndex>,
        window_id: usize,
    ) -> Result<(Vec<SharedKV>, usize), EngineError> {
        let (tokens, abs_offset) =
            self.archive
                .retrieve(window_id)
                .ok_or_else(|| EngineError::RetrievalMiss {
                    reason: format!("window {window_id} is not archived"),
                })?;

        let prior = if window_id > 0 && self.checkpoints.contains(window_id - 1) {
            let (ckpt, _) =
                self.checkpoints
                    .load(window_id - 1)
                    .ok_or_else(|| EngineError::RetrievalMiss {
                        reason: format!("checkpoint for window {} is missing", window_id - 1),
                    })?;
            ckpt
        } else {
            empty_prior(weights)
        };

        let mut kv_cache = prior;
        rs_extend_from_checkpoint_backend(
            larql_inference::WeightsView::dense(weights),
            tokens,
            &mut kv_cache,
            abs_offset,
            self.backend.as_ref(),
            moe_ffn,
            index,
        )?;
        let abs_end = abs_offset + tokens.len() - 1;
        Ok((kv_cache, abs_end))
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
    pub(super) fn process_quant(
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

    pub(super) fn extend_current_quant(
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
            // Mid-window the shadow MUST exist; seeding from an empty
            // prior here would silently drop every in-window token from
            // attention. Fail upward (typed BackendFailure at the
            // KvEngine boundary) instead.
            self.current_window_kv.take()?
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let prof = self.profiling.then_some(&mut self.profile);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let out =
            rs_extend_from_checkpoint_quant(view, index, chunk, prior, abs_start, backend, prof)?;

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

    pub(super) fn current_kv_bytes(&self) -> usize {
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

    pub(super) fn extend_current(
        &mut self,
        weights: &ModelWeights,
        chunk: &[u32],
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<(), EngineError> {
        if chunk.is_empty() {
            return Ok(());
        }

        // `prior_len` is the prior's LOGICAL row count — the window-KV counter
        // mid-window, the checkpoint's row count at a window start, or 0.
        let (mut prior, prior_len) = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let id = self.current_window_id - 1;
                let (ckpt, _) =
                    self.checkpoints
                        .load(id)
                        .ok_or_else(|| EngineError::RetrievalMiss {
                            reason: format!("checkpoint for window {id} is missing"),
                        })?;
                let len = ckpt.first().map_or(0, |(k, _)| k.shape()[0]);
                (ckpt, len)
            } else {
                (empty_prior(weights), 0)
            }
        } else {
            // Mid-window the shadow MUST exist — see extend_current_quant.
            let shadow =
                self.current_window_kv
                    .take()
                    .ok_or_else(|| EngineError::InvariantViolation {
                        what: "mid-window extend with no K/V shadow".into(),
                    })?;
            (shadow, self.current_window_kv_len)
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();

        // In-place fast path: append the chunk's K/V rows into the window's
        // doubling-capacity buffers instead of rebuilding an owned `[len+1]`
        // concat every layer every step (O(window) → O(1) per step). Gated to
        // the Q4K-direct route (with the shared `LARQL_MARKOV_INPLACE_KV`
        // toggle); flags-off keeps the unchanged owned-concat path bit-for-bit,
        // which is what `resident_identity_tests` pins. The window's existing
        // `current_window_kv_len` counter already treats the buffers as
        // over-allocated (the dispatch path does too), so close_window /
        // current_kv_bytes need no change.
        let use_inplace = index.is_some()
            && crate::engines::markov_residual::compute::markov_inplace_kv_enabled()
            && larql_compute::options::q4k_direct_attn_enabled();

        // Both arms restore the shadow on failure, which is what makes a
        // refused chunk rewindable: `current_window_kv_len` and
        // `current_window_tokens` are advanced only after the extend returns,
        // so putting the buffers back at `prior_len` rows restores exactly the
        // window this call started from.
        let outcome = if use_inplace {
            // The in-place path only ever writes past `prior_len`, which the
            // counter never advanced past, so the logical window is already
            // intact — nothing to truncate.
            rs_extend_inplace(
                larql_inference::WeightsView::dense(weights),
                chunk,
                &mut prior,
                prior_len,
                abs_start,
                self.backend.as_ref(),
                moe_ffn,
                index,
            )
            .map(|last| (last, prior_len + chunk.len()))
        } else {
            rs_extend_from_checkpoint_backend(
                larql_inference::WeightsView::dense(weights),
                chunk,
                &mut prior,
                abs_start,
                self.backend.as_ref(),
                moe_ffn,
                index,
            )
            .map(|step| {
                // CPU walk path: narrow arrays, counter == shape[0].
                let rows = prior.first().map_or(0, |(k, _)| k.shape()[0]);
                (step.last_hidden, rows)
            })
            .inspect_err(|_| {
                // The owned-concat path replaces each layer's buffer as it
                // goes and reads a prior by `shape()[0]`, so a half-advanced
                // cache would attend over a token whose step never finished.
                truncate_kv_rows(&mut prior, prior_len);
            })
        };

        let (last_hidden, rows) = match outcome {
            Ok(pair) => pair,
            Err(failure) => {
                self.current_window_kv = Some(prior);
                return Err(failure);
            }
        };
        self.last_hidden = Some(last_hidden);
        self.current_window_kv_len = rows;
        self.current_window_kv = Some(prior);
        self.current_window_tokens.extend_from_slice(chunk);
        Ok(())
    }

    pub(in super::super) fn close_window(&mut self) {
        // W10 Phase B: under HOnly the engine-side window shadow is
        // None; pull the last position's K/V back from the backend
        // (Metal kv cache) via KvDispatch::read_kv_row_at. Without
        // HOnly this branch never fires (kv is always Some) and we
        // slice the engine-side shadow as before.
        let n = self.current_window_kv_len;
        let window_len = self.current_window_tokens.len();
        // Absolute stream position of this window's last token — the value
        // recorded *with* the checkpoint, so a later replay knows where it
        // sat. It is no longer an index into anything: the dispatch path now
        // clips the backend handle to the window (issue #200), so the handle
        // holds this window's rows and not the stream's. The row to read back
        // is therefore its last one. Indexing the handle by absolute position
        // was correct only while the window was not being enforced.
        let abs_end = self.abs_offset + window_len - 1;
        let last_kv: Vec<SharedKV> = match self.current_window_kv.take() {
            Some(kv) => {
                if n == 0 {
                    Vec::new()
                } else {
                    // Shadow is window-local: its last logical row is
                    // `n - 1` regardless of how many windows preceded.
                    kv.iter()
                        .map(|(k, v)| {
                            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
                            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
                            (last_k, last_v)
                        })
                        .collect()
                }
            }
            None => {
                // No CPU shadow — engine ran under HOnly. Read the window's
                // last K/V back from the backend's kv cache. The handle is
                // clipped to the window, so its final row *is* this window's
                // last position; reading an absolute stream index here would
                // now run off the end. If there is no handle or the backend
                // lacks the readback affordance, fall through with an empty
                // checkpoint: the tokens are still archived and the counters
                // reset (a wedged window would otherwise spin `process()`
                // forever), and the mismatched empty checkpoint surfaces as an
                // extend error on the next window instead of silent loss.
                if n == 0 {
                    Vec::new()
                } else if let Some(handle) = self.kv_handle.as_ref() {
                    debug_assert_eq!(
                        n, window_len,
                        "HOnly window shadow counter out of sync with window tokens"
                    );
                    // Window-relative: the clipped handle's last row.
                    let last_row = handle.cached_len().saturating_sub(1);
                    let mut rows = Vec::new();
                    let mut layer = 0;
                    while let Some((k_row, v_row)) = self
                        .backend
                        .as_ref()
                        .read_kv_row_at(handle, layer, last_row)
                    {
                        let kv_dim = k_row.len();
                        let k = Array2::from_shape_vec((1, kv_dim), k_row)
                            .expect("read_kv_row_at returned mismatched length");
                        let v = Array2::from_shape_vec((1, kv_dim), v_row)
                            .expect("read_kv_row_at returned mismatched length");
                        rows.push((k, v));
                        layer += 1;
                    }
                    rows
                } else {
                    Vec::new()
                }
            }
        };
        self.current_window_kv_len = 0;

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
}
