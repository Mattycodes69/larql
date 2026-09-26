//! Standard engine construction, stepping and dispatch helpers.

use larql_inference::async_compute_backend::AsyncComputeBackend;
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_dispatch::helpers::{
    kv_decode_step_from_hidden_via_dispatch, kv_decode_step_from_hidden_via_dispatch_async,
    kv_decode_step_via_dispatch, kv_decode_step_via_dispatch_async,
    kv_prefill_from_hidden_via_dispatch, kv_prefill_from_hidden_via_dispatch_async,
    kv_prefill_via_dispatch, kv_prefill_via_dispatch_async,
};
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend, KvHandle};
use ndarray::Array2;

#[allow(unused_imports)]
use super::*;

impl StandardEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            handles: None,
            prefill_mode: None,
            abs_position: 0,
            row_positions: Default::default(),
            backend: BackendSlot::Sync(backend),
            dequant_scratch: larql_inference::DequantScratch::new(),
            invalidated: None,
        }
    }

    /// Construct with an [`AsyncComputeBackend`]. The engine routes
    /// prefill/decode through async dispatch; output is bit-identical
    /// to [`Self::with_backend`] at Step A3 (parallel-validated) and
    /// faster on Metal once Step A4's deferred dispatch lands.
    pub fn with_async_backend(
        window_size: Option<usize>,
        backend: Box<dyn AsyncComputeBackend>,
    ) -> Self {
        Self {
            window_size,
            handles: None,
            prefill_mode: None,
            abs_position: 0,
            row_positions: Default::default(),
            backend: BackendSlot::Async(backend),
            dequant_scratch: larql_inference::DequantScratch::new(),
            invalidated: None,
        }
    }

    pub(super) fn cache_memory_bytes(&self) -> usize {
        // Two homes, never both: per-layer dispatch puts the K/V in the
        // handles this engine owns; the coarse path hands back a sentinel
        // handle and keeps the cache inside the backend. Summing both is
        // safe — a backend that answers `backend_resident_kv_bytes` with
        // a non-zero figure is by contract reporting K/V that no handle
        // can see, so there is nothing to double-count.
        let handle_bytes: usize = self
            .handles
            .as_ref()
            .map(|handles| {
                // `resident_bytes` (not the per-layer formula) so a
                // whole-model handle reports all its layers, not one.
                handles.iter().map(|h| h.resident_bytes()).sum()
            })
            .unwrap_or(0);
        handle_bytes + self.backend_resident_kv_bytes()
    }

    /// K/V the backend holds internally (Metal's coarse pipeline). Zero
    /// for the async slot: `AsyncComputeBackend` carries no `KvDispatch`,
    /// and no async backend currently owns a cache of its own.
    pub(super) fn backend_resident_kv_bytes(&self) -> usize {
        match &self.backend {
            BackendSlot::Sync(b) => b.as_ref().backend_resident_kv_bytes(),
            BackendSlot::Async(_) => 0,
        }
    }

    /// Shared prefill body — both `prefill` (index=None) and
    /// `prefill_quant` (index=Some) route through here. Matches on the
    /// `BackendSlot` to pick sync vs async dispatch.
    ///
    /// This is the engine's policy point for the dispatch ring's three
    /// outcomes: a refusal terminates as
    /// [`EngineError::Execution`] and no hidden state is produced; a
    /// declining backend stays [`EngineError::BackendFailure`], exactly
    /// as the bare `None` it replaced.
    pub(super) fn do_prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
        index: Option<&larql_inference::larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let (hidden, handles) = match &self.backend {
            BackendSlot::Sync(b) => {
                kv_prefill_via_dispatch(b.as_ref(), view, ffn, token_ids, self.window_size, index)
                    .map_err(EngineError::Execution)?
                    .ok_or_else(|| EngineError::BackendFailure {
                        details: "kv_prefill_via_dispatch returned None".into(),
                    })?
            }
            BackendSlot::Async(b) => kv_prefill_via_dispatch_async(
                b.as_ref(),
                view,
                ffn,
                token_ids,
                self.window_size,
                index,
            )
            .map_err(EngineError::Execution)?
            .ok_or_else(|| EngineError::BackendFailure {
                details: "kv_prefill_via_dispatch_async returned None".into(),
            })?,
        };
        self.handles = Some(handles);
        self.prefill_mode = Some(PrefillDispatchMode::PerLayer);
        // A completed prefill replaces the cache outright, so whatever an
        // earlier failed step left behind is gone with it.
        self.invalidated = None;
        self.abs_position = token_ids.len();
        self.reset_row_positions_after_prefill();
        Ok(hidden)
    }

    /// Multi-modal prefill: accept pre-built initial hidden state from
    /// the host (built via `embed_plan` on an `EmbeddingPlan` that may
    /// contain `Precomputed` vision/audio chunks). Same body as
    /// `do_prefill` minus the embed call; `abs_position` advances by
    /// `initial_hidden.nrows()` instead of by token count. See
    /// ADR-0023 for the seam decision.
    pub(super) fn do_prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        // `token_ids: None` — MM hidden rows (vision/audio) have no 1:1
        // token identities, so PLE inputs cannot be derived here. PLE
        // architectures (Gemma 4 E-series) must prefill via the token
        // entry point (`prefill`), which threads `Some(token_ids)`.
        let (hidden, handles) = match &self.backend {
            BackendSlot::Sync(b) => kv_prefill_from_hidden_via_dispatch(
                b.as_ref(),
                view,
                ffn,
                initial_hidden,
                None,
                self.window_size,
                None,
            ),
            BackendSlot::Async(b) => kv_prefill_from_hidden_via_dispatch_async(
                b.as_ref(),
                view,
                ffn,
                initial_hidden,
                None,
                self.window_size,
                None,
            ),
        }
        .map_err(EngineError::Execution)?
        .ok_or_else(|| EngineError::BackendFailure {
            details: "do_prefill_from_hidden returned None (empty hidden input or \
                      backend dispatch failure)"
                .into(),
        })?;
        self.handles = Some(handles);
        self.prefill_mode = Some(PrefillDispatchMode::PerLayer);
        // A completed prefill replaces the cache outright, so whatever an
        // earlier failed step left behind is gone with it.
        self.invalidated = None;
        // Critical: position pointer must be derived from the hidden
        // row count, NOT from any token count — the input may contain
        // vision rows that aren't tokens. Decode-loop correctness
        // depends on this; off-by-one here garbles the entire
        // continuation. Pinned by the StandardEngine entry-point
        // agreement test in this file's tests module.
        self.abs_position = initial_hidden.nrows();
        self.reset_row_positions_after_prefill();
        Ok(hidden)
    }

    /// Shared decode-step body — both `decode_step` (index=None) and
    /// `decode_step_quant` (index=Some) route through here.
    ///
    /// Same policy point as [`Self::do_prefill`]: a refusal terminates as
    /// [`EngineError::Execution`] and no hidden state escapes.
    ///
    /// **Transactional.** A decode step mutates the cache before it can
    /// know whether it will finish: each layer's attention appends the new
    /// token's K/V, and only then does the FFN get the chance to refuse. A
    /// step that does not complete therefore rewinds every handle to the
    /// length it had on entry, so a caller that fixes the refusal's cause
    /// can drive the same token through the same engine.
    ///
    /// When the rewind cannot be trusted — see [`Self::rewind_is_sound`] —
    /// the error is wrapped as [`EngineError::StateInvalidated`] and the
    /// engine records it, so every later decode refuses instead of
    /// computing from a cache that describes no token sequence. `prefill`
    /// clears the condition, because it replaces the cache outright.
    ///
    /// `abs_position` advances only on success, on every path.
    pub(super) fn do_decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_inference::larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step_input(weights, ffn, StepInput::Token(token_id), index)
    }

    /// One decode step whose new position is a pre-built hidden row —
    /// the decode-time peer of `do_prefill_from_hidden`. Same rewind and
    /// invalidation semantics as `do_decode_step`; only the dispatch
    /// intent differs.
    pub(super) fn do_decode_step_input(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        input: StepInput<'_>,
        index: Option<&larql_inference::larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        if let Some(what) = &self.invalidated {
            return Err(EngineError::InvariantViolation {
                what: format!("decode_step called on an invalidated engine: {what}"),
            });
        }
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let window = self.window_size;
        let abs_position = self.abs_position;
        let backend = &self.backend;
        let handles = self
            .handles
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (handles missing)".into(),
            })?;

        // Snapshot before the first append. Recorded per layer rather than
        // assumed to be uniform: nothing in the trait promises every layer
        // caches the same number of rows, and a wrong assumption here would
        // rewind to a length no layer ever had.
        let entry_lengths: Vec<usize> = handles.iter().map(|h| h.cached_len()).collect();
        let rewindable = Self::rewind_is_sound(window, &entry_lengths);

        let outcome = match (backend, &input) {
            (BackendSlot::Sync(b), StepInput::Token(token_id)) => kv_decode_step_via_dispatch(
                b.as_ref(),
                view,
                ffn,
                handles,
                *token_id,
                abs_position,
                window,
                index,
            ),
            (BackendSlot::Sync(b), StepInput::Hidden(hidden_row)) => {
                kv_decode_step_from_hidden_via_dispatch(
                    b.as_ref(),
                    view,
                    ffn,
                    handles,
                    hidden_row,
                    abs_position,
                    window,
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
            }
            (BackendSlot::Async(b), StepInput::Token(token_id)) => {
                kv_decode_step_via_dispatch_async(
                    b.as_ref(),
                    view,
                    ffn,
                    handles,
                    *token_id,
                    abs_position,
                    window,
                    index,
                )
            }
            (BackendSlot::Async(b), StepInput::Hidden(hidden_row)) => {
                kv_decode_step_from_hidden_via_dispatch_async(
                    b.as_ref(),
                    view,
                    ffn,
                    handles,
                    hidden_row,
                    abs_position,
                    window,
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
            }
        };

        let failure = match outcome {
            Ok(Some(hidden)) => {
                self.record_decode_append(self.abs_position as u64);
                self.abs_position += 1;
                return Ok(hidden);
            }
            // A declining backend leaves the same half-applied step a refusal
            // does, so it gets the same treatment. Distinguishing them here
            // would make the cache's integrity depend on which of two
            // unrelated things went wrong.
            Ok(None) => EngineError::BackendFailure {
                details: "decode step via dispatch returned None".into(),
            },
            Err(refusal) => EngineError::Execution(refusal),
        };

        let rewound = rewindable && Self::rewind(backend, handles, &entry_lengths);
        if rewound {
            return Err(failure);
        }
        let invalidated = failure.invalidating_engine_state();
        self.invalidated = Some(invalidated.to_string());
        Err(invalidated)
    }

    /// Whether rewinding a failed decode step would restore the exact cache
    /// the step started from.
    ///
    /// Unbounded caches only ever append, so truncating to the recorded
    /// length is exact. A windowed cache is different: a step that reaches
    /// the window drops its oldest row to make room, and that row is gone.
    /// Row *count* cannot see this — append-then-drop leaves it unchanged —
    /// so the only sound test is whether every layer had room to spare
    /// before the step began.
    pub(super) fn rewind_is_sound(window: Option<usize>, entry_lengths: &[usize]) -> bool {
        match window {
            None => true,
            // `len < w` is "the step's own append still fits", i.e. it will not
            // push the layer past `w` and trigger the evicting clip.
            Some(w) => entry_lengths.iter().all(|&len| len < w),
        }
    }

    /// Truncate every handle back to its recorded length. All-or-nothing:
    /// one backend that cannot rewind makes the whole cache untrustworthy,
    /// because the layers are only meaningful together.
    pub(super) fn rewind(
        backend: &BackendSlot,
        handles: &mut [KvHandle],
        entry_lengths: &[usize],
    ) -> bool {
        handles
            .iter_mut()
            .zip(entry_lengths)
            .all(|(handle, &len)| match backend {
                BackendSlot::Sync(b) => b.as_ref().truncate_kv(handle, len),
                BackendSlot::Async(b) => b.as_ref().truncate_kv(handle, len),
            })
    }
    /// Rebuild the position map to describe the handles as they stand
    /// after a prefill: one contiguous run per layer, ending at
    /// `abs_position`.
    ///
    /// Sound *only* here. A prefilled cache is contiguous by
    /// construction — a windowed prefill drops its oldest rows, which
    /// shortens the run without perforating it — so the resident rows
    /// and the next position do determine the positions at this one
    /// moment. Every later mutation has to say what it did rather than
    /// have it inferred, which is why nothing else calls this.
    ///
    /// The coarse path keeps an empty map: it has a single whole-model
    /// handle and no per-layer rows to describe.
    pub(super) fn reset_row_positions_after_prefill(&mut self) {
        self.row_positions = match self.prefill_mode {
            Some(PrefillDispatchMode::PerLayer) => {
                let rows: Vec<usize> = self
                    .handles
                    .as_ref()
                    .map(|hs| hs.iter().map(|h| h.cached_len()).collect())
                    .unwrap_or_default();
                larql_inference::kv_row_positions::KvRowPositions::tails_ending_at(
                    self.abs_position as u64,
                    &rows,
                )
            }
            _ => Default::default(),
        };
    }

    /// Record the row every layer just appended at `position`, then
    /// mirror whatever the window clip dropped.
    ///
    /// The clip happens inside dispatch, below this engine, and it keeps
    /// the physical tail. Reproducing that here keeps the map an honest
    /// description of the cache — including where the physical tail is
    /// the wrong set of rows. Correcting it means changing the clip, not
    /// the map: see `clip_layer_to_logical_window`.
    pub(super) fn record_decode_append(&mut self, position: u64) {
        if !matches!(self.prefill_mode, Some(PrefillDispatchMode::PerLayer)) {
            return;
        }
        let Some(handles) = self.handles.as_ref() else {
            return;
        };
        for (layer, handle) in handles.iter().enumerate() {
            let Some(map) = self.row_positions.layer_mut(layer) else {
                continue;
            };
            // An append out of order means the engine's position moved
            // backwards under rows that stayed put — an inconsistency
            // this map exists to expose, so let it stand rather than
            // patch it silently.
            if map.append(position).is_ok() {
                map.retain_tail(handle.cached_len());
            }
        }
    }

    /// Per-layer handles, or `None` on the coarse path / before prefill.
    pub(super) fn layer_handles(&self) -> Option<&[KvHandle]> {
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => self.handles.as_deref(),
            PrefillDispatchMode::Coarse => None,
        }
    }
}
