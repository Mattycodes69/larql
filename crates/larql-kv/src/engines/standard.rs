//! StandardEngine — the production K/V cache, wrapped as a `KvEngine`.
//!
//! Step 3c (2026-05-16): migrated from direct `kv_prefill_run` /
//! `kv_decode_step_run` calls to dispatch through
//! [`larql_inference::EngineBackend`] via
//! [`kv_prefill_via_dispatch`] / [`kv_decode_step_via_dispatch`].
//! Cache state is now `Vec<KvHandle>` (one per layer) instead of
//! `KvCache`. Bit-parity with the legacy path is preserved (verified
//! in this file's parity tests + `larql-kv`'s end-to-end suite).
//!
//! Output is bit-identical to today's `--kv-cache standard` (with
//! `window_size: None`) and `--kv-cache markov-bounded`
//! (with `window_size: Some(N)`).

use ndarray::Array2;

use crate::{EngineInfo, KvEngine};
use larql_inference::async_compute_backend::AsyncComputeBackend;
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{EngineBackend, KvHandle};

mod inherent;
mod per_layer_access;

/// The new position's content for one decode step: a token id (text
/// path — embedded inside the dispatch helper) or a pre-built hidden
/// row (multi-modal / summed-embedding path, MOSS-TTS-Realtime).
enum StepInput<'a> {
    Token(u32),
    Hidden(&'a Array2<f32>),
}

/// Backend slot — `StandardEngine` accepts either a synchronous
/// [`EngineBackend`] (the default `--kv-cache standard` path) or an
/// [`AsyncComputeBackend`] (opt-in via [`StandardEngine::with_async_backend`]).
///
/// The async variant routes prefill/decode through the async helpers
/// in [`larql_inference::kv_dispatch::helpers`]. At Step A3 of the
/// `async-compute-backend.md` migration, async output is bit-identical
/// to sync on CPU; the win is on Metal once Step A4's deferred dispatch
/// lands.
enum BackendSlot {
    Sync(Box<dyn EngineBackend>),
    Async(Box<dyn AsyncComputeBackend>),
}

impl BackendSlot {
    fn name(&self) -> &str {
        match self {
            BackendSlot::Sync(b) => b.name(),
            BackendSlot::Async(b) => b.name(),
        }
    }
}

/// Which dispatch shape populated `handles` at prefill. Decode must
/// follow the recorded mode: the two shapes are not interchangeable
/// (a coarse handle is one whole-model `CpuQ4kCacheHandle`/Metal cache;
/// per-layer handles are one `CpuKvHandle` per layer), and inferring
/// the mode from `handles.len()` conflates "coarse handle" with
/// "1-layer model" — the misdispatch panicked in the backend downcast.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefillDispatchMode {
    /// Backend coarse (fused) path: `handles` is a single whole-model
    /// cache handle from `coarse_prefill`.
    Coarse,
    /// Per-layer dispatch: `handles` has one entry per layer.
    PerLayer,
}

/// Production K/V cache engine. `window_size: None` = unbounded growth
/// (the `--kv-cache standard` flag); `Some(N)` = sliding window (the
/// `--kv-cache markov-bounded --context-window N` flag combo).
///
/// Windowed quant note: the coarse trait surface has no window
/// parameter, so when `window_size` is `Some(N)` the quant entry points
/// decline the coarse path and run per-layer dispatch (which enforces
/// the window via `clip_kv`) — correctness over speed. See
/// [`Self::prefill_quant`].
pub struct StandardEngine {
    window_size: Option<usize>,
    /// One handle per layer (per-layer mode) or a single whole-model
    /// handle (coarse mode); populated by `prefill*`. `None` before
    /// prefill or if the engine has been reset.
    handles: Option<Vec<KvHandle>>,
    /// Which dispatch shape populated `handles` — set together with
    /// `handles` at every prefill entry point; `None` iff `handles`
    /// is `None`. Decode entry points follow this, never handle count.
    prefill_mode: Option<PrefillDispatchMode>,
    /// Tracks the absolute token position of the next token to be
    /// decoded. Set at the end of `prefill` to `prompt_ids.len()`;
    /// incremented after each `decode_step`. The legacy `KvCache` had
    /// its own `next_position` field; this engine tracks it directly.
    abs_position: usize,
    /// Logical position of each resident row, per layer. Kept congruent
    /// with `handles` by every operation that adds or removes rows —
    /// prefill, decode, excise, splice, replace. Empty on the coarse
    /// path, which has no per-layer rows to describe.
    row_positions: larql_inference::kv_row_positions::KvRowPositions,
    backend: BackendSlot,
    /// Engine-owned f32 dequant scratch for the per-layer fallback Q4K path
    /// (`prefill_quant`/`decode_step_quant` populate it; `do_prefill`/
    /// `do_decode_step` resolve attention/FFN through a
    /// `WeightsView::with_scratch` over it). Empty on the dense path. Keeps
    /// `weights` immutable so the engine can hold `Arc<ModelWeights>`.
    dequant_scratch: larql_inference::DequantScratch,
    /// Set when a failed decode step left K/V that could not be rewound,
    /// carrying the rendered cause. While set, decode entry points refuse
    /// with [`EngineError::InvariantViolation`] rather than compute from a
    /// cache that describes no token sequence.
    ///
    /// Cleared by a successful prefill, which replaces the cache outright —
    /// so re-prefilling is the documented way back, and the engine is never
    /// permanently dead.
    invalidated: Option<String>,
}

impl KvEngine for StandardEngine {
    fn name(&self) -> &str {
        "standard"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.cache_memory_bytes();
        EngineInfo {
            name: "standard".into(),
            description: format!(
                "production K/V tensor cache — full FP32 K/V per layer (mem={:.1}MB)",
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
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.do_prefill(weights, ffn, token_ids, None)
    }

    fn supports_multimodal(&self) -> bool {
        // StandardEngine is the first (Phase 1d) engine to implement
        // `prefill_from_hidden`. Six other engines inherit the default
        // `false` per ADR-0023 §"Default-false debt". When each of them
        // gains real MM support (or the eventual `prefill →
        // embed_tokens_pub + prefill_from_hidden` collapse lands across
        // the board), the override here becomes redundant and the
        // trait method itself can be removed.
        true
    }

    fn prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_prefill_from_hidden(weights, ffn, initial_hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step(weights, ffn, token_id, None)
    }

    fn decode_step_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        hidden_row: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step_input(weights, ffn, StepInput::Hidden(hidden_row), None)
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_inference::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Try the backend's coarse (fused) prefill intent first — this
        // is the production-speed Q4K path on CPU (~24 tok/s on Gemma
        // 3 4B vs ~0.4 tok/s through per-layer dispatch). Quant-agnostic:
        // the backend inspects `index` to pick the right kernel.
        //
        // WINDOW: ask the backend to honour this engine's window on the
        // fused path. `coarse_*_windowed` fails closed — a backend that
        // cannot bound BOTH attention and K/V to `window_size` answers
        // `None`, and we fall through to the per-layer path below, which
        // enforces the window via `clip_kv`.
        //
        // That decline is the current answer on every backend, so this
        // is behaviour-preserving today. It replaces a blanket
        // `window_size.is_none()` gate: the reason a windowed engine
        // leaves the fast path is now the backend's own capability
        // answer rather than a rule stated here, so a backend that
        // implements the window starts getting the fused path without
        // this engine changing. That matters — on a host-delegating
        // backend the per-layer route runs the whole forward on the CPU
        // and costs ~2.4x, while the window makes attention *cheaper*.
        let coarse = match &self.backend {
            BackendSlot::Sync(b) => b.as_ref().coarse_prefill_windowed(
                weights,
                token_ids,
                Some(index),
                self.window_size,
            ),
            BackendSlot::Async(b) => b.as_ref().coarse_prefill_windowed(
                weights,
                token_ids,
                Some(index),
                self.window_size,
            ),
        };
        if let Some((hidden, handle)) = coarse {
            // Store as a single-element handles vec — the `KvHandle`
            // wraps the backend's whole-model cache (not per-layer).
            self.handles = Some(vec![handle]);
            self.prefill_mode = Some(PrefillDispatchMode::Coarse);
            self.abs_position = token_ids.len();
            self.reset_row_positions_after_prefill();
            // Same reasoning as the per-layer prefills: the cache is new.
            self.invalidated = None;
            return Ok(hidden);
        }
        // Backend doesn't have a coarse path (e.g. f32 model, or
        // hybrid-MoE / cross-layer-KV models that don't fit the cached
        // shape), or the engine is windowed. Fall back to per-layer
        // dispatch, dequantising attention into the engine-owned scratch
        // (`do_prefill` resolves it through a `WeightsView::with_scratch`)
        // — `weights` stays immutable.
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        // The caller passes `NullFfn` on the quant path (the bench/CLI
        // contract: engines route FFN internally from the vindex — see
        // `NoCacheEngine::prefill_quant`). Forwarding it verbatim would
        // run an identity FFN through `run_ffn` (h + normed(h) garbage)
        // on exactly the archs that decline coarse, so substitute the
        // real Q4K FFN walk built from the vindex.
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            weights,
            index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(backend);
        self.do_prefill(weights, &walk_ffn, token_ids, Some(index))
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_inference::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        // Decode must follow the dispatch mode RECORDED at prefill —
        // never the handle count: a 1-layer model's per-layer prefill
        // also yields exactly one handle, and feeding that per-layer
        // handle to the coarse path is a shape misdispatch (the backend
        // downcast used to panic on it).
        let mode = self
            .prefill_mode
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (handles missing)".into(),
            })?;
        // The coarse branch below bypasses `do_decode_step`, so it needs the
        // invalidation guard in its own right — a guard that only covers the
        // path that *sets* the flag protects the wrong half.
        if let Some(what) = &self.invalidated {
            return Err(EngineError::InvariantViolation {
                what: format!("decode_step_quant called on an invalidated engine: {what}"),
            });
        }
        if mode == PrefillDispatchMode::Coarse {
            let handles = self
                .handles
                .as_mut()
                .ok_or_else(|| EngineError::InvariantViolation {
                    what: "decode_step called before prefill (handles missing)".into(),
                })?;
            // Invariant: Coarse mode stores exactly one whole-model
            // handle (set together with the mode in `prefill_quant`).
            let handle = &mut handles[0];
            // Windowed variant, matching the prefill that minted this
            // handle: a sequence that reached the fused path under a
            // window must keep decoding under it.
            let coarse = match &self.backend {
                BackendSlot::Sync(b) => b.as_ref().coarse_decode_step_windowed(
                    weights,
                    token_id,
                    Some(index),
                    handle,
                    self.abs_position,
                    self.window_size,
                ),
                BackendSlot::Async(b) => b.as_ref().coarse_decode_step_windowed(
                    weights,
                    token_id,
                    Some(index),
                    handle,
                    self.abs_position,
                    self.window_size,
                ),
            };
            return match coarse {
                Some(h) => {
                    self.abs_position += 1;
                    Ok(h)
                }
                // A coarse-prefilled cache CANNOT flow through the
                // per-layer path (whole-model handle vs one handle per
                // layer), so a coarse decode failure is terminal —
                // erroring beats the old behaviour of falling through
                // and panicking in the per-layer downcast.
                None => Err(EngineError::BackendFailure {
                    details: "coarse_decode_step failed on a coarse-prefilled cache; \
                              the whole-model handle cannot be decoded per-layer"
                        .into(),
                }),
            };
        }
        // Per-layer dispatch (recorded mode: PerLayer). Dequantise
        // attention into the engine-owned scratch (idempotent — persists
        // across decode steps); `do_decode_step` resolves it through a
        // `WeightsView::with_scratch`.
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        // Same FFN substitution as `prefill_quant` — the caller's FFN
        // is `NullFfn` by contract on the quant path; route the real
        // Q4K FFN walk from the vindex.
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            weights,
            index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(backend);
        self.do_decode_step(weights, &walk_ffn, token_id, Some(index))
    }

    /// Resident-weights variants (task #16): the caller has already made the
    /// client weights f32-resident, so these take `&weights` (no `&mut` for
    /// `ensure_attn_tensors_dequantised`) and just thread `index` to the
    /// per-layer dispatch. That lets the FFN backend borrow the same `&weights`
    /// concurrently (no borrow conflict) while a Q4K-direct attention kernel
    /// reads packed bytes from `index`. CPU uses per-layer handles (the coarse
    /// path is Metal-only), so these route straight to `do_*`.
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.do_prefill(weights, ffn, token_ids, Some(index))
    }

    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step(weights, ffn, token_id, Some(index))
    }

    fn per_layer_kv_mut(
        &mut self,
    ) -> Option<&mut dyn larql_inference::kv_engine::PerLayerKvAccess> {
        // Only the per-layer dispatch shape can offer row access.
        matches!(self.prefill_mode, Some(PrefillDispatchMode::PerLayer)).then_some(self)
    }

    fn memory_bytes(&self) -> usize {
        self.cache_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.handles
            .as_ref()
            .and_then(|h| h.first())
            .map(|h| h.cached_len())
            .unwrap_or(0)
    }

    fn cold_bytes(&self) -> usize {
        // Standard cache does not have a separate cold tier — the K/V
        // tensors are the state. Sliding-window evictions drop data
        // entirely; nothing is moved to cold.
        0
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `prefill_mode` is the authority (see its declaration): handle
        // count cannot distinguish a coarse handle from a 1-layer model.
        self.prefill_mode.map(|mode| match mode {
            PrefillDispatchMode::Coarse => DispatchPath::Coarse,
            PrefillDispatchMode::PerLayer => DispatchPath::PerLayer,
        })
    }
}

// ─── Per-layer K/V access (substrate only) ───────────────────────────────────
//
// Mechanical row surgery for policy wrappers. `StandardEngine` still
// reports `logical_source_masking: false` — it does not decide which
// rows leave, which layers are touched, or how a removal is undone.
// Only the dense/per-layer prefill path offers this; the coarse quant
// path has a single whole-model handle with no per-layer granularity,
// and `prefill_mode` is what distinguishes them.

#[cfg(test)]
mod tests;
