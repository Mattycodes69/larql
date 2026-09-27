//! `BoundaryKvEngine` — Standard semantics + `larql-boundary` frame emission.

use std::sync::Arc;

use larql_boundary::BoundaryGateConfig;
use larql_inference::async_compute_backend::AsyncComputeBackend;
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend};
use ndarray::Array2;

use crate::engines::boundary_kv::archive::{ArchiveError, BoundaryArchive, InMemoryArchive};
use crate::engines::boundary_kv::identity::BoundaryModelIdentity;
use crate::engines::standard::StandardEngine;
use crate::{EngineInfo, KvEngine};

/// Engine-level configuration.
#[derive(Debug, Clone)]
pub struct BoundaryKvEngineConfig {
    /// Hot-window cap for the inner `StandardEngine`. `None` = unbounded.
    pub window_size: Option<usize>,
    /// Capture a frame whenever the decode position reaches a positive
    /// multiple of `chunk_tokens`. Must be ≥ 1.
    pub chunk_tokens: usize,
    /// Identifies the session's chain in the archive. The archive groups
    /// frames by this id; restoring from a chain requires the same id.
    pub sequence_id: String,
    /// Embedded in every emitted frame for receiver-side verification.
    pub identity: BoundaryModelIdentity,
    /// Codec + threshold configuration for the gate.
    pub gate_config: BoundaryGateConfig,
    /// When true, run the compressed-residual forward to populate
    /// `boundary_agreement`. When false, frames carry `NotChecked` and the
    /// gate falls back per its `require_compressed_agreement` policy. See
    /// `BOUNDARY_REF_PROTOCOL.md` §8 for the cost tradeoff.
    pub verify_agreement: bool,
}

impl BoundaryKvEngineConfig {
    /// Build a default config: chunk_tokens=512, calibration-mode gate (always
    /// `UseBf16`), agreement verification enabled. Caller supplies a sequence
    /// id and a model identity.
    pub fn new(sequence_id: impl Into<String>, identity: BoundaryModelIdentity) -> Self {
        Self {
            window_size: None,
            chunk_tokens: 512,
            sequence_id: sequence_id.into(),
            identity,
            gate_config: BoundaryGateConfig::default(),
            verify_agreement: true,
        }
    }
}

/// Separator between the configured `sequence_id` and the generation
/// counter in [`BoundaryKvEngine::current_sequence_id`]. Re-prefilled
/// engines emit into `<sequence_id>#g<N>` so a new generation's frames
/// never collide with (or interleave into) an earlier chain.
const GENERATION_SEQUENCE_SEPARATOR: &str = "#g";

/// `BoundaryKvEngine` — production-equivalent in-session decode, with frame
/// emission at chunk boundaries.
pub struct BoundaryKvEngine {
    inner: StandardEngine,
    config: BoundaryKvEngineConfig,
    archive: Arc<dyn BoundaryArchive>,
    abs_position: usize,
    /// 0 for the first prompt; incremented on every re-prefill of a used
    /// engine. Selects the archive chain via [`Self::current_sequence_id`].
    generation: u64,
}

impl BoundaryKvEngine {
    /// Construct with the default CPU backend and an in-memory archive.
    pub fn new(config: BoundaryKvEngineConfig) -> Self {
        Self::with_backend(config, cpu_engine_backend())
    }

    /// Construct with a specific compute backend and an in-memory archive.
    pub fn with_backend(config: BoundaryKvEngineConfig, backend: Box<dyn EngineBackend>) -> Self {
        Self::with_backend_and_archive(config, backend, Arc::new(InMemoryArchive::new()))
    }

    /// Construct with a specific async compute backend and an in-memory archive.
    pub fn with_async_backend(
        config: BoundaryKvEngineConfig,
        backend: Box<dyn AsyncComputeBackend>,
    ) -> Self {
        let inner = StandardEngine::with_async_backend(config.window_size, backend);
        Self {
            inner,
            config,
            archive: Arc::new(InMemoryArchive::new()),
            abs_position: 0,
            generation: 0,
        }
    }

    /// Construct with a caller-supplied archive. Use this when you need
    /// durability beyond a single process.
    pub fn with_backend_and_archive(
        config: BoundaryKvEngineConfig,
        backend: Box<dyn EngineBackend>,
        archive: Arc<dyn BoundaryArchive>,
    ) -> Self {
        let inner = StandardEngine::with_backend(config.window_size, backend);
        Self {
            inner,
            config,
            archive,
            abs_position: 0,
            generation: 0,
        }
    }

    /// Borrow the archive (for inspection / chain replay).
    pub fn archive(&self) -> &Arc<dyn BoundaryArchive> {
        &self.archive
    }

    /// Current logical decode position.
    pub fn abs_position(&self) -> usize {
        self.abs_position
    }

    fn chunk_tokens(&self) -> usize {
        self.config.chunk_tokens.max(1)
    }

    fn token_start_of_current_chunk(&self) -> u64 {
        let chunk = self.chunk_tokens() as u64;
        let end = self.abs_position as u64;
        end.saturating_sub(chunk)
    }

    /// True iff `abs_position` (the just-completed step) lands on a chunk
    /// boundary. Position 0 is never a boundary.
    fn at_chunk_boundary(&self) -> bool {
        self.abs_position > 0 && self.abs_position.is_multiple_of(self.chunk_tokens())
    }

    /// The archive chain id frames are currently emitted under.
    ///
    /// Generation 0 (the first prompt) uses the configured `sequence_id`
    /// verbatim; every re-prefill of a used engine starts a fresh chain
    /// (`<sequence_id>#g<N>`). Invariant: frames from two generations never
    /// share a chain — `load_chain` sorts by `token_end`, so mixing
    /// generations would interleave frames and collide `boundary_id`s.
    pub fn current_sequence_id(&self) -> String {
        if self.generation == 0 {
            self.config.sequence_id.clone()
        } else {
            format!(
                "{}{}{}",
                self.config.sequence_id, GENERATION_SEQUENCE_SEPARATOR, self.generation
            )
        }
    }

    /// Mark the start of a prefill. A prefill on a used engine
    /// (`abs_position > 0`) begins a new generation → new archive chain.
    fn begin_generation(&mut self) {
        if self.abs_position > 0 {
            self.generation += 1;
        }
    }

    /// Build + archive a frame from the most recent hidden state. No-ops if
    /// the position is not a chunk boundary or the hidden is empty.
    fn maybe_emit_frame(
        &self,
        weights: &ModelWeights,
        hidden: &Array2<f32>,
    ) -> Result<(), ArchiveError> {
        if !self.at_chunk_boundary() {
            return Ok(());
        }
        if hidden.shape()[0] == 0 || hidden.shape()[1] == 0 {
            return Ok(());
        }
        let frame = crate::engines::boundary_kv::gate::build_frame(
            weights,
            hidden,
            &self.config,
            &self.current_sequence_id(),
            self.token_start_of_current_chunk(),
            self.abs_position as u64,
        );
        self.archive.append(frame)
    }

    /// Prefill through `inner_prefill`, emitting a frame for EVERY chunk
    /// boundary crossed by the prompt (spec §6.1) — interior boundaries
    /// included, not only a prompt length that happens to be an exact
    /// multiple of `chunk_tokens`.
    ///
    /// `StandardEngine::prefill` returns only the final hidden row and
    /// resets its KV state on every call, so interior residuals cannot be
    /// captured from a single call. Instead each boundary-aligned PREFIX is
    /// prefilled through the same inner engine first (each call replaces
    /// the previous state), and the FINAL call is the untouched single-shot
    /// prefill over the whole prompt. Invariants:
    ///
    /// - The returned hidden and the inner KV state are bit-identical to an
    ///   engine that captured no frames (§2.1) — the last `inner_prefill`
    ///   call is exactly the pre-existing single-shot path.
    /// - Each interior frame's residual is exactly `Standard`'s
    ///   last-position hidden for that prefix (same engine, same backend).
    ///
    /// Cost: a prompt spanning B interior boundaries pays B extra prefix
    /// prefills; prompts within one chunk pay nothing.
    fn prefill_chunked<F>(
        &mut self,
        weights: &ModelWeights,
        token_ids: &[u32],
        mut inner_prefill: F,
    ) -> Result<Array2<f32>, EngineError>
    where
        F: FnMut(&mut StandardEngine, &[u32]) -> Result<Array2<f32>, EngineError>,
    {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.begin_generation();
        let n = token_ids.len();
        let chunk = self.chunk_tokens();
        // Interior boundaries: positive multiples of `chunk` strictly inside
        // the prompt. A boundary at exactly `n` is emitted by the final
        // full-prompt prefill below.
        let mut boundary = chunk;
        while boundary < n {
            let hidden = inner_prefill(&mut self.inner, &token_ids[..boundary])?;
            self.abs_position = boundary;
            // Failed emit propagates as engine failure per §8.2 — a frame
            // must never be silently dropped from the resume chain.
            if self.maybe_emit_frame(weights, &hidden).is_err() {
                return Err(EngineError::BackendFailure {
                    details: "boundary frame emit failed".into(),
                });
            }
            boundary += chunk;
        }
        let hidden = inner_prefill(&mut self.inner, token_ids)?;
        self.abs_position = n;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }
}

impl KvEngine for BoundaryKvEngine {
    fn name(&self) -> &str {
        "boundary-kv"
    }

    fn info(&self) -> EngineInfo {
        let inner = self.inner.info();
        let archived = self.archive.total_frames().unwrap_or(0);
        EngineInfo {
            name: "boundary-kv".into(),
            description: format!(
                "Standard KV + boundary-frame emission every {} tokens (archived={archived})",
                self.chunk_tokens(),
            ),
            backend: inner.backend,
            config: format!(
                "chunk_tokens={},sequence_id={},window={}",
                self.chunk_tokens(),
                self.config.sequence_id,
                self.config
                    .window_size
                    .map(|w| w.to_string())
                    .unwrap_or_else(|| "full".into()),
            ),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_chunked(weights, token_ids, |inner, toks| {
            inner.prefill(weights, ffn, toks)
        })
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let hidden = self.inner.decode_step(weights, ffn, token_id)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }

    /// Resident-path prefill: forwards to the inner `StandardEngine`'s
    /// resident form (threads `index` → Q4K-direct attention family) and
    /// keeps the boundary frame emission identical to [`Self::prefill`].
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_chunked(weights, token_ids, |inner, toks| {
            inner.prefill_resident(weights, ffn, index, toks)
        })
    }

    /// Resident-path decode: forwards to the inner `StandardEngine`'s
    /// resident form; frame emission identical to [`Self::decode_step`].
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        let hidden = self
            .inner
            .decode_step_resident(weights, ffn, index, token_id)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.inner.window_tokens()
    }

    fn cold_bytes(&self) -> usize {
        self.inner.cold_bytes()
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_chunked(weights, token_ids, |inner, toks| {
            inner.prefill_quant(weights, ffn, index, toks, backend)
        })
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        let hidden = self
            .inner
            .decode_step_quant(weights, ffn, index, token_id, backend)?;
        self.abs_position += 1;
        if self.maybe_emit_frame(weights, &hidden).is_err() {
            return Err(EngineError::BackendFailure {
                details: "boundary frame emit failed".into(),
            });
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests;
