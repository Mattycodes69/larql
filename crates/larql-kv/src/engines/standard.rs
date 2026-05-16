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
use larql_inference::kv_dispatch::helpers::{
    kv_decode_step_via_dispatch, kv_decode_step_via_dispatch_async, kv_prefill_via_dispatch,
    kv_prefill_via_dispatch_async,
};
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend, KvHandle};

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

/// Production K/V cache engine. `window_size: None` = unbounded growth
/// (the `--kv-cache standard` flag); `Some(N)` = sliding window (the
/// `--kv-cache markov-bounded --context-window N` flag combo).
pub struct StandardEngine {
    window_size: Option<usize>,
    /// One handle per layer; populated by `prefill`. `None` before
    /// prefill or if the engine has been reset.
    handles: Option<Vec<KvHandle>>,
    /// Tracks the absolute token position of the next token to be
    /// decoded. Set at the end of `prefill` to `prompt_ids.len()`;
    /// incremented after each `decode_step`. The legacy `KvCache` had
    /// its own `next_position` field; this engine tracks it directly.
    abs_position: usize,
    backend: BackendSlot,
}

impl StandardEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            handles: None,
            abs_position: 0,
            backend: BackendSlot::Sync(backend),
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
            abs_position: 0,
            backend: BackendSlot::Async(backend),
        }
    }

    fn cache_memory_bytes(&self) -> usize {
        let Some(handles) = self.handles.as_ref() else {
            return 0;
        };
        handles
            .iter()
            .map(|h| {
                // 2 × f32 per cached row (K + V), kv_dim wide.
                h.cached_len() * h.kv_dim() * 2 * std::mem::size_of::<f32>()
            })
            .sum()
    }
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
    ) -> Option<Array2<f32>> {
        let (hidden, handles) = match &self.backend {
            BackendSlot::Sync(b) => {
                kv_prefill_via_dispatch(b.as_ref(), weights, ffn, token_ids, self.window_size)?
            }
            BackendSlot::Async(b) => kv_prefill_via_dispatch_async(
                b.as_ref(),
                weights,
                ffn,
                token_ids,
                self.window_size,
            )?,
        };
        self.handles = Some(handles);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let handles = self.handles.as_mut()?;
        let hidden = match &self.backend {
            BackendSlot::Sync(b) => kv_decode_step_via_dispatch(
                b.as_ref(),
                weights,
                ffn,
                handles,
                token_id,
                self.abs_position,
                self.window_size,
            )?,
            BackendSlot::Async(b) => kv_decode_step_via_dispatch_async(
                b.as_ref(),
                weights,
                ffn,
                handles,
                token_id,
                self.abs_position,
                self.window_size,
            )?,
        };
        self.abs_position += 1;
        Some(hidden)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn engine_name() {
        assert_eq!(StandardEngine::new(None).name(), "standard");
    }

    #[test]
    fn engine_info_unbounded() {
        let info = StandardEngine::new(None).info();
        assert!(info.config.contains("full"));
    }

    #[test]
    fn engine_info_windowed() {
        let info = StandardEngine::new(Some(128)).info();
        assert!(info.config.contains("128"));
    }

    #[test]
    fn memory_zero_before_prefill() {
        let eng = StandardEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn prefill_populates_cache_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0, "cache should be populated");
        assert!(engine.window_tokens() >= 3);
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn cache_grows_with_decode_steps() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let after_one = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
        let after_two = engine.memory_bytes();
        assert!(after_one > after_prefill);
        assert!(after_two > after_one);
    }

    #[test]
    fn sliding_window_clips_cache() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let window = 2usize;
        let mut engine = StandardEngine::new(Some(window));
        // Prefill with 4 tokens — cache should clip to last `window` per layer.
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(
            engine.window_tokens() <= window,
            "expected window_tokens ≤ {window}, got {}",
            engine.window_tokens()
        );
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        assert!(engine.decode_step(&weights, &ffn, 0).is_none());
    }

    // ── Step 4 parity gate ─────────────────────────────────────────────────
    //
    // `StandardEngine` is the engine-trait wrapper over the production K/V
    // cache. Driven through `generate_with_engine`, its token output must
    // be bit-identical to `generate_cached_backend` on the same inputs.
    // This is the unification's bit-parity gate (spec §8.4); failure here
    // blocks Step 5 (default flip).

    use larql_inference::forward::{generate_cached_backend, generate_with_engine};
    use larql_inference::test_utils::make_test_tokenizer;

    fn run_legacy(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        generate_cached_backend(
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            None,
            window,
            |_, _| {},
        )
    }

    fn run_engine(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        let mut engine = StandardEngine::new(window);
        generate_with_engine(
            &mut engine as &mut dyn crate::KvEngine,
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            |_, _| {},
        )
    }

    #[test]
    fn parity_standard_unbounded_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5, 7];
        let max = 6;
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, None);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens to generate_cached_backend (window=None)"
        );
    }

    #[test]
    fn parity_standard_windowed_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[1u32, 2, 3, 4, 5];
        let max = 5;
        // Window smaller than prompt → exercises prefill-time clipping.
        let window = Some(3);
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens to generate_cached_backend (sliding window)"
        );
    }

    #[test]
    fn parity_standard_short_prompt_long_window_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[0u32, 1];
        let max = 4;
        let window = Some(64); // window > prompt — exercises decode-time growth past prompt
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens at short-prompt long-window edge case"
        );
    }

    // ── A5 parity gate ──────────────────────────────────────────────
    //
    // `StandardEngine::with_async_backend(CpuBackend)` must produce
    // bit-identical token streams to `StandardEngine::new(CpuBackend)`.
    // CpuBackend's `AsyncComputeBackend` impl is a degenerate
    // `Ready<T>` wrapper around the sync `KvDispatch` (`A2`), so
    // bit-parity is the trait-shape correctness contract for engine
    // opt-in. Spec: `async-compute-backend.md` §10.5.

    use larql_compute::CpuBackend;
    use larql_inference::AsyncComputeBackend;

    fn run_engine_async(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let mut engine = StandardEngine::with_async_backend(window, backend);
        generate_with_engine(
            &mut engine as &mut dyn crate::KvEngine,
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            |_, _| {},
        )
    }

    #[test]
    fn async_parity_standard_unbounded_matches_sync_engine() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5, 7];
        let max = 6;
        let sync = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
        let asynch = run_engine_async(&weights, &tokenizer, &ffn, prompt, max, None);
        assert_eq!(
            sync, asynch,
            "with_async_backend must produce identical tokens to with_backend (CpuBackend, window=None)"
        );
    }

    #[test]
    fn async_parity_standard_windowed_matches_sync_engine() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[1u32, 2, 3, 4, 5];
        let max = 5;
        let window = Some(3);
        let sync = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        let asynch = run_engine_async(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            sync, asynch,
            "with_async_backend must produce identical tokens to with_backend (CpuBackend, sliding window)"
        );
    }

    #[test]
    fn async_engine_reports_backend_name() {
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let engine = StandardEngine::with_async_backend(None, backend);
        // info() reports the underlying ComputeBackend::name() regardless
        // of which slot variant the engine holds. CpuBackend returns
        // "cpu (BLAS + C Q4 kernel)" or similar — just assert the prefix.
        assert!(
            engine.info().backend.starts_with("cpu"),
            "expected backend name to start with \"cpu\", got {:?}",
            engine.info().backend
        );
    }
}
