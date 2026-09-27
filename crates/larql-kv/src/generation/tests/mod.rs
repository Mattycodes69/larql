use super::*;
use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::{make_test_tokenizer, make_test_weights};

//
// Synthetic engine that returns deterministic hidden states to drive
// the helper through each branch: empty inputs, max_new_tokens=0,
// max_new_tokens=1, normal multi-step generation, prefill failure,
// decode failure.

struct StubEngine {
    cache: Option<KvCache>,
    fail_prefill: bool,
    fail_decode_after: Option<usize>,
    decode_count: usize,
}

impl crate::KvEngine for StubEngine {
    fn name(&self) -> &str {
        "stub"
    }
    fn info(&self) -> crate::EngineInfo {
        crate::EngineInfo {
            name: "stub".into(),
            description: "test fixture".into(),
            backend: "cpu".into(),
            config: String::new(),
        }
    }
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
        if self.fail_prefill {
            return Err(larql_inference::kv_engine::EngineError::BackendFailure {
                details: "test stub: fail_prefill set".into(),
            });
        }
        let (hidden, cache) = kv_prefill_run(
            larql_inference::WeightsView::dense(weights),
            ffn,
            token_ids,
            None,
            None,
            &mut NoopHook,
        )?;
        self.cache = Some(cache);
        Ok(hidden)
    }
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
        self.decode_count += 1;
        if let Some(limit) = self.fail_decode_after {
            if self.decode_count > limit {
                return Err(larql_inference::kv_engine::EngineError::BackendFailure {
                    details: "test stub: fail_decode_after exceeded".into(),
                });
            }
        }
        let cache = self.cache.as_mut().ok_or_else(|| {
            larql_inference::kv_engine::EngineError::InvariantViolation {
                what: "decode_step called before prefill".into(),
            }
        })?;
        kv_decode_step_run(weights, ffn, cache, token_id, None, &mut NoopHook)
    }
    // MM support: drive `generate_with_engine_from_hidden`. We can't
    // recover the original tokens from a pre-built hidden state, so the
    // stub seeds a fresh cache from synthetic ids `0..nrows`. That's a
    // valid populated cache (the from-hidden tests assert control flow
    // — break arms, EOS, max-tokens budget — not bit-parity with a
    // real embed).
    fn supports_multimodal(&self) -> bool {
        true
    }
    fn prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
        if self.fail_prefill {
            return Err(larql_inference::kv_engine::EngineError::BackendFailure {
                details: "test stub: fail_prefill set".into(),
            });
        }
        let ids: Vec<u32> = (0..initial_hidden.nrows() as u32).collect();
        let (hidden, cache) = kv_prefill_run(
            larql_inference::WeightsView::dense(weights),
            ffn,
            &ids,
            None,
            None,
            &mut NoopHook,
        )?;
        self.cache = Some(cache);
        Ok(hidden)
    }
    // Resident-weights path: drive `generate_with_engine_resident`. The
    // stub ignores `index` and reuses the f32 prefill/decode bodies —
    // enough to exercise the wrapper's prefill/decode/break arms.
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        _index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
        self.prefill(weights, ffn, token_ids)
    }
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        _index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, larql_inference::kv_engine::EngineError> {
        self.decode_step(weights, ffn, token_id)
    }
    fn memory_bytes(&self) -> usize {
        0
    }
}

fn fresh_stub() -> StubEngine {
    StubEngine {
        cache: None,
        fail_prefill: false,
        fail_decode_after: None,
        decode_count: 0,
    }
}

//
// Before this PR, `kv_prefill_run` and `kv_decode_step_run` called
// `run_attention*` + `run_ffn` directly, skipping the
// `apply_per_layer_embedding` and `apply_layer_scalar` steps that
// `run_layer_with_ffn` performs. On Gemma-4 (`gemma-4-E4B-it`),
// the missing PLE contribution compounded across decode steps and
// produced garbage (`ッケッケTobchal的存在` after a correct first
// token). These tests pin both phases through the synthetic E2B-like
// fixture so any future regression that drops PLE / layer_scalar
// from the cached path fails locally rather than at the user's
// terminal.

//
// Two pins:
//   1. Bit-identity: a single-Tokens-chunk plan run through
//      embed_plan → generate_with_engine_from_hidden produces the
//      same token stream as generate_with_engine(tokens). This is
//      the analog of the dispatch-level bit-identity test, applied
//      one layer up. If they diverge, something in the wrapper
//      silently dropped a flag/callback/state vs the original.
//   2. max_tokens accounting independent of initial_hidden.nrows().
//      The contract is "max_new_tokens counts decoded tokens only,
//      never prefill rows" — this catches the off-by-one risk that
//      would manifest as captions being one token short, or vision
//      tokens being counted against the user's --max-tokens budget.

//
// The from-hidden wrapper duplicates the prefill→sample→decode loop of
// `generate_with_engine`, including the early-return / break arms:
// zero-max budget, prefill failure, max_new=1, and decode failure
// mid-loop. These mirror the `generate_with_engine_*` stub tests above
// but drive the from-hidden code path (engine.prefill_from_hidden +
// engine.decode_step). The StubEngine seeds a fresh cache from
// synthetic ids, so the assertions are on control flow, not parity.

fn hidden_for(weights: &ModelWeights, rows: usize) -> Array2<f32> {
    let mut h = Array2::<f32>::zeros((rows, weights.hidden_size));
    for r in 0..rows {
        for c in 0..weights.hidden_size {
            h[[r, c]] = ((r * 7 + c * 3) % 11) as f32 * 0.01 - 0.05;
        }
    }
    h
}

//
// The resident wrapper drives `engine.prefill_resident` /
// `engine.decode_step_resident`, threading the `index`. With
// `LARQL_Q4K_DIRECT_ATTN` unset (default), the CPU backend ignores the
// index and runs the f32 path, so a StandardEngine over f32 test
// weights + a Q4K vindex exercises the full happy path. The stub drives
// the zero-max / prefill-failure / max-one / decode-failure break arms.

mod gemma_4_ple_arch_coverage_regression_tes;
mod generate_cached_hooked;
mod generate_with_engine_coverage;
mod generate_with_engine_from_hidden_break_a;
mod generate_with_engine_resident_coverage;
mod phase_1d_3b_generate_with_engine_from_hi;
