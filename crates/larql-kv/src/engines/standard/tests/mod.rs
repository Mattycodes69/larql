use super::*;
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::test_utils::make_test_weights;

//
// The predicate that decides whether a failed decode step can be undone.
// Tested directly because the interesting cases are boundary conditions on
// the window, and driving each of them through a real refusal would need a
// refusing route per case while proving the same three facts.

//
// Verifies that `prefill(tokens)` and
// `prefill_from_hidden(embed_tokens_pub(tokens))` agree on BOTH:
//   (a) the returned hidden state (catches dispatch-level drift),
//   (b) the post-prefill `abs_position` (catches the off-by-one
//       that would silently garble decode-loop continuation).
//
// The `abs_position` check is the load-bearing one — the new line
// in `do_prefill_from_hidden` (`self.abs_position = initial_hidden.nrows()`)
// is the only genuinely new logic in this PR. If it's wrong, the
// first decoded token after MM prefill gets the wrong RoPE position
// and the entire continuation is garbled, but the prefill itself
// looks fine. This test pins that one line.

//
// `StandardEngine` is the engine-trait wrapper over the production K/V
// cache. Driven through `generate_with_engine`, its token output must
// be bit-identical to `generate_cached_backend` on the same inputs.
// This is the unification's bit-parity gate (spec §8.4); failure here
// blocks Step 5 (default flip).

use crate::generation::{generate_cached_backend, generate_with_engine};
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
    let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(window)));
    generate_with_engine(&mut engine, weights, tokenizer, ffn, prompt, max, |_, _| {})
}

// The five parity tests below assert bit-exact equality between
// two code paths that route the same matmuls through different
// dispatch wrappers. BLAS on Windows runs successive matmuls with
// different reduction orders (parallel accumulation), so the two
// paths drift by a few times 1e-3 — enough to flip argmax in a
// token stream and break hidden-state bit-equality. Linux/macOS
// BLAS is deterministic and the property holds there; we keep the
// strict check on those platforms and skip on Windows rather than
// weaken to a fuzzy tolerance that wouldn't catch real bugs.

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
    let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::with_async_backend(
        window, backend,
    )));
    generate_with_engine(&mut engine, weights, tokenizer, ffn, prompt, max, |_, _| {})
}

//
// `prefill_quant` / `decode_step_quant` first try the backend's
// `coarse_prefill` / `coarse_decode_step`. The Q4K-equipped fixture
// (`make_test_q4k_vindex` + `make_test_q4k_weights`) satisfies the
// cached-decode contract, so an UNWINDOWED engine takes the coarse
// path on `CpuBackend` (verified by
// `coarse_prefill_records_coarse_mode_and_decodes_coarse`); the
// per-layer dequant fallback is exercised by the windowed tests
// further down (windowed engines decline coarse by design).

//
// `prefill` and `prefill_quant` reject an empty token slice with
// `EngineError::EmptyPrompt` *before* touching the backend. Pinning
// both arms keeps the early-return invariant covered (the dispatch
// helpers would otherwise return None and the caller could not tell
// "empty input" from "backend failed").

//
// `prefill_resident` / `decode_step_resident` are the
// `--moe-shards`/`LARQL_Q4K_DIRECT_ATTN` entry points: they take
// `&ModelWeights` (immutable — the caller has already made the client
// weights f32-resident) and thread `index` to the per-layer dispatch.
// With `LARQL_Q4K_DIRECT_ATTN` unset (the default in tests) the CPU
// backend ignores the index and reads f32 from `weights.tensors`, so
// driving these with f32 test weights + a Q4K vindex exercises the
// resident code path (`do_prefill` / `do_decode_step` with
// `Some(index)`) without needing the Metal-only direct kernel.

//
// `prefill_quant` / `decode_step_quant` both branch on the
// `BackendSlot` to call `coarse_prefill` / `coarse_decode_step`.
// The sync-slot arms are covered by the Q4K fixture tests above; these
// drive the *async* slot arms. `CpuBackend`'s coarse path returns None
// either way, so the engine still falls through to the per-layer
// dequant fallback — but the async match arm (and its `coarse_*` call)
// is now executed.

//
// The coarse trait surface (`coarse_prefill` / `coarse_decode_step`)
// has no window parameter, so a windowed engine that took the coarse
// path silently attended over the FULL context while `info()`
// reported `window=N`. The fix declines coarse whenever
// `window_size.is_some()` and routes through the per-layer path,
// whose dispatch enforces the window via `clip_kv`. These tests
// assert the actual row accounting, not just shape.

/// Window used by the windowed-quant tests — deliberately smaller
/// than the prompt so prefill-time clipping is exercised.
const QUANT_WINDOW: usize = 2;
/// Decode steps run after prefill in the windowed-quant tests.
const QUANT_DECODE_STEPS: usize = 3;

//
// The bench/CLI passes `NullFfn` on the quant path (engines route
// FFN internally from the vindex — see `NoCacheEngine`). The
// per-layer quant fallback used to forward the caller's `NullFfn`
// verbatim into `run_ffn`, silently producing `h + normed(h)`
// garbage on exactly the archs that decline coarse. Post-fix the
// fallback substitutes a `WalkFfn` built from the vindex; this test
// bit-compares against an explicitly-constructed `WalkFfn` driven
// through the same internals.

/// Window NARROWER than the prompt, so the backend declines the
/// fused path and both engines run the per-layer quant walk this
/// test is about.
///
/// It used to be a window *wider* than the whole run, on the premise
/// that "coarse is declined whenever windowed". That premise is gone:
/// a backend now accepts a window it can honour, and a window wider
/// than the prompt is trivially honourable — so engine A took the
/// fused path while the reference stayed per-layer and the two
/// legitimately disagreed. Narrower-than-prompt is the property that
/// actually forces per-layer now.
///
/// The window then clips, but it clips BOTH engines identically
/// (same `window_size`, same `do_prefill` internals), so the FFN
/// routing this test isolates is unaffected.
const FORCE_PER_LAYER_WINDOW: usize = 2;

//
// `decode_step_quant` used to infer "coarse handle" from
// `handles.len() == 1`, which conflates it with "1-layer model":
// a 1-layer model prefilled via the per-layer path hit the coarse
// downcast (`cpu_q4k_cache_mut`) with a `CpuKvHandle` and panicked.
// The engine now records which mode prefill used and decode follows
// the recorded mode.

mod a5_parity_gate;
mod dispatch_path_reporting;
mod prefill_mode_tracking_bug_fix;
mod rewind_soundness;
mod step_4_parity_gate;
mod windowed_quant_path_honors_the_window_bu;
