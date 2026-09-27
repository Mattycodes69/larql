use super::*;
use crate::KvEngine;
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::test_utils::make_test_weights;

//
// On a CPU backend, `fused_prefill` returns `None`, so the engine
// falls through to `rs_prefill_walk` against the synthetic VectorIndex.
// This exercises the prefill_quant / decode_step_quant branches that the
// Metal-only happy path also takes (apart from the Metal early-return).

//
// The two tests above use `window=None` so the cold tier never
// populates — leaving the walk.rs cold-K/V precompute (lines 66-76)
// and cold-residual decode branch (lines 162-186) uncovered. The
// tests below drive them with a small window + multiple decode steps.

/// Counting FFN that records every `forward` call. Used to prove
/// the executor path actually dispatches through the caller's
/// `FfnBackend` instead of constructing a local `WalkFfn` (the
/// legacy coupling that the migration removes).
struct CountingFfn {
    calls: std::sync::atomic::AtomicUsize,
    hidden: usize,
}
impl larql_inference::ffn::FfnBackend for CountingFfn {
    fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ndarray::Array2::zeros((x.shape()[0], self.hidden))
    }
    fn name(&self) -> &str {
        "counting"
    }
}

/// `Fused`-executor fallback in `*_via_executor` (lines 219-221, 294-296):
/// when the executor advertises fused dispatch, the engine routes back
/// through the legacy `prefill_quant` / `decode_step_quant` path.
struct FusedStubExecutor {
    backend: larql_compute::CpuBackend,
}
impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
    fn backend(&self) -> &dyn larql_compute::ComputeBackend {
        &self.backend
    }
    fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
        larql_inference::layer_executor::ExecutorDispatchKind::Fused
    }
    fn name(&self) -> &str {
        "fused-stub"
    }
}

//
// CpuBackend implements `coarse_prefill_with_state` /
// `coarse_decode_step_with_state_masked` on Q4K-backed vindexes,
// so the dispatch fast path fires on CPU when fed a Q4K fixture.
// This is where the per-layer state-capture branches live —
// including the `append_row` / `grow_capacity_2d` doubling-capacity
// buffers in lines 17-55.

//
// The Option → Result migration added typed-error paths at every
// method entry. These tests pin the entry guards so the
// `EmptyPrompt` / `InvariantViolation` variants stay constructible
// (collapsing them back into a plain `BackendFailure` would
// re-introduce the silent-failure problem the refactor exists to
// fix). Backend-failure / dispatch-None paths are exercised
// through the existing positive-path tests above when their
// synthetic backends behave normally; only the entry guards have
// dedicated negative tests here.

/// Prompt/window pair that makes the prefill overflow so the cold
/// buffers are capacity-padded (`cold_len = 2`, capacity 8).
const EXEC_WINDOW: usize = 2;
const EXEC_PROMPT: [u32; 4] = [0, 1, 2, 3];
const EXEC_NEXT_TOKEN: u32 = 4;

fn assert_bits_eq(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>, ctx: &str) {
    let a_bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let b_bits: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a_bits, b_bits, "{ctx}");
}

/// Stub violating the residual-recompute contract structurally
/// (MLA: K/V routed through a decode-time latent). Everything else
/// inherits the trait defaults from the test fixture's config.
struct MlaStubArch {
    config: larql_inference::larql_models::ModelConfig,
}
impl larql_inference::larql_models::ArchitectureCore for MlaStubArch {
    fn family(&self) -> &str {
        "mla-stub"
    }

    fn config(&self) -> &larql_inference::larql_models::ModelConfig {
        &self.config
    }
}

impl larql_inference::larql_models::LatentAttention for MlaStubArch {
    fn uses_mla(&self) -> bool {
        true
    }
}

impl larql_inference::larql_models::TensorKeys for MlaStubArch {}
impl larql_inference::larql_models::Norms for MlaStubArch {}
impl larql_inference::larql_models::Position for MlaStubArch {}
impl larql_inference::larql_models::Attention for MlaStubArch {}
impl larql_inference::larql_models::FeedForward for MlaStubArch {}
impl larql_inference::larql_models::Embeddings for MlaStubArch {}
impl larql_inference::larql_models::ModelArchitecture for MlaStubArch {}

mod construction;
mod doubling_capacity_hot_kv_consistency_reg;
mod engineerror_surface_coverage;
mod phase_2_executor_driven_path;
mod q4k_dispatch_path_try_prefill_via_dispat;
mod walk_path_overflow_branches_markov_resid;
