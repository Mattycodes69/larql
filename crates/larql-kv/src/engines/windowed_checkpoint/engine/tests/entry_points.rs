//! Entry-point contracts of the `KvEngine` impl: empty prompts, the
//! reported dispatch path, fused-executor fallback, and every way a
//! backend or executor declining a step surfaces as an error instead of
//! a silently stale hidden state.

use super::*;
use larql_compute::kv_dispatch::KvDispatch;
use larql_inference::cpu_engine_backend;
use larql_inference::ffn::{NullFfn, WeightFfn};
use larql_inference::kv_engine::DispatchPath;
use larql_inference::layer_executor::{ExecutorDispatchKind, LayerExecutor, LocalWalkExecutor};
use larql_inference::test_utils::{
    make_test_q4k_vindex, make_test_q4k_weights, make_test_vindex, make_test_weights,
};

const WINDOW: usize = 8;
const PROMPT: [u32; 3] = [0, 1, 2];
const NEXT: u32 = 3;

/// An executor with the trait's default layer bodies: it computes nothing,
/// so every layer it is asked to run comes back `None`.
struct LayerlessExecutor {
    backend: larql_compute::CpuBackend,
    kind: ExecutorDispatchKind,
}

impl LayerlessExecutor {
    fn new(kind: ExecutorDispatchKind) -> Self {
        Self {
            backend: larql_compute::CpuBackend,
            kind,
        }
    }
}

impl LayerExecutor for LayerlessExecutor {
    fn backend(&self) -> &dyn larql_compute::ComputeBackend {
        &self.backend
    }
    fn dispatch_kind(&self) -> ExecutorDispatchKind {
        self.kind
    }
    fn name(&self) -> &str {
        "layerless"
    }
}

fn is_backend_failure(result: Result<Array2<f32>, EngineError>) -> bool {
    matches!(result, Err(EngineError::BackendFailure { .. }))
}

#[test]
fn every_prefill_entry_point_refuses_an_empty_prompt() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let mut engine = WindowedCheckpointEngine::new(WINDOW);

    let empty = |r: Result<Array2<f32>, EngineError>| matches!(r, Err(EngineError::EmptyPrompt));
    assert!(empty(engine.prefill(&weights, &ffn, &[])));
    assert!(empty(engine.prefill_quant(
        &weights,
        &ffn,
        &index,
        &[],
        &*backend
    )));
    assert!(empty(engine.prefill_quant_via_executor(
        &weights,
        &executor,
        &ffn,
        &index,
        &[]
    )));
    assert_eq!(engine.window_tokens(), 0, "nothing was consumed");
}

#[test]
fn dispatch_path_is_none_before_prefill_and_per_layer_after_a_walk() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(WINDOW);
    assert_eq!(engine.dispatch_path(), None);
    engine.prefill(&weights, &ffn, &PROMPT).unwrap();
    assert_eq!(engine.dispatch_path(), Some(DispatchPath::PerLayer));
}

/// A Q4K prefill through the coarse state-capturing dispatch.
fn coarse_prefilled() -> (WindowedCheckpointEngine, ModelWeights, VectorIndex) {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let mut engine = WindowedCheckpointEngine::with_backend(WINDOW, cpu_engine_backend());
    let backend = larql_compute::cpu_backend();
    engine
        .prefill_quant(&weights, &NullFfn, &index, &PROMPT, &*backend)
        .expect("coarse prefill");
    assert!(
        engine.kv_handle.is_some(),
        "fixture must take the coarse path"
    );
    (engine, weights, index)
}

#[test]
fn dispatch_path_reports_coarse_once_the_backend_holds_the_cache() {
    let (engine, _, _) = coarse_prefilled();
    assert_eq!(engine.dispatch_path(), Some(DispatchPath::Coarse));
}

#[test]
fn a_coarse_decode_the_backend_declines_is_a_backend_failure() {
    let (mut engine, weights, index) = coarse_prefilled();
    // A per-layer handle where the coarse cache should be: the CPU
    // backend declines it rather than guess at its layout.
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    engine.kv_handle = Some(larql_compute::CpuBackend.alloc_kv_buffer(0, 1, kv_dim));
    let backend = larql_compute::cpu_backend();
    assert!(is_backend_failure(
        engine.decode_step_quant(&weights, &NullFfn, &index, NEXT, &*backend)
    ));
}

#[test]
fn a_fused_executor_falls_back_to_the_quant_entry_points() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let executor = LayerlessExecutor::new(ExecutorDispatchKind::Fused);
    let mut engine = WindowedCheckpointEngine::with_backend(WINDOW, cpu_engine_backend());
    // The layerless executor would fail any layer it ran; success proves
    // the engine never asked it to.
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &NullFfn, &index, &PROMPT)
        .expect("fused prefill runs the engine's own quant path");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &NullFfn, &index, NEXT)
        .expect("fused decode runs the engine's own quant path");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn an_executor_that_runs_no_layer_fails_prefill_and_decode() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let layerless = LayerlessExecutor::new(ExecutorDispatchKind::PerLayer);

    let mut fresh = WindowedCheckpointEngine::new(WINDOW);
    assert!(is_backend_failure(fresh.prefill_quant_via_executor(
        &weights, &layerless, &NullFfn, &index, &PROMPT
    )));

    let backend = larql_compute::cpu_backend();
    let walker = LocalWalkExecutor::new(&*backend);
    let mut engine = WindowedCheckpointEngine::new(WINDOW);
    engine
        .prefill_quant_via_executor(&weights, &walker, &NullFfn, &index, &PROMPT)
        .unwrap();
    assert!(is_backend_failure(engine.decode_step_quant_via_executor(
        &weights, &layerless, &NullFfn, &index, NEXT
    )));
}

#[test]
fn a_window_shadow_of_the_wrong_depth_fails_the_executor_decode() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let walker = LocalWalkExecutor::new(&*backend);
    let mut engine = WindowedCheckpointEngine::new(WINDOW);
    engine
        .prefill_quant_via_executor(&weights, &walker, &NullFfn, &index, &PROMPT)
        .unwrap();
    engine.current_window_kv.as_mut().unwrap().pop();
    assert!(is_backend_failure(engine.decode_step_quant_via_executor(
        &weights, &walker, &NullFfn, &index, NEXT
    )));
}
