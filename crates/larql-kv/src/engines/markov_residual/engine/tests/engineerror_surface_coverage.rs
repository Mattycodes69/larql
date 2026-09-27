//! EngineError surface coverage

use super::*;

#[test]
fn prefill_returns_empty_prompt_error_on_empty_input() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine.prefill(&weights, &ffn, &[]).unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::EmptyPrompt
    ));
}

#[test]
fn decode_step_returns_invariant_violation_before_prefill() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine.decode_step(&weights, &ffn, 0).unwrap_err();
    assert!(
        matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ),
        "got {err:?}"
    );
}

#[test]
fn prefill_quant_returns_empty_prompt_error_on_empty_input() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine
        .prefill_quant(&weights, &ffn, &index, &[], &*backend)
        .unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::EmptyPrompt
    ));
}

#[test]
fn decode_step_quant_returns_invariant_violation_before_prefill() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine
        .decode_step_quant(&weights, &ffn, &index, 0, &*backend)
        .unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::InvariantViolation { .. }
    ));
}

#[test]
fn prefill_via_executor_returns_empty_prompt_error_on_empty_input() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let backend = larql_compute::CpuBackend;
    let executor = larql_inference::layer_executor::LocalWalkExecutor::new(&backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine
        .prefill_via_executor(&weights, &executor, &ffn, &[])
        .unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::EmptyPrompt
    ));
}

#[test]
fn prefill_quant_via_executor_returns_empty_prompt_error_on_empty_input() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::CpuBackend;
    let executor = larql_inference::layer_executor::LocalWalkExecutor::new(&backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[])
        .unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::EmptyPrompt
    ));
}

#[test]
fn decode_step_quant_via_executor_returns_invariant_violation_before_prefill() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::CpuBackend;
    let executor = larql_inference::layer_executor::LocalWalkExecutor::new(&backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 0)
        .unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::InvariantViolation { .. }
    ));
}
