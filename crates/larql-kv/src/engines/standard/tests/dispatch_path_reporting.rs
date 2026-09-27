//! dispatch_path reporting
//! Empty-prompt guards
//! Resident-weights quant path (task #16)
//! Quant paths through the async backend slot

use super::*;

#[test]
fn dispatch_path_is_none_before_prefill() {
    // Nothing has chosen a shape yet — reporting one would be a guess.
    assert_eq!(StandardEngine::new(None).dispatch_path(), None);
    assert_eq!(StandardEngine::new(Some(4)).dispatch_path(), None);
}

#[test]
fn dispatch_path_reports_coarse_when_the_backend_took_the_fused_path() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let mut engine = StandardEngine::new(None);
    engine
        .prefill_quant(&weights, &NullFfn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill_quant");
    assert_eq!(
        engine.dispatch_path(),
        Some(larql_inference::kv_engine::DispatchPath::Coarse),
        "unwindowed Q4K prefill takes the coarse path on CpuBackend"
    );
}

/// The window gate's observable consequence. A windowed engine MUST
/// decline coarse (the coarse surface has no window parameter, so
/// taking it would silently attend the full context while `info()`
/// advertised `window=N`). Pinning the reported shape means a future
/// change that lets a windowed config onto the fused path fails here
/// rather than quietly returning full-context answers.
#[test]
fn windowed_engine_never_reports_coarse() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let mut engine = StandardEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &NullFfn, &index, &[0u32, 1, 2], &*backend)
        .expect("windowed prefill_quant");
    assert_eq!(
        engine.dispatch_path(),
        Some(larql_inference::kv_engine::DispatchPath::PerLayer),
        "a windowed engine must decline the window-less coarse surface"
    );
}

#[test]
fn dense_prefill_reports_per_layer() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let mut engine = StandardEngine::new(None);
    engine
        .prefill(&weights, &WeightFfn { weights: &weights }, &[0u32, 1, 2])
        .expect("dense prefill");
    assert_eq!(
        engine.dispatch_path(),
        Some(larql_inference::kv_engine::DispatchPath::PerLayer),
        "the dense (no-vindex) path is per-layer by construction"
    );
}

#[test]
fn prefill_quant_cpu_fallback_runs_via_dequant() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    let h = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill_quant Q4K cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_quant_cpu_fallback_extends_cache() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill_quant");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode_step_quant Q4K cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(
        engine.memory_bytes() > mem_before,
        "K/V cache should grow after Q4K decode step"
    );
}

#[test]
fn decode_step_quant_without_prefill_returns_none() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    // self.handles is None → decode_step_quant returns an
    // InvariantViolation error at the `self.handles.as_mut()` guard.
    assert!(engine
        .decode_step_quant(&weights, &ffn, &index, 0, &*backend)
        .is_err());
}

#[test]
fn prefill_empty_prompt_returns_empty_prompt_error() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    let err = engine.prefill(&weights, &ffn, &[]).unwrap_err();
    assert!(
        matches!(err, EngineError::EmptyPrompt),
        "empty prompt must map to EngineError::EmptyPrompt, got {err:?}"
    );
}

#[test]
fn prefill_quant_empty_prompt_returns_empty_prompt_error() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    let err = engine
        .prefill_quant(&weights, &ffn, &index, &[], &*backend)
        .unwrap_err();
    assert!(
        matches!(err, EngineError::EmptyPrompt),
        "empty prompt must map to EngineError::EmptyPrompt, got {err:?}"
    );
}

#[test]
fn prefill_resident_populates_cache_and_returns_hidden() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    let h = engine
        .prefill_resident(&weights, &ffn, &index, &[0u32, 1, 2])
        .expect("prefill_resident");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
    assert!(engine.memory_bytes() > 0, "cache should be populated");
}

#[test]
fn prefill_resident_empty_prompt_returns_empty_prompt_error() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    let err = engine
        .prefill_resident(&weights, &ffn, &index, &[])
        .unwrap_err();
    assert!(matches!(err, EngineError::EmptyPrompt));
}

#[test]
fn decode_step_resident_extends_cache() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    engine
        .prefill_resident(&weights, &ffn, &index, &[0u32, 1])
        .expect("prefill_resident");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_resident(&weights, &ffn, &index, 2)
        .expect("decode_step_resident");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
    assert!(
        engine.memory_bytes() > mem_before,
        "K/V cache should grow after a resident decode step"
    );
}

#[test]
fn decode_step_resident_without_prefill_returns_error() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;
    let mut engine = StandardEngine::new(None);
    // No prefill → handles is None → InvariantViolation at the guard.
    assert!(engine
        .decode_step_resident(&weights, &ffn, &index, 0)
        .is_err());
}

#[test]
fn prefill_quant_async_slot_falls_back_via_dequant() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    use larql_inference::AsyncComputeBackend;
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let async_backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
    let ffn = NullFfn;
    let mut engine = StandardEngine::with_async_backend(None, async_backend);
    let h = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill_quant async-slot fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_quant_async_slot_falls_back_via_dequant() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    use larql_inference::AsyncComputeBackend;
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let async_backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
    let ffn = NullFfn;
    let mut engine = StandardEngine::with_async_backend(None, async_backend);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill_quant async-slot");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode_step_quant async-slot fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(
        engine.memory_bytes() > mem_before,
        "K/V cache should grow after async-slot Q4K decode step"
    );
}
