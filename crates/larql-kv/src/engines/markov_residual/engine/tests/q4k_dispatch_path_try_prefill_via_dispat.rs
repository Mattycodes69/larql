//! Q4K dispatch path (try_prefill_via_dispatch + decode_step_via_dispatch)

use super::*;

#[test]
fn prefill_via_dispatch_with_q4k_vindex_populates_store_and_handle() {
    if crate::engines::w10_enabled() {
        // W10 opt-in deliberately drops these shadows; test verifies the
        // default (Full mask) population, so skip when the env-gated
        // optimisation is active.
        return;
    }
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let h = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("dispatch-path prefill on Q4K vindex");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    // Dispatch path populates kv_handle + abs_position.
    assert!(engine.kv_handle.is_some());
    assert_eq!(engine.abs_position, 3);
    assert!(engine.memory_bytes() > 0);
    let store = engine.store.as_ref().expect("store populated");
    assert_eq!(store.next_position, 3);
    // hot_kv shadow is populated by default (no LARQL_W10_HONLY).
    assert!(store.hot_kv.is_some());
    // No window → no overflow → cold tiers stay None.
    assert!(store.cold_residuals.is_none());
    assert!(store.cold_kv.is_none());
}

#[test]
fn decode_via_dispatch_grows_buffers_in_place() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill dispatch");
    let mem_after_prefill = engine.memory_bytes();
    for tok in 2..6u32 {
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
            .expect("dispatch decode step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }
    assert_eq!(engine.abs_position, 6);
    assert!(engine.memory_bytes() >= mem_after_prefill);
    assert!(engine.kv_handle.is_some());
}

#[test]
fn dispatch_decode_with_profiling_records_w10_stages() {
    if crate::engines::w10_enabled() {
        // The mask cascade changes which timer slots fire (None mask
        // skips state_materialise/append); this test pins Full-mask
        // behaviour. Add a separate test for HOnly/None coverage.
        return;
    }
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None).with_profiling(true);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill");
    engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode 1");
    engine
        .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
        .expect("decode 2");
    // W10 instrumentation: dispatch path bumps state_capture +
    // state_materialise + state_append + decode_total on every step.
    let summary = engine
        .stage_summary()
        .expect("profiler should produce a summary on the dispatch path");
    assert_eq!(summary.engine, "markov-rs");
    assert!(summary.steps >= 2);
    assert!(summary.avg_state_capture_us > 0.0);
    assert!(summary.avg_state_materialise_us > 0.0);
    assert!(summary.avg_state_append_us > 0.0);
    assert!(summary.avg_total_decode_us > 0.0);
}

#[test]
fn dispatch_prefill_with_window_evicts_to_cold_tier() {
    if crate::engines::w10_enabled() {
        // Window-driven cold-tier eviction depends on the Full-mode
        // shadow being populated; HOnly retains it (window != None
        // here so the HOnly branch keeps rs.stored).
        // Test pins Full-mask behaviour for clarity.
        return;
    }
    // window < prompt_len triggers the on-prefill eviction branch
    // inside try_prefill_via_dispatch (cold_residuals + cold_kv get
    // populated from snapshot_evicted_hot_kv).
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
        .expect("prefill with window");
    let store = engine.store.as_ref().expect("store");
    assert!(store.cold_residuals.is_some());
    assert!(store.cold_kv.is_some());
    assert!(engine.window_tokens() <= 2);
    assert!(engine.cold_bytes() > 0);
}

#[test]
fn dispatch_decode_overflow_extends_cold_tier_from_evicted_hot_kv() {
    // Prefill at window cap; each decode step evicts one row into
    // the cold tier. Exercises the post-decode cold-merge branch
    // (lines ~423-462) including the `Some` arm where prior cold
    // K/V already exists and the eviction concatenates.
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill at window cap");
    for tok in 3..6u32 {
        engine
            .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
            .expect("dispatch decode with overflow");
    }
    let store = engine.store.as_ref().expect("store");
    assert!(store.cold_residuals.is_some());
    // Each decode evicts → cold tier grew across steps.
    let cold = store.cold_residuals.as_ref().unwrap();
    assert!(cold[0].shape()[0] >= 3);
    assert!(engine.window_tokens() <= 2);
}
