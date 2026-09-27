//! Construction
//! Prefill → decode cycle
//! Profiling
//! Q4K paths via CPU fallback

use super::*;

#[test]
fn engine_name() {
    assert_eq!(MarkovResidualEngine::new(None).name(), "markov-rs");
}

#[test]
fn engine_memory_zero_before_prefill() {
    let eng = MarkovResidualEngine::new(None);
    assert_eq!(eng.memory_bytes(), 0);
    assert_eq!(eng.window_tokens(), 0);
    assert_eq!(eng.cold_bytes(), 0);
}

#[test]
fn engine_info_full_window() {
    let eng = MarkovResidualEngine::new(None);
    let info = eng.info();
    assert!(
        info.config.contains("full"),
        "expected 'full' in config, got '{}'",
        info.config
    );
}

#[test]
fn engine_info_fixed_window() {
    let eng = MarkovResidualEngine::new(Some(16));
    let info = eng.info();
    assert!(
        info.config.contains("16"),
        "expected window size in config, got '{}'",
        info.config
    );
}

#[test]
fn prefill_stores_residuals_for_all_layers() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(None);
    let h = engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(
        engine.memory_bytes() > 0,
        "store should be non-empty after prefill"
    );
}

#[test]
fn decode_step_produces_finite_logits() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(None);
    engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
    let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(hidden_to_raw_logits(&weights, &h)
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn memory_grows_with_each_decode_step() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(None);
    engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
    let mem_after_prefill = engine.memory_bytes();
    engine.decode_step(&weights, &ffn, 1).expect("decode 1");
    let mem_after_1 = engine.memory_bytes();
    engine.decode_step(&weights, &ffn, 2).expect("decode 2");
    let mem_after_2 = engine.memory_bytes();
    assert!(
        mem_after_1 > mem_after_prefill,
        "memory should grow with decode steps"
    );
    assert!(mem_after_2 > mem_after_1);
}

#[test]
fn window_clipping_limits_hot_store() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(Some(2)); // window=2 tokens
    engine
        .prefill(&weights, &ffn, &[0u32, 1, 2, 3, 4])
        .expect("prefill 5 tokens");
    // After clipping, hot store ≤ window
    assert!(
        engine.window_tokens() <= 2,
        "window_tokens={} should be ≤ 2",
        engine.window_tokens()
    );
    // Cold bytes should now be non-zero (overflow clipped to cold)
    assert!(
        engine.cold_bytes() > 0,
        "cold tier should have bytes after clipping"
    );
}

#[test]
fn multiple_decode_steps_produce_consistent_shapes() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(None);
    engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
    for step in 0..3 {
        let h = engine
            .decode_step(&weights, &ffn, step as u32)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size], "step {step}");
    }
}

#[test]
fn with_profiling_enables_profiling_branch() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(None).with_profiling(true);
    // No decode yet → stage_summary returns None even with profiling on.
    assert!(engine.stage_summary().is_none());

    engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
    engine.decode_step(&weights, &ffn, 2).expect("decode");

    let summary = engine.stage_summary().expect("profiling summary");
    assert_eq!(summary.engine, "markov-rs");
    assert_eq!(summary.steps, 1);
    assert!(summary.avg_total_decode_us > 0.0);
}

#[test]
fn stage_summary_none_without_profiling() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = MarkovResidualEngine::new(None); // profiling: false
    engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
    engine.decode_step(&weights, &ffn, 1).expect("decode");
    assert!(
        engine.stage_summary().is_none(),
        "stage_summary must be None when profiling is disabled"
    );
}

#[test]
fn profiling_decode_path_matches_unprofiled_shape() {
    // Two engines: one profiled, one not. Both should yield hidden states
    // of the same shape after the same prefill+decode sequence.
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut profiled = MarkovResidualEngine::new(None).with_profiling(true);
    let mut plain = MarkovResidualEngine::new(None);
    profiled.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
    plain.prefill(&weights, &ffn, &[0u32, 1]).unwrap();
    let h_p = profiled.decode_step(&weights, &ffn, 2).unwrap();
    let h_n = plain.decode_step(&weights, &ffn, 2).unwrap();
    assert_eq!(h_p.shape(), h_n.shape());
}

#[test]
fn prefill_q4k_cpu_fallback_runs_walk_path() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    // `NullFfn` satisfies the trait without borrowing `weights`, which is
    // `&mut` here. The engine ignores the FFN parameter on this path.
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let h = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill_quant cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_q4k_cpu_fallback_extends_store() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill_quant");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode_step_quant cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(
        engine.memory_bytes() > mem_before,
        "store should grow after decode_step_quant"
    );
}
