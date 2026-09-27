//! Phase 2: executor-driven path

use super::*;

#[test]
fn prefill_quant_via_executor_runs_through_local_walk() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("executor prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_quant_via_executor_extends_store() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
        .expect("prefill");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
        .expect("decode");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > mem_before);
}

#[test]
fn executor_path_honors_ffn_parameter() {
    // Pass a counting stub. If the engine constructs its own
    // WalkFfn internally (the legacy bug we're fixing) the counter
    // stays at zero.
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);

    let ffn = CountingFfn {
        calls: std::sync::atomic::AtomicUsize::new(0),
        hidden: weights.hidden_size,
    };
    let mut engine = MarkovResidualEngine::new(None);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("prefill via executor");

    let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
    // Prefill runs FFN once per layer.
    assert_eq!(
        call_count, weights.num_layers,
        "executor path should dispatch FFN through the supplied backend \
         once per layer; got {call_count} for {} layers — engine is \
         likely constructing its own FFN internally",
        weights.num_layers
    );
}

#[test]
fn prefill_quant_via_executor_with_window_populates_cold_tier() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
        .expect("prefill with overflow");
    assert!(engine.window_tokens() <= 2);
    assert!(engine.cold_bytes() > 0);
}

/// Drive `decode_step_quant_via_executor`'s `cold_kv` branch (lines
/// 315-333): prefill with overflow so the engine pre-computes
/// cold_kv during prefill, then run a single decode step that
/// combines cold_kv + hot K/V for attention.
#[test]
fn decode_step_quant_via_executor_uses_cold_kv_branch() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
        .expect("prefill overflow → cold_kv populated");
    // First decode reads cold_kv branch (rs.cold_kv = Some(_)).
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 4)
        .expect("decode via cold_kv branch");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// Drive the cold_residuals branch when cold_kv has been cleared
/// (the second decode after overflow). At line ~399 the engine
/// clears cold_kv when a new overflow happens, then subsequent
/// decodes recompute K/V from cold_residuals.
#[test]
fn decode_step_quant_via_executor_hits_cold_residuals_branch() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
        .expect("prefill");
    // First decode clears cold_kv via overflow.
    engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 4)
        .expect("first decode");
    // Second decode: cold_kv is None, exercises the recompute_kv
    // from cold_residuals branch.
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 5)
        .expect("decode via cold_residuals recompute");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn fused_executor_falls_back_to_legacy_quant_path() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let exec = FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    };
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let h = engine
        .prefill_quant_via_executor(&weights, &exec, &ffn, &index, &[0u32, 1])
        .expect("fused fallback prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    let h2 = engine
        .decode_step_quant_via_executor(&weights, &exec, &ffn, &index, 2)
        .expect("fused fallback decode");
    assert_eq!(h2.shape(), &[1, weights.hidden_size]);
}
