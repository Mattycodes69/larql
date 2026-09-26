//! Doubling-capacity + hot_kv-consistency regressions (executor path)
//! §4 architecture-precondition gate

use super::*;

#[test]
fn decode_via_executor_padded_cold_buffers_match_trimmed_reference() {
    // The executor decode must read `cold_len` / `hot_len`, not the
    // buffers' capacity. Decode the same logical state over the
    // padded buffers and over buffers trimmed to exactly `cold_len`
    // (the layout the pre-migration code assumed) — outputs must be
    // bit-identical. The padded run previously attended six
    // zero-K/V phantom rows per layer.
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    use ndarray::s;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;

    let run = |trim: bool| -> ndarray::Array2<f32> {
        let mut engine = MarkovResidualEngine::new(Some(EXEC_WINDOW));
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &EXEC_PROMPT)
            .expect("prefill with overflow");
        {
            let store = engine.store.as_mut().unwrap();
            let c = store.cold_len;
            assert!(
                store.cold_kv.as_ref().unwrap()[0].0.shape()[0] > c,
                "fixture must be capacity-padded"
            );
            if trim {
                for layer in store.cold_residuals.as_mut().unwrap().iter_mut() {
                    *layer = layer.slice(s![..c, ..]).to_owned();
                }
                for (k, v) in store.cold_kv.as_mut().unwrap().iter_mut() {
                    *k = k.slice(s![..c, ..]).to_owned();
                    *v = v.slice(s![..c, ..]).to_owned();
                }
            }
        }
        engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, EXEC_NEXT_TOKEN)
            .expect("executor decode")
    };

    let h_padded = run(false);
    let h_trimmed = run(true);
    assert_bits_eq(
        &h_padded,
        &h_trimmed,
        "executor decode attended capacity slack instead of cold_len rows",
    );
}

#[test]
fn decode_via_executor_after_walk_prefill_drops_hot_kv_and_stays_consistent() {
    // The executor decode discards `run_decode_layer`'s returned K/V
    // while `hot_len` grows. Carrying the walk-prefill `hot_kv`
    // through unchanged violated the store invariant (kv[l] and
    // stored[l] agree on the first hot_len rows) and made the
    // post-step window clip slice out of bounds. The fixed path
    // drops the cache; attention never reads it here, so runs with
    // and without a prefill-seeded hot_kv must be bit-identical.
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;

    let run = |drop_hot_kv_after_prefill: bool| -> ndarray::Array2<f32> {
        let mut engine = MarkovResidualEngine::new(Some(EXEC_WINDOW));
        // Walk-path prefill captures hot_kv (2 rows, no overflow).
        engine
            .prefill_quant(
                &weights,
                &ffn,
                &index,
                &EXEC_PROMPT[..EXEC_WINDOW],
                &*backend,
            )
            .expect("walk prefill");
        assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
        if drop_hot_kv_after_prefill {
            engine.store.as_mut().unwrap().hot_kv = None;
        }
        // Two decode steps: the first overflows the window (the
        // out-of-bounds clip on the old code), the second reads
        // whatever state the first left behind.
        engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
            .expect("executor decode 1");
        let store = engine.store.as_ref().unwrap();
        assert!(
            store.hot_kv.is_none(),
            "executor decode must not carry a hot_kv it did not extend"
        );
        engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 3)
            .expect("executor decode 2")
    };

    let h_with_seeded_cache = run(false);
    let h_without_cache = run(true);
    assert_bits_eq(
        &h_with_seeded_cache,
        &h_without_cache,
        "stale hot_kv leaked into the executor decode path",
    );
}

#[test]
fn prefill_refuses_arch_violating_residual_recompute_preconditions() {
    use larql_inference::ffn::NullFfn;
    let mut weights = make_test_weights();
    weights.arch = Box::new(MlaStubArch {
        config: weights.arch.config().clone(),
    });
    assert!(!weights.arch.kv_recomputable_from_residuals());

    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(None);
    let err = engine.prefill(&weights, &ffn, &[0u32, 1]).unwrap_err();
    assert!(
        matches!(
            err,
            larql_inference::kv_engine::EngineError::InvariantViolation { .. }
        ),
        "precondition violation must fail closed, got {err:?}"
    );
    assert!(
        engine.store.is_none(),
        "no state may be built after refusal"
    );

    // Same gate on the quant entry point.
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let err = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .unwrap_err();
    assert!(matches!(
        err,
        larql_inference::kv_engine::EngineError::InvariantViolation { .. }
    ));
}
