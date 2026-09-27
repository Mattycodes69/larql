//! Windowed quant path honors the window (bug fix)
//! Per-layer quant fallback routes the real FFN (bug fix)

use super::*;

#[test]
fn prefill_quant_windowed_bounds_cached_rows_to_window() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let prompt = [0u32, 1, 2, 3, 4]; // 5 tokens > QUANT_WINDOW
    let mut engine = StandardEngine::new(Some(QUANT_WINDOW));
    engine
        .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
        .expect("windowed prefill_quant");
    // The load-bearing assertion: the cache must hold at most
    // `window` rows. On the (window-less) coarse path this is the
    // full prompt length — the bug this test pins.
    assert!(
        engine.window_tokens() <= QUANT_WINDOW,
        "windowed quant prefill must bound cached rows to window={QUANT_WINDOW}, \
         got {} (coarse path ignores the window)",
        engine.window_tokens()
    );
    // Decode past the window: the bound must hold on every step.
    for step in 0..QUANT_DECODE_STEPS {
        let token = (5 + step) as u32;
        engine
            .decode_step_quant(&weights, &ffn, &index, token, &*backend)
            .expect("windowed decode_step_quant");
        assert!(
            engine.window_tokens() <= QUANT_WINDOW,
            "window bound violated after decode step {step}: {} rows",
            engine.window_tokens()
        );
    }
}

/// Windowed quant output must be bit-identical to the per-layer
/// windowed path — because post-fix it IS the per-layer windowed
/// path. The reference drives the engine internals directly with an
/// explicitly-constructed `WalkFfn` (the substitution
/// `prefill_quant` performs itself), so any divergence means the
/// quant entry points stopped enforcing the window or stopped
/// routing the real FFN.
#[test]
fn windowed_quant_path_matches_per_layer_windowed_reference() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let prompt = [0u32, 1, 2, 3, 4];
    let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

    // Engine A: public quant entry points, caller passes NullFfn
    // (the bench-harness contract — engines route FFN internally).
    let ffn = NullFfn;
    let mut engine_a = StandardEngine::new(Some(QUANT_WINDOW));
    let h_a = engine_a
        .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
        .expect("engine A windowed prefill_quant");

    // Engine B: the per-layer windowed reference — same dequant
    // scratch handling + explicit WalkFfn, driven through the
    // internal per-layer body.
    let mut engine_b = StandardEngine::new(Some(QUANT_WINDOW));
    larql_inference::vindex::ensure_attn_tensors_dequantised(
        &mut engine_b.dequant_scratch,
        &weights,
        &index,
    );
    let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
        &weights,
        &index,
        larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
    )
    .with_backend(&*backend);
    let h_b = engine_b
        .do_prefill(&weights, &walk_ffn, &prompt, Some(&index))
        .expect("engine B per-layer windowed prefill");
    assert_eq!(
        bits(&h_a),
        bits(&h_b),
        "windowed quant prefill must match the per-layer windowed reference bit-for-bit"
    );

    for step in 0..QUANT_DECODE_STEPS {
        let token = (5 + step) as u32;
        let h_a = engine_a
            .decode_step_quant(&weights, &ffn, &index, token, &*backend)
            .expect("engine A windowed decode_step_quant");
        let h_b = engine_b
            .do_decode_step(&weights, &walk_ffn, token, Some(&index))
            .expect("engine B per-layer windowed decode");
        assert_eq!(
            bits(&h_a),
            bits(&h_b),
            "windowed quant decode step {step} must match the per-layer reference bit-for-bit"
        );
    }
}

#[test]
fn quant_fallback_with_null_ffn_matches_explicit_walk_ffn_reference() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let prompt = [0u32, 1, 2];
    let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

    // Engine A: public quant entry points with NullFfn.
    let ffn = NullFfn;
    let mut engine_a = StandardEngine::new(Some(FORCE_PER_LAYER_WINDOW));
    let h_a = engine_a
        .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
        .expect("engine A prefill_quant");
    // Pin the premise: this test is about the PER-LAYER fallback, so
    // it is only meaningful if engine A actually took it. Without
    // this, a future widening of what the fused path accepts would
    // make the comparison silently test nothing.
    assert_eq!(
        engine_a.dispatch_path(),
        Some(larql_inference::kv_engine::DispatchPath::PerLayer),
        "engine A must be on the per-layer path for this comparison to mean anything"
    );

    // Engine B: same internals, explicit WalkFfn.
    let mut engine_b = StandardEngine::new(Some(FORCE_PER_LAYER_WINDOW));
    larql_inference::vindex::ensure_attn_tensors_dequantised(
        &mut engine_b.dequant_scratch,
        &weights,
        &index,
    );
    let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
        &weights,
        &index,
        larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
    )
    .with_backend(&*backend);
    let h_b = engine_b
        .do_prefill(&weights, &walk_ffn, &prompt, Some(&index))
        .expect("engine B reference prefill");
    assert_eq!(
        bits(&h_a),
        bits(&h_b),
        "quant-fallback prefill with NullFfn must substitute the real WalkFfn \
         (bit-parity with the explicit-WalkFfn reference)"
    );

    for step in 0..QUANT_DECODE_STEPS {
        let token = (3 + step) as u32;
        let h_a = engine_a
            .decode_step_quant(&weights, &ffn, &index, token, &*backend)
            .expect("engine A decode_step_quant");
        let h_b = engine_b
            .do_decode_step(&weights, &walk_ffn, token, Some(&index))
            .expect("engine B reference decode");
        assert_eq!(
            bits(&h_a),
            bits(&h_b),
            "quant-fallback decode step {step} with NullFfn must match the \
             explicit-WalkFfn reference bit-for-bit"
        );
    }
}
