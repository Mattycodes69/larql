//! Prefill-mode tracking (bug fix)

use super::*;

#[test]
fn one_layer_per_layer_prefill_then_decode_step_quant_does_not_panic() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights_layers};
    /// A single decoder layer — the count that used to be
    /// indistinguishable from the coarse single-handle shape.
    const ONE_LAYER: usize = 1;
    let weights = make_test_q4k_weights_layers(ONE_LAYER);
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let prompt = [0u32, 1, 2];
    let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

    let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
        &weights,
        &index,
        larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
    )
    .with_backend(&*backend);

    // Engine A: per-layer prefill via the resident entry point
    // (per-layer handles — for a 1-layer model that is exactly one
    // handle), then decode through the public quant entry point.
    let mut engine_a = StandardEngine::new(None);
    engine_a
        .prefill_resident(&weights, &walk_ffn, &index, &prompt)
        .expect("engine A per-layer prefill_resident");
    assert_eq!(
        engine_a.handles.as_ref().map(|h| h.len()),
        Some(ONE_LAYER),
        "1-layer per-layer prefill must produce exactly one per-layer handle"
    );
    // Pre-fix: panics in `cpu_q4k_cache_mut` (coarse downcast on a
    // per-layer `CpuKvHandle`). Post-fix: follows the recorded
    // per-layer mode.
    let ffn = NullFfn;
    let h_a = engine_a
        .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
        .expect("decode_step_quant after per-layer prefill on a 1-layer model");

    // Engine B: pure per-layer reference for the same step.
    let mut engine_b = StandardEngine::new(None);
    engine_b
        .prefill_resident(&weights, &walk_ffn, &index, &prompt)
        .expect("engine B per-layer prefill_resident");
    larql_inference::vindex::ensure_attn_tensors_dequantised(
        &mut engine_b.dequant_scratch,
        &weights,
        &index,
    );
    let h_b = engine_b
        .do_decode_step(&weights, &walk_ffn, 3, Some(&index))
        .expect("engine B per-layer reference decode");
    assert_eq!(
        bits(&h_a),
        bits(&h_b),
        "1-layer decode_step_quant must produce per-layer parity, not a coarse \
         misdispatch"
    );
}

/// After a *coarse* quant prefill the engine must keep decoding
/// coarse — the recorded mode, not the handle count, drives the
/// dispatch. (Unwindowed engine + the coarse-capable Q4K fixture
/// takes the coarse path on CPU; the handle is the whole-model
/// `CpuQ4kCacheHandle`.)
#[test]
fn coarse_prefill_records_coarse_mode_and_decodes_coarse() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let prompt = [0u32, 1, 2];
    let mut engine = StandardEngine::new(None);
    engine
        .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
        .expect("coarse prefill_quant");
    assert_eq!(
        engine.prefill_mode,
        Some(PrefillDispatchMode::Coarse),
        "unwindowed quant prefill on a coarse-capable backend must record Coarse"
    );
    let cached_before = engine.window_tokens();
    engine
        .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
        .expect("coarse decode_step_quant");
    assert_eq!(
        engine.window_tokens(),
        cached_before + 1,
        "coarse decode must extend the whole-model cache by exactly one row"
    );
}
