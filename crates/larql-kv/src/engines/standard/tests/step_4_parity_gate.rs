//! Step 4 parity gate

use super::*;

#[cfg(not(windows))]
#[test]
fn parity_standard_unbounded_matches_legacy() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = &[2u32, 3, 5, 7];
    let max = 6;
    let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, None);
    let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
    assert_eq!(
        engine, legacy,
        "engine dispatch must produce identical tokens to generate_cached_backend (window=None)"
    );
}

#[cfg(not(windows))]
#[test]
fn parity_standard_windowed_matches_legacy() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = &[1u32, 2, 3, 4, 5];
    let max = 5;
    // Window smaller than prompt → exercises prefill-time clipping.
    let window = Some(3);
    let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
    let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
    assert_eq!(
        engine, legacy,
        "engine dispatch must produce identical tokens to generate_cached_backend (sliding window)"
    );
}

#[test]
fn parity_standard_short_prompt_long_window_matches_legacy() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = &[0u32, 1];
    let max = 4;
    let window = Some(64); // window > prompt — exercises decode-time growth past prompt
    let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
    let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
    assert_eq!(
        engine, legacy,
        "engine dispatch must produce identical tokens at short-prompt long-window edge case"
    );
}

/// Gemma-4 PLE + layer_scalar parity (issue #98 regression, engine
/// level): StandardEngine's dispatch path must apply the same
/// per-layer PLE + `layer_scalar` sequence as the legacy
/// `kv_prefill_run` / `kv_decode_step_run` reference. The synthetic
/// E2B-like fixture carries non-zero PLE tensors and a non-identity
/// `layer_scalar`, so dropping either step diverges bit-visibly.
/// Several decode steps deep because #98's signature was "first
/// token fine, later tokens garbage".
#[cfg(not(windows))]
#[test]
fn standard_engine_matches_legacy_on_ple_arch() {
    use crate::generation::{kv_decode_step_run, kv_prefill_run};
    use larql_inference::forward::NoopHook;
    use larql_inference::test_utils::make_synthetic_e2b_like_weights;

    const DECODE_STEPS: usize = 4;

    let weights = make_synthetic_e2b_like_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = [0u32, 1, 2];

    let mut engine = StandardEngine::new(None);
    let h_engine = engine
        .prefill(&weights, &ffn, &prompt)
        .expect("engine PLE prefill");
    let (h_legacy, mut cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        Some(&larql_compute::CpuBackend),
        &mut NoopHook,
    )
    .expect("legacy PLE prefill");
    let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    assert_eq!(
        bits(&h_engine),
        bits(&h_legacy),
        "PLE prefill hidden must match legacy bit-for-bit"
    );

    for step in 0..DECODE_STEPS {
        let token = (3 + step) as u32;
        let h_engine = engine
            .decode_step(&weights, &ffn, token)
            .expect("engine PLE decode");
        let h_legacy = kv_decode_step_run(
            &weights,
            &ffn,
            &mut cache,
            token,
            Some(&larql_compute::CpuBackend),
            &mut NoopHook,
        )
        .expect("legacy PLE decode");
        assert_eq!(
            bits(&h_engine),
            bits(&h_legacy),
            "PLE decode step {step} hidden must match legacy bit-for-bit"
        );
    }
}
