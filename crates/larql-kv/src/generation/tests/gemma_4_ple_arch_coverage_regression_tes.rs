//! Gemma-4 PLE arch coverage (regression test for issue #98)

use super::*;

/// `kv_prefill_run` must execute cleanly on a PLE arch — the
/// fixture's PLE keys + projection tensors / norms / gates must be
/// reachable from the prefill loop without dimension mismatch or
/// panic. Value-level pins live in `tests/dispatch_parity.rs` (PLE
/// parity vs the dispatch helpers, bit-exact); this test asserts
/// finiteness + correct hidden-dim shape.
#[test]
fn kv_prefill_run_works_on_synthetic_e2b_ple_arch() {
    let weights = larql_inference::test_utils::make_synthetic_e2b_like_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = [0u32, 1, 2];
    let (last_hidden, cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("PLE-arch prefill should not fail");
    assert_eq!(last_hidden.shape(), &[1, weights.hidden_size]);
    assert!(
        last_hidden.iter().all(|v| v.is_finite()),
        "prefill output must be finite"
    );
    assert_eq!(cache.next_position, prompt.len());
}

/// `kv_decode_step_run` must execute cleanly on a PLE arch for at
/// least three successive steps. Issue #98's signature was: step 1
/// looks fine, steps 2+ degrade. Driving three steps exercises the
/// per-decode-step PLE recompute (`precompute_per_layer_inputs(..,
/// &[token_id])`) under the same code path that produced the
/// regression.
#[test]
fn kv_decode_step_run_works_for_multiple_steps_on_synthetic_e2b_ple_arch() {
    let weights = larql_inference::test_utils::make_synthetic_e2b_like_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt = [0u32, 1];
    let (_h_prefill, mut cache) = kv_prefill_run(
        larql_inference::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
        &mut NoopHook,
    )
    .expect("PLE-arch prefill should not fail");

    for step in 0..3 {
        let h_step = kv_decode_step_run(&weights, &ffn, &mut cache, 0u32, None, &mut NoopHook)
            .unwrap_or_else(|e| panic!("decode step {step} failed: {e:?}"));
        assert_eq!(h_step.shape(), &[1, weights.hidden_size]);
        assert!(
            h_step.iter().all(|v| v.is_finite()),
            "decode step {step} output must be finite"
        );
    }
    assert_eq!(cache.next_position, prompt.len() + 3);
}

/// `generate_cached_constrained` carries its own inline prefill +
/// decode loops (it cannot reuse `kv_prefill_run` because of the
/// mask hook), so those loops must run the same per-layer sequence
/// — attention → FFN → PLE → layer_scalar. With a no-op mask the
/// token stream must match `generate_cached` (which drives the
/// oracle loops) exactly; on the E2B fixture a constrained path
/// that drops PLE / layer_scalar computes different logits and
/// diverges.
#[cfg(not(windows))]
#[test]
fn generate_cached_constrained_matches_generate_cached_on_ple_arch() {
    let weights = larql_inference::test_utils::make_synthetic_e2b_like_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = [0u32, 1, 2];
    const MAX_NEW_TOKENS: usize = 6;

    let mut toks_oracle: Vec<u32> = Vec::new();
    generate_cached(
        &weights,
        &tokenizer,
        &ffn,
        &prompt,
        MAX_NEW_TOKENS,
        |id, _| toks_oracle.push(id),
    );
    let mut toks_constrained: Vec<u32> = Vec::new();
    generate_cached_constrained(
        &weights,
        &tokenizer,
        &ffn,
        &prompt,
        MAX_NEW_TOKENS,
        |_, _| {},
        |id, _| toks_constrained.push(id),
    );
    assert!(
        !toks_oracle.is_empty(),
        "oracle generation must emit tokens"
    );
    assert_eq!(
        toks_oracle, toks_constrained,
        "no-op-mask constrained generation must match generate_cached"
    );
}
