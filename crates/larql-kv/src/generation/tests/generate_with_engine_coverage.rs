//! generate_with_engine coverage

use super::*;

#[test]
fn generate_cached_returns_token_ids() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut decoded_tokens: Vec<String> = Vec::new();
    let ids = generate_cached(&weights, &tokenizer, &ffn, &[0u32, 1], 3, |_id, text| {
        decoded_tokens.push(text.to_string())
    });
    assert!(ids.len() <= 3, "should generate at most 3 tokens");
    assert_eq!(
        ids.len(),
        decoded_tokens.len(),
        "callback called once per token"
    );
}

#[test]
fn generate_cached_with_window_limits_cache() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let ids =
        generate_cached_with_window(&weights, &tokenizer, &ffn, &[0u32], 4, Some(2), |_, _| {});
    assert!(ids.len() <= 4);
}

#[test]
fn generate_with_engine_empty_prompt_returns_empty() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine(&mut eng, &weights, &tokenizer, &ffn, &[], 5, |_, _| {});
    assert!(out.is_empty());
}

#[test]
fn generate_with_engine_zero_max_returns_empty() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &[0u32, 1],
        0,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn generate_with_engine_max_one_returns_single_token() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &[0u32, 1],
        1,
        |_, _| {},
    );
    assert_eq!(out.len(), 1);
}

#[test]
fn generate_with_engine_multi_step_fires_callback_per_token() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let mut callbacks = 0usize;
    let out = generate_with_engine(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &[0u32, 1],
        4,
        |_, _| callbacks += 1,
    );
    assert_eq!(out.len(), callbacks);
    assert!(out.len() <= 4);
}

#[test]
fn generate_with_engine_prefill_failure_returns_empty() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut stub = fresh_stub();
    stub.fail_prefill = true;
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out = generate_with_engine(&mut eng, &weights, &tokenizer, &ffn, &[0u32], 3, |_, _| {});
    assert!(out.is_empty());
}

#[test]
fn generate_with_engine_decode_failure_breaks_loop_early() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let mut stub = fresh_stub();
    stub.fail_decode_after = Some(1);
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out = generate_with_engine(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &[0u32, 1],
        5,
        |_, _| {},
    );
    assert!(
        out.len() <= 2,
        "should break after decode failure, got {} tokens",
        out.len()
    );
}

#[test]
fn generate_cached_backend_cpu() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let ids = generate_cached_backend(
        &weights,
        &tokenizer,
        &ffn,
        &[2u32, 3],
        2,
        None,
        None,
        |_, _| {},
    );
    assert!(ids.len() <= 2);
}

#[test]
fn generate_cached_constrained_restricts_tokens() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let allowed: std::collections::HashSet<u32> = (0u32..8).collect();
    let ids = generate_cached_constrained(
        &weights,
        &tokenizer,
        &ffn,
        &[0u32],
        3,
        |_generated, logits| {
            for (id, logit) in logits.iter_mut().enumerate() {
                if !allowed.contains(&(id as u32)) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        },
        |_, _| {},
    );
    for &id in &ids {
        assert!(
            allowed.contains(&id),
            "generated token {id} outside allowed set"
        );
    }
}

#[test]
fn generate_cached_empty_prompt() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let ids = generate_cached(&weights, &tokenizer, &ffn, &[], 2, |_, _| {});
    assert!(ids.len() <= 2);
}
