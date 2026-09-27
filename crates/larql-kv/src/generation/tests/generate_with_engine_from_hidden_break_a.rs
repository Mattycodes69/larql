//! generate_with_engine_from_hidden break-arm coverage

use super::*;

#[test]
fn from_hidden_zero_max_returns_empty() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let h = hidden_for(&weights, 3);
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out =
        generate_with_engine_from_hidden(&mut eng, &weights, &tokenizer, &ffn, &h, 0, |_, _| {});
    assert!(out.is_empty());
}

#[test]
fn from_hidden_prefill_failure_returns_empty() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let h = hidden_for(&weights, 3);
    let mut stub = fresh_stub();
    stub.fail_prefill = true;
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out =
        generate_with_engine_from_hidden(&mut eng, &weights, &tokenizer, &ffn, &h, 4, |_, _| {});
    assert!(out.is_empty());
}

#[test]
fn from_hidden_max_one_returns_single_token() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let h = hidden_for(&weights, 2);
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out =
        generate_with_engine_from_hidden(&mut eng, &weights, &tokenizer, &ffn, &h, 1, |_, _| {});
    assert_eq!(out.len(), 1);
}

#[test]
fn from_hidden_decode_failure_breaks_loop_early() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let h = hidden_for(&weights, 2);
    let mut stub = fresh_stub();
    stub.fail_decode_after = Some(1);
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out =
        generate_with_engine_from_hidden(&mut eng, &weights, &tokenizer, &ffn, &h, 5, |_, _| {});
    assert!(
        out.len() <= 2,
        "should break after decode failure, got {} tokens",
        out.len()
    );
}

#[test]
fn from_hidden_multi_step_fires_callback_per_token() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let h = hidden_for(&weights, 2);
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let mut callbacks = 0usize;
    let out =
        generate_with_engine_from_hidden(&mut eng, &weights, &tokenizer, &ffn, &h, 4, |_, _| {
            callbacks += 1
        });
    assert_eq!(out.len(), callbacks);
    assert!(out.len() <= 4);
}
