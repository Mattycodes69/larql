//! generate_with_engine_resident coverage

use super::*;

#[test]
fn resident_happy_path_via_standard_engine() {
    use crate::engines::standard::StandardEngine;
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;
    let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let mut callbacks = 0usize;
    let out = generate_with_engine_resident(
        &mut engine,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1, 2],
        4,
        |_, _| callbacks += 1,
    );
    assert_eq!(out.len(), callbacks, "callback fires once per token");
    assert!(out.len() <= 4);
}

#[test]
fn resident_zero_max_returns_empty() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1],
        0,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn resident_empty_prompt_returns_empty() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[],
        4,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn resident_prefill_failure_returns_empty() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut stub = fresh_stub();
    stub.fail_prefill = true;
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32],
        3,
        |_, _| {},
    );
    assert!(out.is_empty());
}

#[test]
fn resident_max_one_returns_single_token() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut eng = crate::AnyEngine::Kv(Box::new(fresh_stub()));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
        &[0u32, 1],
        1,
        |_, _| {},
    );
    assert_eq!(out.len(), 1);
}

#[test]
fn resident_decode_failure_breaks_loop_early() {
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_q4k_vindex(&weights);
    let ffn = WeightFfn { weights: &weights };
    let mut stub = fresh_stub();
    stub.fail_decode_after = Some(1);
    let mut eng = crate::AnyEngine::Kv(Box::new(stub));
    let out = generate_with_engine_resident(
        &mut eng,
        &weights,
        &tokenizer,
        &ffn,
        &index,
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
