//! Phase 1d.3b: generate_with_engine_from_hidden contracts

use super::*;

#[test]
fn wrapper_text_only_plan_matches_generate_with_engine() {
    use crate::engines::standard::StandardEngine;
    use crate::AnyEngine;
    use larql_compute::forward::{embed_plan, EmbeddingPlan};
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let tokens = [0u32, 1, 2, 3];
    let max_new = 4usize;

    // Path A: text path. Post kv-engine-retrieval-trait-split,
    // engines are wrapped in AnyEngine for uniform dispatch.
    let mut engine_a = AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let mut emitted_a: Vec<(u32, String)> = Vec::new();
    let ids_a = generate_with_engine(
        &mut engine_a,
        &weights,
        &tokenizer,
        &ffn,
        &tokens,
        max_new,
        |id, s| emitted_a.push((id, s.to_string())),
    );

    // Path B: single-Tokens-chunk plan → embed_plan → wrapper.
    let mut engine_b = AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let plan = EmbeddingPlan::from_tokens(&tokens);
    let initial_hidden = embed_plan(&weights, &plan);
    let mut emitted_b: Vec<(u32, String)> = Vec::new();
    let ids_b = generate_with_engine_from_hidden(
        &mut engine_b,
        &weights,
        &tokenizer,
        &ffn,
        &initial_hidden,
        max_new,
        |id, s| emitted_b.push((id, s.to_string())),
    );

    assert_eq!(
        ids_a, ids_b,
        "text path and from-hidden wrapper must produce identical token streams \
         on a single-Tokens-chunk plan"
    );
    assert_eq!(
        emitted_a, emitted_b,
        "streaming callback must fire identically across paths \
         (same id + same decoded text per token)"
    );
}

#[test]
fn wrapper_max_tokens_independent_of_hidden_rows() {
    // Build a hidden state with MORE rows than max_new_tokens to
    // confirm the budget isn't accidentally tangled with prefill
    // length. If it were, generation would terminate early (or not
    // at all) depending on the off-by-one's direction.
    use crate::engines::standard::StandardEngine;
    use larql_compute::forward::{embed_plan, EmbeddingPlan};
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };

    let prefill_rows = 7usize;
    let max_new = 3usize;
    let tokens: Vec<u32> = (0u32..prefill_rows as u32).collect();
    let plan = EmbeddingPlan::from_tokens(&tokens);
    let initial_hidden = embed_plan(&weights, &plan);
    assert_eq!(initial_hidden.nrows(), prefill_rows);

    let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let ids = generate_with_engine_from_hidden(
        &mut engine,
        &weights,
        &tokenizer,
        &ffn,
        &initial_hidden,
        max_new,
        |_, _| {},
    );

    // Wrapper may terminate early on EOS (stop-token in the
    // synthetic stream), but must NEVER exceed max_new even though
    // initial_hidden has more rows than max_new.
    assert!(
        ids.len() <= max_new,
        "wrapper decoded {} tokens but max_new_tokens={max_new}; \
         prefill rows ({prefill_rows}) leaked into the token budget",
        ids.len(),
    );
}

#[test]
fn wrapper_zero_hidden_rows_returns_empty() {
    use crate::engines::standard::StandardEngine;
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let empty = Array2::<f32>::zeros((0, weights.hidden_size));
    let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
    let ids = generate_with_engine_from_hidden(
        &mut engine,
        &weights,
        &tokenizer,
        &ffn,
        &empty,
        5,
        |_, _| {},
    );
    assert!(ids.is_empty(), "zero-row hidden should yield empty stream");
}
