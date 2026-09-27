//! Client-supplied sizes are bounded, and an over-bound value is refused
//! rather than clamped.

use larql_server::routes::limits::{
    generation_tokens, proto_count, within, MAX_GENERATION_TOKENS, MAX_RESULT_ROWS,
};

#[test]
fn a_value_within_its_bound_passes_through() {
    assert_eq!(within("top", 7, 10), Ok(7));
    assert_eq!(within("top", 10, 10), Ok(10));
}

#[test]
fn a_value_over_its_bound_is_refused_with_the_field_named() {
    let err = within("limit", MAX_RESULT_ROWS + 1, MAX_RESULT_ROWS).unwrap_err();
    assert!(err.contains("limit"), "{err}");
}

#[test]
fn an_absurd_max_tokens_is_refused_before_it_reaches_the_decoder() {
    assert_eq!(generation_tokens("max_tokens", None, 16), Ok(16));
    assert!(generation_tokens("max_tokens", Some(usize::MAX), 16).is_err());
    assert!(generation_tokens("max_tokens", Some(MAX_GENERATION_TOKENS + 1), 16).is_err());
    assert_eq!(
        generation_tokens("max_tokens", Some(MAX_GENERATION_TOKENS), 16),
        Ok(MAX_GENERATION_TOKENS)
    );
}

#[test]
fn a_zero_proto_count_means_the_default_and_others_are_bounded() {
    assert_eq!(proto_count("top", 0, 5, 100), Ok(5));
    assert_eq!(proto_count("top", 42, 5, 100), Ok(42));
    assert!(proto_count("top", u32::MAX, 5, 100).is_err());
}
