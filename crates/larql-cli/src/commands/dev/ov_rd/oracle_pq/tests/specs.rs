//! The probe-spec parsers and the conditional-quotient guards.

use super::super::specs::{
    is_bos_or_previous_attention, parse_code_class_collapse_specs,
    parse_code_substitution_to_specs, parse_conditional_quotient_guards, ConditionalQuotientGuard,
};

const PROSE: &str = "natural_prose";
/// Attention rows over three positions, peaked at BOS, at the previous
/// position, and at the query position itself.
const TO_BOS: [f32; 3] = [0.8, 0.1, 0.1];
const TO_PREVIOUS: [f32; 3] = [0.1, 0.8, 0.1];
const TO_SELF: [f32; 3] = [0.1, 0.1, 0.8];

#[test]
fn bos_or_previous_attention_reads_the_causal_argmax() {
    assert!(is_bos_or_previous_attention(2, &TO_BOS));
    assert!(is_bos_or_previous_attention(2, &TO_PREVIOUS));
    assert!(!is_bos_or_previous_attention(2, &TO_SELF));
    assert!(!is_bos_or_previous_attention(2, &[]));
}

#[test]
fn guards_keep_the_oracle_code_only_in_prose() {
    use ConditionalQuotientGuard::*;
    for guard in [EarlyProsePosition, EarlyProseBosPrev, ProseBosPrev] {
        assert!(!guard.keeps_secondary_oracle("code", 0, 1, &TO_BOS));
    }
    assert!(EarlyProsePosition.keeps_secondary_oracle(PROSE, 1, 1, &TO_SELF));
    assert!(!EarlyProsePosition.keeps_secondary_oracle(PROSE, 2, 1, &TO_BOS));
    assert!(EarlyProseBosPrev.keeps_secondary_oracle(PROSE, 2, 2, &TO_PREVIOUS));
    assert!(!EarlyProseBosPrev.keeps_secondary_oracle(PROSE, 2, 1, &TO_PREVIOUS));
    assert!(ProseBosPrev.keeps_secondary_oracle(PROSE, 2, 0, &TO_BOS));
    assert!(!ProseBosPrev.keeps_secondary_oracle(PROSE, 2, 0, &TO_SELF));
}

#[test]
fn guards_parse_by_name_or_label_once_each() {
    let guards =
        parse_conditional_quotient_guards("prose_bos_prev, G_prose_bos_prev_guard").unwrap();
    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].label(), "G_prose_bos_prev_guard");
    let err = parse_conditional_quotient_guards("bogus").unwrap_err();
    assert!(err
        .to_string()
        .contains("unsupported conditional quotient guard"));
}

#[test]
fn class_collapse_specs_name_unnamed_specs_and_refuse_malformed_ones() {
    let specs = parse_code_class_collapse_specs("6+10:13|7:10; x y=1:2").unwrap();
    assert_eq!(specs[0].name, "collapse0_6_10_to_13_and_7_to_10");
    assert_eq!(
        specs[0].label(),
        "collapse0_6_10_to_13_and_7_to_10=6+10:13|7:10"
    );
    assert_eq!(specs[1].name, "x_y");
    for (spec, fragment) in [
        ("a=6", "expected sources:target"),
        ("a=x:1", "invalid class-collapse source code"),
        ("a=:1", "has no sources"),
        ("a=1:2|1:3", "appears in more than one mapping"),
        ("a=1:x", "invalid class-collapse target code"),
        ("a=", "has no mappings"),
    ] {
        let err = parse_code_class_collapse_specs(spec).unwrap_err();
        assert!(err.to_string().contains(fragment), "{spec}: {err}");
    }
}

#[test]
fn substitution_targets_are_majority_or_a_code() {
    let specs = parse_code_substitution_to_specs("Majority, 3").unwrap();
    assert_eq!(specs.len(), 2);
    let err = parse_code_substitution_to_specs("x").unwrap_err();
    assert!(err.to_string().contains("invalid code substitution target"));
}
