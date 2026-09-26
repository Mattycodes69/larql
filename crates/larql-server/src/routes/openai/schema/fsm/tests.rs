use super::*;
use std::collections::BTreeMap;

fn assert_accepts(schema: Schema, json: &str) {
    let mut fsm = Fsm::new(schema);
    let res = fsm.step_str(json);
    assert_eq!(res, StepResult::Ok, "rejected accepting case: {json:?}");
    assert!(fsm.is_complete(), "not complete after: {json:?}");
}

fn assert_rejects(schema: Schema, json: &str) {
    let mut fsm = Fsm::new(schema);
    let res = fsm.step_str(json);
    let complete = fsm.is_complete();
    assert!(
        res == StepResult::Reject || !complete,
        "accepted should-reject: {json:?}"
    );
}

// ── Schema::Any (structural-only) ─────────────────────────────────

#[test]
fn any_accepts_basic_values() {
    for s in [
        "{}", "[]", r#""abc""#, "42", "-3.14", "true", "false", "null",
    ] {
        assert_accepts(Schema::Any, s);
    }
}

#[test]
fn any_accepts_nested() {
    assert_accepts(Schema::Any, r#"{"a":{"b":[1,2,{"c":true}]}}"#);
}

#[test]
fn any_rejects_garbage() {
    assert_rejects(Schema::Any, "}");
    assert_rejects(Schema::Any, "[,]");
}

#[test]
fn any_string_escapes() {
    assert_accepts(Schema::Any, r#""hello \"world\"""#);
    assert_accepts(Schema::Any, r#""line\nbreak""#);
}

// ── Object schema ─────────────────────────────────────────────────

fn obj(props: &[(&str, Schema)], required: &[&str], strict: bool) -> Schema {
    let mut p = BTreeMap::new();
    for (k, v) in props {
        p.insert((*k).into(), v.clone());
    }
    Schema::object(ObjectSchema {
        properties: p,
        required: required.iter().map(|s| s.to_string()).collect(),
        additional: if strict {
            None
        } else {
            Some(Box::new(Schema::Any))
        },
    })
}

#[test]
fn object_strict_rejects_unknown_key() {
    let s = obj(&[("a", Schema::number())], &[], true);
    assert_rejects(s, r#"{"b":1}"#);
}

#[test]
fn object_strict_accepts_known_key() {
    let s = obj(&[("a", Schema::number())], &[], true);
    assert_accepts(s, r#"{"a":1}"#);
}

#[test]
fn object_required_must_appear() {
    let s = obj(
        &[("a", Schema::number()), ("b", Schema::string())],
        &["a", "b"],
        true,
    );
    assert_rejects(s.clone(), r#"{"a":1}"#); // missing b
    assert_accepts(s, r#"{"a":1,"b":"x"}"#);
}

#[test]
fn object_typed_value_string_rejects_number() {
    let s = obj(&[("name", Schema::string())], &[], true);
    assert_rejects(s, r#"{"name":42}"#);
}

#[test]
fn object_integer_rejects_decimal() {
    let s = obj(&[("n", Schema::integer())], &[], true);
    assert_rejects(s.clone(), r#"{"n":1.5}"#);
    assert_accepts(s, r#"{"n":42}"#);
}

// ── Array schema ──────────────────────────────────────────────────

#[test]
fn array_typed_items_string_rejects_number() {
    let s = Schema::array(Schema::string());
    assert_rejects(s, r#"["a", 1]"#);
}

#[test]
fn array_typed_items_string_accepts_strings() {
    let s = Schema::array(Schema::string());
    assert_accepts(s, r#"["a","b","c"]"#);
}

#[test]
fn array_min_items_rejects_short() {
    let s = Schema::Array(ArraySchema {
        items: Box::new(Schema::Any),
        min: Some(2),
        max: None,
    });
    assert_rejects(s, "[1]");
}

// ── String schema ─────────────────────────────────────────────────

#[test]
fn string_const_only_exact_match() {
    let s = Schema::String(StringSchema {
        r#const: Some("hello".into()),
        ..Default::default()
    });
    assert_accepts(s.clone(), r#""hello""#);
    assert_rejects(s, r#""world""#);
}

#[test]
fn string_const_rejects_diverging_prefix_early() {
    // The FSM should reject the first non-matching character without
    // waiting for the closing quote.
    let s = Schema::String(StringSchema {
        r#const: Some("hello".into()),
        ..Default::default()
    });
    let mut fsm = Fsm::new(s);
    assert_eq!(fsm.step_str(r#""he"#), StepResult::Ok);
    assert_eq!(fsm.step('y'), StepResult::Reject);
}

#[test]
fn string_enum_accepts_member() {
    let s = Schema::String(StringSchema {
        r#enum: Some(vec!["a".into(), "b".into(), "c".into()]),
        ..Default::default()
    });
    assert_accepts(s.clone(), r#""b""#);
    assert_rejects(s, r#""z""#);
}

// ── Number schema ─────────────────────────────────────────────────

#[test]
fn number_minmax_via_object_wrapper() {
    // Numbers validate their bounds at the terminator (`,` / `}` / EOS).
    // Wrap inside an object so the terminator fires as part of the
    // outer dispatch.
    let s = obj(
        &[(
            "n",
            Schema::Number(NumberSchema {
                integer: false,
                minimum: Some(0.0),
                maximum: Some(10.0),
            }),
        )],
        &[],
        true,
    );
    assert_accepts(s.clone(), r#"{"n":5}"#);
    assert_rejects(s.clone(), r#"{"n":11}"#);
    assert_rejects(s, r#"{"n":-1}"#);
}

// ── OneOf ─────────────────────────────────────────────────────────

#[test]
fn oneof_commits_on_string_vs_number() {
    let s = Schema::OneOf(vec![Schema::string(), Schema::number()]);
    assert_accepts(s.clone(), r#""hi""#);
    assert_accepts(s, "42");
}

#[test]
fn oneof_branches_with_same_prefix_resolve() {
    // Two object schemas distinguished by the constant value of `name`.
    let a = obj(
        &[(
            "name",
            Schema::String(StringSchema {
                r#const: Some("alpha".into()),
                ..Default::default()
            }),
        )],
        &["name"],
        true,
    );
    let b = obj(
        &[(
            "name",
            Schema::String(StringSchema {
                r#const: Some("beta".into()),
                ..Default::default()
            }),
        )],
        &["name"],
        true,
    );
    let s = Schema::OneOf(vec![a, b]);
    assert_accepts(s.clone(), r#"{"name":"alpha"}"#);
    assert_accepts(s.clone(), r#"{"name":"beta"}"#);
    assert_rejects(s, r#"{"name":"gamma"}"#);
}

// ── Const ─────────────────────────────────────────────────────────

#[test]
fn const_literal_matches_canonical() {
    let s = Schema::Const(serde_json::json!(42));
    assert_accepts(s, "42");

    let s = Schema::Const(serde_json::json!("hello"));
    assert_accepts(s, r#""hello""#);

    let s = Schema::Const(serde_json::json!(true));
    assert_accepts(s, "true");

    let s = Schema::Const(serde_json::json!(null));
    assert_accepts(s, "null");
}

// ── Completion / depth ────────────────────────────────────────────

#[test]
fn is_complete_only_after_root_closes() {
    let mut fsm = Fsm::any();
    assert!(!fsm.is_complete());
    assert_eq!(fsm.step_str("{"), StepResult::Ok);
    assert!(!fsm.is_complete());
    assert_eq!(fsm.step_str("}"), StepResult::Ok);
    assert!(fsm.is_complete());
}

// ── N0.6 emission-time key discipline ─────────────────────────
//
// The mask samples token by token, so a closed object must refuse
// a doomed key AT THE KEY'S FIRST CHARACTER (not after its colon)
// and must refuse a comma no further key could legally follow —
// otherwise constrained generation dead-ends mid-emission. Both
// were found live by the N0.6-on-V3 gates.

#[test]
fn strict_object_rejects_a_doomed_key_at_its_first_char() {
    let s = obj(&[("name", Schema::number())], &[], true);
    let mut fsm = Fsm::new(s);
    assert_eq!(fsm.step_str(r#"{"g"#), StepResult::Reject);
}

#[test]
fn strict_object_rejects_reopening_a_seen_key() {
    let s = obj(
        &[("a", Schema::number()), ("b", Schema::number())],
        &[],
        true,
    );
    let mut fsm = Fsm::new(s);
    assert_eq!(fsm.step_str(r#"{"a":1,"a"#), StepResult::Reject);
}

#[test]
fn open_object_keys_stay_unconstrained() {
    let s = obj(&[("a", Schema::number())], &[], false);
    assert_accepts(s, r#"{"anything":true}"#);
}

#[test]
fn trailing_comma_is_rejected() {
    let s = obj(&[("a", Schema::number())], &[], false);
    let mut fsm = Fsm::new(s);
    assert_eq!(fsm.step_str(r#"{"a":1,"#), StepResult::Ok);
    assert_eq!(fsm.step_str("}"), StepResult::Reject);
}

#[test]
fn comma_with_no_viable_key_left_is_rejected() {
    // Every property emitted on a strict object: `}` is the only
    // legal continuation — a comma could never be followed.
    let s = obj(&[("a", Schema::number())], &[], true);
    let mut fsm = Fsm::new(s);
    assert_eq!(fsm.step_str(r#"{"a":1"#), StepResult::Ok);
    assert_eq!(fsm.step_str(","), StepResult::Reject);
}

#[test]
fn strict_object_still_walks_its_full_emission() {
    let s = obj(
        &[("a", Schema::number()), ("b", Schema::number())],
        &["a", "b"],
        true,
    );
    assert_accepts(s, r#"{"a":1,"b":2}"#);
}

#[test]
fn depth_tracks_open_containers() {
    let mut fsm = Fsm::any();
    assert_eq!(fsm.step_str("[[["), StepResult::Ok);
    assert_eq!(fsm.depth(), 3);
    assert_eq!(fsm.step_str("]]"), StepResult::Ok);
    assert_eq!(fsm.depth(), 1);
}
