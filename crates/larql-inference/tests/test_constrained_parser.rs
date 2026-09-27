//! JSON Schema → AST refusals and edge shapes (`constrained::parser`,
//! `constrained::ast`).

use larql_inference::constrained::{parse_schema, ObjectSchema, Schema};
use serde_json::json;

fn refusal(schema: serde_json::Value) -> String {
    parse_schema(&schema).unwrap_err()
}

#[test]
fn an_empty_enum_is_refused() {
    assert!(refusal(json!({"enum": []})).contains("at least one value"));
}

#[test]
fn one_of_must_be_a_non_empty_array() {
    assert!(refusal(json!({"oneOf": {"type": "string"}})).contains("must be an array"));
    assert!(refusal(json!({"anyOf": []})).contains("at least one branch"));
}

#[test]
fn a_type_array_of_one_is_that_type() {
    let s = parse_schema(&json!({"type": ["boolean"]})).unwrap();
    assert!(matches!(s, Schema::Boolean));
}

#[test]
fn an_empty_type_array_is_refused() {
    assert!(refusal(json!({"type": []})).contains("type [] is empty"));
}

#[test]
fn a_type_that_is_neither_string_nor_array_is_refused() {
    assert!(refusal(json!({"type": 3})).contains("type must be a string or array"));
}

#[test]
fn an_unknown_type_is_refused() {
    assert!(refusal(json!({"type": "date"})).contains("unknown type"));
}

#[test]
fn boolean_and_null_types_parse() {
    assert!(matches!(
        parse_schema(&json!({"type": "boolean"})).unwrap(),
        Schema::Boolean
    ));
    assert!(matches!(
        parse_schema(&json!({"type": "null"})).unwrap(),
        Schema::Null
    ));
}

#[test]
fn object_fields_of_the_wrong_shape_are_refused() {
    assert!(refusal(json!({"type": "object", "properties": []}))
        .contains("properties must be an object"));
    assert!(
        refusal(json!({"type": "object", "required": "a"})).contains("required must be an array")
    );
    assert!(refusal(json!({"type": "object", "required": [1]}))
        .contains("required[] entries must be strings"));
    assert!(
        refusal(json!({"type": "object", "additionalProperties": 1}))
            .contains("additionalProperties must be bool or schema")
    );
}

#[test]
fn a_bad_property_schema_is_refused() {
    assert!(
        refusal(json!({"type": "object", "properties": {"a": {"type": "date"}}}))
            .contains("unknown type")
    );
}

#[test]
fn additional_properties_can_be_a_schema() {
    let s = parse_schema(&json!({
        "type": "object",
        "additionalProperties": {"type": "integer"}
    }))
    .unwrap();
    let Schema::Object(o) = s else {
        panic!("expected object")
    };
    assert!(matches!(o.additional.as_deref(), Some(Schema::Number(_))));
}

#[test]
fn an_array_without_items_accepts_anything() {
    let Schema::Array(a) = parse_schema(&json!({"type": "array"})).unwrap() else {
        panic!("expected array")
    };
    assert!(matches!(*a.items, Schema::Any));
}

#[test]
fn empty_strict_object_has_no_fields_and_no_extras() {
    let o = ObjectSchema::empty_strict();
    assert!(o.properties.is_empty() && o.required.is_empty() && o.additional.is_none());
}
