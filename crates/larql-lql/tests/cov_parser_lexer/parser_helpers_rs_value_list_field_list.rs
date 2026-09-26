//! parser/helpers.rs — value list, field list, compare-op + token errors

use super::*;

#[test]
fn value_list_with_multiple_items() {
    // parse_value LParen branch + comma loop (helpers.rs 305-316, line 314).
    let stmt = ok(r#"SELECT * FROM EDGES WHERE relation IN ("a", "b", "c");"#);
    match stmt {
        Statement::Select { conditions, .. } => match &conditions[0].value {
            larql_lql::ast::Value::List(items) => assert_eq!(items.len(), 3),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
}

#[test]
fn empty_value_list() {
    // parse_value LParen with immediate RParen (skips the item loop entirely).
    let stmt = ok("SELECT * FROM EDGES WHERE relation IN ();");
    match stmt {
        Statement::Select { conditions, .. } => match &conditions[0].value {
            larql_lql::ast::Value::List(items) => assert!(items.is_empty()),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
}

#[test]
fn value_not_a_value_is_error() {
    // parse_value fallthrough error (helpers.rs 318): a bare keyword in value
    // position (FROM) is not a value.
    err("SELECT * FROM EDGES WHERE a = FROM;");
}

#[test]
fn star_in_middle_of_field_list() {
    // parse_field Star arm (helpers.rs 193-196): reached via a `*` AFTER a
    // comma, so parse_field_list's leading-star shortcut does not apply.
    let stmt = ok("SELECT entity, * FROM EDGES;");
    match stmt {
        Statement::Select { fields, .. } => assert_eq!(fields.len(), 2),
        other => panic!("got {other:?}"),
    }
}

#[test]
fn field_not_a_name_is_error() {
    // parse_field fallthrough error (helpers.rs 208-211): an integer after a
    // comma is not a field name.
    err("SELECT entity, 7 FROM EDGES;");
}

#[test]
fn missing_comparison_operator_is_error() {
    // parse_compare_op error (helpers.rs 265-268).
    err("SELECT * FROM EDGES WHERE a b;");
}

#[test]
fn expect_token_mismatch_is_error() {
    // expect_token error (helpers.rs 443-446): INSERT column list missing the
    // comma between `entity` and `relation`.
    err(r#"INSERT INTO EDGES (entity relation, target) VALUES ("a", "b", "c");"#);
}

#[test]
fn expect_ident_eq_mismatch_is_error() {
    // expect_ident_eq error (helpers.rs 461-465): first column must be `entity`.
    err(r#"INSERT INTO EDGES (foo, relation, target) VALUES ("a", "b", "c");"#);
}

#[test]
fn expect_ident_eq_accepts_keyword_column_names() {
    // expect_ident_eq keyword arm (helpers.rs 457-460): `relation` is a
    // keyword but is accepted as the column identifier via as_field_name.
    let stmt = ok(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#);
    assert!(matches!(stmt, Statement::Insert { .. }));
}

#[test]
fn expect_field_name_error() {
    // expect_field_name fallthrough error (helpers.rs 482-485): a string
    // literal is not a valid field name in a WHERE condition.
    err(r#"DELETE FROM EDGES WHERE "lit" = 1;"#);
}

#[test]
fn expect_u32_rejects_non_positive_int() {
    // expect_u32 error (helpers.rs 413-416): TOP wants a positive integer.
    err(r#"WALK "x" TOP -5;"#);
    err(r#"WALK "x" TOP "five";"#);
    // Past u32: refused, not truncated to 0.
    err(r#"WALK "x" TOP 4294967296;"#);
    ok(r#"WALK "x" TOP 4294967295;"#);
}

#[test]
fn expect_f32_accepts_integer_and_rejects_string() {
    // expect_f32 IntegerLit arm (helpers.rs 426-429): CONFIDENCE 1 (int).
    let stmt =
        ok(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a","b","c") CONFIDENCE 1;"#);
    match stmt {
        Statement::Insert { confidence, .. } => {
            assert!((confidence.unwrap() - 1.0).abs() < 1e-6);
        }
        other => panic!("got {other:?}"),
    }
    // expect_f32 error arm (helpers.rs 430-433).
    err(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a","b","c") CONFIDENCE "hi";"#);
}

#[test]
fn parse_range_rejects_inverted_bounds() {
    // parse_range start > end error (helpers.rs 23-27).
    err(r#"WALK "x" LAYERS 30-3;"#);
}

#[test]
fn parse_range_valid() {
    let stmt = ok(r#"WALK "x" LAYERS 0-33;"#);
    match stmt {
        Statement::Walk { layers, .. } => {
            let r = layers.expect("range");
            assert_eq!(r.start, 0);
            assert_eq!(r.end, 33);
        }
        other => panic!("got {other:?}"),
    }
}
