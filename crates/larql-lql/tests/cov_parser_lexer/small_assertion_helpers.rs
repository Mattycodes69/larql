//! small assertion helpers
//! lexer.rs — numeric literals

use super::*;

#[test]
fn every_keyword_usable_as_select_field_name() {
    // Hits lexer.rs `as_field_name` for all 100 keyword variants, plus the
    // `Token::Keyword(kw)` arm of helpers.rs::parse_field.
    for kw in ALL_KEYWORDS {
        let sql = format!("SELECT {kw} FROM EDGES;");
        let stmt = ok(&sql);
        match stmt {
            Statement::Select { fields, .. } => {
                assert_eq!(fields.len(), 1, "field count for {kw}");
            }
            other => panic!("expected Select for {kw}, got {other:?}"),
        }
    }
}

#[test]
fn keyword_field_name_in_where_and_order_by() {
    // expect_field_name (helpers.rs 476-481 keyword arm) via WHERE + ORDER BY.
    // Uses keywords that collide with real column names.
    let stmt = ok("SELECT * FROM EDGES WHERE layer = 5 ORDER BY confidence DESC;");
    match stmt {
        Statement::Select {
            conditions, order, ..
        } => {
            assert_eq!(conditions.len(), 1);
            assert_eq!(conditions[0].field, "layer");
            let o = order.expect("order present");
            assert_eq!(o.field, "confidence");
            assert!(o.descending);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn keyword_field_name_in_update_set() {
    // expect_field_name keyword arm via UPDATE ... SET <kw> = ...
    let stmt = ok(r#"UPDATE EDGES SET confidence = 0.9 WHERE relation = "x";"#);
    match stmt {
        Statement::Update { set, .. } => {
            assert_eq!(set[0].field, "confidence");
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn multi_digit_float_fraction_loop() {
    // CONFIDENCE <float> → NumberLit. A multi-digit fraction drives the inner
    // fractional-digit while-loop in lexer.rs::read_number (line ~588).
    let stmt = ok(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c") CONFIDENCE 0.8755;"#,
    );
    match stmt {
        Statement::Insert { confidence, .. } => {
            let c = confidence.expect("confidence present");
            assert!((c - 0.8755).abs() < 1e-4, "got {c}");
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn integer_then_dot_not_a_float() {
    // A digit run followed by '.' where the dot is NOT followed by a digit:
    // read_number must emit an IntegerLit and leave the Dot. `5.` then EOF —
    // the trailing Dot is a trailing token → parse error, but it confirms the
    // integer (not float) branch was taken without panicking.
    err("SELECT * FROM EDGES LIMIT 5.;");
}

#[test]
fn negative_integer_value_in_condition() {
    // parse_value Dash → IntegerLit branch (helpers.rs 295-298).
    let stmt = ok("SELECT * FROM EDGES WHERE layer = -3;");
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(
                conditions[0].value,
                larql_lql::ast::Value::Integer(-3)
            ));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn negative_float_value_in_condition() {
    // parse_value Dash → NumberLit branch (helpers.rs 291-293).
    let stmt = ok("SELECT * FROM EDGES WHERE confidence = -1.5;");
    match stmt {
        Statement::Select { conditions, .. } => match conditions[0].value {
            larql_lql::ast::Value::Number(n) => assert!((n + 1.5).abs() < 1e-6),
            ref other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
}

#[test]
fn dash_not_followed_by_number_is_error() {
    // parse_value Dash → neither Number nor Integer (helpers.rs 299-302).
    err(r#"SELECT * FROM EDGES WHERE entity = -"oops";"#);
}
