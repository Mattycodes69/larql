//! SELECT

use super::*;

#[test]
fn parse_select_star() {
    let stmt = parse("SELECT * FROM EDGES;").unwrap();
    match stmt {
        Statement::Select { fields, .. } => {
            assert_eq!(fields.len(), 1);
            assert!(matches!(fields[0], Field::Star));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_named_fields() {
    let stmt = parse(
        r#"SELECT entity, relation, target, confidence FROM EDGES WHERE entity = "France" ORDER BY confidence DESC LIMIT 10;"#,
    ).unwrap();
    match stmt {
        Statement::Select {
            fields,
            conditions,
            order,
            limit,
            ..
        } => {
            assert_eq!(fields.len(), 4);
            assert_eq!(conditions.len(), 1);
            let ord = order.unwrap();
            assert!(ord.descending);
            assert_eq!(limit, Some(10));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_multiple_conditions() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE relation = "capital-of" AND confidence > 0.5;"#)
        .unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[0].op, CompareOp::Eq));
            assert!(matches!(conditions[1].op, CompareOp::Gt));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_by_layer_and_feature() {
    let stmt = parse("SELECT * FROM EDGES WHERE layer = 26 AND feature = 9515;").unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[0].value, Value::Integer(26)));
            assert!(matches!(conditions[1].value, Value::Integer(9515)));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_nearest() {
    let stmt = parse(
        r#"SELECT entity, target, distance FROM EDGES NEAREST TO "Mozart" AT LAYER 26 LIMIT 20;"#,
    )
    .unwrap();
    match stmt {
        Statement::Select { nearest, limit, .. } => {
            let n = nearest.unwrap();
            assert_eq!(n.entity, "Mozart");
            assert_eq!(n.layer, 26);
            assert_eq!(limit, Some(20));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_no_where() {
    let stmt = parse("SELECT * FROM EDGES LIMIT 5;").unwrap();
    match stmt {
        Statement::Select {
            conditions, limit, ..
        } => {
            assert!(conditions.is_empty());
            assert_eq!(limit, Some(5));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_order_asc() {
    let stmt = parse("SELECT * FROM EDGES ORDER BY layer ASC;").unwrap();
    match stmt {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_order_default_asc() {
    let stmt = parse("SELECT * FROM EDGES ORDER BY layer;").unwrap();
    match stmt {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        _ => panic!("expected Select"),
    }
}
