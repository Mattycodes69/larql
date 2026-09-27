//! INSERT
//! DELETE
//! UPDATE
//! MERGE

use super::*;

#[test]
fn parse_insert_minimal() {
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            entity,
            relation,
            target,
            layer,
            confidence,
            alpha,
            mode,
        } => {
            assert_eq!(entity, "John Coyle");
            assert_eq!(relation, "lives-in");
            assert_eq!(target, "Colchester");
            assert!(layer.is_none());
            assert!(confidence.is_none());
            assert!(alpha.is_none());
            assert_eq!(mode, InsertMode::Knn);
        }
        _ => panic!("expected Insert"),
    }
}

#[test]
fn parse_insert_with_layer_and_confidence() {
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "occupation", "engineer") AT LAYER 26 CONFIDENCE 0.8;"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            layer,
            confidence,
            alpha,
            ..
        } => {
            assert_eq!(layer, Some(26));
            assert!((confidence.unwrap() - 0.8).abs() < 0.01);
            assert!(alpha.is_none());
        }
        _ => panic!("expected Insert"),
    }
}

#[test]
fn parse_insert_with_alpha() {
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon") ALPHA 0.5;"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            alpha,
            layer,
            confidence,
            ..
        } => {
            assert!((alpha.unwrap() - 0.5).abs() < 1e-6);
            assert!(layer.is_none());
            assert!(confidence.is_none());
        }
        _ => panic!("expected Insert"),
    }
}

#[test]
fn parse_insert_with_layer_confidence_alpha() {
    // All three optional clauses can coexist in any order encountered.
    let stmt = parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon") AT LAYER 24 CONFIDENCE 0.95 ALPHA 0.3;"#,
    ).unwrap();
    match stmt {
        Statement::Insert {
            layer,
            confidence,
            alpha,
            ..
        } => {
            assert_eq!(layer, Some(24));
            assert!((confidence.unwrap() - 0.95).abs() < 1e-6);
            assert!((alpha.unwrap() - 0.3).abs() < 1e-6);
        }
        _ => panic!("expected Insert"),
    }
}

#[test]
fn parse_delete_single_condition() {
    let stmt = parse(r#"DELETE FROM EDGES WHERE entity = "outdated_fact";"#).unwrap();
    match stmt {
        Statement::Delete { conditions } => {
            assert_eq!(conditions.len(), 1);
            assert_eq!(conditions[0].field, "entity");
        }
        _ => panic!("expected Delete"),
    }
}

#[test]
fn parse_delete_multiple_conditions() {
    let stmt = parse(r#"DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";"#)
        .unwrap();
    match stmt {
        Statement::Delete { conditions } => assert_eq!(conditions.len(), 2),
        _ => panic!("expected Delete"),
    }
}

#[test]
fn parse_delete_by_layer() {
    let stmt = parse(r#"DELETE FROM EDGES WHERE entity = "outdated" AND layer = 26;"#).unwrap();
    match stmt {
        Statement::Delete { conditions } => {
            assert_eq!(conditions.len(), 2);
            assert!(matches!(conditions[1].value, Value::Integer(26)));
        }
        _ => panic!("expected Delete"),
    }
}

#[test]
fn parse_update_single_set() {
    let stmt = parse(
        r#"UPDATE EDGES SET target = "London" WHERE entity = "John Coyle" AND relation = "lives-in";"#,
    ).unwrap();
    match stmt {
        Statement::Update { set, conditions } => {
            assert_eq!(set.len(), 1);
            assert_eq!(set[0].field, "target");
            assert_eq!(conditions.len(), 2);
        }
        _ => panic!("expected Update"),
    }
}

#[test]
fn parse_update_multiple_assignments() {
    let stmt = parse(
        r#"UPDATE EDGES SET target = "London", confidence = 0.9 WHERE entity = "John Coyle";"#,
    )
    .unwrap();
    match stmt {
        Statement::Update { set, conditions } => {
            assert_eq!(set.len(), 2);
            assert_eq!(conditions.len(), 1);
        }
        _ => panic!("expected Update"),
    }
}

#[test]
fn parse_merge_minimal() {
    let stmt = parse(r#"MERGE "source.vindex";"#).unwrap();
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "source.vindex");
            assert!(target.is_none());
            assert!(conflict.is_none());
        }
        _ => panic!("expected Merge"),
    }
}

#[test]
fn parse_merge_into_no_conflict() {
    let stmt = parse(r#"MERGE "source.vindex" INTO "target.vindex";"#).unwrap();
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "source.vindex");
            assert_eq!(target.as_deref(), Some("target.vindex"));
            assert!(conflict.is_none());
        }
        _ => panic!("expected Merge"),
    }
}

#[test]
fn parse_merge_into_with_conflict() {
    let stmt =
        parse(r#"MERGE "medical.vindex" INTO "gemma3.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#)
            .unwrap();
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "medical.vindex");
            assert_eq!(target.as_deref(), Some("gemma3.vindex"));
            assert_eq!(conflict, Some(ConflictStrategy::HighestConfidence));
        }
        _ => panic!("expected Merge"),
    }
}

#[test]
fn parse_merge_keep_source() {
    let stmt = parse(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT KEEP_SOURCE;"#).unwrap();
    match stmt {
        Statement::Merge { conflict, .. } => {
            assert_eq!(conflict, Some(ConflictStrategy::KeepSource))
        }
        _ => panic!("expected Merge"),
    }
}

#[test]
fn parse_merge_keep_target() {
    let stmt = parse(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT KEEP_TARGET;"#).unwrap();
    match stmt {
        Statement::Merge { conflict, .. } => {
            assert_eq!(conflict, Some(ConflictStrategy::KeepTarget))
        }
        _ => panic!("expected Merge"),
    }
}
