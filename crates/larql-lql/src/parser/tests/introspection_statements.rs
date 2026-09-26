//! INTROSPECTION STATEMENTS
//! SHOW ENTITIES
//! REBALANCE
//! SHOW COMPACT STATUS
//! COMPACT
//! STATS
//! PIPE OPERATOR
//! COMPARISON OPERATORS
//! COMMENTS AND WHITESPACE
//! ERROR CASES
//! FULL DEMO SCRIPT FROM SPEC v0.3 — every statement parses

use super::*;

#[test]
fn parse_show_relations_minimal() {
    let stmt = parse("SHOW RELATIONS;").unwrap();
    match stmt {
        Statement::ShowRelations {
            layer,
            with_examples,
            mode,
        } => {
            assert!(layer.is_none());
            assert!(!with_examples);
            assert_eq!(mode, DescribeMode::Brief); // Brief is the default
        }
        _ => panic!("expected ShowRelations"),
    }
}

#[test]
fn parse_show_relations_with_examples() {
    let stmt = parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
    match stmt {
        Statement::ShowRelations { with_examples, .. } => assert!(with_examples),
        _ => panic!("expected ShowRelations"),
    }
}

#[test]
fn parse_show_relations_at_layer() {
    let stmt = parse("SHOW RELATIONS AT LAYER 26;").unwrap();
    match stmt {
        Statement::ShowRelations { layer, .. } => assert_eq!(layer, Some(26)),
        _ => panic!("expected ShowRelations"),
    }
}

#[test]
fn parse_show_relations_verbose() {
    let stmt = parse("SHOW RELATIONS VERBOSE;").unwrap();
    match stmt {
        Statement::ShowRelations { mode, .. } => assert_eq!(mode, DescribeMode::Verbose),
        _ => panic!("expected ShowRelations"),
    }
}

#[test]
fn parse_show_relations_raw() {
    let stmt = parse("SHOW RELATIONS RAW;").unwrap();
    match stmt {
        Statement::ShowRelations { mode, .. } => assert_eq!(mode, DescribeMode::Raw),
        _ => panic!("expected ShowRelations"),
    }
}

#[test]
fn parse_show_relations_verbose_with_examples() {
    let stmt = parse("SHOW RELATIONS VERBOSE WITH EXAMPLES;").unwrap();
    match stmt {
        Statement::ShowRelations {
            mode,
            with_examples,
            ..
        } => {
            assert_eq!(mode, DescribeMode::Verbose);
            assert!(with_examples);
        }
        _ => panic!("expected ShowRelations"),
    }
}

#[test]
fn parse_show_layers_minimal() {
    let stmt = parse("SHOW LAYERS;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => assert!(range.is_none()),
        _ => panic!("expected ShowLayers"),
    }
}

#[test]
fn parse_show_layers_with_range() {
    let stmt = parse("SHOW LAYERS RANGE 0-10;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => {
            let r = range.unwrap();
            assert_eq!(r.start, 0);
            assert_eq!(r.end, 10);
        }
        _ => panic!("expected ShowLayers"),
    }
}

#[test]
fn parse_show_layers_bare_range() {
    let stmt = parse("SHOW LAYERS 0-10;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => {
            let r = range.unwrap();
            assert_eq!(r.start, 0);
            assert_eq!(r.end, 10);
        }
        _ => panic!("expected ShowLayers"),
    }
}

#[test]
fn parse_show_features_minimal() {
    let stmt = parse("SHOW FEATURES 26;").unwrap();
    match stmt {
        Statement::ShowFeatures {
            layer,
            conditions,
            limit,
        } => {
            assert_eq!(layer, 26);
            assert!(conditions.is_empty());
            assert!(limit.is_none());
        }
        _ => panic!("expected ShowFeatures"),
    }
}

#[test]
fn parse_show_features_with_where_and_limit() {
    let stmt = parse(r#"SHOW FEATURES 26 WHERE relation = "capital-of" LIMIT 5;"#).unwrap();
    match stmt {
        Statement::ShowFeatures {
            layer,
            conditions,
            limit,
        } => {
            assert_eq!(layer, 26);
            assert_eq!(conditions.len(), 1);
            assert_eq!(limit, Some(5));
        }
        _ => panic!("expected ShowFeatures"),
    }
}

#[test]
fn parse_show_models() {
    let stmt = parse("SHOW MODELS;").unwrap();
    assert!(matches!(stmt, Statement::ShowModels));
}

#[test]
fn parse_show_entities_minimal() {
    let stmt = parse("SHOW ENTITIES;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert!(layer.is_none());
            assert!(limit.is_none());
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[test]
fn parse_show_entities_bare_layer() {
    let stmt = parse("SHOW ENTITIES 26;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert_eq!(layer, Some(26));
            assert!(limit.is_none());
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[test]
fn parse_show_entities_at_layer_with_limit() {
    let stmt = parse("SHOW ENTITIES AT LAYER 26 LIMIT 50;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert_eq!(layer, Some(26));
            assert_eq!(limit, Some(50));
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[test]
fn parse_show_entities_limit_only() {
    let stmt = parse("SHOW ENTITIES LIMIT 100;").unwrap();
    match stmt {
        Statement::ShowEntities { layer, limit } => {
            assert!(layer.is_none());
            assert_eq!(limit, Some(100));
        }
        _ => panic!("expected ShowEntities"),
    }
}

#[test]
fn parse_rebalance_minimal() {
    let stmt = parse("REBALANCE;").unwrap();
    match stmt {
        Statement::Rebalance {
            max_iters,
            floor,
            ceiling,
        } => {
            assert!(max_iters.is_none());
            assert!(floor.is_none());
            assert!(ceiling.is_none());
        }
        _ => panic!("expected Rebalance"),
    }
}

#[test]
fn parse_rebalance_until_converged() {
    let stmt = parse("REBALANCE UNTIL CONVERGED;").unwrap();
    assert!(matches!(stmt, Statement::Rebalance { .. }));
}

#[test]
fn parse_rebalance_max_iters() {
    let stmt = parse("REBALANCE MAX 32;").unwrap();
    match stmt {
        Statement::Rebalance { max_iters, .. } => assert_eq!(max_iters, Some(32)),
        _ => panic!("expected Rebalance"),
    }
}

#[test]
fn parse_rebalance_floor_ceiling() {
    let stmt = parse("REBALANCE FLOOR 0.3 CEILING 0.9;").unwrap();
    match stmt {
        Statement::Rebalance { floor, ceiling, .. } => {
            assert!((floor.unwrap() - 0.3).abs() < 1e-6);
            assert!((ceiling.unwrap() - 0.9).abs() < 1e-6);
        }
        _ => panic!("expected Rebalance"),
    }
}

#[test]
fn parse_rebalance_all_clauses() {
    let stmt = parse("REBALANCE UNTIL CONVERGED MAX 16 FLOOR = 0.25 CEILING = 0.95;").unwrap();
    match stmt {
        Statement::Rebalance {
            max_iters,
            floor,
            ceiling,
        } => {
            assert_eq!(max_iters, Some(16));
            assert!((floor.unwrap() - 0.25).abs() < 1e-6);
            assert!((ceiling.unwrap() - 0.95).abs() < 1e-6);
        }
        _ => panic!("expected Rebalance"),
    }
}

#[test]
fn parse_show_compact_status() {
    let stmt = parse("SHOW COMPACT STATUS;").unwrap();
    assert!(matches!(stmt, Statement::ShowCompactStatus));
}

#[test]
fn parse_show_compact_status_no_semicolon() {
    let stmt = parse("SHOW COMPACT STATUS").unwrap();
    assert!(matches!(stmt, Statement::ShowCompactStatus));
}

#[test]
fn parse_compact_minor() {
    let stmt = parse("COMPACT MINOR;").unwrap();
    assert!(matches!(stmt, Statement::CompactMinor));
}

#[test]
fn parse_compact_major() {
    let stmt = parse("COMPACT MAJOR;").unwrap();
    assert!(matches!(
        stmt,
        Statement::CompactMajor {
            full: false,
            lambda: None
        }
    ));
}

#[test]
fn parse_compact_major_full() {
    let stmt = parse("COMPACT MAJOR FULL;").unwrap();
    assert!(matches!(
        stmt,
        Statement::CompactMajor {
            full: true,
            lambda: None
        }
    ));
}

#[test]
fn parse_compact_major_with_lambda() {
    let stmt = parse("COMPACT MAJOR WITH LAMBDA = 0.001;").unwrap();
    match stmt {
        Statement::CompactMajor { full, lambda } => {
            assert!(!full);
            assert!((lambda.unwrap() - 0.001).abs() < 1e-6);
        }
        _ => panic!("expected CompactMajor"),
    }
}

#[test]
fn parse_stats_no_path() {
    let stmt = parse("STATS;").unwrap();
    assert!(matches!(stmt, Statement::Stats { vindex: None }));
}

#[test]
fn parse_stats_with_path() {
    let stmt = parse(r#"STATS "gemma3.vindex";"#).unwrap();
    match stmt {
        Statement::Stats { vindex } => assert_eq!(vindex.as_deref(), Some("gemma3.vindex")),
        _ => panic!("expected Stats"),
    }
}

#[test]
fn parse_stats_no_semicolon() {
    let stmt = parse("STATS").unwrap();
    assert!(matches!(stmt, Statement::Stats { vindex: None }));
}

#[test]
fn parse_pipe_walk_to_explain() {
    let stmt = parse(
        r#"WALK "The capital of France is" TOP 5 |> EXPLAIN WALK "The capital of France is";"#,
    )
    .unwrap();
    match stmt {
        Statement::Pipe { left, right } => {
            assert!(matches!(*left, Statement::Walk { .. }));
            assert!(matches!(*right, Statement::Explain { .. }));
        }
        _ => panic!("expected Pipe"),
    }
}

#[test]
fn parse_select_neq() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE relation != "morphological";"#).unwrap();
    match stmt {
        Statement::Select { conditions, .. } => assert!(matches!(conditions[0].op, CompareOp::Neq)),
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_gte_lte() {
    let stmt = parse("SELECT * FROM EDGES WHERE layer >= 20 AND layer <= 30;").unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(conditions[0].op, CompareOp::Gte));
            assert!(matches!(conditions[1].op, CompareOp::Lte));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_like() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE entity LIKE "Fran%";"#).unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(conditions[0].op, CompareOp::Like))
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_select_in() {
    let stmt = parse(r#"SELECT * FROM EDGES WHERE entity IN ("France", "Germany");"#).unwrap();
    match stmt {
        Statement::Select { conditions, .. } => {
            assert!(matches!(conditions[0].op, CompareOp::In));
            if let Value::List(items) = &conditions[0].value {
                assert_eq!(items.len(), 2);
            } else {
                panic!("expected list value");
            }
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_with_leading_comment() {
    let stmt = parse("-- This is a comment\nSTATS;").unwrap();
    assert!(matches!(stmt, Statement::Stats { .. }));
}

#[test]
fn parse_with_trailing_comment() {
    let stmt = parse("STATS; -- trailing comment").unwrap();
    assert!(matches!(stmt, Statement::Stats { .. }));
}

#[test]
fn parse_multiline_statement() {
    let stmt = parse("SELECT *\n  FROM EDGES\n  WHERE layer = 26\n  LIMIT 5;").unwrap();
    match stmt {
        Statement::Select {
            conditions, limit, ..
        } => {
            assert_eq!(conditions.len(), 1);
            assert_eq!(limit, Some(5));
        }
        _ => panic!("expected Select"),
    }
}

#[test]
fn parse_error_unknown_statement() {
    assert!(parse("FOOBAR;").is_err());
}

#[test]
fn parse_error_walk_missing_prompt() {
    assert!(parse("WALK TOP 5;").is_err());
}

#[test]
fn parse_error_select_missing_from() {
    assert!(parse(r#"SELECT * WHERE entity = "x";"#).is_err());
}

#[test]
fn parse_error_insert_missing_values() {
    assert!(parse("INSERT INTO EDGES (entity, relation, target);").is_err());
}

#[test]
fn parse_error_show_invalid_noun() {
    assert!(parse("SHOW FOOBAR;").is_err());
}

#[test]
fn parse_error_empty_input() {
    assert!(parse("").is_err());
}

#[test]
fn parse_error_comment_only() {
    assert!(parse("-- just a comment").is_err());
}

#[test]
fn parse_demo_script_act1() {
    parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;"#).unwrap();
    parse(r#"USE "gemma3-4b.vindex";"#).unwrap();
    parse("STATS;").unwrap();
}

#[test]
fn parse_demo_script_act2() {
    parse("SHOW RELATIONS WITH EXAMPLES;").unwrap();
    parse(r#"DESCRIBE "France";"#).unwrap();
    parse(r#"DESCRIBE "Einstein";"#).unwrap();
    parse(r#"DESCRIBE "def" SYNTAX;"#).unwrap();
}

#[test]
fn parse_demo_script_act3() {
    parse(r#"WALK "France" TOP 10;"#).unwrap();
    parse(r#"EXPLAIN WALK "The capital of France is";"#).unwrap();
    parse(r#"INFER "The capital of France is" TOP 5 COMPARE;"#).unwrap();
}

#[test]
fn parse_demo_script_act4() {
    parse(r#"DESCRIBE "John Coyle";"#).unwrap();
    parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
    ).unwrap();
    parse(r#"DESCRIBE "John Coyle";"#).unwrap();
}

#[test]
fn parse_demo_script_act5() {
    parse(r#"DIFF "gemma3-4b.vindex" CURRENT;"#).unwrap();
    parse(r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#).unwrap();
}
