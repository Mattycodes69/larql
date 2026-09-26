//! parser/lifecycle.rs — USE / COMPILE / DIFF / EXTRACT / COMPACT
//! Cross-cutting: statement dispatch, pipe, trailing-token, empty input

use super::*;

#[test]
fn use_vindex_model_and_remote() {
    match ok(r#"USE "gemma3-4b.vindex";"#) {
        Statement::Use { target } => {
            assert!(matches!(target, larql_lql::ast::UseTarget::Vindex(_)))
        }
        other => panic!("got {other:?}"),
    }
    // USE MODEL ... AUTO_EXTRACT
    match ok(r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#) {
        Statement::Use { target } => match target {
            larql_lql::ast::UseTarget::Model { auto_extract, .. } => assert!(auto_extract),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
    // USE MODEL without AUTO_EXTRACT
    match ok(r#"USE MODEL "google/gemma-3-4b-it";"#) {
        Statement::Use { target } => match target {
            larql_lql::ast::UseTarget::Model { auto_extract, .. } => assert!(!auto_extract),
            other => panic!("got {other:?}"),
        },
        other => panic!("got {other:?}"),
    }
    // USE REMOTE — lifecycle.rs 202-204.
    match ok(r#"USE REMOTE "http://localhost:8080";"#) {
        Statement::Use { target } => {
            assert!(matches!(target, larql_lql::ast::UseTarget::Remote(_)))
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compile_into_model_and_vindex() {
    match ok(r#"COMPILE CURRENT INTO MODEL "out.safetensors";"#) {
        Statement::Compile { target, .. } => {
            assert_eq!(target, larql_lql::ast::CompileTarget::Model)
        }
        other => panic!("got {other:?}"),
    }
    // INTO VINDEX (vindex is a bare identifier, not a keyword).
    match ok(r#"COMPILE "src.vindex" INTO VINDEX "out.vindex";"#) {
        Statement::Compile { target, .. } => {
            assert_eq!(target, larql_lql::ast::CompileTarget::Vindex)
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compile_into_unrecognised_target_is_error() {
    // lifecycle.rs 75-76: INTO followed by neither MODEL nor `vindex` ident.
    err(r#"COMPILE CURRENT INTO "out.safetensors";"#);
}

#[test]
fn compile_into_vindex_on_conflict_strategies() {
    for (s, want) in [
        ("LAST_WINS", larql_lql::ast::CompileConflict::LastWins),
        (
            "HIGHEST_CONFIDENCE",
            larql_lql::ast::CompileConflict::HighestConfidence,
        ),
        ("FAIL", larql_lql::ast::CompileConflict::Fail),
    ] {
        let sql = format!(r#"COMPILE "s.vindex" INTO VINDEX "o.vindex" ON CONFLICT {s};"#);
        match ok(&sql) {
            Statement::Compile { on_conflict, .. } => assert_eq!(on_conflict, Some(want)),
            other => panic!("got {other:?}"),
        }
    }
}

#[test]
fn compile_on_conflict_invalid_strategy_is_error() {
    // lifecycle.rs 112-116: bad strategy keyword after ON CONFLICT.
    err(r#"COMPILE "s.vindex" INTO VINDEX "o.vindex" ON CONFLICT WALK;"#);
}

#[test]
fn compile_on_conflict_rejected_for_model_target() {
    // lifecycle.rs 119-123: ON CONFLICT is only valid for INTO VINDEX.
    err(r#"COMPILE CURRENT INTO MODEL "o.safetensors" ON CONFLICT LAST_WINS;"#);
}

#[test]
fn extract_with_inference_all_and_legacy_weights() {
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH INFERENCE;"#) {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Inference)
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH ALL;"#) {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::All)
        }
        other => panic!("got {other:?}"),
    }
    // Legacy WITH WEIGHTS → Inference.
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH WEIGHTS;"#) {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Inference)
        }
        other => panic!("got {other:?}"),
    }
    // Default (no WITH) → Browse, with COMPONENTS + LAYERS.
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS FFN_GATE LAYERS 0-5;"#) {
        Statement::Extract {
            extract_level,
            layers,
            ..
        } => {
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Browse);
            assert!(layers.is_some());
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn extract_with_unknown_keyword_is_error() {
    // lifecycle.rs 40: WITH followed by neither INFERENCE/ALL/WEIGHTS.
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" WITH WALK;"#);
}

#[test]
fn extract_format_names_a_generation_or_stays_absent() {
    use larql_lql::ast::ExtractFormat;
    // Explicit requests parse, case-insensitively (bare identifiers, as
    // with COMPILE INTO VINDEX).
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" FORMAT VINDEX2;"#) {
        Statement::Extract { format, .. } => assert_eq!(format, Some(ExtractFormat::Vindex2)),
        other => panic!("got {other:?}"),
    }
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex" FORMAT vindex3 WITH INFERENCE;"#) {
        Statement::Extract {
            format,
            extract_level,
            ..
        } => {
            assert_eq!(format, Some(ExtractFormat::Vindex3));
            assert_eq!(extract_level, larql_lql::ast::ExtractLevel::Inference);
        }
        other => panic!("got {other:?}"),
    }
    // Absence is "no preference" — a distinct value from either request,
    // resolved by the vindex crate's policy site, not the parser.
    match ok(r#"EXTRACT MODEL "m" INTO "o.vindex";"#) {
        Statement::Extract { format, .. } => assert_eq!(format, None),
        other => panic!("got {other:?}"),
    }
    // FORMAT followed by anything else is a parse error.
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" FORMAT GGUF;"#);
}

#[test]
fn diff_with_all_optional_clauses() {
    let stmt = ok(r#"DIFF "a.vindex" "b.vindex" LAYER 5 RELATION "capital-of" LIMIT 10;"#);
    match stmt {
        Statement::Diff {
            layer,
            relation,
            limit,
            into_patch,
            ..
        } => {
            assert_eq!(layer, Some(5));
            assert_eq!(relation.as_deref(), Some("capital-of"));
            assert_eq!(limit, Some(10));
            assert!(into_patch.is_none());
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn diff_relations_alias_and_into_patch() {
    // RELATIONS keyword variant + INTO PATCH terminal clause (lifecycle 152-173).
    let stmt = ok(r#"DIFF CURRENT "b.vindex" RELATIONS "x" INTO PATCH "p.vlp";"#);
    match stmt {
        Statement::Diff {
            relation,
            into_patch,
            ..
        } => {
            assert_eq!(relation.as_deref(), Some("x"));
            assert_eq!(into_patch.as_deref(), Some("p.vlp"));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn diff_minimal() {
    let stmt = ok(r#"DIFF "a.vindex" "b.vindex";"#);
    assert!(matches!(stmt, Statement::Diff { .. }));
}

#[test]
fn compact_minor_and_major_variants() {
    assert!(matches!(ok("COMPACT MINOR;"), Statement::CompactMinor));

    // MAJOR plain.
    match ok("COMPACT MAJOR;") {
        Statement::CompactMajor { full, lambda } => {
            assert!(!full);
            assert!(lambda.is_none());
        }
        other => panic!("got {other:?}"),
    }
    // MAJOR FULL (FULL spelled as ident).
    match ok("COMPACT MAJOR FULL;") {
        Statement::CompactMajor { full, .. } => assert!(full),
        other => panic!("got {other:?}"),
    }
    // MAJOR ALL (FULL via the ALL keyword — lifecycle.rs 227-230).
    match ok("COMPACT MAJOR ALL;") {
        Statement::CompactMajor { full, .. } => assert!(full),
        other => panic!("got {other:?}"),
    }
    // MAJOR WITH LAMBDA = <f>.
    match ok("COMPACT MAJOR WITH LAMBDA = 0.25;") {
        Statement::CompactMajor { lambda, .. } => {
            assert!((lambda.unwrap() - 0.25).abs() < 1e-6)
        }
        other => panic!("got {other:?}"),
    }
    // MAJOR FULL WITH LAMBDA = <f> (both clauses).
    match ok("COMPACT MAJOR FULL WITH LAMBDA = 0.5;") {
        Statement::CompactMajor { full, lambda } => {
            assert!(full);
            assert!((lambda.unwrap() - 0.5).abs() < 1e-6);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compact_major_with_lambda_missing_eq_is_error() {
    // lifecycle.rs 244-246: LAMBDA must be followed by '='.
    err("COMPACT MAJOR WITH LAMBDA 0.25;");
}

#[test]
fn compact_major_with_non_lambda_is_error() {
    // lifecycle.rs 250-253: WITH must be followed by LAMBDA in COMPACT MAJOR.
    err("COMPACT MAJOR WITH FULL;");
}

#[test]
fn compact_unknown_subcommand_is_error() {
    // lifecycle.rs 262-265: neither MINOR nor MAJOR.
    err("COMPACT EVERYTHING;");
}

#[test]
fn unknown_leading_keyword_is_error() {
    // parse_statement fallthrough (mod.rs 85-88): no statement keyword.
    err("FROM EDGES;");
}

#[test]
fn empty_input_is_error() {
    // Empty token stream → parse_statement sees Eof → error.
    err("");
    err("   ");
    err("-- only a comment");
}

#[test]
fn trailing_token_after_statement_is_error() {
    // mod.rs 51-56: extra tokens after a complete statement (no pipe).
    err("STATS extra;");
}

#[test]
fn select_default_source_and_nearest_clause() {
    // SELECT ... FROM EDGES NEAREST TO "X" AT LAYER N (query.rs nearest path).
    let stmt = ok(r#"SELECT * FROM EDGES NEAREST TO "France" AT LAYER 14;"#);
    match stmt {
        Statement::Select { nearest, .. } => {
            let n = nearest.expect("nearest present");
            assert_eq!(n.entity, "France");
            assert_eq!(n.layer, 14);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn select_from_features_and_entities() {
    assert!(matches!(
        ok("SELECT * FROM FEATURES;"),
        Statement::Select {
            source: larql_lql::ast::SelectSource::Features,
            ..
        }
    ));
    assert!(matches!(
        ok("SELECT * FROM ENTITIES;"),
        Statement::Select {
            source: larql_lql::ast::SelectSource::Entities,
            ..
        }
    ));
}

#[test]
fn order_by_asc_explicit_and_default() {
    match ok("SELECT * FROM EDGES ORDER BY confidence ASC;") {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        other => panic!("got {other:?}"),
    }
    // No ASC/DESC → default ascending.
    match ok("SELECT * FROM EDGES ORDER BY confidence;") {
        Statement::Select { order, .. } => assert!(!order.unwrap().descending),
        other => panic!("got {other:?}"),
    }
}
