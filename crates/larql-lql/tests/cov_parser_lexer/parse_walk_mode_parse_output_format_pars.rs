//! parse_walk_mode / parse_output_format / parse_conflict_strategy
//! parse_component / parse_component_list
//! try_parse_layer_band (helpers.rs 33-60)

use super::*;

#[test]
fn walk_modes_all_three() {
    for (m, want) in [
        ("HYBRID", larql_lql::ast::WalkMode::Hybrid),
        ("PURE", larql_lql::ast::WalkMode::Pure),
        ("DENSE", larql_lql::ast::WalkMode::Dense),
    ] {
        let sql = format!(r#"WALK "x" MODE {m};"#);
        match ok(&sql) {
            Statement::Walk { mode, .. } => assert_eq!(mode, Some(want)),
            other => panic!("got {other:?}"),
        }
    }
}

#[test]
fn walk_mode_invalid_is_error() {
    // parse_walk_mode error (helpers.rs 76-79).
    err(r#"WALK "x" MODE TOP;"#);
}

#[test]
fn compile_output_formats_both() {
    match ok(r#"COMPILE CURRENT INTO MODEL "out.safetensors" FORMAT SAFETENSORS;"#) {
        Statement::Compile { format, .. } => {
            assert_eq!(format, Some(larql_lql::ast::OutputFormat::Safetensors));
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"COMPILE CURRENT INTO MODEL "out.gguf" FORMAT GGUF;"#) {
        Statement::Compile { format, .. } => {
            assert_eq!(format, Some(larql_lql::ast::OutputFormat::Gguf));
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn compile_output_format_invalid_is_error() {
    // parse_output_format error (helpers.rs 93-96).
    err(r#"COMPILE CURRENT INTO MODEL "out" FORMAT WALK;"#);
}

#[test]
fn merge_conflict_strategies_all_three() {
    for (s, want) in [
        ("KEEP_SOURCE", larql_lql::ast::ConflictStrategy::KeepSource),
        ("KEEP_TARGET", larql_lql::ast::ConflictStrategy::KeepTarget),
        (
            "HIGHEST_CONFIDENCE",
            larql_lql::ast::ConflictStrategy::HighestConfidence,
        ),
    ] {
        let sql = format!(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT {s};"#);
        match ok(&sql) {
            Statement::Merge { conflict, .. } => assert_eq!(conflict, Some(want)),
            other => panic!("got {other:?}"),
        }
    }
}

#[test]
fn merge_conflict_strategy_invalid_is_error() {
    // parse_conflict_strategy error (helpers.rs 114-117).
    err(r#"MERGE "a.vindex" INTO "b.vindex" ON CONFLICT FAIL;"#);
}

#[test]
fn merge_without_clauses() {
    let stmt = ok(r#"MERGE "a.vindex";"#);
    match stmt {
        Statement::Merge {
            source,
            target,
            conflict,
        } => {
            assert_eq!(source, "a.vindex");
            assert!(target.is_none());
            assert!(conflict.is_none());
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn component_list_via_keywords() {
    let stmt = ok(
        r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS, ATTN_OV, ATTN_QK;"#,
    );
    match stmt {
        Statement::Extract { components, .. } => {
            assert_eq!(components.unwrap().len(), 6);
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn component_list_via_unquoted_idents() {
    // parse_component Ident arm (helpers.rs 157-169): components given as bare
    // identifiers rather than keyword tokens. `gate`/`down` etc are NOT
    // keywords, but the full names are — so use names the lexer treats as
    // idents only when *not* keywords. `ffn_gate` lexes as a keyword, so to
    // hit the Ident arm we need a non-keyword spelling that the matcher still
    // accepts. The component matcher lowercases, and the lexer keyword set is
    // upper-only-by-canonical; `Ffn_Gate` still lexes to the keyword. So we
    // instead reach the Ident arm with the unknown-component error path below
    // and rely on keyword tokens for the happy path. This test asserts the
    // unknown-component Ident error (helpers.rs 165).
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS not_a_component;"#);
}

#[test]
fn component_invalid_token_is_error() {
    // parse_component fallthrough error (helpers.rs 170-173): a number is not a
    // component name.
    err(r#"EXTRACT MODEL "m" INTO "o.vindex" COMPONENTS 5;"#);
}

#[test]
fn describe_with_layer_bands() {
    // SYNTAX / KNOWLEDGE / OUTPUT bands + ALL LAYERS.
    match ok(r#"DESCRIBE "France" SYNTAX;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::Syntax))
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"DESCRIBE "France" KNOWLEDGE;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::Knowledge))
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"DESCRIBE "France" OUTPUT;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::Output))
        }
        other => panic!("got {other:?}"),
    }
    match ok(r#"DESCRIBE "France" ALL LAYERS;"#) {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(larql_lql::ast::LayerBand::All))
        }
        other => panic!("got {other:?}"),
    }
}

#[test]
fn describe_all_not_followed_by_layers_backtracks() {
    // try_parse_layer_band ALL arm with no LAYERS following: pos is restored
    // and None returned (helpers.rs 42-44). With `ALL` not followed by LAYERS,
    // the band is left unset and the describe loop breaks; the stray `ALL`
    // keyword then becomes a trailing token, so the overall parse is an error.
    // The backtrack restore (helpers.rs 42-43) is still executed en route.
    err(r#"DESCRIBE "France" ALL;"#);
}
