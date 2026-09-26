//! EXPLAIN

use super::*;

#[test]
fn parse_explain_walk_minimal() {
    let stmt = parse(r#"EXPLAIN WALK "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Explain {
            prompt,
            mode,
            layers,
            verbose,
            ..
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(mode, ExplainMode::Walk);
            assert!(layers.is_none());
            assert!(!verbose);
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_walk_with_layers_and_verbose() {
    let stmt = parse(r#"EXPLAIN WALK "prompt" LAYERS 24-33 VERBOSE;"#).unwrap();
    match stmt {
        Statement::Explain {
            layers, verbose, ..
        } => {
            let l = layers.unwrap();
            assert_eq!(l.start, 24);
            assert_eq!(l.end, 33);
            assert!(verbose);
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_infer_minimal() {
    let stmt = parse(r#"EXPLAIN INFER "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Explain {
            prompt,
            mode,
            layers,
            band,
            verbose,
            top,
            relations_only,
            with_attention,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(mode, ExplainMode::Infer);
            assert!(layers.is_none());
            assert!(band.is_none());
            assert!(!verbose);
            assert!(top.is_none());
            assert!(!relations_only);
            assert!(!with_attention);
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_infer_with_options() {
    let stmt = parse(r#"EXPLAIN INFER "test prompt" LAYERS 20-30 VERBOSE TOP 10;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            layers,
            verbose,
            top,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            let l = layers.unwrap();
            assert_eq!(l.start, 20);
            assert_eq!(l.end, 30);
            assert!(verbose);
            assert_eq!(top, Some(10));
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_walk_with_top() {
    let stmt = parse(r#"EXPLAIN WALK "test" TOP 5;"#).unwrap();
    match stmt {
        Statement::Explain { mode, top, .. } => {
            assert_eq!(mode, ExplainMode::Walk);
            assert_eq!(top, Some(5));
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_infer_with_band() {
    let stmt = parse(r#"EXPLAIN INFER "test" KNOWLEDGE;"#).unwrap();
    match stmt {
        Statement::Explain { mode, band, .. } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert_eq!(band, Some(LayerBand::Knowledge));
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_infer_relations_only() {
    let stmt = parse(r#"EXPLAIN INFER "test" RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            relations_only,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert!(relations_only);
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_infer_with_attention() {
    let stmt = parse(r#"EXPLAIN INFER "test" WITH ATTENTION;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            with_attention,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert!(with_attention);
        }
        _ => panic!("expected Explain"),
    }
}

#[test]
fn parse_explain_infer_all_options() {
    let stmt =
        parse(r#"EXPLAIN INFER "test" KNOWLEDGE TOP 1 RELATIONS ONLY WITH ATTENTION;"#).unwrap();
    match stmt {
        Statement::Explain {
            mode,
            band,
            top,
            relations_only,
            with_attention,
            ..
        } => {
            assert_eq!(mode, ExplainMode::Infer);
            assert_eq!(band, Some(LayerBand::Knowledge));
            assert_eq!(top, Some(1));
            assert!(relations_only);
            assert!(with_attention);
        }
        _ => panic!("expected Explain"),
    }
}
