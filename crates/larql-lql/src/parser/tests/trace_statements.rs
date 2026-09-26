//! TRACE STATEMENTS
//! Range validation
//! Keyword field name mapping

use super::*;

#[test]
fn parse_trace_minimal() {
    let stmt = parse(r#"TRACE "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Trace {
            prompt,
            answer,
            decompose,
            layers,
            positions,
            save,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert!(answer.is_none());
            assert!(!decompose);
            assert!(layers.is_none());
            assert!(positions.is_none());
            assert!(save.is_none());
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn parse_trace_with_for_token() {
    let stmt = parse(r#"TRACE "The capital of France is" FOR "Paris";"#).unwrap();
    match stmt {
        Statement::Trace { prompt, answer, .. } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(answer.unwrap(), "Paris");
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn parse_trace_decompose_with_layers() {
    let stmt = parse(r#"TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;"#).unwrap();
    match stmt {
        Statement::Trace {
            decompose, layers, ..
        } => {
            assert!(decompose);
            let r = layers.unwrap();
            assert_eq!(r.start, 22);
            assert_eq!(r.end, 27);
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn parse_trace_save() {
    let stmt = parse(r#"TRACE "The capital of France is" SAVE "france.trace";"#).unwrap();
    match stmt {
        Statement::Trace { save, .. } => {
            assert_eq!(save.unwrap(), "france.trace");
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn parse_trace_positions_all() {
    let stmt = parse(r#"TRACE "The capital of France is" POSITIONS ALL;"#).unwrap();
    match stmt {
        Statement::Trace { positions, .. } => {
            assert_eq!(positions.unwrap(), TracePositionMode::All);
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn parse_trace_positions_last() {
    let stmt = parse(r#"TRACE "The capital of France is" POSITIONS LAST;"#).unwrap();
    match stmt {
        Statement::Trace { positions, .. } => {
            assert_eq!(positions.unwrap(), TracePositionMode::Last);
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn parse_trace_full() {
    let stmt = parse(
        r#"TRACE "The capital of France is" FOR "Paris" DECOMPOSE LAYERS 22-27 SAVE "out.trace";"#,
    )
    .unwrap();
    match stmt {
        Statement::Trace {
            prompt,
            answer,
            decompose,
            layers,
            save,
            ..
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(answer.unwrap(), "Paris");
            assert!(decompose);
            assert_eq!(layers.as_ref().unwrap().start, 22);
            assert_eq!(save.unwrap(), "out.trace");
        }
        _ => panic!("expected Trace"),
    }
}

#[test]
fn range_invalid_start_greater_than_end() {
    let result = parse("SHOW LAYERS 10-5;");
    assert!(result.is_err(), "range 10-5 should fail");
}

#[test]
fn range_valid_same_start_end() {
    let stmt = parse("SHOW LAYERS 5-5;").unwrap();
    match stmt {
        Statement::ShowLayers { range } => {
            let r = range.unwrap();
            assert_eq!(r.start, 5);
            assert_eq!(r.end, 5);
        }
        _ => panic!("expected ShowLayers"),
    }
}

#[test]
fn keyword_field_names_consistent() {
    use crate::lexer::Keyword;
    // Key field-name keywords that must map correctly
    assert_eq!(Keyword::Layer.as_field_name(), "layer");
    assert_eq!(Keyword::Confidence.as_field_name(), "confidence");
    assert_eq!(Keyword::Relation.as_field_name(), "relation");
    assert_eq!(Keyword::FfnGate.as_field_name(), "ffn_gate");
    assert_eq!(Keyword::FfnDown.as_field_name(), "ffn_down");
    assert_eq!(Keyword::AttnOv.as_field_name(), "attn_ov");
    assert_eq!(Keyword::AutoExtract.as_field_name(), "auto_extract");
}

#[test]
fn parser_rejects_trailing_tokens_after_semicolon() {
    let result = parse(r#"STATS; SELECT * FROM EDGES;"#);
    assert!(result.is_err(), "single-statement parser must reject tails");
}

#[test]
fn parser_rejects_trailing_identifier_without_semicolon() {
    let result = parse(r#"STATS unexpected"#);
    assert!(result.is_err(), "single-statement parser must consume EOF");
}
