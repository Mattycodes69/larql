//! INFER / TRACE / EXPLAIN INFER mode variants

use super::*;

#[test]
fn infer_with_top_5_runs() {
    let (mut session, dir) = full_vindex_session("infer_top5");
    let stmt = parser::parse(r#"INFER "[1] [2] [3]" TOP 5;"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER TOP 5");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn infer_default_top_runs() {
    let (mut session, dir) = full_vindex_session("infer_default");
    let stmt = parser::parse(r#"INFER "[7]";"#).unwrap();
    let _ = session.execute(&stmt).expect("INFER default top");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_band_clause_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_band");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]" KNOWLEDGE;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER KNOWLEDGE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_relations_only_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_rel_only");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1]" RELATIONS ONLY;"#).unwrap();
    let _ = session
        .execute(&stmt)
        .expect("EXPLAIN INFER RELATIONS ONLY");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_attention_and_band_runs() {
    // WITH ATTENTION engages the f32 attention-capture path; combined
    // with KNOWLEDGE band exercises the band_to_layer_range filter
    // inside the per-layer render loop.
    let (mut session, dir) = full_vindex_session("explain_infer_attn_band");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]" KNOWLEDGE WITH ATTENTION;"#).unwrap();
    let _ = session
        .execute(&stmt)
        .expect("EXPLAIN INFER KNOWLEDGE WITH ATTENTION");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_attention_and_relations_only_runs() {
    // WITH ATTENTION + RELATIONS ONLY hits the compact-format render
    // branch with the labelled-hits-empty short-circuit, since the
    // synthetic vindex has no relation classifier.
    let (mut session, dir) = full_vindex_session("explain_infer_attn_relonly");
    let stmt = parser::parse(r#"EXPLAIN INFER "[5]" RELATIONS ONLY WITH ATTENTION;"#).unwrap();
    let _ = session
        .execute(&stmt)
        .expect("EXPLAIN INFER RELATIONS ONLY WITH ATTENTION");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_layers_range_filters_trace() {
    // LAYERS m-n exercises the layer_range filter in the per-layer
    // render loop.
    let (mut session, dir) = full_vindex_session("explain_infer_layers_range");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]" LAYERS 0-0;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER LAYERS 0-0");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_syntax_band_runs() {
    // SYNTAX band — different layer_range than KNOWLEDGE; exercises
    // the LayerBand::Syntax arm of band_to_layer_range.
    let (mut session, dir) = full_vindex_session("explain_infer_syntax");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]" SYNTAX;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER SYNTAX");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_output_band_runs() {
    // OUTPUT band — third arm of band_to_layer_range.
    let (mut session, dir) = full_vindex_session("explain_infer_output");
    let stmt = parser::parse(r#"EXPLAIN INFER "[1] [2]" OUTPUT;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER OUTPUT");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explain_infer_with_top_clause_runs() {
    let (mut session, dir) = full_vindex_session("explain_infer_top");
    let stmt = parser::parse(r#"EXPLAIN INFER "[5]" TOP 3;"#).unwrap();
    let _ = session.execute(&stmt).expect("EXPLAIN INFER TOP 3");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_for_specific_token_runs() {
    let (mut session, dir) = full_vindex_session("trace_for");
    let stmt = parser::parse(r#"TRACE "[1] [2]" FOR "[5]";"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE FOR token");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_positions_all_runs() {
    let (mut session, dir) = full_vindex_session("trace_positions_all");
    let stmt = parser::parse(r#"TRACE "[1] [2]" POSITIONS ALL;"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE POSITIONS ALL");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_save_writes_file() {
    let (mut session, dir) = full_vindex_session("trace_save");
    let save_path = dir.join("trace_output.json");
    let stmt = parser::parse(&format!(
        r#"TRACE "[1] [2]" POSITIONS ALL SAVE "{}";"#,
        lql_path(&save_path)
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("TRACE SAVE");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trace_with_layers_clause_runs() {
    let (mut session, dir) = full_vindex_session("trace_layers");
    let stmt = parser::parse(r#"TRACE "[1]" LAYERS 0-1;"#).unwrap();
    let _ = session.execute(&stmt).expect("TRACE LAYERS");
    let _ = std::fs::remove_dir_all(&dir);
}
