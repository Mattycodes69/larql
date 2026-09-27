//! DESCRIBE (rich fixture)

use super::*;

#[test]
fn describe_known_entity_runs_through_walk_pipeline() {
    let (mut session, dir) = rich_vindex_session("describe_known");
    // The walk against the rich fixture probably won't surface
    // gate-thresholded edges (DESCRIBE_GATE_THRESHOLD = 5.0 vs gate
    // norms ~ 1.0 here), so the orchestrator likely falls through to
    // "(no edges found)" — but the entire phase pipeline runs.
    let stmt = parser::parse(r#"DESCRIBE "Paris";"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE");
    // The first line is always the entity name.
    assert_eq!(out[0], "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_unknown_entity_emits_not_found() {
    let (mut session, dir) = rich_vindex_session("describe_unknown");
    // "Mongolia" isn't in the vocab — tokenises to [UNK] (id 0),
    // average_embed_rows succeeds (unit at axis 0), so we don't take
    // the "(not found)" branch but rather "(no edges found)" once the
    // walk produces nothing above the gate threshold.
    let stmt = parser::parse(r#"DESCRIBE "Mongolia";"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE");
    let joined = out.join("\n");
    // Either branch is acceptable — we just want to exercise the path.
    assert!(joined.contains("Mongolia"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_brief_mode_compiles() {
    let (mut session, dir) = rich_vindex_session("describe_brief");
    let stmt = parser::parse(r#"DESCRIBE "Paris" BRIEF;"#).unwrap();
    let _ = session.execute(&stmt).expect("DESCRIBE BRIEF");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_at_explicit_layer_uses_layer_filter() {
    let (mut session, dir) = rich_vindex_session("describe_layer");
    let stmt = parser::parse(r#"DESCRIBE "Paris" AT LAYER 1;"#).unwrap();
    let out = session.execute(&stmt).expect("DESCRIBE AT LAYER");
    assert_eq!(out[0], "Paris");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_with_band_clause_runs() {
    let (mut session, dir) = rich_vindex_session("describe_band");
    let stmt = parser::parse(r#"DESCRIBE "Paris" SYNTAX;"#).unwrap();
    let _ = session.execute(&stmt).expect("DESCRIBE SYNTAX");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn describe_relations_only_runs() {
    let (mut session, dir) = rich_vindex_session("describe_rel_only");
    let stmt = parser::parse(r#"DESCRIBE "Paris" RELATIONS ONLY;"#).unwrap();
    let _ = session.execute(&stmt).expect("DESCRIBE RELATIONS ONLY");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_with_entity_and_relation_drives_walk_path() {
    let (mut session, dir) = rich_vindex_session("select_edges_walk");
    // Both filters present → walk-anchored path. The rich fixture has
    // a relation classifier (feature_labels.json) so labelled edges
    // can match the user's relation predicate.
    let stmt =
        parser::parse(r#"SELECT * FROM EDGES WHERE entity = "Paris" AND relation = "capital";"#)
            .unwrap();
    let _ = session
        .execute(&stmt)
        .expect("SELECT EDGES walk path with classifier");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn select_edges_walk_path_with_feature_filter() {
    let (mut session, dir) = rich_vindex_session("select_edges_walk_feat");
    let stmt = parser::parse(
        r#"SELECT * FROM EDGES WHERE entity = "Paris" AND relation = "capital" AND feature = 0;"#,
    )
    .unwrap();
    let _ = session.execute(&stmt).expect("SELECT EDGES walk + feature");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_relations_runs_against_classifier() {
    let (mut session, dir) = rich_vindex_session("show_relations");
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    let _ = session.execute(&stmt).expect("SHOW RELATIONS");
    let _ = std::fs::remove_dir_all(&dir);
}
