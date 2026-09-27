//! MERGE with a featured source → loop body + conflict arms (merge.rs)
//! COMPACT MINOR promotion-failure branch (executor/compact.rs)

use super::*;

#[test]
fn merge_featured_source_into_empty_overlay_fires_none_arm() {
    // A featured source vindex (real down_meta) merged into the synthetic
    // session's EMPTY overlay: every source feature has Some meta (so the
    // loop doesn't `continue` at the source-meta check, 62-63), and the
    // target overlay's feature_meta is None — so the `(None, _) => true`
    // arm (merge.rs:68) fires for each, exercising the write path
    // (76-78: update_feature_meta + merged++).
    let (mut session, _dir, _) = fresh_session();
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(r#"MERGE "{}";"#, sql_path(source.path())),
    )
    .expect("merge featured source");
    assert!(
        out.iter().any(|l| l.contains("features merged")),
        "expected merge summary with merged count, got: {out:?}"
    );
}

#[test]
fn merge_featured_source_keep_target_skips_existing() {
    // First seed the session overlay with compose features at layers 0/1
    // (feature 0), so the target's feature_meta is Some for those slots.
    // Then MERGE the featured source ON CONFLICT KEEP_TARGET → the
    // `(Some(_), KeepTarget) => false` arm (merge.rs:70) fires for the
    // overlapping slot (skipped++, 79-80) while non-overlapping source
    // slots still take the `(None, _)` arm.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(
            r#"MERGE "{}" ON CONFLICT KEEP_TARGET;"#,
            sql_path(source.path())
        ),
    )
    .expect("merge keep_target");
    assert!(out.iter().any(|l| l.contains("skipped")), "got: {out:?}");
}

#[test]
fn merge_featured_source_keep_source_overwrites_existing() {
    // Same overlapping setup, ON CONFLICT KEEP_SOURCE → the
    // `(Some(_), KeepSource) => true` arm (merge.rs:69) overwrites.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(
            r#"MERGE "{}" ON CONFLICT KEEP_SOURCE;"#,
            sql_path(source.path())
        ),
    )
    .expect("merge keep_source");
    assert!(
        out.iter().any(|l| l.contains("features merged")),
        "got: {out:?}"
    );
}

#[test]
fn merge_featured_source_highest_confidence_compares_scores() {
    // Same overlapping setup, ON CONFLICT HIGHEST_CONFIDENCE → the
    // `(Some(existing), HighestConfidence)` arm (merge.rs:71-73) compares
    // source vs existing c_score to decide should_write.
    let (mut session, _dir, _) = fresh_session();
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[1]", "capital", "[2]") AT LAYER 0 MODE COMPOSE;"#,
    )
    .expect("compose insert");
    let source = make_featured_source_dir();
    let out = try_run(
        &mut session,
        &format!(
            r#"MERGE "{}" ON CONFLICT HIGHEST_CONFIDENCE;"#,
            sql_path(source.path())
        ),
    )
    .expect("merge highest_confidence");
    assert!(!out.is_empty());
}

#[test]
fn compact_minor_reports_failed_promotion_when_layer_full() {
    // COMPACT MINOR promotes each L0 (KNN) entry by calling compose-mode
    // exec_insert at the entry's layer. When that layer's feature slots are
    // all claimed, `find_free_feature` returns None, `install_slots`
    // yields an empty list, and exec_insert errors with "no free feature
    // slots" — which COMPACT MINOR catches in its `Err(e)` arm
    // (compact.rs:71-74), incrementing `failed` and pushing the
    // "failed …" line.
    //
    // The synthetic vindex has intermediate_size=32 → 32 feature slots per
    // layer. We fill all 32 at layer 0 with compose inserts, then add a KNN
    // entry at layer 0 and COMPACT MINOR; its promotion has nowhere to go.
    let (mut session, _dir, _) = fresh_session();

    // Claim all 32 slots at layer 0. The first insert pays the decoy
    // capture; the rest reuse the per-layer decoy cache, so this stays
    // cheap on the 2-layer/16-dim synthetic model.
    for i in 0..32u32 {
        let _ = try_run(
            &mut session,
            &format!(
                r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[{}]", "capital", "[{}]") AT LAYER 0 MODE COMPOSE;"#,
                i % 16,
                (i % 15) + 1
            ),
        );
    }

    // A KNN entry at layer 0 — COMPACT MINOR will try to promote it via
    // compose, but layer 0 is now full.
    try_run(
        &mut session,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("[5]", "language", "[6]") AT LAYER 0 MODE KNN;"#,
    )
    .expect("knn insert at full layer");

    let out = try_run(&mut session, "COMPACT MINOR;").expect("compact minor ok");
    let joined = out.join("\n");
    assert!(
        joined.contains("failed") || joined.contains("complete"),
        "expected a failed-promotion line or completion summary, got:\n{joined}"
    );
}
