//! DIFF INTO PATCH

use super::*;

#[test]
fn diff_into_patch_writes_vlp_file() {
    let dir_a = make_test_vindex_dir("diff_into_patch_a");
    let dir_b = make_test_vindex_dir("diff_into_patch_b");
    let mut session = Session::new();
    let patch_path =
        std::env::temp_dir().join(format!("larql_diff_patch_{}.vlp", std::process::id()));
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" INTO PATCH "{}";"#,
        lql_path(&dir_a),
        lql_path(&dir_b),
        lql_path(&patch_path)
    ))
    .unwrap();
    let _ = session.execute(&stmt).expect("DIFF INTO PATCH");
    let _ = std::fs::remove_file(&patch_path);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_changes_reports_modified_added_removed() {
    // DIFF base vs modified should surface all three status categories
    // exercised by the `match (meta_a, meta_b) { ... }` arm.
    let dir_a = make_test_vindex_dir("diff_changes_a");
    let dir_b = make_modified_test_vindex_dir("diff_changes_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}";"#,
        lql_path(&dir_a),
        lql_path(&dir_b),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("modified"),
        "expected 'modified' status, got: {joined}"
    );
    assert!(
        joined.contains("removed"),
        "expected 'removed' status, got: {joined}"
    );
    assert!(
        joined.contains("added"),
        "expected 'added' status, got: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_no_changes_reports_no_differences() {
    // DIFF a fixture against itself: every (Some,Some) cell hits the
    // "tokens equal AND c_score within 0.01" continue arm and the
    // (None,None) arm continues; the final no-differences message
    // should appear.
    let dir_a = make_test_vindex_dir("diff_self_a");
    let dir_b = make_test_vindex_dir("diff_self_b"); // same content
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}";"#,
        lql_path(&dir_a),
        lql_path(&dir_b),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF self");
    let joined = out.join("\n");
    assert!(
        joined.contains("no differences found"),
        "expected no-differences message, got: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_into_patch_with_real_changes_serialises_all_op_types() {
    // INTO PATCH on diverging fixtures should produce a .vlp file with
    // Update + Delete + Insert ops corresponding to modified/removed/added.
    let dir_a = make_test_vindex_dir("diff_patch_real_a");
    let dir_b = make_modified_test_vindex_dir("diff_patch_real_b");
    let mut session = Session::new();
    let patch_path = std::env::temp_dir().join(format!(
        "larql_diff_real_patch_{}_{}.vlp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    ));
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" INTO PATCH "{}";"#,
        lql_path(&dir_a),
        lql_path(&dir_b),
        lql_path(&patch_path),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF INTO PATCH real");
    let joined = out.join("\n");

    // The summary line includes counts for inserts/updates/deletes.
    assert!(
        joined.contains("Extracted:"),
        "expected 'Extracted:' summary, got: {joined}"
    );
    assert!(
        joined.contains("inserts") && joined.contains("updates") && joined.contains("deletes"),
        "expected all three op kinds in summary, got: {joined}"
    );
    assert!(
        patch_path.exists(),
        "patch file should be written at {}",
        patch_path.display()
    );

    let _ = std::fs::remove_file(&patch_path);
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_current_resolves_to_active_vindex() {
    // VindexRef::Current resolves to the session's active backend path.
    let (mut session, dir_a) = vindex_session("diff_current_a");
    let dir_b = make_modified_test_vindex_dir("diff_current_b");
    let stmt = parser::parse(&format!(r#"DIFF CURRENT "{}";"#, lql_path(&dir_b),)).unwrap();
    let out = session
        .execute(&stmt)
        .expect("DIFF CURRENT against another path");
    let joined = out.join("\n");
    assert!(
        joined.contains(dir_a.to_string_lossy().as_ref()),
        "DIFF header should reference current vindex path, got: {joined}",
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_layer_filter_excludes_other_layers() {
    // LAYER 1 should restrict the diff to layer 1 — modifications on
    // layer 0 (Paris → Madrid, French → None) must not appear.
    let dir_a = make_test_vindex_dir("diff_layer_filter_a");
    let dir_b = make_modified_test_vindex_dir("diff_layer_filter_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" LAYER 1;"#,
        lql_path(&dir_a),
        lql_path(&dir_b),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF LAYER 1");
    let joined = out.join("\n");
    // Layer 1 only adds Rome; should NOT show Madrid/French
    assert!(
        !joined.contains("Madrid"),
        "LAYER 1 filter should exclude L0 changes, got: {joined}"
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn diff_with_explicit_limit_caps_displayed_diffs() {
    // LIMIT 1 should display at most 1 diff row even though the
    // modified fixture has 3 changes.
    let dir_a = make_test_vindex_dir("diff_limit_cap_a");
    let dir_b = make_modified_test_vindex_dir("diff_limit_cap_b");
    let mut session = Session::new();
    let stmt = parser::parse(&format!(
        r#"DIFF "{}" "{}" LIMIT 1;"#,
        lql_path(&dir_a),
        lql_path(&dir_b),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("DIFF LIMIT 1");
    let joined = out.join("\n");
    // Only one diff row should appear; the summary line still mentions LIMIT.
    let diff_rows: usize = out
        .iter()
        .filter(|l| l.starts_with("L0") || l.starts_with("L1"))
        .count();
    assert!(
        diff_rows <= 1,
        "LIMIT 1 should cap diff rows, got {} rows in: {joined}",
        diff_rows,
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}
