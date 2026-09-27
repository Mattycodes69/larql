//! Session state: no backend
//! USE errors
//! Lifecycle: error cases without valid model/vindex
//! Mutation: no-backend errors
//! INFER
//! is_readable_token
//! is_content_token
//! SHOW MODELS works without backend
//! Pipe: errors propagate
//! Format helpers

use super::*;

#[test]
fn no_backend_stats() {
    let mut session = Session::new();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, LqlError::NoBackend));
}

#[test]
fn no_backend_walk() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::NoBackend));
}

#[test]
fn no_backend_describe() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_select() {
    let mut session = Session::new();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_explain() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_relations() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_layers() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW LAYERS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_features() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW FEATURES 26;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn use_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE "/nonexistent/path/fake.vindex";"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn use_model_fails_on_nonexistent() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE MODEL "/nonexistent/model";"#).unwrap();
    let result = session.execute(&stmt);
    // Should fail to resolve the model path
    assert!(result.is_err());
}

#[test]
fn use_with_corrupt_index_json_errors() {
    // load_vindex_config maps malformed JSON into VindexError::Parse;
    // use_cmd wraps it as LqlError::Execution. Exercises the error
    // branch on line 32-33 of use_cmd.rs.
    let dir = std::env::temp_dir().join(format!("larql_use_corrupt_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("index.json"), "{ this is not valid json").unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("vindex config") || msg.contains("config") || msg.contains("Parse"),
        "expected config-load error, got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn use_with_corrupt_knn_store_warns_and_continues() {
    // Build a valid vindex, then drop a corrupt knn_store.bin into
    // the dir. USE should not fail — the load failure is logged to
    // stderr and the session keeps going with an empty store.
    let dir = make_test_vindex_dir("use_corrupt_knn");
    std::fs::write(
        dir.join(larql_vindex::format::filenames::KNN_STORE_BIN),
        b"this is not a valid knn store",
    )
    .unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    let _ = session
        .execute(&stmt)
        .expect("USE should tolerate corrupt knn_store.bin");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn use_with_corrupt_memit_store_warns_and_continues() {
    // Same shape as the knn_store case: a malformed memit_store.json
    // shouldn't block USE — the session falls back to MemitStore::new().
    let dir = make_test_vindex_dir("use_corrupt_memit");
    std::fs::write(dir.join("memit_store.json"), "not valid json {{{{").unwrap();

    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    let _ = session
        .execute(&stmt)
        .expect("USE should tolerate corrupt memit_store.json");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn use_model_auto_extract_parses() {
    // Verify AUTO_EXTRACT parses correctly (loading will fail for nonexistent model)
    let mut session = Session::new();
    let stmt = parser::parse(r#"USE MODEL "/nonexistent/model" AUTO_EXTRACT;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
}

#[test]
fn extract_fails_on_nonexistent_model() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"EXTRACT MODEL "/nonexistent/model" INTO "/tmp/test_extract_out.vindex";"#)
            .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn compile_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn diff_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DIFF "/nonexistent/a.vindex" "/nonexistent/b.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

#[test]
fn insert_no_backend() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#)
            .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn delete_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DELETE FROM EDGES WHERE entity = "x";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn update_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"UPDATE EDGES SET target = "y" WHERE entity = "x";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn merge_nonexistent_source() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"MERGE "/nonexistent/source.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

#[test]
fn infer_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"INFER "test" TOP 5;"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn readable_tokens() {
    assert!(is_readable_token("French"));
    assert!(is_readable_token("Paris"));
    assert!(is_readable_token("capital-of"));
    assert!(is_readable_token("is"));
    assert!(is_readable_token("Europe"));
}

#[test]
fn unreadable_tokens() {
    assert!(!is_readable_token("ইসলামাবাদ"));
    assert!(!is_readable_token("южна"));
    assert!(!is_readable_token("ളാ"));
    assert!(!is_readable_token("ڪ"));
    assert!(!is_readable_token(""));
}

#[test]
fn content_tokens_pass() {
    assert!(is_content_token("French"));
    assert!(is_content_token("Paris"));
    assert!(is_content_token("Europe"));
    assert!(is_content_token("Mozart"));
    assert!(is_content_token("composer"));
    assert!(is_content_token("Berlin"));
    assert!(is_content_token("IBM"));
    assert!(is_content_token("Facebook"));
}

#[test]
fn stop_words_rejected() {
    assert!(!is_content_token("the"));
    assert!(!is_content_token("from"));
    assert!(!is_content_token("for"));
    assert!(!is_content_token("with"));
    assert!(!is_content_token("this"));
    assert!(!is_content_token("about"));
    assert!(!is_content_token("which"));
    assert!(!is_content_token("first"));
    assert!(!is_content_token("after"));
}

#[test]
fn short_tokens_rejected() {
    assert!(!is_content_token("a"));
    assert!(!is_content_token("of"));
    assert!(!is_content_token("is"));
    assert!(!is_content_token("-"));
    assert!(!is_content_token("lö"));
    assert!(!is_content_token("par"));
}

#[test]
fn code_tokens_rejected() {
    assert!(!is_content_token("trialComponents"));
    assert!(!is_content_token("NavigationBar"));
    assert!(!is_content_token("LastName"));
}

#[test]
fn show_models_no_crash() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

#[test]
fn pipe_error_propagates() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"STATS |> WALK "test";"#).unwrap();
    assert!(session.execute(&stmt).is_err());
}

#[test]
fn format_number_small() {
    assert_eq!(format_number(42), "42");
    assert_eq!(format_number(999), "999");
}

#[test]
fn format_number_thousands() {
    assert_eq!(format_number(1_000), "1.0K");
    assert_eq!(format_number(10_240), "10.2K");
    assert_eq!(format_number(348_160), "348.2K");
}

#[test]
fn format_number_millions() {
    assert_eq!(format_number(1_000_000), "1.00M");
    assert_eq!(format_number(2_917_432), "2.92M");
}

#[test]
fn format_bytes_small() {
    assert_eq!(format_bytes(512), "512 B");
}

#[test]
fn format_bytes_kb() {
    assert_eq!(format_bytes(2048), "2.0 KB");
}

#[test]
fn format_bytes_mb() {
    let mb = 5 * 1_048_576;
    assert_eq!(format_bytes(mb), "5.0 MB");
}

#[test]
fn format_bytes_gb() {
    let gb = 6_420_000_000;
    assert!(format_bytes(gb).contains("GB"));
}
