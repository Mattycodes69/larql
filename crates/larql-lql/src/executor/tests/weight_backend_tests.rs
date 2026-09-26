//! Weight backend tests

use super::*;

#[test]
fn weight_backend_stats() {
    let mut session = weight_session();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt).unwrap();
    assert!(result.iter().any(|l| l.contains("test/model")));
    assert!(result.iter().any(|l| l.contains("live weights")));
    assert!(result.iter().any(|l| l.contains("2"))); // num_layers
}

#[test]
fn weight_backend_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("requires a vindex"),
        "expected vindex error, got: {msg}"
    );
    assert!(
        msg.contains("EXTRACT"),
        "should suggest EXTRACT, got: {msg}"
    );
}

#[test]
fn weight_backend_describe_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_select_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_explain_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_insert_requires_vindex() {
    let mut session = weight_session();
    let stmt =
        parser::parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#)
            .unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex") || msg.contains("mutation requires"));
}

#[test]
fn weight_backend_show_relations_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_compile_current_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("EXTRACT") || msg.contains("vindex"));
}

#[test]
fn weight_backend_show_models_works() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}
