//! INFER (walk-FFN, compare, knn-override, default top, error)

use super::*;

#[test]
fn infer_renders_walk_predictions() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [
                    {"token": "Paris", "probability": 0.7},
                    {"token": "Lyon", "probability": 0.1}
                ],
                "latency_ms": 12.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"INFER "The capital of France is" TOP 2;"#);
    assert!(joined.contains("Predictions (walk FFN)"));
    assert!(joined.contains("Paris"));
    assert!(joined.contains("70.00%"));
    assert!(joined.contains("ms (remote)"));
}

#[test]
fn infer_compare_mode_renders_walk_and_dense() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "walk": [{"token": "Paris", "probability": 0.7}],
                "walk_ms": 10.0,
                "dense": [{"token": "Paris", "probability": 0.65}],
                "dense_ms": 80.0,
                "latency_ms": 92.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"INFER "test" TOP 1 COMPARE;"#);
    assert!(joined.contains("Predictions (walk)"));
    assert!(joined.contains("Predictions (dense)"));
    // walk_ms / dense_ms branch.
    assert!(joined.contains("ms"));
}

#[test]
fn infer_renders_knn_override_note() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(200)
        .with_body(
            serde_json::json!({
                "predictions": [],
                "knn_override": {"token": "Atlantis", "cosine": 0.91, "layer": 5,
                                 "model_top1": {"token": "Greece", "probability": 0.3}},
                "latency_ms": 12.0,
            })
            .to_string(),
        )
        .create();
    let joined = ok_joined(&mut session, r#"INFER "any";"#);
    assert!(joined.contains("KNN override: Atlantis"));
    assert!(joined.contains("note: KNN override"));
    assert!(joined.contains("model_top1=Greece"));
}

#[test]
fn infer_errors_on_http_502() {
    let (mut server, mut session) = connect();
    server
        .mock("POST", "/v1/infer")
        .with_status(502)
        .with_body("bad gateway")
        .create();
    let err = run(&mut session, r#"INFER "p";"#).unwrap_err();
    assert!(err.contains("502"), "got: {err}");
}
