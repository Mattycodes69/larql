//! `/v1/completions` honours the server-side inference deadline and
//! reports a generation failure instead of hanging.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

async fn post_completion(timeout: std::time::Duration) -> (StatusCode, String) {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    post_completion_to(model, timeout).await
}

async fn post_completion_to(
    model: std::sync::Arc<larql_server::state::LoadedModel>,
    timeout: std::time::Duration,
) -> (StatusCode, String) {
    let state = common::state_with_timeout(vec![model], timeout);
    let app = larql_server::routes::single_model_router(state);
    let body = serde_json::json!({
        "model": "synthetic",
        "prompt": "the capital of France is",
        "max_tokens": 4,
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// A deadline short enough to expire while generation is held.
const HELD_DEADLINE: std::time::Duration = std::time::Duration::from_millis(50);

#[tokio::test]
async fn a_completion_past_the_deadline_is_a_gateway_timeout() {
    let (model, _fixture) = common::model_with_q4k_weights("synthetic");
    // Hold the generation lock so the request cannot finish before its
    // deadline: a bare 1ns deadline races a 4-token completion, and a fast
    // runner can win that race.
    let held = model.lock_weights_for_gen().expect("fixture weights load");
    let (status, body) = post_completion_to(model.clone(), HELD_DEADLINE).await;
    drop(held);
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(body.contains("timeout"), "{body}");
}

#[tokio::test]
async fn a_zero_deadline_disables_the_timeout() {
    let (status, body) = post_completion(std::time::Duration::ZERO).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn a_model_whose_weights_cannot_load_fails_the_request() {
    let model = common::model_infer_enabled("synthetic");
    let (status, body) = post_completion_to(model, std::time::Duration::ZERO).await;
    assert!(
        status.is_server_error() || status.is_client_error(),
        "{status}: {body}"
    );
}

#[tokio::test]
async fn a_bitnet_stream_whose_artifacts_are_missing_reports_the_load_error() {
    let Ok(mut model) = std::sync::Arc::try_unwrap(common::model_infer_enabled("synthetic")) else {
        panic!("a freshly built model has one owner");
    };
    model.config.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
    let state =
        common::state_with_timeout(vec![std::sync::Arc::new(model)], std::time::Duration::ZERO);
    let app = larql_server::routes::single_model_router(state);
    let body = serde_json::json!({
        "model": "synthetic",
        "prompt": "hello",
        "max_tokens": 2,
        "stream": true,
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("failed to load bitnet model"), "{text}");
}
