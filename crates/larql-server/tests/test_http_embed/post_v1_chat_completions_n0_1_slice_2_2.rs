//! POST /v1/chat/completions — N0.1 slice 2

use super::*;

#[tokio::test]
async fn http_openai_embeddings_400_uses_nested_envelope() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": [0u32, 1u32], "encoding_format": "binary"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("encoding_format='binary'"),
        "message should reference the bad input; got {msg:?}"
    );
}

#[tokio::test]
async fn http_openai_embeddings_empty_uses_nested_envelope() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embeddings", serde_json::json!({"input": []})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
}

#[tokio::test]
async fn http_openai_completions_400_uses_nested_envelope() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "stream": true, "echo": true, "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
}

#[tokio::test]
async fn http_openai_completions_503_uses_nested_envelope() {
    // model has infer_disabled = true → ServiceUnavailable.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "service_unavailable_error");
}

#[tokio::test]
async fn http_openai_chat_completions_400_uses_nested_envelope() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({"messages": []}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "invalid_request_error");
}

#[tokio::test]
async fn http_openai_chat_completions_503_uses_nested_envelope() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v = body_json(resp.into_body()).await;
    assert_openai_error_envelope(&v, "service_unavailable_error");
}
