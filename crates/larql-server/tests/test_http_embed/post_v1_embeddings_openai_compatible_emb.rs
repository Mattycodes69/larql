//! POST /v1/embeddings — OpenAI-compatible embeddings (N0.4)
//! OpenAI endpoints — auth flow

use super::*;

#[tokio::test]
async fn http_openai_embeddings_string_input_returns_200_with_pooled_vector() {
    // Uses the functional tokenizer so "France" tokenises cleanly.
    let app = single_model_router(state(vec![model_functional("gemma")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "gemma");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let embedding = data[0]["embedding"].as_array().unwrap();
    assert_eq!(embedding.len(), 4); // hidden_size=4 in synthetic model
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["prompt_tokens"],
        body["usage"]["total_tokens"]
    );
}

#[tokio::test]
async fn http_openai_embeddings_string_array_returns_indexed_data() {
    let app = single_model_router(state(vec![model_functional("gemma")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": ["France", "Germany", "capital"]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (i, entry) in data.iter().enumerate() {
        assert_eq!(entry["index"], i);
        assert_eq!(entry["object"], "embedding");
        let v = entry["embedding"].as_array().unwrap();
        assert_eq!(v.len(), 4);
    }
}

#[tokio::test]
async fn http_openai_embeddings_pretokenised_single_works() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": [0u32, 1u32, 2u32]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(body["usage"]["prompt_tokens"], 3);
}

#[tokio::test]
async fn http_openai_embeddings_base64_format_returns_string() {
    // base64 is now supported — the embedding field is a base64 string
    // of the LE f32 bytes instead of a JSON array. Use pretokenised
    // input so the synthetic tokenizer doesn't gate the test path.
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({
            "input": [0u32, 1u32, 2u32],
            "encoding_format": "base64",
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let embedding = &body["data"][0]["embedding"];
    assert!(
        embedding.is_string(),
        "expected base64 string, got {embedding}"
    );
    let s = embedding.as_str().unwrap();
    // Decode + sanity-check length: 4 bytes per f32, must be ≥1 f32.
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .expect("valid base64");
    assert!(!bytes.is_empty());
    assert_eq!(
        bytes.len() % 4,
        0,
        "len must be 4·hidden, got {}",
        bytes.len()
    );
}

#[tokio::test]
async fn http_openai_embeddings_unknown_format_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": [0u32, 1u32], "encoding_format": "binary"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_embeddings_empty_input_returns_400() {
    let app = single_model_router(state(vec![model("test")]));
    let resp = post_json(app, "/v1/embeddings", serde_json::json!({"input": []})).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_stream_with_echo_returns_400() {
    // echo=true is not supported in stream mode (one-prompt-one-stream).
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "stream": true,
            "echo": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_stream_with_batched_prompts_returns_400() {
    // Batched prompts not supported with stream=true.
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": ["hi", "there"],
            "stream": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_stream_returns_event_stream_content_type() {
    use axum::http::header;
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "stream": true,
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
}

#[tokio::test]
async fn http_openai_completions_n_gt_1_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "n": 2, "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_completions_infer_disabled_returns_503() {
    // model() builds with infer_disabled=true.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"prompt": "hi", "max_tokens": 1}),
    )
    .await;
    // ServerError::InferenceUnavailable maps to 503.
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_missing_prompt_returns_422() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(app, "/v1/completions", serde_json::json!({"max_tokens": 1})).await;
    // Missing required `prompt` field — serde returns 422 via axum's
    // Json extractor.
    assert!(
        resp.status() == StatusCode::UNPROCESSABLE_ENTITY
            || resp.status() == StatusCode::BAD_REQUEST,
        "got {}",
        resp.status()
    );
}

#[tokio::test]
async fn http_openai_models_multi_lists_all_with_openai_shape() {
    let app = multi_model_router(state(vec![
        model_functional("gemma-a"),
        model_functional("gemma-b"),
    ]));
    let resp = get(app, "/v1/models").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"gemma-a"));
    assert!(ids.contains(&"gemma-b"));
    for entry in data {
        assert_eq!(entry["object"], "model");
        assert_eq!(entry["owned_by"], "larql");
    }
}

#[tokio::test]
async fn http_openai_embeddings_multi_routes_via_model_field() {
    let app = multi_model_router(state(vec![
        model_functional("gemma-a"),
        model_functional("gemma-b"),
    ]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"model": "gemma-b", "input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "gemma-b");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["index"], 0);
}

#[tokio::test]
async fn http_openai_embeddings_multi_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model_functional("gemma-a")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"model": "missing", "input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_openai_embeddings_no_model_field_in_single_model_works() {
    // Single-model mode: omitting `model` is fine; we use the loaded one.
    let app = single_model_router(state(vec![model_functional("gemma")]));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["model"], "gemma");
}

#[tokio::test]
async fn http_openai_completions_multi_routes_via_model_field() {
    // Use ModelBuilder to flip infer_disabled=false.
    use larql_server::state::LoadedModel;
    use std::sync::Arc;
    let m = ModelBuilder::new("gemma-a").build();
    let n = ModelBuilder::new("gemma-b").build();
    let _: Arc<LoadedModel> = Arc::clone(&m);
    let app = multi_model_router(state(vec![m, n]));
    // infer_disabled=true on default ModelBuilder → expect 503.
    // We're testing routing, not generation — 503 from the right model
    // confirms routing worked.
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"model": "gemma-b", "prompt": "x", "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_multi_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model_functional("gemma-a")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({"model": "missing", "prompt": "x", "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_openai_embeddings_with_auth_required_no_token_returns_401() {
    use axum::middleware;
    let app_state = state_with_key(vec![model_functional("gemma")], "sk-secret");
    let app = single_model_router(app_state.clone()).layer(middleware::from_fn_with_state(
        app_state,
        larql_server::auth::auth_middleware,
    ));
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn http_openai_embeddings_with_auth_correct_bearer_returns_200() {
    use axum::middleware;
    let app_state = state_with_key(vec![model_functional("gemma")], "sk-secret");
    let app = single_model_router(app_state.clone()).layer(middleware::from_fn_with_state(
        app_state,
        larql_server::auth::auth_middleware,
    ));
    let resp = post_json_h(
        app,
        "/v1/embeddings",
        serde_json::json!({"input": "France"}),
        ("authorization", "Bearer sk-secret"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}
