//! HTTP integration tests: embed, logits, token encode/decode (single + multi).

#[path = "../common/mod.rs"]
mod common;
use common::*;

use axum::body::Body;
use axum::http::Request;
use axum::http::StatusCode;
use larql_server::http::BINARY_FFN_CONTENT_TYPE;
use tower::ServiceExt;

fn binary_embed_body(token_ids: &[u32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + token_ids.len() * 4);
    body.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &token_id in token_ids {
        body.extend_from_slice(&token_id.to_le_bytes());
    }
    body
}

fn binary_logits_body(values: &[f32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(values.len() * 4);
    for &value in values {
        body.extend_from_slice(&value.to_le_bytes());
    }
    body
}

async fn post_binary(app: axum::Router, path: &str, body: Vec<u8>) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", BINARY_FFN_CONTENT_TYPE)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

// POST /v1/embed

// GET /v1/embed/{token_id}  (single-token lookup)

// POST /v1/logits

// GET /v1/token/decode

// GET /v1/token/encode

// POST /v1/embeddings — OpenAI-compatible embeddings (N0.4)

// POST /v1/completions — OpenAI-compatible completions (N0.2)
//
// These tests exercise request validation (the parts that don't
// require a real model + weights). End-to-end generation is exercised
// via the `larql run` CLI smoke test against a real vindex.

// OpenAI endpoints — multi-model routing
//
// In multi-model mode the client passes `model` in the request body
// (OpenAI convention). The endpoints route to the right loaded vindex
// without needing a path-prefixed `/v1/{model_id}/...` URL.

// OpenAI endpoints — auth flow

// POST /v1/chat/completions — N0.1 slice 2

// REV4 — OpenAI error envelope shape.
//
// /v1/embeddings, /v1/completions, /v1/chat/completions must return
// the nested `{error: {message, type, param, code}}` shape so the
// OpenAI Python and JS SDKs parse errors without special-casing.
// LARQL paradigm endpoints keep the flat `{error: "msg"}` shape
// (covered by other tests in this file).

fn assert_openai_error_envelope(v: &serde_json::Value, expected_type: &str) {
    let err = v
        .get("error")
        .and_then(|e| e.as_object())
        .expect("response body must be {\"error\": {...}} (nested)");
    assert!(
        err.get("message").and_then(|m| m.as_str()).is_some(),
        "error.message must be a non-null string; got {:?}",
        err.get("message")
    );
    assert_eq!(
        err.get("type").and_then(|t| t.as_str()),
        Some(expected_type),
        "error.type mismatch"
    );
    assert!(
        err.contains_key("param"),
        "error.param key must be present (even if null) — SDKs hard-key on it"
    );
    assert!(
        err.contains_key("code"),
        "error.code key must be present (even if null) — SDKs hard-key on it"
    );
}

mod multi_model_routes;
mod post_v1_chat_completions_n0_1_slice_2;
mod post_v1_chat_completions_n0_1_slice_2_2;
mod post_v1_embed;
mod post_v1_embeddings_openai_compatible_emb;
