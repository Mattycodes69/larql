//! The embed, logits and token routes under the multi-model `/v1/{model_id}/…`
//! prefix resolve the named model.

use super::*;

fn app() -> axum::Router {
    multi_model_router(state(vec![model("a"), model("b")]))
}

#[tokio::test]
async fn multi_model_embed_resolves_the_named_model() {
    let resp = post_json(
        app(),
        "/v1/b/embed",
        serde_json::json!({"token_ids": [0, 1]}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(body_json(resp.into_body()).await["seq_len"], 2);

    let missing = post_json(
        app(),
        "/v1/nope/embed",
        serde_json::json!({"token_ids": [0]}),
    )
    .await;
    assert_eq!(missing.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn multi_model_logits_resolves_the_named_model() {
    let resp = post_binary(app(), "/v1/a/logits", binary_logits_body(&[0.0; 4])).await;
    assert_ne!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "the route exists for a loaded model"
    );
}

#[tokio::test]
async fn multi_model_token_routes_resolve_the_named_model() {
    let enc = get(app(), "/v1/a/token/encode?text=hi").await;
    assert_ne!(enc.status(), StatusCode::NOT_FOUND);
    let dec = get(app(), "/v1/a/token/decode?ids=0").await;
    assert_ne!(dec.status(), StatusCode::NOT_FOUND);
}
