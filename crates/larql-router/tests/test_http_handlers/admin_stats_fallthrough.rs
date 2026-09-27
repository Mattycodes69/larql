//! `/v1/stats` proxying skips a shard that answers non-2xx and returns the
//! next shard's stats; `/metrics` serves Prometheus text when a registry
//! is installed.

use super::*;

/// A shard that is up but has no `/v1/stats` route (404s every request).
async fn spawn_statless_shard() -> SocketAddr {
    let app = axum::Router::new().route("/v1/health", get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

async fn get_path(app: axum::Router, uri: &str) -> axum::http::Response<Body> {
    app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn stats_skips_a_non_2xx_shard_and_proxies_the_next() {
    let statless = spawn_statless_shard().await;
    let (good, _calls) = spawn_fake_shard().await;
    let app = make_router(&format!("0-1=http://{statless},2-3=http://{good}"));

    let resp = get_path(app, "/v1/stats").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "application/json"
    );
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["hidden_size"], 2560);
}

#[tokio::test]
async fn stats_503s_when_every_shard_is_non_2xx() {
    let statless = spawn_statless_shard().await;
    let app = make_router(&format!("0-3=http://{statless}"));
    let resp = get_path(app, "/v1/stats").await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    assert_eq!(&body[..], br#"{"error":"no shard reachable"}"#);
}

#[tokio::test]
async fn metrics_serves_prometheus_text_and_503s_without_registry() {
    let without = make_router("0-3=http://127.0.0.1:1");
    let resp = get_path(without, "/metrics").await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/plain; charset=utf-8"
    );

    let state = Arc::new(AppState {
        static_shards: parse_shards("0-3=http://127.0.0.1:1").unwrap(),
        grid: None,
        client: reqwest::Client::new(),
        metrics: Some(larql_router::metrics::RouterMetrics::new()),
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
        openai_responses: Default::default(),
    });
    let resp = get_path(build_router(state), "/metrics").await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&body).contains("larql_router_build_info"));
}
