//! Dense walk-ffn handler, request parsing and proxy tests.

use super::*;

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = make_router("0-3=http://127.0.0.1:1"); // shard URL unused for /v1/health
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn walk_ffn_rejects_invalid_json_body() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("not json"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("invalid JSON"));
}

#[tokio::test]
async fn walk_ffn_rejects_missing_layer_field() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"foo":"bar"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("must provide"));
}

#[tokio::test]
async fn walk_ffn_rejects_empty_layer_list() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layers":[]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_rejects_layer_outside_shard_map() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":99}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("no owning shard"));
}

#[tokio::test]
async fn walk_ffn_proxies_single_shard_json_unchanged() {
    let (addr, calls) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr}"));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":2}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["results"].is_array());
    let stored = calls.inner.lock().await.clone();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0]["layer"], 2);
}

#[tokio::test]
async fn walk_ffn_fans_out_to_multiple_shards_and_merges() {
    let (addr_a, _calls_a) = spawn_fake_shard().await;
    let (addr_b, _calls_b) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr_a},4-7=http://{addr_b}"));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layers":[1,5]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    // Sorted by layer.
    assert_eq!(results[0]["layer"], 1);
    assert_eq!(results[1]["layer"], 5);
    // latency_ms is the max of both shards (both reported 5.5).
    assert!((v["latency_ms"].as_f64().unwrap() - 5.5).abs() < 1e-6);
}

#[tokio::test]
async fn walk_ffn_binary_single_shard_round_trips() {
    let (addr, _) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr}"));
    // Binary body: 4-byte little-endian layer id only.
    let body = 2u32.to_le_bytes().to_vec();
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, BINARY_CT)
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let resp_ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string());
    assert_eq!(resp_ct.as_deref(), Some(BINARY_CT));
    let echoed = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    assert_eq!(echoed.as_ref(), body.as_slice());
}

#[tokio::test]
async fn walk_ffn_binary_rejects_truncated_header() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, BINARY_CT)
                .body(Body::from(vec![0u8, 1u8])) // < 4 bytes
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn walk_ffn_binary_rejects_multi_shard_fanout() {
    let app = make_router("0-3=http://127.0.0.1:1,4-7=http://127.0.0.1:2");
    // Binary batch header: layers 1 and 5 live on different shards.
    let mut body = Vec::new();
    body.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // BATCH_MARKER
    body.extend_from_slice(&2u32.to_le_bytes()); // n=2
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&5u32.to_le_bytes());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, BINARY_CT)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("binary fan-out"));
}

#[tokio::test]
async fn stats_proxies_to_first_reachable_shard() {
    let (addr, _) = spawn_fake_shard().await;
    let app = make_router(&format!("0-3=http://{addr}"));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["hidden_size"], 2560);
}

#[tokio::test]
async fn stats_returns_503_when_no_shard_reachable() {
    let app = make_router("0-3=http://127.0.0.1:1"); // unreachable
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].as_str().unwrap().contains("no shard reachable"));
}

#[tokio::test]
async fn walk_ffn_routes_via_grid_when_grid_state_is_set() {
    use larql_router::grid::{GridState, ServerEntry};
    use parking_lot::RwLock;
    use std::collections::HashMap;

    let (addr, calls) = spawn_fake_shard().await;
    let grid = Arc::new(RwLock::new(GridState::default()));
    grid.write().register(ServerEntry {
        server_id: "grid-srv".into(),
        listen_url: format!("http://{addr}"),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 9,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 0,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
        serves_openai: false,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
        openai_responses: Default::default(),
    });
    let app = build_router(state);

    // Request layer 3 — covered by the grid, not the static map.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layer":3}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let stored = calls.inner.lock().await.clone();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0]["layer"], 3);

    // /v1/stats hits the grid first; should reach the same fake shard.
    let stats = app
        .oneshot(
            Request::builder()
                .uri("/v1/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stats.status(), StatusCode::OK);
    let body = axum::body::to_bytes(stats.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["hidden_size"], 2560);
}

#[tokio::test]
async fn walk_ffn_grid_layer_missing_falls_back_to_static_shards() {
    use larql_router::grid::GridState;
    use parking_lot::RwLock;

    let (addr, _calls) = spawn_fake_shard().await;
    let grid = Arc::new(RwLock::new(GridState::default()));
    // No servers registered — grid lookup returns None for every layer.
    // Static map covers it.

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards(&format!("0-9=http://{addr}")).unwrap(),
        grid: Some(grid),
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
        openai_responses: Default::default(),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":5}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn walk_ffn_500s_on_shard_connection_failure() {
    let app = make_router("0-3=http://127.0.0.1:1"); // unreachable
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":0}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
}
