//! ADR-0018: MoE expert routing dispatch

use super::*;

/// MoE request with no grid configured 503s — there's no static-shard
/// fallback path for expert routing.
#[tokio::test]
async fn moe_request_without_grid_returns_503() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"layer":0,"experts":[0,3]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]
        .as_str()
        .unwrap()
        .contains("MoE routing requires"));
}

/// MoE request against a grid with no shard owning the requested
/// `(layer, expert)` returns 503.
#[tokio::test]
async fn moe_request_with_no_owner_returns_503() {
    use larql_router::grid::{GridState, ServerEntry};
    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    {
        let mut g = grid.write();
        g.register(ServerEntry {
            server_id: "moe-a".into(),
            listen_url: "http://moe-a".into(),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 3,
            serves_openai: false,
        });
    }
    let shards = parse_shards("99-100=http://unused:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: shards,
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
                .body(Body::from(r#"{"layer":0,"experts":[99],"model_id":"m"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]
        .as_str()
        .unwrap()
        .contains("(layer 0, expert 99)"));
}

/// MoE dispatch with two expert shards and an `experts` request —
/// fans out to both shards, merges the responses.
#[tokio::test]
async fn moe_request_fans_out_to_owning_shards_and_merges() {
    use larql_router::grid::{GridState, ServerEntry};

    let (addr_lo, _calls_lo) = spawn_fake_shard().await;
    let (addr_hi, _calls_hi) = spawn_fake_shard().await;

    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    {
        let mut g = grid.write();
        g.register(ServerEntry {
            server_id: "moe-lo".into(),
            listen_url: format!("http://{addr_lo}"),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 0,
            expert_end: 3,
            serves_openai: false,
        });
        g.register(ServerEntry {
            server_id: "moe-hi".into(),
            listen_url: format!("http://{addr_hi}"),
            model_id: "m".into(),
            layer_start: 0,
            layer_end: 0,
            vindex_hash: "h".into(),
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: 0,
            last_seen: std::time::Instant::now(),
            layer_latencies: std::collections::HashMap::new(),
            req_per_sec: 0.0,
            rtt_ms: None,
            expert_start: 4,
            expert_end: 7,
            serves_openai: false,
        });
    }
    let shards = parse_shards("99-100=http://unused:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: shards,
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
                .body(Body::from(
                    r#"{"layer":0,"experts":[0,3,5,7],"model_id":"m"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "MoE fan-out should merge two shard responses"
    );
    let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["results"].is_array(), "merged envelope must have results");
}

/// A walk-ffn call that 502s should increment the `error_5xx` counter
/// on the registry. Proves the instrumentation hook in the handler
/// fires on the error path.
#[tokio::test]
async fn walk_ffn_5xx_increments_error_counter() {
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};

    let shards = parse_shards("0-3=http://127.0.0.1:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let metrics = RouterMetrics::new();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: None,
        client,
        metrics: Some(metrics.clone()),
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
                .body(Body::from(r#"{"layer":0}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

    let text = encode_metrics_text(&metrics).unwrap();
    assert!(
        text.contains("larql_router_walk_ffn_requests_total{status=\"error_5xx\"} 1"),
        "expected error_5xx=1, got:\n{text}"
    );
}
