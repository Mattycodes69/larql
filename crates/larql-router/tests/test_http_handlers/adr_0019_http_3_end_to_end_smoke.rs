//! ADR-0019: HTTP/3 end-to-end smoke

use super::*;

#[cfg(feature = "http3")]
#[tokio::test]
async fn moe_fanout_dispatches_through_h3_client_when_configured() {
    use larql_router::grid::{GridState, ServerEntry};
    use larql_router_protocol::transport::h3::{serve_axum, server_endpoint, H3Client};
    use larql_router_protocol::transport::quic::self_signed_tls;
    use tokio::sync::Mutex;

    let _ = rustls::crypto::ring::default_provider().install_default();

    // ── Stand up a single h3 echo server that records every body it sees.
    let recorded: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded_handler = recorded.clone();
    let h3_app = axum::Router::new().route(
        "/v1/walk-ffn",
        axum::routing::post(move |body: axum::extract::Json<Value>| {
            let recorded = recorded_handler.clone();
            async move {
                recorded.lock().await.push(body.0.clone());
                axum::Json(json!({
                    "results": [{"layer": 0, "expert": 0, "out": "ok"}],
                    "latency_ms": 1.0
                }))
            }
        }),
    );
    let tls = self_signed_tls("h3-shard").expect("self_signed_tls");
    let endpoint = server_endpoint("127.0.0.1:0".parse().unwrap(), &tls).expect("server_endpoint");
    let h3_addr = endpoint.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let _ = serve_axum(endpoint, h3_app).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ── Grid: one MoE shard owning experts 0-7 of layer 0, listen URL
    //    pointing at the h3 listener. The router's dispatch path will
    //    parse `host:port` out of this and call H3Client::post_json.
    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    {
        let mut g = grid.write();
        g.register(ServerEntry {
            server_id: "moe-h3".into(),
            listen_url: format!("http://127.0.0.1:{}", h3_addr.port()),
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
            expert_end: 7,
            serves_openai: false,
        });
    }

    // ── Router AppState with the matching H3Client.
    let shards = parse_shards("99-100=http://unused:1").unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let h3_client =
        Arc::new(H3Client::new("127.0.0.1:0".parse().unwrap(), None).expect("h3 client"));
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: Some(grid),
        client,
        metrics: None,
        h3_client: Some(h3_client),
        hedge_after: None,
        openai_responses: Default::default(),
    });
    let app = build_router(state);

    // ── Issue a MoE request — four picked experts all on the same shard.
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
        "router→shard h3 dispatch must return OK"
    );

    let recv = recorded.lock().await;
    assert_eq!(recv.len(), 1, "h3 shard saw exactly one sub-request");
    let layer_experts = recv[0]["layer_experts"].as_array().expect("layer_experts");
    assert_eq!(layer_experts.len(), 1);
    // The router groups all four picked experts onto the single shard
    // that owns 0-7, so the sub-request payload lists all of them.
    let experts = layer_experts[0]["experts"].as_array().expect("experts");
    let ids: Vec<u64> = experts.iter().filter_map(|v| v.as_u64()).collect();
    assert_eq!(ids, vec![0, 3, 5, 7]);
}

/// ADR-0020 — when every replica that owns the requested layer is at
/// or above the configured saturation ceiling, the router must emit
/// `503 Service Unavailable` (not `400 Bad Request`), set the
/// `Retry-After` hint, and bump the `route_saturation_total` counter.
#[tokio::test]
async fn walk_ffn_returns_503_with_retry_after_when_replicas_saturated() {
    use larql_router::grid::{GridState, ServerEntry};
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};
    use parking_lot::RwLock;
    use std::collections::HashMap;

    let grid = Arc::new(RwLock::new(GridState::default()));
    // One owner, requests_in_flight already at the ceiling.
    grid.write().register(ServerEntry {
        server_id: "saturated".into(),
        listen_url: "http://unreachable:9".into(),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 9,
        vindex_hash: "h".into(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 8,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
        serves_openai: false,
    });
    grid.write().set_saturation_ceiling(Some(8));

    let metrics = RouterMetrics::new();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: parse_shards("99-100=http://unused:1").unwrap(),
        grid: Some(grid),
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
                .body(Body::from(r#"{"model_id":"m","layer":3}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers()
            .get(header::RETRY_AFTER)
            .map(|v| v.to_str().unwrap()),
        Some("0.5"),
        "Retry-After hint must be set on saturation 503s"
    );
    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        v["error"].as_str().unwrap().contains("saturation ceiling"),
        "error body should explain saturation; got {}",
        v["error"]
    );

    let text = encode_metrics_text(&metrics).unwrap();
    assert!(
        text.lines()
            .any(|l| l.starts_with("larql_router_route_saturation_total ") && l.ends_with(" 1")),
        "route_saturation_total must increment exactly once; got:\n{text}"
    );
}
