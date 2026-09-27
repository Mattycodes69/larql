//! ADR-0017: /metrics endpoint

use super::*;

/// `/metrics` returns Prometheus text format when a registry is wired
/// in. Every documented metric family appears in the output with a
/// pre-touched zero value (so dashboards don't see "missing metric"
/// for a freshly-started router).
#[tokio::test]
async fn metrics_endpoint_serves_prometheus_text_with_zero_values() {
    use larql_router::metrics::RouterMetrics;

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
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/plain"),
        "/metrics must serve text/plain, got {ct}"
    );
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    for required in [
        "larql_router_build_info",
        "larql_router_grid_servers",
        "larql_router_grid_models",
        "larql_router_grid_coverage_gaps",
        "larql_router_grid_elevated_ranges",
        "larql_router_target_replicas",
        "larql_router_grid_registers_total",
        "larql_router_grid_deregisters_total",
        "larql_router_rebalancer_actions_total",
        "larql_router_rtt_probes_total",
        "larql_router_walk_ffn_requests_total",
        "larql_router_walk_ffn_duration_seconds",
    ] {
        assert!(
            text.contains(required),
            "/metrics output missing {required}; got:\n{text}"
        );
    }
}

/// `/metrics` returns 503 when the AppState lacks a registry —
/// integration tests sometimes build a router without one, and the
/// handler shouldn't panic.
#[tokio::test]
async fn metrics_endpoint_returns_503_when_no_registry() {
    let app = make_router("0-3=http://127.0.0.1:1");
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}
