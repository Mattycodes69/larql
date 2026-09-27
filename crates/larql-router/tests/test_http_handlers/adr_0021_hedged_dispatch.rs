//! ADR-0021 hedged dispatch

use super::*;

#[tokio::test]
async fn walk_ffn_hedge_fires_when_primary_is_slow() {
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};

    // Primary delays 500ms; hedge fires after 30ms; secondary responds
    // ~immediately. The hedge must fire (counter +1) AND win (counter +1).
    let (_slow_addr, _slow_calls, _fast_addr, fast_calls, _b_addr, _b_calls, grid) =
        build_hedge_topology(Duration::from_millis(500)).await;

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
        hedge_after: Some(Duration::from_millis(30)),
        openai_responses: Default::default(),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layers":[0,1]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = encode_metrics_text(&metrics).unwrap();
    let line = |needle: &str| {
        text.lines()
            .find(|l| l.starts_with(needle))
            .map(String::from)
    };
    let fires = line("larql_router_route_hedge_fires_total ").unwrap();
    let wins = line("larql_router_route_hedge_wins_total ").unwrap();
    assert!(
        fires.ends_with(" 1"),
        "hedge_fires_total expected 1, got line: {fires}"
    );
    assert!(
        wins.ends_with(" 1"),
        "hedge_wins_total expected 1, got line: {wins}"
    );

    // The fast secondary received the layer-0 sub-request.
    let fast_seen = fast_calls.inner.lock().await.clone();
    assert!(
        !fast_seen.is_empty(),
        "secondary (fast_a) must have served the hedged sub-request"
    );
}

#[tokio::test]
async fn walk_ffn_hedge_does_not_fire_on_fast_primary() {
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};

    // Primary delay 0 — no hedge should fire. hedge_after stays set,
    // so any spurious firing would surface as a non-zero counter.
    let (_slow_addr, _slow_calls, _fast_addr, fast_calls, _b_addr, _b_calls, grid) =
        build_hedge_topology(Duration::from_millis(0)).await;

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
        hedge_after: Some(Duration::from_millis(500)),
        openai_responses: Default::default(),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layers":[0,1]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = encode_metrics_text(&metrics).unwrap();
    let line = |needle: &str| {
        text.lines()
            .find(|l| l.starts_with(needle))
            .map(String::from)
    };
    let fires = line("larql_router_route_hedge_fires_total ").unwrap();
    let wins = line("larql_router_route_hedge_wins_total ").unwrap();
    assert!(
        fires.ends_with(" 0"),
        "hedge_fires_total expected 0 when primary is fast, got: {fires}"
    );
    assert!(
        wins.ends_with(" 0"),
        "hedge_wins_total expected 0, got: {wins}"
    );
    // Secondary should NOT have received anything.
    let fast_seen = fast_calls.inner.lock().await.clone();
    assert!(
        fast_seen.is_empty(),
        "secondary should not be touched when primary wins, but saw {fast_seen:?}"
    );
}

#[tokio::test]
async fn walk_ffn_no_hedge_when_only_one_replica() {
    use larql_router::grid::{GridState, ServerEntry};
    use larql_router::metrics::{encode_metrics_text, RouterMetrics};
    use std::collections::HashMap;

    // Topology has no secondary for any layer; even with hedging
    // configured, the hedge path is skipped because route_with_rank
    // returns only one URL.
    let (a_addr, _a_calls) = spawn_fake_shard().await;
    let (b_addr, _b_calls) = spawn_fake_shard().await;
    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    for (id, addr, ls, le) in [("a", a_addr, 0u32, 0u32), ("b", b_addr, 1u32, 1u32)] {
        grid.write().register(ServerEntry {
            server_id: id.into(),
            listen_url: format!("http://{addr}"),
            model_id: "m".into(),
            layer_start: ls,
            layer_end: le,
            vindex_hash: "h".into(),
            shard_sha256: String::new(),
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
    }

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
        hedge_after: Some(Duration::from_millis(20)),
        openai_responses: Default::default(),
    });
    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/walk-ffn")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"model_id":"m","layers":[0,1]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let text = encode_metrics_text(&metrics).unwrap();
    let fires_line = text
        .lines()
        .find(|l| l.starts_with("larql_router_route_hedge_fires_total "))
        .unwrap();
    assert!(
        fires_line.ends_with(" 0"),
        "no secondary exists — hedge must never fire; got: {fires_line}"
    );
}
