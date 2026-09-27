//! End-to-end tests for the router's HTTP surface (`/v1/walk-ffn`,
//! `/v1/stats`, `/v1/health`).
//!
//! Each test stands up a loopback "fake shard" that echoes the request
//! back as JSON, points the router at it via `--shards`, then drives the
//! router via real HTTP requests. This exercises `handle_walk_ffn`,
//! `handle_walk_ffn_inner`, `proxy_raw`, `handle_stats`, and `handle_health`
//! end-to-end.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Request, StatusCode};
use axum::routing::{get, post};
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tower::ServiceExt; // for `oneshot`

use larql_router::http::{build_router, AppState, BINARY_CT};
use larql_router::shards::parse_shards;

#[derive(Clone, Default)]
struct ShardCalls {
    inner: Arc<Mutex<Vec<Value>>>,
}

async fn fake_walk_ffn(
    State(calls): State<ShardCalls>,
    body: axum::extract::Json<Value>,
) -> Json<Value> {
    calls.inner.lock().await.push(body.0.clone());

    // Echo back the layer(s) plus a fake latency so the router's merge
    // path has a concrete max latency to surface.
    let body = &body.0;
    let layer = body.get("layer").and_then(|v| v.as_u64()).unwrap_or(0);
    let layers = body
        .get("layers")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let results: Vec<Value> = if layers.is_empty() {
        vec![json!({"layer": layer, "value": "ok"})]
    } else {
        layers
            .iter()
            .map(|l| json!({"layer": l.as_u64().unwrap_or(0), "value": "ok"}))
            .collect()
    };
    Json(json!({
        "results": results,
        "latency_ms": 5.5,
    }))
}

async fn fake_walk_ffn_binary(
    State(_calls): State<ShardCalls>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    // For binary requests we mirror the body back so the router's
    // proxy_raw path can be inspected.
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, BINARY_CT)
        .body(Body::from(body))
        .unwrap()
}

async fn fake_stats() -> Json<Value> {
    Json(json!({"hidden_size": 2560, "num_layers": 34}))
}

async fn spawn_fake_shard() -> (SocketAddr, ShardCalls) {
    let calls = ShardCalls::default();
    let app_calls = calls.clone();
    let app = axum::Router::new()
        .route(
            "/v1/walk-ffn",
            post(
                |st: State<ShardCalls>, req: axum::extract::Request| async move {
                    let is_binary = req
                        .headers()
                        .get(header::CONTENT_TYPE)
                        .and_then(|v| v.to_str().ok())
                        .map(|ct| ct.starts_with(BINARY_CT))
                        .unwrap_or(false);
                    if is_binary {
                        let body = axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024)
                            .await
                            .unwrap();
                        fake_walk_ffn_binary(st, body).await
                    } else {
                        let body = axum::body::to_bytes(req.into_body(), 64 * 1024 * 1024)
                            .await
                            .unwrap();
                        let json: Value = serde_json::from_slice(&body).unwrap();
                        fake_walk_ffn(st, axum::extract::Json(json))
                            .await
                            .into_response()
                    }
                },
            ),
        )
        .route("/v1/stats", get(fake_stats))
        .with_state(app_calls);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, calls)
}

use axum::response::IntoResponse;

fn make_router(static_shards: &str) -> axum::Router {
    let shards = parse_shards(static_shards).unwrap();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let state = Arc::new(AppState {
        static_shards: shards,
        grid: None,
        client,
        metrics: None,
        #[cfg(feature = "http3")]
        h3_client: None,
        hedge_after: None,
        openai_responses: Default::default(),
    });
    build_router(state)
}

//
// Phase 4c: prove the full router→server h3 wire works.
// 1. Spin up an h3 axum listener that records the MoE sub-request body.
// 2. Configure AppState with the matching H3Client (no fingerprint pin —
//    LAN/dev mode).
// 3. Issue a MoE `experts` request to the router and assert the h3 server
//    received the rewritten `layer_experts` payload.

/// Like `spawn_fake_shard`, but the `/v1/walk-ffn` handler waits
/// `delay` before responding. Used to construct a deliberately-slow
/// primary that the hedge should beat.
async fn spawn_slow_shard(delay: Duration) -> (SocketAddr, ShardCalls) {
    let calls = ShardCalls::default();
    let app_calls = calls.clone();
    let app = axum::Router::new()
        .route(
            "/v1/walk-ffn",
            post(
                move |st: State<ShardCalls>, body: axum::extract::Json<Value>| async move {
                    tokio::time::sleep(delay).await;
                    fake_walk_ffn(st, body).await
                },
            ),
        )
        .with_state(app_calls);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (addr, calls)
}

/// Build a 2-replica grid where layer 0 has a slow primary + fast
/// secondary, layer 1 has a single fast owner. Multi-layer requests
/// force the fan-out path; the slow primary triggers the hedge.
async fn build_hedge_topology(
    slow_delay: Duration,
) -> (
    SocketAddr,
    ShardCalls,
    SocketAddr,
    ShardCalls,
    SocketAddr,
    ShardCalls,
    Arc<parking_lot::RwLock<larql_router::grid::GridState>>,
) {
    use larql_router::grid::{GridState, ServerEntry};
    use std::collections::HashMap;

    let (slow_addr, slow_calls) = spawn_slow_shard(slow_delay).await;
    let (fast_addr, fast_calls) = spawn_fake_shard().await;
    let (b_addr, b_calls) = spawn_fake_shard().await;

    let grid = Arc::new(parking_lot::RwLock::new(GridState::default()));
    // Layer 0 owned by slow_a (primary, lower in_flight = priority) +
    // fast_a (secondary). Heartbeats set in_flight so the comparator
    // picks slow_a first deterministically.
    grid.write().register(ServerEntry {
        server_id: "slow-a".into(),
        listen_url: format!("http://{slow_addr}"),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 0,
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
    grid.write().register(ServerEntry {
        server_id: "fast-a".into(),
        listen_url: format!("http://{fast_addr}"),
        model_id: "m".into(),
        layer_start: 0,
        layer_end: 0,
        vindex_hash: "h".into(),
        shard_sha256: String::new(),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 5,
        last_seen: std::time::Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
        expert_start: 0,
        expert_end: 0,
        serves_openai: false,
    });
    // Layer 1 owned by a single fast shard.
    grid.write().register(ServerEntry {
        server_id: "b".into(),
        listen_url: format!("http://{b_addr}"),
        model_id: "m".into(),
        layer_start: 1,
        layer_end: 1,
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
    (
        slow_addr, slow_calls, fast_addr, fast_calls, b_addr, b_calls, grid,
    )
}

mod admin_stats_fallthrough;
mod adr_0017_metrics_endpoint;
mod adr_0018_moe_expert_routing_dispatch;
mod adr_0019_http_3_end_to_end_smoke;
mod adr_0021_hedged_dispatch;
mod moe_hedge_and_failure;
mod walk_ffn_handlers;
