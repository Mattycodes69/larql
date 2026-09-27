//! Unauthenticated admin endpoints: metrics, health and stats.

use crate::dispatch::unique_candidate_urls;
use crate::metrics::encode_metrics_text;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::Json;
use serde_json::Value;
use std::sync::Arc;

#[allow(unused_imports)]
use super::*;

/// ADR-0017 — Prometheus text-format `/metrics` endpoint. Unauth,
/// same model as `/v1/health`. Returns 503 with a short body when
/// the router was built without a metrics registry (test harness).
pub async fn handle_metrics(State(state): State<Arc<AppState>>) -> Response {
    let Some(metrics) = &state.metrics else {
        return Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body("metrics registry not installed".to_string().into())
            .unwrap();
    };
    match encode_metrics_text(metrics) {
        Ok(text) => Response::builder()
            .status(StatusCode::OK)
            .header(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )
            .body(text.into())
            .unwrap(),
        Err(e) => Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
            .body(format!("metrics encode failed: {e}").into())
            .unwrap(),
    }
}

pub async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Proxy `/v1/stats` to the first reachable shard so that clients
/// connecting via `RemoteWalkBackend` (which reads `hidden_size` from
/// `/v1/stats`) work transparently through the router.
pub async fn handle_stats(State(state): State<Arc<AppState>>) -> Response {
    let grid_urls = if let Some(grid) = &state.grid {
        grid.read().all_shard_urls()
    } else {
        Vec::new()
    };
    let candidates = unique_candidate_urls(grid_urls, &state.static_shards);
    for url in candidates {
        let stats_url = format!("{url}/v1/stats");
        if let Ok(resp) = state.client.get(&stats_url).send().await {
            if resp.status().is_success() {
                if let Ok(bytes) = resp.bytes().await {
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(bytes))
                        .unwrap();
                }
            }
        }
    }
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(r#"{"error":"no shard reachable"}"#))
        .unwrap()
}
