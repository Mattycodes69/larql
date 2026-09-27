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

const PLAIN_TEXT: &str = "text/plain; charset=utf-8";
/// Prometheus text exposition format, version 0.0.4.
const PROMETHEUS_TEXT: &str = "text/plain; version=0.0.4; charset=utf-8";
const JSON: &str = "application/json";

/// A fixed-status response with a body and content type.
fn respond(status: StatusCode, content_type: &str, body: impl Into<axum::body::Body>) -> Response {
    let mut response = Response::new(body.into());
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_str(content_type).expect("content types are static ASCII"),
    );
    response
}

/// ADR-0017 — Prometheus text-format `/metrics` endpoint. Unauth,
/// same model as `/v1/health`. Returns 503 with a short body when
/// the router was built without a metrics registry (test harness).
pub async fn handle_metrics(State(state): State<Arc<AppState>>) -> Response {
    let Some(metrics) = &state.metrics else {
        let body = "metrics registry not installed".to_string();
        return respond(StatusCode::SERVICE_UNAVAILABLE, PLAIN_TEXT, body);
    };
    metrics_response(encode_metrics_text(metrics))
}

/// Render an encoded registry as the `/metrics` response: Prometheus text
/// on success, a 500 naming the encoder failure otherwise.
pub(super) fn metrics_response(encoded: Result<String, prometheus::Error>) -> Response {
    match encoded {
        Ok(text) => respond(StatusCode::OK, PROMETHEUS_TEXT, text),
        Err(e) => {
            let body = format!("metrics encode failed: {e}");
            respond(StatusCode::INTERNAL_SERVER_ERROR, PLAIN_TEXT, body)
        }
    }
}

pub async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Proxy `/v1/stats` to the first reachable shard so that clients
/// connecting via `RemoteWalkBackend` (which reads `hidden_size` from
/// `/v1/stats`) work transparently through the router. A shard that
/// errors or answers non-2xx is skipped in favour of the next.
pub async fn handle_stats(State(state): State<Arc<AppState>>) -> Response {
    let grid_urls = if let Some(grid) = &state.grid {
        grid.read().all_shard_urls()
    } else {
        Vec::new()
    };
    let candidates = unique_candidate_urls(grid_urls, &state.static_shards);
    for url in candidates {
        let stats_url = format!("{url}/v1/stats");
        let Ok(resp) = state.client.get(&stats_url).send().await else {
            continue;
        };
        if !resp.status().is_success() {
            continue;
        }
        if let Ok(bytes) = resp.bytes().await {
            return respond(StatusCode::OK, JSON, bytes);
        }
    }
    respond(
        StatusCode::SERVICE_UNAVAILABLE,
        JSON,
        r#"{"error":"no shard reachable"}"#,
    )
}
