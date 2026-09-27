//! `POST /v1/walk-ffn`: dense fan-out across layer shards.

use crate::dispatch::{
    build_subrequest_body, group_layers_by_url, hedged_post_json, merge_shard_responses,
    HedgeOutcome,
};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use larql_router_protocol::walk_ffn as walk_ffn_wire;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[allow(unused_imports)]
use super::*;

/// `POST /v1/walk-ffn` entry point. Errors are normalised to JSON
/// regardless of the request content-type so clients always see the same
/// envelope.
pub async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    // ADR-0017 — observe duration + status. The timer starts before
    // the inner handler runs and stops on every exit path, including
    // early errors.
    let timer = state.metrics.as_ref().map(|m| {
        m.walk_ffn_duration_seconds
            .with_label_values(&[])
            .start_timer()
    });
    let result = handle_walk_ffn_inner(state.clone(), request).await;
    if let Some(t) = timer {
        t.observe_duration();
    }
    if let Some(m) = &state.metrics {
        let label = match &result {
            Ok(_) => "success",
            Err((status, _)) if status.is_client_error() => "error_4xx",
            Err(_) => "error_5xx",
        };
        m.walk_ffn_requests_total.with_label_values(&[label]).inc();
    }
    match result {
        Ok(r) => r,
        Err((status, msg)) => {
            let body = format!(r#"{{"error":{}}}"#, serde_json::Value::String(msg));
            let mut builder = Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json");
            // ADR-0020 — clients can use Retry-After to back off from
            // a saturated router rather than hammering it. 0.5s
            // matches the doc default; any 503 emitted by this
            // handler is currently saturation-driven.
            if status == StatusCode::SERVICE_UNAVAILABLE {
                builder = builder.header(header::RETRY_AFTER, "0.5");
            }
            builder.body(axum::body::Body::from(body)).unwrap()
        }
    }
}

pub(super) async fn handle_walk_ffn_inner(
    state: Arc<AppState>,
    request: axum::extract::Request,
) -> Result<Response, (StatusCode, String)> {
    let is_binary = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(is_binary_content_type)
        .unwrap_or(false);

    let body_bytes = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("read body: {e}")))?;

    let (spec, model_id_owned) = extract_request_spec_and_model_id(&body_bytes, is_binary)
        .map_err(|m| (StatusCode::BAD_REQUEST, m))?;

    if spec.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty layer list".to_string()));
    }

    // ADR-0018 — MoE dispatch branches off here. Dense path continues
    // through the rest of the function unchanged.
    if let RequestSpec::Moe(pairs) = &spec {
        return handle_moe_dispatch(state, model_id_owned.as_deref(), pairs).await;
    }
    let layers: Vec<usize> = match spec {
        RequestSpec::Dense(l) => l,
        RequestSpec::Moe(_) => unreachable!("handled above"),
    };

    let mid = model_id_owned.as_deref();
    let layer_urls = match state.resolve_all(mid, &layers).await {
        Ok(map) => map,
        Err(missing) => {
            // ADR-0020 — distinguish "no shard owns this layer"
            // (400) from "shards own it but all are saturated"
            // (503). Saturation increments a counter so operators
            // can see the load-shedding signal.
            let saturated = match &state.grid {
                Some(grid) => grid.read().has_owners_for(mid, missing as u32),
                None => false,
            };
            if saturated {
                if let Some(m) = &state.metrics {
                    m.route_saturation_total.inc();
                }
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "layer {missing}: every replica is at or above the configured \
                         saturation ceiling — retry shortly"
                    ),
                ));
            }
            return Err((
                StatusCode::BAD_REQUEST,
                format!("layer {missing} has no owning shard in this router"),
            ));
        }
    };

    let unique_urls: std::collections::HashSet<&String> = layer_urls.values().collect();

    if unique_urls.len() == 1 || layers.len() == 1 {
        // All layers on the same shard — proxy raw bytes unchanged.
        let url = layer_urls.values().next().unwrap();
        let ct = if is_binary {
            BINARY_CT
        } else {
            "application/json"
        };
        return proxy_raw(&state.client, url, body_bytes, ct).await;
    }

    // Multi-shard dispatch.
    if is_binary {
        return Err((
            StatusCode::BAD_REQUEST,
            "binary fan-out across multiple shards is not supported; use JSON or split by shard"
                .to_string(),
        ));
    }

    let body_value: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let by_url = group_layers_by_url(&layer_urls);

    // ADR-0021 — derive a secondary URL per primary group, if hedging
    // is enabled AND the grid actually offers a second replica. Static-
    // shard fall-back groups (no grid replica) get None and dispatch
    // through the non-hedged path.
    let hedge_after = state.hedge_after;
    let secondary_by_primary: HashMap<String, Option<String>> = if hedge_after.is_some() {
        let mut out = HashMap::with_capacity(by_url.len());
        if let Some(grid) = &state.grid {
            let guard = grid.read();
            for (primary, shard_layers) in &by_url {
                // Any layer in the group resolves the same replica set
                // (groups share a primary URL → same owning shard range).
                let probe_layer = *shard_layers.first().unwrap_or(&0) as u32;
                let ranked = guard.route_with_rank(mid, probe_layer, 2);
                // Pick the first ranked URL that isn't the primary —
                // route_with_rank's ordering can change between the
                // resolve_all snapshot and this read if a heartbeat
                // landed in between.
                let secondary = ranked.into_iter().find(|u| u != primary);
                out.insert(primary.clone(), secondary);
            }
        }
        out
    } else {
        HashMap::new()
    };

    let mut handles = Vec::new();
    for (url, shard_layers) in &by_url {
        let sub_body = build_subrequest_body(&body_value, shard_layers);
        let client = state.client.clone();
        let primary = url.clone();
        let secondary = secondary_by_primary.get(url).and_then(|s| s.clone());
        handles.push(tokio::spawn(async move {
            let (result, outcome) = hedged_post_json(
                &client,
                &primary,
                secondary.as_deref(),
                hedge_after,
                walk_ffn_wire::PATH,
                &sub_body,
            )
            .await;
            (result, outcome)
        }));
    }

    let joined: Vec<(Result<Value, String>, HedgeOutcome)> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|jh| jh.unwrap_or_else(|e| (Err(e.to_string()), HedgeOutcome::default())))
        .collect();

    // Surface hedge outcomes to metrics before the early-return on
    // shard-error so even a failed hedge still increments the counter.
    if let Some(m) = &state.metrics {
        for (_, outcome) in &joined {
            if outcome.fired {
                m.route_hedge_fires_total.inc();
            }
            if outcome.won {
                m.route_hedge_wins_total.inc();
            }
        }
    }

    let responses: Vec<Value> = joined
        .into_iter()
        .map(|(result, _)| result)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard error: {e}")))?;

    let merged = merge_shard_responses(&responses);
    let json_bytes = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}
