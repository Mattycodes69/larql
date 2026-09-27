//! ADR-0018 MoE dispatch: per-expert shard resolution, H3 and raw proxying.

#[cfg(feature = "http3")]
use crate::dispatch::HedgeOutcome;
use crate::dispatch::{hedged_post_json, merge_shard_responses};
use axum::body::Bytes;
use axum::http::{header, StatusCode};
use axum::response::Response;
use larql_router_protocol::walk_ffn as walk_ffn_wire;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[allow(unused_imports)]
use super::*;

/// ADR-0018 — MoE dispatch path. For each `(layer, [experts])` entry:
///
///   1. Resolve every `(layer, expert)` pair to its owning shard via
///      `GridState::route_all_experts`. Grid-only — MoE has no static
///      shard fallback.
///   2. Group the pairs by destination URL so each shard gets one
///      sub-request carrying every `(layer, expert)` it owns from this
///      call.
///   3. Build a JSON body per shard in the same `layer_experts` shape
///      the caller sent.
///   4. Fan out in parallel; merge responses with the existing
///      [`merge_shard_responses`] envelope.
///
/// Routing requires a live grid — MoE deployments never use static
/// `--shards`. If `state.grid` is `None` the handler 503s with a
/// helpful message.
pub(super) async fn handle_moe_dispatch(
    state: Arc<AppState>,
    model_id: Option<&str>,
    pairs: &[(usize, Vec<u32>)],
) -> Result<Response, (StatusCode, String)> {
    let Some(grid) = &state.grid else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "MoE routing requires a self-assembling grid (--grid-port); \
             this router was started in static --shards-only mode"
                .to_string(),
        ));
    };

    // Flatten the (layer, [experts]) list into individual (layer, expert)
    // pairs that route_all_experts can resolve.
    let flat: Vec<(usize, u32)> = pairs
        .iter()
        .flat_map(|(layer, experts)| experts.iter().map(move |&e| (*layer, e)))
        .collect();

    let layer_expert_urls = {
        let guard = grid.read();
        guard
            .route_all_experts(model_id, &flat)
            .map_err(|(layer, expert)| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no shard owns (layer {layer}, expert {expert}) in this router"),
                )
            })?
    };

    // Group (layer, expert) pairs by destination URL → per-shard
    // sub-request payload.
    let mut by_url: HashMap<String, HashMap<usize, Vec<u32>>> = HashMap::new();
    for ((layer, expert), url) in &layer_expert_urls {
        by_url
            .entry(url.clone())
            .or_default()
            .entry(*layer)
            .or_default()
            .push(*expert);
    }

    // ADR-0021 — derive a secondary URL per primary group when hedging
    // is enabled. Hedge only fires on the reqwest path; the h3 fan-out
    // already gets per-stream independence (ADR-0019), and a hedged h3
    // helper isn't in scope for this ADR.
    let hedge_after = state.hedge_after;
    let secondary_by_primary: HashMap<String, Option<String>> = if hedge_after.is_some() {
        let mut out = HashMap::with_capacity(by_url.len());
        let guard = grid.read();
        for (primary, layer_to_experts) in &by_url {
            // Any (layer, expert) in this group is owned by `primary`;
            // they share an owning replica set, so the first pair's
            // secondary is the secondary for the whole group.
            let (probe_layer, probe_expert) = layer_to_experts
                .iter()
                .next()
                .and_then(|(l, exs)| exs.first().map(|e| (*l as u32, *e)))
                .unwrap_or((0, 0));
            let ranked = guard.route_expert_with_rank(model_id, probe_layer, probe_expert, 2);
            let secondary = ranked.into_iter().find(|u| u != primary);
            out.insert(primary.clone(), secondary);
        }
        out
    } else {
        HashMap::new()
    };

    let mut handles = Vec::new();
    for (url, layer_to_experts) in by_url {
        let layer_experts_json: Vec<Value> = layer_to_experts
            .into_iter()
            .map(|(layer, mut experts)| {
                experts.sort_unstable();
                serde_json::json!({ "layer": layer, "experts": experts })
            })
            .collect();
        let mut sub_body = serde_json::Map::new();
        if let Some(mid) = model_id {
            sub_body.insert("model_id".into(), Value::String(mid.to_string()));
        }
        sub_body.insert("layer_experts".into(), Value::Array(layer_experts_json));
        let sub_body = Value::Object(sub_body);

        // ADR-0019 — when the operator opted into `--http3-shards`,
        // dispatch the MoE sub-request through h3 instead of
        // reqwest. h3 gives per-stream independence over QUIC, which
        // is the whole point: parallel per-token expert sub-requests
        // to the same shard stop blocking each other on TCP HoL.
        #[cfg(feature = "http3")]
        if let Some(h3) = state.h3_client.clone() {
            let h3_handle: tokio::task::JoinHandle<(Result<Value, String>, HedgeOutcome)> =
                tokio::spawn(async move {
                    let r = dispatch_via_h3(h3, url.clone(), sub_body).await;
                    (r, HedgeOutcome::default())
                });
            handles.push(h3_handle);
            continue;
        }

        let client = state.client.clone();
        let secondary = secondary_by_primary.get(&url).and_then(|s| s.clone());
        let primary = url.clone();
        handles.push(tokio::spawn(async move {
            hedged_post_json(
                &client,
                &primary,
                secondary.as_deref(),
                hedge_after,
                walk_ffn_wire::PATH,
                &sub_body,
            )
            .await
        }));
    }

    let mut responses = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok((Ok(v), outcome)) => {
                if let Some(m) = &state.metrics {
                    if outcome.fired {
                        m.route_hedge_fires_total.inc();
                    }
                    if outcome.won {
                        m.route_hedge_wins_total.inc();
                    }
                }
                responses.push(v)
            }
            Ok((Err(e), outcome)) => {
                if let Some(m) = &state.metrics {
                    if outcome.fired {
                        m.route_hedge_fires_total.inc();
                    }
                    if outcome.won {
                        m.route_hedge_wins_total.inc();
                    }
                }
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!("MoE sub-request to shard failed: {e}"),
                ));
            }
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("MoE dispatch task join: {e}"),
                ))
            }
        }
    }

    let merged = merge_shard_responses(&responses);
    let body = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("encode: {e}")))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .unwrap())
}

/// ADR-0019 — issue one MoE sub-request via the HTTP/3 transport.
///
/// Parses the shard URL (`http://host:port` or `https://host:port`)
/// into the `(SocketAddr, server_name)` pair that
/// [`larql_router_protocol::transport::h3::H3Client::post_json`]
/// expects, serializes the JSON body, and returns the parsed
/// response. Feature-gated under `http3` so the dense build never
/// pays the h3 dispatch cost.
#[cfg(feature = "http3")]
pub(super) async fn dispatch_via_h3(
    client: Arc<larql_router_protocol::transport::h3::H3Client>,
    shard_url: String,
    body: Value,
) -> Result<Value, String> {
    // `shard_url` looks like `http://10.0.0.11:8080` or `http://shard-a:8080`.
    // Strip scheme, split host:port, resolve to a SocketAddr. h3 ignores
    // the URL scheme; what matters is the UDP socket + SNI name.
    let trimmed = shard_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/');
    let (host, port_str) = trimmed
        .rsplit_once(':')
        .ok_or_else(|| format!("shard URL {shard_url:?} missing :port"))?;
    let port: u16 = port_str
        .parse()
        .map_err(|e| format!("shard URL {shard_url:?} bad port: {e}"))?;

    use std::net::ToSocketAddrs;
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {host}:{port}: {e}"))?
        .next()
        .ok_or_else(|| format!("no address for {host}:{port}"))?;

    let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("encode body: {e}"))?;
    let resp = client
        .post_json(addr, host, walk_ffn_wire::PATH, body_bytes.into())
        .await
        .map_err(|e| format!("h3 post: {e}"))?;
    if resp.status >= 400 {
        return Err(format!("shard returned HTTP {}", resp.status));
    }
    serde_json::from_slice(&resp.body).map_err(|e| format!("decode response: {e}"))
}

/// Forward raw bytes to a shard, passing the Content-Type header through.
pub(super) async fn proxy_raw(
    client: &reqwest::Client,
    base_url: &str,
    body: Bytes,
    ct: &str,
) -> Result<Response, (StatusCode, String)> {
    let url = format!("{base_url}{}", walk_ffn_wire::PATH);
    let resp = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, ct)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard {base_url}: {e}")))?;

    let status = resp.status();
    let resp_ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let resp_bytes = resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("read shard response: {e}")))?;

    Ok(Response::builder()
        .status(status.as_u16())
        .header(header::CONTENT_TYPE, resp_ct)
        .body(axum::body::Body::from(resp_bytes))
        .unwrap())
}
