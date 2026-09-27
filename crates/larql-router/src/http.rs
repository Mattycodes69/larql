//! HTTP server for the router: `AppState`, the `/v1/walk-ffn` handler, the
//! `/v1/stats` proxy, the `/v1/health` heartbeat, and the axum `Router`
//! factory. Moved out of `main.rs` so integration tests can build a Router
//! against a mock shard backend without spawning the binary.

use std::collections::HashMap;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use parking_lot::RwLock;

use crate::dispatch::resolve_static_only;
use crate::grid::GridState;
use crate::metrics::RouterMetrics;
use crate::shards::{find_shard_for_layer, Shard};

use larql_router_protocol::walk_ffn as walk_ffn_wire;
/// Content-Type used by the FFN binary protocol. JSON requests use the
/// standard `application/json`. Single-sourced in the shared walk-ffn
/// codec (ROADMAP hardening item 16); re-exported under the historical
/// `larql_router::http::BINARY_CT` path.
pub use larql_router_protocol::walk_ffn::BINARY_CT;

mod admin;
mod moe_dispatch;
mod request;
mod walk_ffn;
pub use admin::*;
use moe_dispatch::*;
pub use request::*;
pub use walk_ffn::*;

/// Shared HTTP service state. Holds the static shard map, an optional
/// grid handle, and a single reqwest client (whose connection pool is
/// reused across all outbound shard calls).
pub struct AppState {
    pub static_shards: Vec<Shard>,
    pub grid: Option<Arc<RwLock<GridState>>>,
    pub client: reqwest::Client,
    /// ADR-0017 — shared metrics registry. `None` disables
    /// observation (used by some integration tests that don't need
    /// the dependency); production paths always carry a value.
    pub metrics: Option<Arc<RouterMetrics>>,
    /// ADR-0019 — optional HTTP/3 shard transport. When `Some(...)`,
    /// the MoE expert fan-out path dispatches through h3 instead of
    /// reqwest. The dense path keeps reqwest unchanged because the
    /// HTTP/3 win (per-stream independence) only matters for
    /// parallel per-token fan-outs. Always `None` when the crate is
    /// built without the `http3` feature.
    #[cfg(feature = "http3")]
    pub h3_client: Option<Arc<larql_router_protocol::transport::h3::H3Client>>,
    /// ADR-0021 — hedged-dispatch delay. When `Some(d)`, the
    /// multi-shard fan-out picks a secondary replica per sub-request
    /// and dispatches it `d` after the primary if the primary hasn't
    /// responded yet. `None` disables hedging (pre-ADR-0021
    /// behaviour); operators opt in via `--hedge-after-ms M`.
    pub hedge_after: Option<std::time::Duration>,
    /// N0-router — sticky routes for the Responses API: which grid
    /// server produced (and stores) each proxied response id, so
    /// `previous_response_id` chains and by-id retrieval land on the
    /// same server. Bounded FIFO; see `openai::responses`.
    pub openai_responses: crate::openai::ResponseRouteStore,
}

impl AppState {
    /// Resolve every layer to its owning shard URL. Grid lookups take
    /// priority; any layer not covered by the grid falls back to the
    /// static shard map. Returns `Err(first uncovered layer)`.
    pub async fn resolve_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        if let Some(grid) = &self.grid {
            let guard = grid.read();
            let mut out = HashMap::with_capacity(layers.len());
            let mut static_needed: Vec<usize> = Vec::new();
            for &layer in layers {
                match guard.route(model_id, layer as u32) {
                    Some(url) => {
                        out.insert(layer, url);
                    }
                    None => static_needed.push(layer),
                }
            }
            drop(guard);
            for layer in static_needed {
                match find_shard_for_layer(&self.static_shards, layer) {
                    Some(s) => {
                        out.insert(layer, s.url.clone());
                    }
                    None => return Err(layer),
                }
            }
            return Ok(out);
        }
        resolve_static_only(&self.static_shards, layers)
    }
}

/// Build the axum `Router` for the public HTTP surface. Held separate
/// from the binary's `main()` so integration tests can mount it onto an
/// in-process listener.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(walk_ffn_wire::PATH, post(handle_walk_ffn))
        .route("/v1/health", get(handle_health))
        .route("/v1/stats", get(handle_stats))
        .route("/metrics", get(handle_metrics))
        // N0-router — the OpenAI surface on the grid front door; each
        // request proxies to a `serves_openai` grid server (openai.rs).
        .route("/v1/models", get(crate::openai::handle_models))
        .route(
            "/v1/chat/completions",
            post(crate::openai::handle_chat_completions),
        )
        .route("/v1/completions", post(crate::openai::handle_completions))
        .route("/v1/embeddings", post(crate::openai::handle_embeddings))
        .route(
            crate::openai::RESPONSES_PATH,
            post(crate::openai::responses::handle_responses),
        )
        .route(
            "/v1/responses/{response_id}",
            get(crate::openai::responses::handle_get_response)
                .delete(crate::openai::responses::handle_delete_response),
        )
        .with_state(state)
}

#[cfg(test)]
mod tests;
