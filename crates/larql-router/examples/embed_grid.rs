//! Programmatic GridState wiring.
//!
//! Demonstrates the router as a library: build a GridState by hand,
//! register a couple of serving servers + a Mode B spare, query
//! routes, inspect coverage gaps, and exercise the rebalancer's
//! state-shaping methods directly. No gRPC server, no HTTP listener,
//! no tokio loop running — just the in-memory data structures.
//!
//! Run with `cargo run -p larql-router --example embed_grid`.

use std::collections::HashMap;
use std::time::Instant;

use larql_router::grid::{GridState, ServerEntry};

fn server(
    server_id: &str,
    listen_url: &str,
    model_id: &str,
    layer_start: u32,
    layer_end: u32,
) -> ServerEntry {
    ServerEntry {
        server_id: server_id.into(),
        listen_url: listen_url.into(),
        model_id: model_id.into(),
        layer_start,
        layer_end,
        vindex_hash: format!("hash-{server_id}"),
        cpu_pct: 0.0,
        ram_used: 0,
        requests_in_flight: 0,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
    }
}

fn main() {
    let mut grid = GridState::default();

    // Register two serving shards of gemma-3-4b: layers 0-14 on shard-a,
    // layers 15-29 on shard-b. (32-layer model — 30-31 left uncovered to
    // showcase coverage_gaps.)
    grid.register(server("a", "http://shard-a:9181", "gemma3:4b", 0, 14));
    grid.register(server("b", "http://shard-b:9182", "gemma3:4b", 15, 29));

    // Routing a covered layer returns the owning shard's URL.
    println!("== Routing ==");
    let route_5 = grid.route(Some("gemma3:4b"), 5).unwrap();
    println!("  layer  5 -> {route_5}");
    let route_20 = grid.route(Some("gemma3:4b"), 20).unwrap();
    println!("  layer 20 -> {route_20}");
    println!(
        "  layer 30 -> {:?}  (no shard)",
        grid.route(Some("gemma3:4b"), 30)
    );

    // route_all batches the lookup; returns Err(first uncovered layer).
    let plan = grid.route_all(Some("gemma3:4b"), &[0, 5, 14, 15, 29]);
    println!("\n== Batched route_all ==");
    println!("  contiguous coverage: {plan:?}");
    let partial = grid.route_all(Some("gemma3:4b"), &[0, 5, 30, 31]);
    println!("  partial coverage:    {partial:?}");

    // Hot-shard elevation. Once a range crosses the configured per-shard
    // req/sec threshold the rebalancer raises its effective target by 1
    // so the under-replication tick will pull a spare.
    grid.set_target_replicas(1);
    grid.mark_elevated("gemma3:4b", 0, 14);
    println!("\n== Hot-shard elevation ==");
    println!(
        "  effective_target_for(0-14) = {}",
        grid.effective_target_for("gemma3:4b", 0, 14)
    );
    println!(
        "  under_replicated_ranges    = {:?}",
        grid.under_replicated_ranges()
    );
    grid.demote_elevated("gemma3:4b", 0, 14);

    // Coverage gaps + over/under-replication ledger.
    println!("\n== Coverage + replication ==");
    println!("  coverage_gaps              = {:?}", grid.coverage_gaps());
    println!(
        "  under_replicated_ranges    = {:?}",
        grid.under_replicated_ranges()
    );
    println!(
        "  over_replicated_ranges     = {:?}",
        grid.over_replicated_ranges()
    );

    // status_response is what the gRPC `status` RPC returns; the
    // `larql-router status` CLI sub-command formats it for humans.
    let snap = grid.status_response();
    println!("\n== status_response ==");
    println!("  servers reported: {}", snap.servers.len());
    println!(
        "  shards in model:  {}",
        snap.models
            .iter()
            .find(|m| m.model_id == "gemma3:4b")
            .map(|m| m.shards.len())
            .unwrap_or(0)
    );
}
