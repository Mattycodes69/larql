//! Criterion benchmarks for the grid routing hot-path (ADR-0012).
//!
//! Measures ns/op for the operations that run on every inference request
//! going through the router.
//!
//! Run with:
//!   cargo bench -p larql-router --bench routing
//!
//! All bench IDs use server counts and layer counts, not model names.

use std::collections::HashMap;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

use larql_router::grid::{GridState, ServerEntry};
use larql_router_protocol::LayerLatency;

const SERVER_COUNTS: &[(usize, &str)] = &[(1, "1srv"), (10, "10srv"), (100, "100srv")];
const LAYER_COUNTS: &[(usize, &str)] = &[(30, "30layers"), (62, "62layers")];

fn make_entry(id: usize, layer_start: u32, layer_end: u32) -> ServerEntry {
    ServerEntry {
        server_id: format!("srv-{id}"),
        listen_url: format!("http://10.0.0.{id}:8080"),
        model_id: "bench-model".into(),
        layer_start,
        layer_end,
        vindex_hash: format!("hash-{id}"),
        cpu_pct: 0.0,
        ram_used: 4 * 1024 * 1024 * 1024,
        requests_in_flight: id as u32 % 10,
        last_seen: Instant::now(),
        layer_latencies: HashMap::new(),
        req_per_sec: 0.0,
        rtt_ms: None,
    }
}

/// Build a worst-case state: `n_servers` each owning all `n_layers`
/// layers (full replication). Useful as an upper bound but not what
/// real deployments look like.
fn build_state(n_servers: usize, n_layers: usize) -> GridState {
    let mut state = GridState::default();
    for i in 0..n_servers {
        state.register(make_entry(i, 0, (n_layers - 1) as u32));
    }
    state
}

/// Build a production-shape grid: a model with `n_layers` layers
/// partitioned into `n_shards` contiguous slices, each slice
/// replicated `n_replicas` times. Total servers = n_shards ×
/// n_replicas, replicas-per-layer = n_replicas (constant).
fn build_realistic_state(n_layers: usize, n_shards: usize, n_replicas: usize) -> GridState {
    let mut state = GridState::default();
    let layers_per_shard = n_layers / n_shards;
    for shard_idx in 0..n_shards {
        let layer_start = (shard_idx * layers_per_shard) as u32;
        let layer_end = if shard_idx == n_shards - 1 {
            (n_layers - 1) as u32
        } else {
            ((shard_idx + 1) * layers_per_shard - 1) as u32
        };
        for replica_idx in 0..n_replicas {
            let server_id = shard_idx * n_replicas + replica_idx;
            state.register(make_entry(server_id, layer_start, layer_end));
        }
    }
    state
}

// ── route() hot path ──────────────────────────────────────────────────────────

fn bench_route_single_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_single_layer");
    for &(n_servers, slabel) in SERVER_COUNTS {
        let state = build_state(n_servers, 30);
        group.bench_with_input(BenchmarkId::new(slabel, n_servers), &n_servers, |b, _| {
            b.iter(|| state.route(Some("bench-model"), 15));
        });
    }
    group.finish();
}

// ── route_all() — full forward pass routing ───────────────────────────────────

fn bench_route_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_all");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            let state = build_state(n_servers, n_layers);
            let layers: Vec<usize> = (0..n_layers).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &layers,
                |b, layers| {
                    b.iter(|| state.route_all(Some("bench-model"), layers));
                },
            );
        }
    }
    group.finish();
}

// ── update_heartbeat() — load metric update ───────────────────────────────────

fn bench_heartbeat_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/heartbeat_update");
    for &(n_servers, slabel) in SERVER_COUNTS {
        let mut state = build_state(n_servers, 30);
        let server_ids: Vec<String> = (0..n_servers).map(|i| format!("srv-{i}")).collect();
        let layer_stats: Vec<LayerLatency> = (0..30u32)
            .map(|l| LayerLatency {
                layer: l,
                avg_ms: 2.0,
                p99_ms: 5.0,
            })
            .collect();
        group.bench_with_input(
            BenchmarkId::new(slabel, n_servers),
            &server_ids,
            |b, ids| {
                b.iter(|| {
                    // Update the first server's heartbeat.
                    state.update_heartbeat(&ids[0], 50.0, 2 << 30, 5, layer_stats.clone(), 0.0);
                });
            },
        );
    }
    group.finish();
}

// ── Production-shape route() — replicas-per-layer stays at n_replicas ────────

/// Production-shape scenarios: (n_shards, n_replicas, label).
/// Total servers = n_shards × n_replicas; replicas-per-layer = n_replicas.
const REALISTIC_TOPOLOGIES: &[(usize, usize, &str)] = &[
    (2, 2, "2shards_x2"),   // 4 servers
    (5, 2, "5shards_x2"),   // 10 servers
    (10, 2, "10shards_x2"), // 20 servers
    (10, 3, "10shards_x3"), // 30 servers
    (20, 2, "20shards_x2"), // 40 servers
];

fn bench_route_single_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_realistic");
    for &(n_shards, n_replicas, label) in REALISTIC_TOPOLOGIES {
        let state = build_realistic_state(30, n_shards, n_replicas);
        group.bench_with_input(
            BenchmarkId::new(label, n_shards * n_replicas),
            &(),
            |b, _| {
                // Route the middle layer — typical request shape.
                b.iter(|| state.route(Some("bench-model"), 15));
            },
        );
    }
    group.finish();
}

fn bench_route_all_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/route_all_realistic");
    let scenarios = &[
        (30, 2, 2, "30layers_2shards_x2"),
        (30, 5, 2, "30layers_5shards_x2"),
        (30, 10, 2, "30layers_10shards_x2"),
        (62, 10, 2, "62layers_10shards_x2"),
        (62, 20, 2, "62layers_20shards_x2"),
    ];
    for &(n_layers, n_shards, n_replicas, label) in scenarios {
        let state = build_realistic_state(n_layers, n_shards, n_replicas);
        let layers: Vec<usize> = (0..n_layers).collect();
        group.bench_with_input(
            BenchmarkId::new(label, n_shards * n_replicas),
            &layers,
            |b, layers| {
                b.iter(|| state.route_all(Some("bench-model"), layers));
            },
        );
    }
    group.finish();
}

// ── Single register: cost of one rebuild_route_table call ────────────────────

/// Measure the cost of *one* server joining a grid of size N. Each
/// `register()` triggers exactly one `rebuild_route_table()` over
/// N+1 servers, so this isolates the per-rebuild cost that the
/// `register_cascade` bench below conflates across N registrations.
fn bench_single_register(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/single_register");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &(n_servers, n_layers),
                |b, &(ns, nl)| {
                    b.iter_batched(
                        || build_state(ns, nl),
                        |mut state| {
                            // One register = one rebuild over (ns + 1) servers.
                            state.register(make_entry(ns, 0, (nl - 1) as u32));
                            state
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

// ── register_cascade — building a grid from scratch (N registers) ────────────

/// Build an N-server grid from empty by calling `register()` N times.
/// This is O(N² × L) because each register triggers a full
/// `rebuild_route_table()` over the growing set. Useful as a
/// worst-case "cold start" measurement but not representative of
/// real per-join cost — for that, use [`bench_single_register`].
fn bench_register_cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing/register_cascade");
    for &(n_servers, slabel) in SERVER_COUNTS {
        for &(n_layers, llabel) in LAYER_COUNTS {
            group.bench_with_input(
                BenchmarkId::new(format!("{slabel}_{llabel}"), n_servers * n_layers),
                &(n_servers, n_layers),
                |b, &(ns, nl)| {
                    b.iter(|| {
                        let mut state = GridState::default();
                        for i in 0..ns {
                            state.register(make_entry(i, 0, (nl - 1) as u32));
                        }
                        state
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_route_single_layer,
    bench_route_all,
    bench_route_single_realistic,
    bench_route_all_realistic,
    bench_heartbeat_update,
    bench_single_register,
    bench_register_cascade,
);
criterion_main!(benches);
