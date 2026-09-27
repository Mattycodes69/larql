//! `larql dev <subcmd>` — the research / interpretability subcommand group.
//!
//! Compiled only with the `research` cargo feature. Legacy top-level
//! names (`larql walk`, …) reach these through the argv trampoline in
//! `crate::trampoline`.

use clap::Subcommand;

use super::ov_rd;
use crate::commands::extraction::*;

// ══════════════════════════════════════════════════════════════════════
// Research subcommand group — `larql dev <subcmd>`.
//
// Everything in here is unchanged from the pre-redesign top-level surface
// except its invocation path. A small argv trampoline in `main()` rewrites
// `larql <legacy-name>` → `larql dev <legacy-name>` so existing scripts
// continue to work without a breaking change.
// ══════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
pub(crate) enum DevCommand {
    /// Extract edges from FFN weights. Zero forward passes.
    WeightExtract(weight_walk_cmd::WeightWalkArgs),

    /// Extract routing edges from attention OV circuits. Zero forward passes.
    AttentionExtract(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),

    /// Capture residual stream vectors for entities via forward passes.
    Residuals(residuals_cmd::ResidualsArgs),

    /// Run full forward pass and predict next token.
    Predict(predict_cmd::PredictArgs),

    /// Build gate index for graph-based FFN (offline, run once per model).
    IndexGates(index_gates_cmd::IndexGatesArgs),

    /// Walk the model as a local vector index — gate KNN + down token lookup.
    Walk(walk_cmd::WalkArgs),

    /// Capture and compare attention patterns across prompts.
    AttentionCapture(attention_capture_cmd::AttentionCaptureArgs),

    /// Extract attention template circuits from QK weight decomposition.
    QkTemplates(qk_templates_cmd::QkTemplatesArgs),

    /// SVD rank analysis of attention QK products.
    QkRank(qk_rank_cmd::QkRankArgs),

    /// Extract interpretable modes from low-rank QK heads via SVD → gate projection.
    QkModes(qk_modes_cmd::QkModesArgs),

    /// Map attention OV circuits to FFN gate features.
    OvGate(ov_gate_cmd::OvGateArgs),

    /// OV rate-distortion and residual-table attention compilation experiments.
    OvRd(ov_rd::cmd::OvRdArgs),

    /// Discover attention → FFN circuits from weight decomposition.
    CircuitDiscover(circuit_discover_cmd::CircuitDiscoverArgs),

    /// Bottleneck analysis of attention components.
    AttnBottleneck(attn_bottleneck_cmd::AttnBottleneckArgs),

    /// Bottleneck analysis of FFN components.
    FfnBottleneck(ffn_bottleneck_cmd::FfnBottleneckArgs),

    /// Measure overlap between entity-routed and ground-truth gate features.
    FfnOverlap(ffn_overlap_cmd::FfnOverlapArgs),

    /// Knowledge graph retrieval benchmark.
    KgBench(kg_bench_cmd::KgBenchArgs),

    /// Trace residual stream trajectories on the sphere across layers.
    TrajectoryTrace(trajectory_trace_cmd::TrajectoryTraceArgs),

    /// Test rank-k projection through the residual stream.
    ProjectionTest(projection_test_cmd::ProjectionTestArgs),

    /// Extract OV fingerprint basis from attention weights.
    FingerprintExtract(fingerprint_extract_cmd::FingerprintExtractArgs),

    /// Test rule-based bottleneck — if-else rules replace early layers.
    BottleneckTest(bottleneck_test_cmd::BottleneckTestArgs),

    /// Embedding jump — raw token embeddings → projected L13 → decoder.
    EmbeddingJump(embedding_jump_cmd::EmbeddingJumpArgs),

    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    /// Measure round-trip latency breakdown against a remote FFN server.
    FfnLatency(ffn_latency_cmd::FfnLatencyArgs),
}

pub(crate) fn run_dev(cmd: DevCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        DevCommand::WeightExtract(a) => weight_walk_cmd::run(a),
        DevCommand::AttentionExtract(a) => attention_walk_cmd::run(a),
        DevCommand::VectorExtract(a) => vector_extract_cmd::run(a),
        DevCommand::Residuals(a) => residuals_cmd::run(a),
        DevCommand::Predict(a) => predict_cmd::run(a),
        DevCommand::IndexGates(a) => index_gates_cmd::run(a),
        DevCommand::Walk(a) => walk_cmd::run(a),
        DevCommand::AttentionCapture(a) => attention_capture_cmd::run(a),
        DevCommand::QkTemplates(a) => qk_templates_cmd::run(a),
        DevCommand::QkRank(a) => qk_rank_cmd::run(a),
        DevCommand::QkModes(a) => qk_modes_cmd::run(a),
        DevCommand::OvGate(a) => ov_gate_cmd::run(a),
        DevCommand::OvRd(a) => ov_rd::cmd::run(a),
        DevCommand::CircuitDiscover(a) => circuit_discover_cmd::run(a),
        DevCommand::AttnBottleneck(a) => attn_bottleneck_cmd::run(a),
        DevCommand::FfnBottleneck(a) => ffn_bottleneck_cmd::run(a),
        DevCommand::FfnOverlap(a) => ffn_overlap_cmd::run(a),
        DevCommand::KgBench(a) => kg_bench_cmd::run(a),
        DevCommand::TrajectoryTrace(a) => trajectory_trace_cmd::run(a),
        DevCommand::ProjectionTest(a) => projection_test_cmd::run(a),
        DevCommand::FingerprintExtract(a) => fingerprint_extract_cmd::run(a),
        DevCommand::BottleneckTest(a) => bottleneck_test_cmd::run(a),
        DevCommand::EmbeddingJump(a) => embedding_jump_cmd::run(a),
        DevCommand::Bfs(a) => bfs_cmd::run(a),
        DevCommand::FfnLatency(a) => ffn_latency_cmd::run(a),
    }
}
