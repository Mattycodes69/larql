#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::type_complexity)]
// Architectural lints suppressed crate-wide:
//
//  - `too_many_arguments`: most command runners have wide signatures
//    (`run(args, output, callbacks, ...)`) reflecting CLI surface, not
//    internal coupling. Reducing to ≤7 args would mean introducing
//    parameter structs across every command, which is a bigger refactor
//    than a lint cleanup.
//  - `large_enum_variant`: a few command-arg enums (notably `OvRdArgs`)
//    have one or two large variants. Boxing them would change call
//    sites everywhere.
//
// Mechanical lints (iter-on-map-keys, ptr_arg, needless_deref,
// map_entry, etc.) are fixed in place rather than suppressed.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::large_enum_variant)]

use clap::{Parser, Subcommand};

mod anyres_tiler;
mod backend_select;
mod commands;
mod formatting;
mod image_input;
mod utils;

use commands::dev::*;
use commands::diagnostics::*;
use commands::extraction::*;
use commands::primary::*;
use commands::query::*;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL — decompile transformer weights into a queryable vindex"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// ══════════════════════════════════════════════════════════════════════
// Top-level commands
//
// Grouped in --help output via `next_help_heading`:
//   * (unspecified)   — Primary user verbs
//   * "Build"         — Extract / compile / publish
//   * "Query"         — Graph file introspection (legacy pre-LQL)
//   * "LQL"           — Query-language surface
//   * "Server"        — Serve a vindex
//   * "Research"      — `larql dev <subcmd>`
// ══════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum Commands {
    // ── Primary user-facing ─────────────────────────────────────────
    /// Run inference (one-shot if prompt is given, chat if not).
    Run(run_cmd::RunArgs),

    /// Interactive chat — alias for `run <model>` with no prompt.
    Chat(ChatArgs),

    /// Download a vindex from HuggingFace and cache it locally.
    Pull(pull_cmd::PullArgs),

    /// Manage HuggingFace *model* repos (safetensors + tokenizer + config).
    /// Companion to `pull` (which is vindex-only). Use `model pull` to
    /// stage a raw HF model for `convert safetensors-to-vindex`.
    Model(model_cmd::ModelArgs),

    /// Register a local vindex directory with the cache so `run` / `list`
    /// / `show` can find it by shorthand.
    Link(link_cmd::LinkArgs),

    /// List cached vindexes.
    List(list_cmd::ListArgs),

    /// Show metadata for a vindex.
    Show(show_cmd::ShowArgs),

    /// Carve a subset of a vindex (client / server / browse / router slice).
    Slice(slice_cmd::SliceArgs),

    /// Publish a vindex to HuggingFace — full vindex plus slice siblings.
    Publish(publish_cmd::PublishArgs),

    /// Validate the VINDEX3 production registry (registry/index.json +
    /// registry/models/*.json).
    #[command(subcommand)]
    Registry(registry_cmd::RegistryCommand),

    /// Remove a cached vindex.
    Rm(rm_cmd::RmArgs),

    /// Benchmark decode throughput on a real vindex (Metal / CPU / Ollama).
    Bench(bench::BenchArgs),

    /// DEC residual-replay loadgen — capture real per-layer residuals, then
    /// replay batch × wire × dispatch sweeps against an expert server
    /// (docs/dec-funnel.md).
    DecBench(dec_bench::DecBenchArgs),

    /// K3 serving ledger — miss budget, weight touch, dense-precision
    /// frontier and speculative block economics, derived from the
    /// checkpoint's own tensor table (docs/dec-funnel.md).
    K3Ledger(k3_ledger::K3LedgerArgs),

    /// Split-axis accuracy suite — parametric vs in-context vs conflict,
    /// scored with top-1 match and Shannon bits-per-token.
    Accuracy(accuracy_cmd::AccuracyArgs),

    /// Shannon-style next-token bit measurements and demo compression.
    #[command(subcommand)]
    Shannon(shannon_cmd::ShannonCommand),

    // ── Server ──────────────────────────────────────────────────────
    #[command(next_help_heading = "Server")]
    /// Serve a vindex over HTTP + gRPC.
    Serve(serve_cmd::ServeArgs),

    #[command(next_help_heading = "Server")]
    /// Ask a running LARQL server what it will and will not do
    /// (`GET /v1/capabilities`). Distinct from `capabilities`, which
    /// reports what this release recognises rather than what one
    /// server offers.
    ServerCapabilities(server_capabilities_cmd::ServerCapabilitiesArgs),

    // ── LQL ─────────────────────────────────────────────────────────
    #[command(next_help_heading = "LQL")]
    /// Launch the LQL interactive REPL.
    Repl,

    #[command(next_help_heading = "LQL")]
    /// Execute a one-shot LQL statement.
    Lql(LqlArgs),

    // ── Build / extract ─────────────────────────────────────────────
    #[command(next_help_heading = "Build")]
    /// Build a .vindex by decompiling a HuggingFace model.
    Extract(extract_index_cmd::ExtractIndexArgs),

    #[command(next_help_heading = "Build")]
    /// Backwards-compat alias for `extract` (identical behavior).
    ExtractIndex(extract_index_cmd::ExtractIndexArgs),

    #[command(next_help_heading = "Build")]
    /// Build a custom vindex from a Vindexfile (declarative: FROM + PATCH + INSERT).
    Build(build_cmd::BuildArgs),

    #[command(next_help_heading = "Build")]
    /// Compile vindex patches into model weights (AOT compilation).
    Compile(compile_cmd::CompileArgs),

    #[command(next_help_heading = "Build")]
    /// Convert between model formats (GGUF ↔ vindex, safetensors → vindex).
    Convert(convert_cmd::ConvertArgs),

    #[command(next_help_heading = "Build")]
    /// HuggingFace Hub: upload a vindex.
    Hf(hf_cmd::HfArgs),

    #[command(next_help_heading = "Build")]
    /// Verify vindex file integrity (SHA256 checksums).
    Verify(verify_cmd::VerifyArgs),

    #[command(next_help_heading = "Build")]
    /// Engine diagnostic — print which kernel paths fire for a vindex.
    Diag(diag_cmd::DiagArgs),

    #[command(next_help_heading = "Build")]
    /// Cross-backend numerical parity diff (CPU vs Metal vs reference).
    Parity(parity::ParityArgs),

    #[command(next_help_heading = "Build")]
    /// Expert-selection locality over a routing trace: does speculative
    /// decoding amortise the expert bank, and can a hot cache work?
    /// Collect the trace with `LARQL_MOE_ROUTE_TRACE=<path> larql shannon score`.
    MoeLocality(moe_locality::MoeLocalityArgs),

    // ── Factory (docs/vindex-factory.md) ─────────────────────────────
    #[command(next_help_heading = "Factory", subcommand)]
    /// Vindex Factory recipe tooling — validate a recipe, compute its
    /// build_id (docs/vindex-factory.md).
    Recipe(recipe_cmd::RecipeCommand),

    #[command(next_help_heading = "Factory")]
    /// Print this release's capability manifest as JSON — which
    /// architectures it recognises and what each one supports.
    Capabilities,

    #[command(next_help_heading = "Factory", name = "inspect-hf")]
    /// Machine-readable architecture inventory of an HF checkpoint dir —
    /// identity, per-layer attention policy, tensors, and every config key
    /// this build does not consume.
    InspectHf(inspect_hf_cmd::InspectHfArgs),

    #[command(next_help_heading = "Factory", subcommand)]
    /// VINDEX3 container programme verbs (`plan`: semantic
    /// representability check before conversion).
    Vindex3(vindex3_cmd::Vindex3Command),

    #[command(next_help_heading = "Factory", subcommand)]
    /// Render a Hub model card for a build (docs/vindex-factory.md §9).
    Card(card_cmd::CardCommand),

    #[command(next_help_heading = "Factory")]
    /// Serve a stored physical-plan search record over MCP, read-only
    /// (docs/represent-optimizer-mcp.md §4h).
    OptimizerMcp(optimizer_mcp::OptimizerMcpArgs),

    // ── Query (legacy, pre-LQL graph-file surface) ──────────────────
    #[command(next_help_heading = "Query")]
    /// Query a graph file for facts.
    Query(query_cmd::QueryArgs),

    #[command(next_help_heading = "Query")]
    /// Describe an entity (all edges).
    Describe(describe_cmd::DescribeArgs),

    #[command(next_help_heading = "Query")]
    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    #[command(next_help_heading = "Query")]
    /// Validate a graph file.
    Validate(validate_cmd::ValidateArgs),

    #[command(next_help_heading = "Query")]
    /// Merge multiple graph files.
    Merge(merge_cmd::MergeArgs),

    #[command(next_help_heading = "Query")]
    /// Filter graph edges by confidence, layer, selectivity, relation, source.
    Filter(filter_cmd::FilterArgs),

    // ── Research / power-user tooling ───────────────────────────────
    #[command(next_help_heading = "Research", subcommand)]
    /// Research / interpretability tools (weight-extract, qk-rank, …).
    Dev(DevCommand),
}

// ══════════════════════════════════════════════════════════════════════
// Research subcommand group — `larql dev <subcmd>`.
//
// Everything in here is unchanged from the pre-redesign top-level surface
// except its invocation path. A small argv trampoline in `main()` rewrites
// `larql <legacy-name>` → `larql dev <legacy-name>` so existing scripts
// continue to work without a breaking change.
// ══════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum DevCommand {
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

// ══════════════════════════════════════════════════════════════════════
// Minor glue types
// ══════════════════════════════════════════════════════════════════════

#[derive(clap::Args)]
struct ChatArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    model: String,

    /// Max tokens to generate per chat response.
    #[arg(short = 'n', long = "max-tokens", default_value = "64")]
    max_tokens: usize,

    /// Route FFN to a remote larql-server.
    #[arg(long, value_name = "URL")]
    ffn: Option<String>,
    routed_from: Option<String>,

    /// HTTP timeout in seconds for --ffn.
    #[arg(long, default_value = "60")]
    ffn_timeout_secs: u64,

    /// Verbose load / timing output.
    #[arg(short, long)]
    verbose: bool,
}

impl From<ChatArgs> for run_cmd::RunArgs {
    fn from(c: ChatArgs) -> Self {
        run_cmd::RunArgs {
            model: c.model,
            prompt: None,
            max_tokens: c.max_tokens,
            top: 1,
            kv_cache: run_cmd::KvCacheKind::Standard,
            context_window: 0,
            engine: None,
            continuation: None,
            continuation_options: Vec::new(),
            ffn: c.ffn,
            routed_from: c.routed_from,
            emit_ids: false,
            ffn_timeout_secs: c.ffn_timeout_secs,
            metal: false,
            verbose: c.verbose,
            experts: false,
            experts_dir: None,
            ops: Vec::new(),
            constrained: false,
            moe_shards: None,
            moe_units_manifest: None,
            moe_dispatch: "streaming".to_string(),
            moe_predispatch_iters: 1,
            ffn_dispatch: "streaming".to_string(),
            ffn_predispatch_iters: 1,
            // Chat doesn't accept images today; default to empty for the
            // shim. When chat-with-images becomes a thing (Phase 2+), the
            // ChatArgs struct will grow its own --image flag.
            image: Vec::new(),
            mm_weights: None,
            v3_shards: Vec::new(),
            v3_ffn_shards: Vec::new(),
            v3_ffn_wire: None,
            v3_profile: None,
            v3_shard_token_env: None,
            // Chat is text-only today; speech arrives via `run --speak`
            // (and later a chat session feeding the speech stream).
            speak: false,
            plugin: Default::default(),
            voice: None,
            codec_cmd: None,
            speech_out: None,
            play: false,
            max_frames: 0,
            greedy: false,
            seed: 0,
            q4: false,
        }
    }
}

#[derive(clap::Args)]
struct LqlArgs {
    /// LQL statement (e.g. `WALK "The capital of France is" TOP 5;`).
    statement: String,
}

// ══════════════════════════════════════════════════════════════════════
// Main entry + argv trampoline
// ══════════════════════════════════════════════════════════════════════

/// Research subcommands previously lived at the top level. Rewrite
/// `larql <legacy-name> …` → `larql dev <legacy-name> …` before clap
/// parses so existing scripts keep working.
const LEGACY_DEV_NAMES: &[&str] = &[
    "weight-extract",
    "attention-extract",
    "vector-extract",
    "residuals",
    "predict",
    "index-gates",
    "walk",
    "attention-capture",
    "qk-templates",
    "qk-rank",
    "qk-modes",
    "ov-gate",
    "circuit-discover",
    "attn-bottleneck",
    "ffn-bottleneck",
    "ffn-overlap",
    "kg-bench",
    "trajectory-trace",
    "projection-test",
    "fingerprint-extract",
    "bottleneck-test",
    "embedding-jump",
    "bfs",
    "ffn-latency",
];

fn rewrite_legacy_argv(args: Vec<String>) -> Vec<String> {
    if args.len() >= 2 && LEGACY_DEV_NAMES.contains(&args[1].as_str()) {
        let mut rewritten = Vec::with_capacity(args.len() + 1);
        rewritten.push(args[0].clone());
        rewritten.push("dev".to_string());
        rewritten.extend(args.into_iter().skip(1));
        return rewritten;
    }
    args
}

fn main() {
    // Windows defaults the main thread to a 1 MiB stack, which our large
    // clap-derived `Commands` enum overflows during parse_from in debug
    // builds. Spawn the real entrypoint on a worker thread with a roomy
    // stack so the binary behaves the same as on Linux/macOS (where the
    // default main stack is ~8 MiB).
    let code = std::thread::Builder::new()
        .name("larql-main".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn larql-main thread")
        .join()
        .expect("larql-main thread panicked");
    // Flush the latent-mask channel survival counts, if that probe was
    // collecting them. No-op unless `LARQL_MOE_LATENT_STATS` is set.
    larql_compute::cpu::ops::moe::latent_mask::dump_stats();
    std::process::exit(code);
}

fn real_main() -> i32 {
    let raw_args: Vec<String> = std::env::args().collect();
    let args = rewrite_legacy_argv(raw_args);
    let cli = Cli::parse_from(args);

    let result = match cli.command {
        // ── Primary ──
        Commands::Run(args) => run_cmd::run(args),
        Commands::Chat(args) => run_cmd::run(args.into()),
        Commands::Bench(args) => bench::run(args),
        Commands::DecBench(args) => dec_bench::run(args),
        Commands::K3Ledger(args) => k3_ledger::run(args),
        Commands::Accuracy(args) => accuracy_cmd::run(args),
        Commands::Shannon(cmd) => shannon_cmd::run(cmd),
        Commands::Pull(args) => pull_cmd::run(args),
        Commands::Model(args) => model_cmd::run(args),
        Commands::Link(args) => link_cmd::run(args),
        Commands::List(args) => list_cmd::run(args),
        Commands::Show(args) => show_cmd::run(args),
        Commands::Slice(args) => slice_cmd::run(args),
        Commands::Publish(args) => publish_cmd::run(args),
        Commands::Registry(cmd) => registry_cmd::run(cmd),
        Commands::Rm(args) => rm_cmd::run(args),

        // ── Build / extract ──
        Commands::Extract(args) => extract_index_cmd::run(args),
        Commands::ExtractIndex(args) => extract_index_cmd::run(args),
        Commands::Build(args) => build_cmd::run(args),
        Commands::Compile(args) => compile_cmd::run(args),
        Commands::Convert(args) => convert_cmd::run(args),
        Commands::Hf(args) => hf_cmd::run(args),
        Commands::Verify(args) => verify_cmd::run(args),
        Commands::Diag(args) => diag_cmd::run(args),
        Commands::Parity(args) => parity::run(args),
        Commands::MoeLocality(args) => moe_locality::run(args),

        // ── Query (legacy graph-file surface) ──
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Merge(args) => merge_cmd::run(args),
        Commands::Filter(args) => filter_cmd::run(args),

        // ── LQL ──
        Commands::Repl => {
            larql_lql::run_repl();
            Ok(())
        }
        Commands::Lql(args) => match larql_lql::run_batch(&args.statement) {
            Ok(lines) => {
                for line in &lines {
                    println!("{line}");
                }
                Ok(())
            }
            Err(e) => Err(e),
        },

        // ── Factory ──
        Commands::Recipe(cmd) => recipe_cmd::run(cmd),
        Commands::Capabilities => capabilities_cmd::run(),
        Commands::ServerCapabilities(args) => server_capabilities_cmd::run(args),
        Commands::InspectHf(args) => inspect_hf_cmd::run(args),
        Commands::Vindex3(cmd) => vindex3_cmd::run(cmd),
        Commands::OptimizerMcp(args) => optimizer_mcp::run(args),
        Commands::Card(cmd) => card_cmd::run(cmd),

        // ── Serve (exec into larql-server) ──
        Commands::Serve(args) => serve_cmd::run_serve(args),

        // ── Research / dev tools ──
        Commands::Dev(cmd) => run_dev(cmd),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        return 1;
    }
    0
}

fn run_dev(cmd: DevCommand) -> Result<(), Box<dyn std::error::Error>> {
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

#[cfg(test)]
mod trampoline_tests {
    use super::*;

    fn args(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn primary_verb_is_untouched() {
        let input = args(&["larql", "run", "gemma3-4b.vindex", "hello"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn top_level_extract_is_untouched() {
        let input = args(&["larql", "extract", "google/gemma-3-4b-it", "-o", "out"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn extract_index_alias_is_untouched() {
        // `extract-index` is a distinct top-level variant, not a legacy
        // research command — must not be rewritten to `dev extract-index`.
        let input = args(&["larql", "extract-index", "google/gemma-3-4b-it"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn legacy_research_verb_is_rewritten() {
        let input = args(&[
            "larql",
            "walk",
            "--index",
            "x.vindex",
            "--prompt",
            "hi",
            "--predict",
        ]);
        let out = rewrite_legacy_argv(input);
        assert_eq!(
            out,
            args(&[
                "larql",
                "dev",
                "walk",
                "--index",
                "x.vindex",
                "--prompt",
                "hi",
                "--predict"
            ])
        );
    }

    /// Every legacy name must rewrite to a subcommand that ACTUALLY
    /// EXISTS.
    ///
    /// `legacy_research_flag_names_all_rewrite` below only asserts that
    /// the rewrite happens — it passes just as happily when the target
    /// is gone, and three dead entries (`extract-routes`, `ffn-bench`,
    /// `ffn-throughput`) survived behind it until 2026-08-22. The
    /// failure mode is user-visible and confusing: `larql ffn-bench`
    /// was rewritten to `larql dev ffn-bench`, which clap then rejected
    /// with a "did you mean" for a *different* command.
    #[test]
    fn every_legacy_name_maps_to_a_real_dev_subcommand() {
        use clap::CommandFactory;
        let cli = Cli::command();
        let dev = cli
            .get_subcommands()
            .find(|c| c.get_name() == "dev")
            .expect("`dev` subcommand exists");
        let live: Vec<&str> = dev.get_subcommands().map(|c| c.get_name()).collect();
        let dead: Vec<&&str> = LEGACY_DEV_NAMES
            .iter()
            .filter(|n| !live.contains(&**n))
            .collect();
        assert!(
            dead.is_empty(),
            "LEGACY_DEV_NAMES rewrites these to `larql dev <name>`, but no such \
             subcommand exists — the rewrite turns a clean top-level error into a \
             misleading one: {dead:?}"
        );
    }

    #[test]
    fn legacy_research_flag_names_all_rewrite() {
        // Spot-check each legacy name survives the rewrite.
        for name in LEGACY_DEV_NAMES {
            let input = args(&["larql", name, "--help"]);
            let out = rewrite_legacy_argv(input);
            assert_eq!(out[0], "larql");
            assert_eq!(out[1], "dev");
            assert_eq!(out[2], *name);
            assert_eq!(out[3], "--help");
        }
    }

    #[test]
    fn no_args_returns_unchanged() {
        let input = args(&["larql"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn unknown_verb_is_not_rewritten() {
        // If `larql typo-command` comes in, don't wrap in `dev` — let
        // clap produce its own "unrecognized subcommand" error.
        let input = args(&["larql", "typo-command"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn rewrite_preserves_argument_count_plus_one() {
        let input = args(&["larql", "walk", "--flag", "value"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out.len(), input.len() + 1);
    }
}

#[cfg(test)]
mod documentation_tests {
    use clap::CommandFactory;

    #[test]
    fn current_documentation_facts_match_clap() {
        let facts: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/generated/current-facts.json"))
                .unwrap();
        let command = super::Cli::command();
        let vindex3 = command.find_subcommand("vindex3").unwrap();
        let mut names: Vec<_> = vindex3.get_subcommands().map(|c| c.get_name()).collect();
        names.sort_unstable();
        assert_eq!(facts["commands"]["larql_vindex3"], serde_json::json!(names));
    }
}
