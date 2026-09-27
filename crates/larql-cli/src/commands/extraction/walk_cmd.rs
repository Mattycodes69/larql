use std::path::PathBuf;
use std::time::Instant;

#[cfg(unix)]
extern crate libc;

/// Current process RSS in megabytes (best-effort).
fn rss_mb() -> f64 {
    #[cfg(unix)]
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        // macOS: ru_maxrss is bytes. Linux: kilobytes.
        #[cfg(target_os = "macos")]
        let bytes = usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        let bytes = (usage.ru_maxrss as u64) * 1024;
        bytes as f64 / (1024.0 * 1024.0)
    }
    #[cfg(not(unix))]
    {
        0.0
    }
}

use clap::Args;
use larql_inference::InferenceModel;
use larql_vindex::{
    load_vindex_embeddings, load_vindex_tokenizer, ndarray, IndexLoadCallbacks,
    SilentLoadCallbacks, VectorIndex,
};

/// Log to stderr only if verbose.
macro_rules! vlog {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose { eprintln!($($arg)*); }
    };
}

mod predict;
mod print;
mod q4k;
mod stream;
use predict::*;
use print::*;
use q4k::*;
use stream::*;

#[derive(Args)]
pub struct WalkArgs {
    /// Prompt text to walk through the model.
    #[arg(short, long)]
    pub prompt: String,

    /// Path to a .vindex directory (self-contained, no model needed).
    #[arg(long)]
    pub index: Option<PathBuf>,

    /// Model path or HuggingFace model ID (needed for --predict/--compare,
    /// or when not using --index).
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to extracted ffn_gate vectors (alternative to --index).
    #[arg(long)]
    pub gate_vectors: Option<PathBuf>,

    /// Path to extracted ffn_down vectors (alternative to --index).
    #[arg(long)]
    pub down_vectors: Option<PathBuf>,

    /// Top-K features per layer for the gate KNN. Default: unlimited
    /// (`usize::MAX`) — matches the server's `WalkFfn::new_unlimited`
    /// behavior and sidesteps quality drift on stale/low-K vindexes.
    /// Pass an explicit `N` to cap for speed/memory trade-offs.
    #[arg(short = 'k', long, default_value_t = usize::MAX)]
    pub top_k: usize,

    /// Layers to walk. Comma-separated or range (e.g., "26,27,28" or "24-33").
    /// Default: all layers.
    #[arg(short, long)]
    pub layers: Option<String>,

    /// Number of top predictions to show.
    #[arg(long, default_value = "10")]
    pub predict_top_k: usize,

    /// Max tokens to generate autoregressively when `--predict` is set.
    /// `1` reproduces the old "next-token-only" behavior.
    #[arg(long, default_value = "1")]
    pub max_tokens: usize,

    /// KV cache strategy for autoregressive decode.
    /// See `larql run --help` for the full menu.
    #[arg(long, default_value = "standard",
          value_parser = crate::commands::primary::run_cmd::parse_kv_cache)]
    pub kv_cache: crate::commands::primary::run_cmd::KvCacheKind,

    /// Sliding-window size when `--kv-cache markov-bounded`.
    #[arg(long, default_value = "0")]
    pub context_window: usize,

    /// KV engine spec — overrides `--kv-cache` when set. See `larql run
    /// --help` for the full syntax. Falls back to the `LARQL_KV_ENGINE`
    /// env var when unset.
    #[arg(long, value_name = "SPEC")]
    pub engine: Option<String>,

    /// Run full forward pass with walk FFN and show predictions (requires --model).
    #[arg(long)]
    pub predict: bool,

    /// Compare walk FFN predictions against dense ground truth (requires --model).
    #[arg(long)]
    pub compare: bool,

    /// Number of down tokens to show per feature.
    #[arg(long, default_value = "5")]
    pub down_top_k: usize,

    /// Show verbose loading and timing info.
    #[arg(short, long)]
    pub verbose: bool,

    /// Run autoregressive generation through the Metal Q4K pipeline:
    /// fused `full_pipeline_q4` prefill + `decode_token` KV-cached decode.
    /// Works for pre-norm (Llama, Mistral) and post-norm + QK-norm
    /// (Gemma 3, Gemma 4) architectures. Requires a Q4K vindex and a
    /// build with `--features gpu` on an M-series Mac.
    #[arg(long)]
    pub metal: bool,

    /// Route the FFN to a remote `larql-server` via `POST /v1/walk-ffn`
    /// (with `full_output: true`). Attention still runs locally; the FFN
    /// per-layer call lands on the server. Incompatible with `--compare`
    /// — the comparison backends expect local FFN weights.
    ///
    /// Example: `--ffn-remote http://127.0.0.1:8080`
    #[arg(long, value_name = "URL")]
    pub ffn_remote: Option<String>,

    /// Per-request HTTP timeout (seconds) for `--ffn-remote`.
    #[arg(long, default_value = "60")]
    pub ffn_remote_timeout_secs: u64,

    /// Dense FFN dispatch strategy when `--ffn-remote` is set.
    ///
    ///   streaming  (default) — sequential per-layer round-trips (exact).
    ///   batch      — all layers fired in parallel, then injected (approximate).
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub ffn_dispatch: String,

    /// Number of predispatch iterations per token when `--ffn-dispatch batch`.
    #[arg(long, default_value = "1", value_name = "N")]
    pub ffn_predispatch_iters: usize,
}

struct VerboseLoadCallbacks;

impl IndexLoadCallbacks for VerboseLoadCallbacks {
    fn on_file_start(&mut self, component: &str, path: &str) {
        eprintln!("Loading {component}: {path}");
    }
    fn on_progress(&mut self, records: usize) {
        eprint!("\r  {records} records...");
    }
    fn on_file_done(&mut self, component: &str, records: usize, elapsed_ms: f64) {
        eprintln!(
            "\r  {component}: {records} records ({:.1}s)",
            elapsed_ms / 1000.0
        );
    }
}

pub fn run(args: WalkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    // Validated once, here, because `run` fans out to several forward paths
    // and only some of them read the spec. Rejecting an unparseable one up
    // front beats warning and running `standard` anyway: a typo'd engine that
    // silently exercises the default is the same class of failure as a flag
    // dropped entirely (issue #199), and the one a caller is least likely to
    // notice.
    validate_engine_spec(requested_engine_spec(&args).as_deref())?;
    let load_start = Instant::now();

    // Load the index — either from .vindex or from separate NDJSON files
    let index = if let Some(ref vindex_path) = args.index {
        vlog!(verbose, "Loading vindex: {}", vindex_path.display());
        if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        }
    } else if let Some(ref gate_path) = args.gate_vectors {
        let mut idx = if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_gates(gate_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_gates(gate_path, &mut cb)?
        };
        if let Some(ref down_path) = args.down_vectors {
            if verbose {
                let mut cb = VerboseLoadCallbacks;
                idx.load_down_meta(down_path, &mut cb)?;
            } else {
                let mut cb = SilentLoadCallbacks;
                idx.load_down_meta(down_path, &mut cb)?;
            }
        }
        idx
    } else {
        return Err("Either --index (vindex directory) or --gate-vectors required".into());
    };

    vlog!(
        verbose,
        "Index: {} layers, {} gate vectors, {} down meta entries ({:.1}s)",
        index.num_layers,
        index.total_gate_vectors(),
        index.total_down_meta(),
        load_start.elapsed().as_secs_f64()
    );
    // RSS at this point = attn + embed + norms (gate vectors demand-paged,
    // not yet faulted in). Useful for the "7 GB" claim in demos.
    vlog!(
        verbose,
        "  RSS at load: {:.1} GB (gate vectors not yet resident)",
        rss_mb() / 1024.0
    );

    // Parse layer selection
    let all_layers = index.loaded_layers();
    let layers = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => all_layers.clone(),
    };

    if args.predict || args.compare {
        if let Some(model_name) = args.model.as_deref() {
            // Load from safetensors
            run_with_model(model_name, &args, &index, &layers)?;
        } else if let Some(ref vindex_path) = args.index {
            // Try loading weights from vindex
            run_with_vindex_weights(vindex_path, &args, &index, &layers, verbose)?;
        } else {
            return Err(
                "--model or --index (with --include-weights) required for --predict".into(),
            );
        }
    } else if let Some(ref vindex_path) = args.index {
        run_vindex_walk(vindex_path, &args, &index, &layers)?;
    } else {
        let model_name = args
            .model
            .as_deref()
            .ok_or("--model required for embedding walk (or use --index for standalone)")?;
        run_model_embedding_walk(model_name, &args, &index, &layers)?;
    }

    Ok(())
}

/// Walk using embeddings from the .vindex directory. No model needed.
fn run_vindex_walk(
    vindex_path: &std::path::Path,
    args: &WalkArgs,
    index: &VectorIndex,
    layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    vlog!(verbose, "Loading embeddings from vindex...");
    let (embed, embed_scale) = load_vindex_embeddings(vindex_path)?;
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens: {:?})",
        args.prompt,
        token_ids.len(),
        token_ids
    );

    let last_tok = *token_ids.last().ok_or("empty prompt")?;
    let embed_row = embed.row(last_tok as usize);
    let query: ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    let token_str = tokenizer
        .decode(&[last_tok], true)
        .unwrap_or_else(|_| format!("T{last_tok}"));
    vlog!(
        verbose,
        "Query: embedding for {:?} (T{last_tok})",
        token_str.trim()
    );

    let walk_start = Instant::now();
    let trace = index.walk(&query, layers, args.top_k);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    print_walk_trace(&trace, args.down_top_k);

    eprintln!(
        "\nWalk: {} layers, top-{}, {:.1}ms ({:.2}ms/layer)",
        layers.len(),
        args.top_k,
        walk_ms,
        walk_ms / layers.len() as f64
    );

    Ok(())
}

/// Walk using the model's embedding for the last token as the query vector.
fn run_model_embedding_walk(
    model_name: &str,
    args: &WalkArgs,
    index: &VectorIndex,
    layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    vlog!(verbose, "Loading model: {}", model_name);
    let model = InferenceModel::load(model_name)?;
    let weights = model.weights();

    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens: {:?})",
        args.prompt,
        token_ids.len(),
        token_ids
    );

    let last_tok = *token_ids.last().ok_or("empty prompt")?;
    let embed_scale = weights.arch.embed_scale_multiplier();
    let embed_row = weights.embed.row(last_tok as usize);
    let query: ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    let token_str = model
        .tokenizer()
        .decode(&[last_tok], true)
        .unwrap_or_else(|_| format!("T{last_tok}"));
    vlog!(
        verbose,
        "Query: embedding for {:?} (T{last_tok})",
        token_str.trim()
    );

    let walk_start = Instant::now();
    let trace = index.walk(&query, layers, args.top_k);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    print_walk_trace(&trace, args.down_top_k);

    eprintln!(
        "\nWalk: {} layers, top-{}, {:.1}ms ({:.2}ms/layer)",
        layers.len(),
        args.top_k,
        walk_ms,
        walk_ms / layers.len() as f64
    );

    Ok(())
}

/// Walk with full forward pass — uses WalkFfn as the FFN backend.
/// Walk with full forward pass — loads model from safetensors.
fn run_with_model(
    model_name: &str,
    args: &WalkArgs,
    index: &VectorIndex,
    _layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    vlog!(args.verbose, "Loading model: {}", model_name);
    let model_start = Instant::now();
    let model = InferenceModel::load(model_name)?;
    vlog!(
        args.verbose,
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        model_start.elapsed().as_secs_f64()
    );

    run_predict_inner(model.weights(), model.tokenizer(), args, index)
}

/// Walk with full forward pass — loads weights from vindex (no safetensors).
fn run_with_vindex_weights(
    vindex_path: &std::path::Path,
    args: &WalkArgs,
    index: &VectorIndex,
    _layers: &[usize],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    vlog!(verbose, "Loading model weights from vindex...");
    let load_start = Instant::now();

    let mut cb: Box<dyn IndexLoadCallbacks> = if verbose {
        Box::new(VerboseLoadCallbacks)
    } else {
        Box::new(SilentLoadCallbacks)
    };
    // Route Q4 vindexes through the dedicated loader + predict path.
    // `load_model_weights` rejects quantised vindexes (it only knows how to
    // reconstruct the float ModelWeights), so we branch on `config.quant`
    // BEFORE calling it to avoid a confusing error for Q4 users.
    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    if cfg.quant == larql_vindex::QuantFormat::Q4K {
        let mut weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut *cb)?;
        let tokenizer = load_vindex_tokenizer(vindex_path)?;
        vlog!(
            verbose,
            "  {} layers, hidden_size={} (Q4_K, {:.1}s)",
            weights.num_layers,
            weights.hidden_size,
            load_start.elapsed().as_secs_f64()
        );
        // RSS now = attn weights + embeddings + norms. FFN payload (gate_vectors,
        // interleaved_kquant) is demand-paged; pages fault in during inference.
        vlog!(verbose, "  RSS after weights: {:.1} GB", rss_mb() / 1024.0);
        if args.ffn_remote.is_some() {
            return run_predict_q4k_remote(&mut weights, &tokenizer, args, vindex_path);
        }
        return run_predict_q4k(&mut weights, &tokenizer, args, index);
    }

    // Remote FFN: load weights with a pre-mmap filter that skips the
    // FFN tensors — they live on the remote server, the client heap
    // shouldn't carry them. Peak RSS drops to attention + embed +
    // norms + lm_head only.
    let load_opts = larql_vindex::LoadWeightsOptions {
        skip_ffn: args.ffn_remote.is_some(),
        ..Default::default()
    };
    if load_opts.skip_ffn {
        vlog!(
            verbose,
            "  remote FFN configured — skipping FFN tensors at load"
        );
    }
    let weights = larql_vindex::load_model_weights_with_opts(vindex_path, &mut *cb, load_opts)?;
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    vlog!(
        verbose,
        "  {} layers, hidden_size={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        load_start.elapsed().as_secs_f64()
    );

    run_predict_inner(&weights, &tokenizer, args, index)
}

/// The engine the caller asked for, if any: `--engine` first, then
/// `LARQL_KV_ENGINE`. One resolver so the validation in [`run`], the
/// rejection in [`run_predict_q4k`] and the builder in [`generate_stream`]
/// cannot disagree about whether an engine was requested.
fn requested_engine_spec(args: &WalkArgs) -> Option<String> {
    args.engine
        .clone()
        .or_else(|| std::env::var("LARQL_KV_ENGINE").ok())
}

/// Reject a spec no engine answers to.
///
/// Split out as a pure function so the message is testable without a model.
/// Warning and running `standard` instead — the old behaviour — makes a typo'd
/// engine indistinguishable from the default, which is the same failure as
/// dropping the flag entirely (issue #199).
fn validate_engine_spec(spec: Option<&str>) -> Result<(), String> {
    match spec {
        Some(s) if larql_kv::EngineKind::from_name(s).is_none() => Err(format!(
            "unknown --engine {s:?}; supported: {}",
            larql_kv::EngineKind::supported_names().join(", ")
        )),
        _ => Ok(()),
    }
}

/// The refusal owed to a caller who named an engine on a path with no KV
/// cache to put it in.
fn engine_unsupported_on_uncached_path(spec: &str) -> String {
    format!(
        "--engine {spec:?} is not honoured on the CPU Q4K generation path, \
         which is not token-cached and so has no KV engine to select. \
         Use --metal for the KV-cached path, or `larql bench --engine` \
         to compare engines."
    )
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}

#[cfg(test)]
mod tests;
