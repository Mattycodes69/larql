//! Streaming token generation over any `FfnBackend`.

use larql_inference::ModelWeights;
use larql_vindex::tokenizers;

#[allow(unused_imports)]
use super::*;

/// Stream autoregressive generation to stdout, token by token, using
/// a CPU KV cache.
///
/// **Phase 1 (prefill)**: full forward pass over the prompt, capturing
/// post-RoPE K and post-V-norm V per layer → initial KV cache.
/// **Phase 2 (decode)**: per-step — embed new token (one row), run a
/// decode-step attention that attends new Q against cached K/V +
/// appends new K/V to the cache, FFN, next layer. Per-step cost is
/// O(cached_len × hidden) instead of O(cached_len² × hidden) without
/// the cache.
///
/// Backend-agnostic — works with `WalkFfn` (local), `RemoteWalkBackend`
/// (FFN over HTTP), or any other `FfnBackend` impl.
pub(super) fn generate_stream(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    ffn: &dyn larql_inference::FfnBackend,
    initial_ids: &[u32],
    args: &WalkArgs,
    verbose: bool,
) -> Vec<u32> {
    use crate::commands::primary::run_cmd::KvCacheKind;
    use std::io::Write;
    let mut stdout = std::io::stdout();
    let max_tokens = args.max_tokens;

    // Auto-detected compute backend. On macOS with the `gpu` feature
    // this is Metal; otherwise CPU BLAS. Note the Metal backend has a
    // FLOP threshold (~500M) below which it stays on CPU — single-token
    // decode-step matmuls (m=1 × k×n) are ~5-7M FLOP and fall under
    // that limit, so projections run on CPU BLAS even when Metal is
    // available. Real GPU wins require either the Q4K `full_pipeline`
    // (already wired via `--metal` on Q4K vindexes) or batched decode.
    let backend = larql_inference::default_engine_backend();
    // Captured for the verbose label after `backend` is consumed by the
    // engine builder.
    let backend_name = backend.name().to_string();

    // Unified `KvEngine` dispatch. Resolution precedence:
    //   1. `--engine SPEC` flag (parsed by `EngineKind::from_name`)
    //   2. `LARQL_KV_ENGINE` env var (same parser)
    //   3. `--kv-cache standard|markov-bounded|none` legacy mapping
    // CLI flag wins over env var; env var wins over `--kv-cache`. See
    // `crates/larql-inference/docs/specs/kv-engine-unification.md` §6.
    use larql_kv::EngineKind;
    let engine_spec = requested_engine_spec(args);
    let (kind, label) = match engine_spec {
        Some(spec) => {
            let kind = EngineKind::from_name(&spec).unwrap_or_else(|| {
                eprintln!(
                    "warning: unknown --engine spec {spec:?}, falling back to standard (unbounded)"
                );
                EngineKind::Standard { window_size: None }
            });
            let label = match &kind {
                EngineKind::Standard { window_size: None } => "engine=standard",
                EngineKind::Standard {
                    window_size: Some(_),
                } => "engine=standard (windowed)",
                EngineKind::NoCache => "engine=no-cache",
                EngineKind::MarkovResidual { .. } => "engine=markov-rs",
                EngineKind::WindowedCheckpoint { .. } => "engine=windowed-checkpoint",
                EngineKind::TurboQuant { .. } => "engine=turbo-quant",
                EngineKind::Apollo { .. } => "engine=apollo",
                EngineKind::BoundaryKv { .. } => "engine=boundary-kv",
                EngineKind::MarkovResidualCodec { .. } => "engine=markov-rs-codec",
                EngineKind::BoundaryPerLayer { .. } => "engine=boundary-per-layer",
                EngineKind::SemanticPromotion { .. } => "engine=semantic-promotion",
            };
            (kind, label)
        }
        None => match args.kv_cache {
            KvCacheKind::Standard => (
                EngineKind::Standard { window_size: None },
                "standard KV cache",
            ),
            KvCacheKind::MarkovBounded => (
                EngineKind::Standard {
                    window_size: if args.context_window > 0 {
                        Some(args.context_window)
                    } else {
                        None
                    },
                },
                "Markov-bounded KV cache",
            ),
            KvCacheKind::None => (EngineKind::NoCache, "no cache (O(N²))"),
        },
    };
    let mut engine = kind.build(backend);
    let generated = larql_kv::generation::generate_with_engine(
        &mut engine,
        weights,
        tokenizer,
        ffn,
        initial_ids,
        max_tokens,
        |_id, tok| {
            print!("{tok}");
            let _ = stdout.flush();
        },
    );
    println!();
    if verbose {
        // Honest reporting: the backend is `backend.name()` but the
        // Metal path only actually dispatches when matmul size exceeds
        // the calibrated FLOP threshold. Decode-step matmuls on 4B are
        // typically below that, so labelling "via metal" would be a
        // lie. Report both the detected backend AND note that single-
        // token decode stays on CPU regardless.
        eprintln!(
            "  Generated {} tokens ({}) — backend={} (decode matmuls usually below GPU threshold)",
            generated.len(),
            label,
            backend_name,
        );
    }
    generated
}

pub(super) fn is_stop_token(s: &str) -> bool {
    larql_inference::vindex::is_end_of_turn(s)
}
