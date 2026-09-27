//! `larql parity` — cross-backend numerical diff for inference components.
//!
//! Diffs the same input through multiple backends (slow naive reference,
//! production CPU, Metal, HF — backends added incrementally) and reports
//! the first checkpoint where they diverge beyond `--tolerance`.
//!
//! v1 (this file) ships:
//!   - `--component moe-expert` — single expert forward (gate / up / act / down)
//!   - `--component moe-block`  — full MoE block (router → top-K → experts → sum → norm)
//!   - backends: `reference` (slow naive), `cpu` (production)
//!
//! v2 (planned) — Metal as a third backend, attention/dense-ffn/layer/forward
//! components. v3 — HF Python sidecar for ground-truth reference.
//!
//! What shipped is in `crates/larql-cli/CHANGELOG.md` (2026-05-10); the
//! remaining open scoping work is `ROADMAP.md` → "P2: parity polish".

// Without the `gpu`+macOS feature the real `run` (and everything it calls —
// the naive reference impls, MoE helpers, dump utilities) is `#[cfg]`'d out,
// leaving only the stub `run` below. That code is live in the gpu build, so
// it is not truly dead — silence dead-code only in the gpu-off config rather
// than gating every helper individually.
#![cfg_attr(not(all(feature = "gpu", target_os = "macos")), allow(dead_code))]

use clap::Args;

#[cfg(all(feature = "gpu", target_os = "macos"))]
use crate::commands::primary::cache;
use larql_compute::Activation;
use larql_models::weights::{per_layer_ffn_key, PER_LAYER_FFN_DOWN, PER_LAYER_FFN_GATE_UP};
#[cfg(all(feature = "gpu", target_os = "macos"))]
use larql_vindex::{load_model_weights_kquant, load_vindex_config, SilentLoadCallbacks};

mod layer_diff;
mod moe;
mod reference;
// The components are reached only from the Metal-backed `run`.
#[cfg(all(feature = "gpu", target_os = "macos"))]
use layer_diff::*;
#[cfg(all(feature = "gpu", target_os = "macos"))]
use moe::*;
use reference::*;

// ── Component / backend taxonomies ────────────────────────────────────────────

/// Inference checkpoints that can be diffed independently.
const COMPONENTS: &[&str] = &[
    "moe-expert", // single expert forward (gate/up/act/down)
    "moe-block",  // full MoE block (router → top-K → experts → sum → norm)
    "lm-head",    // final projection parity (Q4_K vs f32 reference)
    "layer",      // full hybrid-MoE layer: CPU vs Metal, per-layer residual diff
];

/// Backends available as comparison targets.
///
/// `reference` is the slow naive triple-loop CPU baseline. `cpu` is the
/// production path under test. `metal` is the GPU backend (v2 — used by
/// `--component layer`).
const BACKENDS: &[&str] = &[
    "reference", // slow naive baseline (moe-expert, moe-block)
    "cpu",       // production CPU path
    "metal",     // Metal GPU backend (layer component)
];

#[derive(Args)]
pub struct ParityArgs {
    /// Vindex directory, `hf://` URL, or cache shorthand. Same resolution
    /// as `larql run`.
    pub model: String,

    /// Inference checkpoint to diff. v1: `moe-expert`, `moe-block`.
    #[arg(long, default_value = "moe-block")]
    pub component: String,

    /// Layer index. Default 0.
    #[arg(long, default_value = "0")]
    pub layer: usize,

    /// Expert index (used when `--component moe-expert`).
    #[arg(long, default_value = "0")]
    pub expert: usize,

    /// Comma-separated list of backends to run. v1: `reference,cpu`.
    /// First backend in the list is the reference; subsequent backends
    /// are diffed against it.
    #[arg(long, default_value = "reference,cpu")]
    pub backends: String,

    /// Prompt for `--component layer` (drives the actual forward pass).
    /// For `moe-expert`/`moe-block`, the prompt seeds a synthetic residual
    /// if provided; otherwise a deterministic sin-pattern is used.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Random-ish seed for the synthetic residual. Ignored when `--prompt`
    /// is set. Default 0 produces the canonical sin pattern.
    #[arg(long, default_value = "0")]
    pub seed: u32,

    /// Max element-wise abs diff allowed before declaring divergence. The
    /// right value depends on component depth — per-expert ≈ 1e-3, full
    /// forward needs more headroom for accumulated f32 noise.
    #[arg(long, default_value = "1e-3")]
    pub tolerance: f64,

    /// Print intermediate values at each checkpoint, not just diffs.
    #[arg(long, short)]
    pub verbose: bool,
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
pub fn run(_args: ParityArgs) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "`larql parity` requires the `gpu` feature on macOS — Metal is the reference \
         backend this command compares CPU output against."
            .into(),
    )
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
pub fn run(args: ParityArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !COMPONENTS.contains(&args.component.as_str()) {
        return Err(format!(
            "unknown --component '{}'. Available: {}",
            args.component,
            COMPONENTS.join(", ")
        )
        .into());
    }

    // `layer` component always uses metal+cpu internally; other components
    // need the backends list validated and require ≥2.
    if args.component != "layer" {
        let backends: Vec<&str> = args.backends.split(',').map(|s| s.trim()).collect();
        for b in &backends {
            if !BACKENDS.contains(b) {
                return Err(format!(
                    "unknown backend '{}'. Available: {}",
                    b,
                    BACKENDS.join(", ")
                )
                .into());
            }
        }
        if backends.len() < 2 {
            return Err("need at least 2 backends to diff (default is `reference,cpu`)".into());
        }
    }

    // ── Resolve + load vindex ────────────────────────────────────────────────
    let path = cache::resolve_model(&args.model)?;
    let config = load_vindex_config(&path)?;
    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&path, &mut cb)?;
    let arch = &*weights.arch;

    println!("Vindex:    {}", path.display());
    println!("Model:     {}", config.model);
    println!("Component: {}", args.component);
    println!("Layer:     {}", args.layer);
    println!();

    if args.component == "layer" {
        return run_layer_diff(&path, &config, &args);
    }

    // lm-head parity is backend-agnostic (Q4_K matvec vs f32 reference) —
    // works on any vindex that has an lm_head, MoE or dense. The moe-*
    // components need an expert store, which pure MoE (GPT-OSS, OLMoE,
    // GraniteMoE) has exactly as hybrid does — gating on hybrid alone was
    // the `is_hybrid_moe()`-only assumption this codebase keeps finding.
    if !(arch.is_moe() || arch.is_hybrid_moe()) && args.component != "lm-head" {
        return Err(format!(
            "vindex {} is not MoE — moe-* components are MoE-only",
            args.model
        )
        .into());
    }

    let backends: Vec<&str> = args.backends.split(',').map(|s| s.trim()).collect();
    println!("Backends:  {}", backends.join(" → "));
    println!();

    match args.component.as_str() {
        "moe-expert" => run_moe_expert(&config, &weights, &args, &backends),
        "moe-block" => run_moe_block(&config, &weights, &args, &backends),
        "lm-head" => run_lm_head(&path, &config, &weights, &args, &backends),
        _ => unreachable!("validated above"),
    }
}

// ── lm-head: Q4_K-vs-reference logits for the final projection ───────────────
//
// Diagnostic motivation: a 2026-04-27 silent-corruption bug had the writer
// emit Q4_K (`format/weights/write_kquant`) while `lm_head_knn_backend` dispatched
// `q4_matvec` (Q4_0). Same byte-rate per element (0.5625 B/elem) → identical
// file size → no validation caught the format collision → multilingual
// gibberish under `--metal`. This component diffs the actual on-disk Q4_K
// lm_head against an f32 reference computed from `weights.lm_head` (the model's
// HF-loaded tied embedding for Gemma 3/4 / Llama-tied / etc.). Any future
// format swap (Q4_K → Q4_KF, transposition, scale offset, ...) makes the
// top-1 token mismatch loud.

fn run_lm_head(
    path: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    weights: &larql_models::ModelWeights,
    args: &ParityArgs,
    backends: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_compute::CpuBackend;
    use larql_vindex::SilentLoadCallbacks;

    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    println!("hidden={hidden}, vocab={vocab}");

    // Build the same residual the moe-block / moe-expert variants use so a
    // cross-component diff at the same prompt seed is straightforward.
    let h = make_residual(hidden, args.seed);

    // Reference: f32 dot product against `weights.lm_head` (tied embedding
    // for Gemma 3 / Gemma 4 / Llama; explicit lm_head row for untied).
    let lm = &weights.lm_head;
    if lm.is_empty() {
        return Err("model has no lm_head loaded — re-run extract with weights enabled".into());
    }
    let ref_scores: Vec<f32> = lm
        .rows()
        .into_iter()
        .map(|row| row.iter().zip(h.iter()).map(|(a, b)| a * b).sum())
        .collect();

    // Vindex side: load the index *here* (separately from the f32 weights
    // load that load_model_weights_kquant did) so we exercise the production
    // `open_inference_vindex` path including `load_lm_head_kquant`.
    let mut cb = SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(path, &mut cb)?;
    let _ = index.load_lm_head(path);
    let _ = index.load_lm_head_kquant(path);
    let has_q4 = index.has_lm_head_kquant();
    let has_full = index.has_lm_head();
    println!(
        "lm_head sources: q4_mmap={has_q4}  f32_mmap={has_full}  tied_embed={}",
        weights.lm_head.shape()[0] == config.vocab_size
    );

    // The cpu backend's lm_head_knn_backend does Q4_K matvec when the
    // q4 mmap is present, falls back to f16 mmap, then f32 BLAS. We
    // diff each available source against the reference so a regression
    // in any one path stands out.
    let cpu = CpuBackend;
    let h1d = ndarray::Array1::from_vec(h.clone());

    let mut traces: Vec<(&str, Vec<f32>)> = vec![("reference (f32 dot)", ref_scores.clone())];

    if backends.contains(&"cpu") {
        let hits = index.lm_head_knn_backend(&h1d, vocab.min(8), &cpu);
        if !hits.is_empty() {
            // hits is (token, score) sorted descending. Reconstruct a
            // sparse score vector for the diff helper.
            let mut sparse = vec![f32::NEG_INFINITY; vocab];
            for (tok, score) in &hits {
                sparse[*tok as usize] = *score;
            }
            traces.push(("cpu (lm_head_knn_backend)", sparse));
        } else {
            println!(
                "  WARN: lm_head_knn_backend returned empty — vindex has no lm_head sources \
                 (no lm_head_q4.bin, no lm_head.bin, no f16 mmap), and tied-embed fallback \
                 lives in larql-inference. Re-run via `larql run` for the production path."
            );
        }
    }

    println!();
    println!("=== lm-head top-1 token comparison ===");
    let (ref_name, ref_v) = &traces[0];
    let ref_top1 = argmax(ref_v);
    println!("  {ref_name:<28}  top-1 token = {ref_top1}");
    for (name, v) in traces.iter().skip(1) {
        let top1 = argmax(v);
        let verdict = if top1 == ref_top1 {
            "✓ matches reference"
        } else {
            "✗ DIFFERENT TOP-1 — likely format mismatch (Q4_K vs Q4_0, transposition, ...)"
        };
        println!("  {name:<28}  top-1 token = {top1}   {verdict}");
    }
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Vindex helpers ────────────────────────────────────────────────────────────

fn expert_bytes(
    weights: &larql_models::ModelWeights,
    layer: usize,
    expert: usize,
) -> Result<(&[u8], &[u8]), Box<dyn std::error::Error>> {
    let gu_key = per_layer_ffn_key(layer, expert, PER_LAYER_FFN_GATE_UP);
    let dn_key = per_layer_ffn_key(layer, expert, PER_LAYER_FFN_DOWN);
    let gu = weights
        .get_packed_bytes(&gu_key)
        .ok_or_else(|| format!("missing per-layer entry: {gu_key}"))?;
    let dn = weights
        .get_packed_bytes(&dn_key)
        .ok_or_else(|| format!("missing per-layer entry: {dn_key}"))?;
    Ok((gu, dn))
}

fn pre_experts_norm_for(weights: &larql_models::ModelWeights, layer: usize) -> &[f32] {
    weights
        .arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

fn post_experts_norm_for(weights: &larql_models::ModelWeights, layer: usize) -> &[f32] {
    weights
        .arch
        .moe_post_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[])
}

fn router_proj_for(
    weights: &larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let key = arch
        .moe_router_key(layer)
        .ok_or("arch has no router_proj key for this layer")?;
    weights
        .vectors
        .get(&key)
        .cloned()
        .ok_or_else(|| format!("router_proj not found in weights: {key}").into())
}

fn router_per_expert_scale_for(
    weights: &larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Vec<f32> {
    arch.moe_router_per_expert_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .cloned()
        .unwrap_or_default()
}

fn router_norm_for(
    weights: &larql_models::ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Vec<f32> {
    arch.moe_router_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .cloned()
        .unwrap_or_default()
}

fn activation_for(arch: &dyn larql_models::ModelArchitecture) -> Activation {
    match arch.activation() {
        larql_models::Activation::GeluTanh => Activation::GeluTanh,
        _ => Activation::Silu,
    }
}

fn make_residual(hidden: usize, seed: u32) -> Vec<f32> {
    // Deterministic per-(hidden, seed) sin pattern. seed=0 reproduces the
    // canonical pattern used by the bench / parity tests.
    let phase = (seed as f32) * 0.001;
    (0..hidden)
        .map(|i| ((i as f32 + 1.0) * 0.0007 + phase).sin())
        .collect()
}

// ── Diff reporter ─────────────────────────────────────────────────────────────

fn diff_against_first(traces: &[(&str, Vec<f32>)], tolerance: f64) {
    let (ref_name, ref_v) = &traces[0];
    println!(
        "Reference backend: {ref_name}  (first {} elems used as the truth)",
        ref_v.len()
    );
    let n = ref_v.len();
    print!("  {ref_name:<10} [0..3] = [");
    for (i, x) in ref_v.iter().take(3).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:+.4e}", x);
    }
    println!("]");

    for (name, v) in traces.iter().skip(1) {
        if v.len() != n {
            println!(
                "  {name:<10} LENGTH MISMATCH: ref.len={n}, {name}.len={}",
                v.len()
            );
            continue;
        }
        let mut max_abs = 0.0f64;
        let mut max_idx = 0;
        let mut max_a = 0.0f32;
        let mut max_b = 0.0f32;
        let mut nan = 0;
        for (i, (a, b)) in ref_v.iter().zip(v.iter()).enumerate() {
            if a.is_nan() || b.is_nan() {
                nan += 1;
                continue;
            }
            let d = ((a - b) as f64).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
                max_a = *a;
                max_b = *b;
            }
        }
        let verdict = if max_abs < tolerance {
            "✓ within tolerance"
        } else if max_abs < tolerance * 100.0 {
            "⚠ small drift"
        } else {
            "✗ DIVERGENCE"
        };
        print!("  {name:<10} [0..3] = [");
        for (i, x) in v.iter().take(3).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:+.4e}", x);
        }
        println!("]");
        println!(
            "             max |Δ|={:.3e}  at idx {}  (ref={:+.4e}, {name}={:+.4e})  {verdict}",
            max_abs, max_idx, max_a, max_b
        );
        if nan > 0 {
            println!("             NaN count: {nan}");
        }
    }
}

fn dump3(label: &str, v: &[f32]) {
    let n = v.len().min(3);
    print!("  {label}: [");
    for (i, x) in v.iter().take(n).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:+.6e}", x);
    }
    if v.len() > n {
        print!(", …]  ({} elems)", v.len());
    } else {
        print!("]");
    }
    println!();
}
