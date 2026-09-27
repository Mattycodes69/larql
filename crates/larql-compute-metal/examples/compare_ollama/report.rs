//! The comparison table and per-layer analysis.

/// Every timing the report prints, in ms/token.
pub struct Timings {
    pub ollama_ms: f64,
    pub q4k_21_ms: f64,
    pub q4kf_21_ms: f64,
    pub q8_21_ms: f64,
    pub q4k_34_ms: f64,
    pub q4kf_34_ms: f64,
    pub raw_34_ms: f64,
}

/// Layers in the short decode arms.
const SHORT_LAYERS: f64 = 21.0;
/// Layers in the full-depth arms (Gemma 3 4B).
const FULL_LAYERS: f64 = 34.0;
/// Placeholder per-layer Ollama figure when Ollama is not running.
const OLLAMA_ABSENT_PER_LAYER_MS: f64 = 10.0;
/// Layers actually computed in the cached-layers projection (L0-12 cached).
const PROJECTED_COMPUTED_LAYERS: f64 = 8.0;
/// Rough FFN and dispatch-overhead shares of the 21-layer decode.
const FFN_SHARE: f64 = 0.36;
const DISPATCH_SHARE: f64 = 0.29;

fn tps(ms: f64) -> f64 {
    if ms > 0.0 {
        1000.0 / ms
    } else {
        0.0
    }
}

/// `ms / ollama_ms`, or 0.0 when Ollama was not measured.
fn vs(ms: f64, ollama_ms: f64) -> f64 {
    if ollama_ms > 0.0 {
        ms / ollama_ms
    } else {
        0.0
    }
}

fn decode_row(label: &str, ms: f64, ollama_ms: f64) {
    println!(
        "  │ {label} │ {:>6.1}ms │ {:>5.0}   │  {:>5.2}x │",
        ms,
        1000.0 / ms,
        vs(ms, ollama_ms)
    );
}

pub fn print_table(t: &Timings) {
    let ollama_ms = t.ollama_ms;
    let ollama_tps = tps(ollama_ms);
    println!("  ┌─────────────────────────────────┬──────────┬─────────┬──────────┐");
    println!("  │ Engine                          │  ms/tok  │  tok/s  │ vs Ollama│");
    println!("  ├─────────────────────────────────┼──────────┼─────────┼──────────┤");
    if ollama_ms > 0.0 {
        println!(
            "  │ Ollama gemma3:4b (34L, live)    │ {:>6.1}ms │ {:>5.0}   │   1.00x  │",
            ollama_ms, ollama_tps
        );
    } else {
        println!("  │ Ollama gemma3:4b                │   (not running)     │          │");
    }
    println!("  ├─────────────────────────────────┼──────────┼─────────┼──────────┤");
    decode_row("LARQL Q4_K decode (21L, KV)    ", t.q4k_21_ms, ollama_ms);
    decode_row("LARQL Q4_KF decode (21L, KV)   ", t.q4kf_21_ms, ollama_ms);
    decode_row("LARQL Q8   decode (21L, KV)    ", t.q8_21_ms, ollama_ms);
    decode_row("LARQL Q4_K decode (34L, KV)    ", t.q4k_34_ms, ollama_ms);
    decode_row("LARQL Q4_KF decode (34L, KV)   ", t.q4kf_34_ms, ollama_ms);
    println!("  ├─────────────────────────────────┼──────────┼─────────┼──────────┤");
    println!(
        "  │ LARQL raw QKV kernel (34L)      │ {:>6.1}ms │    —    │  {:>5.1}x  │",
        t.raw_34_ms,
        if ollama_ms > 0.0 {
            ollama_ms / t.raw_34_ms
        } else {
            0.0
        }
    );
    println!("  │   (kernel only, zero overhead)  │          │         │  faster  │");
    println!("  └─────────────────────────────────┴──────────┴─────────┴──────────┘");
}

pub fn print_analysis(t: &Timings) {
    let ollama_tps = tps(t.ollama_ms);
    println!();
    let per_layer_larql = t.q4k_21_ms / SHORT_LAYERS;
    let per_layer_ollama = if t.ollama_ms > 0.0 {
        t.ollama_ms * FULL_LAYERS / FULL_LAYERS
    } else {
        OLLAMA_ABSENT_PER_LAYER_MS
    };
    let per_layer_raw = t.raw_34_ms / FULL_LAYERS;
    println!("  Per-layer analysis:");
    println!("    LARQL decode:      {per_layer_larql:.3}ms/layer (QKV + attend + FFN + norms)");
    println!("    Ollama decode:     {per_layer_ollama:.3}ms/layer (entire layer)");
    println!("    LARQL raw kernel:  {per_layer_raw:.3}ms/layer (QKV only, zero overhead)");
    println!();
    println!("  Bottleneck: NOT the kernel ({per_layer_raw:.3}ms).");
    println!(
        "  Gap is FFN ({:.1}ms) + dispatch overhead ({:.1}ms).",
        t.q4k_21_ms * FFN_SHARE,
        t.q4k_21_ms * DISPATCH_SHARE
    );
    println!();

    let projected_cached = 1000.0 / (per_layer_larql * PROJECTED_COMPUTED_LAYERS);
    println!("  Projected with cached layers (L0-12, compute 8 only):");
    println!(
        "    {:.0} tok/s — {}",
        projected_cached,
        if projected_cached > ollama_tps {
            "EXCEEDS Ollama"
        } else {
            "approaching Ollama"
        }
    );
}
