//! Head-to-head: LARQL vs Ollama tok/s on the same machine, same moment.
//!
//! Runs LARQL decode (Q4_K, Q8, raw kernel) then queries Ollama's API,
//! prints a single comparison table. This is THE benchmark to run.
//!
//! Usage: cargo run --release --features gpu -p larql-compute --example compare_ollama
//!
//! Requires: ollama running locally with gemma3:4b loaded.
//!
//! Layout: `compare_ollama/layers.rs` (synthetic weights + pipeline
//! layers), `kernel_bench.rs` (isolated raw-kernel timings),
//! `ollama.rs` (the live query), `report.rs` (table + analysis).

extern crate blas_src;

#[cfg(target_os = "macos")]
#[path = "compare_ollama/kernel_bench.rs"]
mod kernel_bench;
#[cfg(target_os = "macos")]
#[path = "compare_ollama/layers.rs"]
mod layers;
#[cfg(target_os = "macos")]
#[path = "compare_ollama/ollama.rs"]
mod ollama;
#[cfg(target_os = "macos")]
#[path = "compare_ollama/report.rs"]
mod report;

/// Timed decode iterations per arm.
#[cfg(target_os = "macos")]
const TIMED_ITERS: usize = 20;
/// Untimed warm-up decodes for the 21-layer arms.
#[cfg(target_os = "macos")]
const WARMUP_SHORT: usize = 5;
/// Untimed warm-up decodes for the 34-layer arms.
#[cfg(target_os = "macos")]
const WARMUP_FULL: usize = 3;
/// Layers in the short decode arms.
#[cfg(target_os = "macos")]
const SHORT_LAYERS: usize = 21;

/// Reset the KV cache, warm up, then time `n` decodes; ms per token.
#[cfg(target_os = "macos")]
fn time_decode(
    metal: &dyn larql_compute::prelude::ComputeBackend,
    layers: &[larql_compute::FullPipelineLayer],
    x: &[f32],
    warmup: usize,
    n: usize,
) -> f64 {
    use layers::{HIDDEN, INTER};
    metal.reset_kv_cache();
    for _ in 0..warmup {
        let _ = metal.decode_token(layers, x, HIDDEN, INTER);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = metal.decode_token(layers, x, HIDDEN, INTER);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / n as f64
}

fn main() {
    #[cfg(not(target_os = "macos"))]
    {
        println!("Run on macOS with --features gpu");
    }

    #[cfg(target_os = "macos")]
    {
        use larql_compute::prelude::*;
        use layers::{build_layers, pipeline_layers, AttnWeights, HIDDEN};

        let metal_raw = larql_compute_metal::MetalBackend::new().expect("Metal required");
        let metal: &dyn ComputeBackend = &metal_raw;
        let n = TIMED_ITERS;

        println!("╔═══════════════════════════════════════════════════╗");
        println!("║         LARQL vs Ollama — Head to Head            ║");
        println!("╚═══════════════════════════════════════════════════╝");
        println!();
        println!("  Machine:  M3 Max, macOS");
        println!("  Model:    Gemma 3 4B (hidden=2560, inter=10240)");
        println!();

        let x: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 * 0.001).sin()).collect();

        // ── LARQL Q4_K decode (21 layers) ──
        let data_21 = build_layers(SHORT_LAYERS);
        let q4k_21 = pipeline_layers(&data_21, AttnWeights::Q4K);
        let q4k_21_ms = time_decode(metal, &q4k_21, &x, WARMUP_SHORT, n);

        // ── LARQL Q8 decode (21 layers) ──
        let q8_21 = pipeline_layers(&data_21, AttnWeights::Q8);
        let q8_21_ms = time_decode(metal, &q8_21, &x, WARMUP_SHORT, n);

        // ── LARQL Q4_K decode (34 layers) ──
        let data_34 = build_layers(kernel_bench::LAYERS);
        let q4k_34 = pipeline_layers(&data_34, AttnWeights::Q4K);
        let q4k_34_ms = time_decode(metal, &q4k_34, &x, WARMUP_FULL, n);

        // ── LARQL Q4_KF (full attention) decode (21 + 34 layers) ──
        //
        // The headline-fastest path on Gemma 3 4B per the README — uses
        // the llama.cpp-exact `q4kf_proj` / `q4kf_qkv_proj` kernel for
        // attention as well as FFN. The Q4_K variants above keep
        // attention as the GGUF-default Q4_K layout; flipping to Q4_KF
        // reuses the same f32-input fused matvec kernel for every
        // projection, which on M3 measures faster than the Q4_K-attn
        // dual-path.
        let q4kf_21 = pipeline_layers(&data_21, AttnWeights::Q4KF);
        let q4kf_21_ms = time_decode(metal, &q4kf_21, &x, WARMUP_SHORT, n);
        let q4kf_34 = pipeline_layers(&data_34, AttnWeights::Q4KF);
        let q4kf_34_ms = time_decode(metal, &q4kf_34, &x, WARMUP_FULL, n);

        // ── LARQL raw QKV kernel (34 layers, zero overhead) ──
        let raw_34_ms = kernel_bench::raw_qkv_ms(&metal_raw, &data_34[0], &x, n);

        // ── Isolated FFN / O-proj / dispatch-floor breakdown (34 layers) ──
        kernel_bench::print_component_breakdown(&metal_raw, &data_34[0], q4k_34_ms, raw_34_ms, n);

        // ── Ollama (live query) ──
        let timings = report::Timings {
            ollama_ms: ollama::ollama_ms_per_token(),
            q4k_21_ms,
            q4kf_21_ms,
            q8_21_ms,
            q4k_34_ms,
            q4kf_34_ms,
            raw_34_ms,
        };

        // ── Results + analysis ──
        report::print_table(&timings);
        report::print_analysis(&timings);
    }
}
