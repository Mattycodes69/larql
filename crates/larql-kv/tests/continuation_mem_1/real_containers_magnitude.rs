//! real containers (magnitude)

use super::*;

#[test]
#[ignore = "real container: LARQL_MEM1_QWEN"]
fn real_dense_qwen3_0_6b() {
    real(
        "LARQL_MEM1_QWEN",
        "qwen3-0.6b",
        Journey {
            prefill: tokens(1000, 128),
            resume: tokens(3000, 32),
            decode: tokens(4000, 16),
        },
        false,
        false,
    );
}

#[test]
#[ignore = "real container: LARQL_MEM1_GEMMA (sequence crosses the sliding window)"]
fn real_sliding_gemma3_4b() {
    let n: usize = std::env::var("LARQL_MEM1_GEMMA_PREFILL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1100);
    real(
        "LARQL_MEM1_GEMMA",
        "gemma3-4b-it",
        Journey {
            prefill: tokens(1000, n),
            resume: tokens(3000, 16),
            decode: tokens(4000, 8),
        },
        false,
        true,
    );
}

#[test]
#[ignore = "real container: LARQL_MEM1_OUTE"]
fn real_convqkv_mamba2_oute_250m() {
    real(
        "LARQL_MEM1_OUTE",
        "oute-mamba2attn-250m",
        Journey {
            prefill: tokens(1000, 64),
            resume: tokens(3000, 16),
            decode: tokens(4000, 16),
        },
        true,
        false,
    );
}

#[test]
#[ignore = "real container: LARQL_MEM1_KIMI (short sequences only)"]
fn real_kda_mla_kimi_s7() {
    real(
        "LARQL_MEM1_KIMI",
        "kimi-linear-48b-s7",
        Journey {
            prefill: tokens(1000, 16),
            resume: tokens(3000, 4),
            decode: tokens(4000, 4),
        },
        true,
        false,
    );
}

/// D6: oute's recurrent-operator windows on the serial reference backend,
/// the integrity-clean source for M8's per-call copy clause.
#[test]
#[ignore = "real container: LARQL_MEM1_OUTE (serial reference backend)"]
fn real_recurrent_windows_oute_reference() {
    let _serial = serial();
    let dir = std::env::var_os("LARQL_MEM1_OUTE").expect("set LARQL_MEM1_OUTE");
    let subject = subjects::open(std::path::Path::new(&dir), "oute-mamba2attn-250m.reference");
    let journey = Journey {
        prefill: tokens(1000, 16),
        resume: tokens(3000, 4),
        decode: tokens(4000, 4),
    };
    let extra = json!({ "container": dir.to_string_lossy(), "purpose": "D6: integrity-clean recurrent windows" });
    measure_subject(&subject, &ReferenceBackend::new(), &journey, true, extra);
}
