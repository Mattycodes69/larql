use super::*;
use crate::accuracy::cosine_similarity;

/// TurboQuant's codebooks are optimised for unit-norm vectors (the natural
/// distribution of K/V heads after QK-norm), so unit-norm inputs measure
/// the codec at its operating point (cos ≈ 0.9954 at 4-bit).
/// Generate a unit-norm vector using a simple LCG (no external rand dep).
/// Uses lower 32 bits of the state for uniform [0, 1) values.
fn unit_norm_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let raw: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect();
    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-12 {
        raw.iter().map(|v| v / norm).collect()
    } else {
        raw
    }
}

// ── Measured codec quality floors (2026-07-30, unit-sigma codebooks) ─────
//
// Mean round-trip values over 500 Gaussian-direction unit vectors
// (`codec_mean_cosine_and_norm_meet_floor`):
//   4-bit: mean cos 0.9954–0.9960 across d ∈ {32, 64, 128, 256}
//   3-bit: mean cos 0.9830–0.9844
//   mean decoded norm ≥ 0.9836 at both bit-widths
// The mis-scaled codebooks (sigma trained √2 too small + silent D256
// fallback for d ∉ {128, 256}) gave 0.9908 at 4-bit d=256, 0.947 at
// d=64, 0.919 at d=32, and decoded norms of 0.95 / 0.70 / 0.55.
// Floors sit just under today's measured values so a codebook
// regression trips the assert instead of hiding under slack.
const MEAN_COS_FLOOR_4BIT: f64 = 0.995;
const MEAN_COS_FLOOR_3BIT: f64 = 0.982;
const MEAN_NORM_FLOOR: f64 = 0.98;
/// Single-vector floors for the LCG fixtures below (measured
/// 2026-07-30: 0.9955 / 0.9956 at 4-bit, 0.9839 at 3-bit). The LCG
/// produces structured low-bit coordinates, so these are
/// deterministic per (dim, seed).
const MIN_COS_4BIT: f64 = 0.995;
const MIN_COS_3BIT: f64 = 0.982;

/// Gaussian-direction unit vectors at every supported block dim.
/// Pins the accuracy floor AND the absence of the old
/// d ∉ {128, 256} fallback cliff.
#[test]
fn codec_mean_cosine_and_norm_meet_floor() {
    use rand::prelude::*;
    use rand_distr::Normal;
    const N_VECTORS: usize = 500;
    for bits in [3u8, 4] {
        for d in [32usize, 64, 128, 256] {
            let tq = TurboQuant::new(bits);
            let mut rng = StdRng::seed_from_u64(777);
            let dist = Normal::new(0.0f32, 1.0).unwrap();
            let mut cos_sum = 0.0f64;
            let mut norm_sum = 0.0f64;
            for _ in 0..N_VECTORS {
                let raw: Vec<f32> = (0..d).map(|_| rng.sample(dist)).collect();
                let nrm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
                let x: Vec<f32> = raw.iter().map(|v| v / nrm).collect();
                let dec = tq.decode_vector(&tq.encode_vector(&x), d);
                cos_sum += cosine_similarity(&x, &dec);
                norm_sum += dec.iter().map(|v| v * v).sum::<f32>().sqrt() as f64;
            }
            let mean_cos = cos_sum / N_VECTORS as f64;
            let mean_norm = norm_sum / N_VECTORS as f64;
            let cos_floor = match bits {
                4 => MEAN_COS_FLOOR_4BIT,
                _ => MEAN_COS_FLOOR_3BIT,
            };
            assert!(
                mean_cos > cos_floor,
                "bits={bits} d={d}: mean cos {mean_cos:.4} < floor {cos_floor}"
            );
            assert!(
                mean_norm > MEAN_NORM_FLOOR,
                "bits={bits} d={d}: mean decoded norm {mean_norm:.4} < floor {MEAN_NORM_FLOOR}"
            );
        }
    }
}

// ── Codec roundtrip quality ───────────────────────────────────────────────

#[test]
fn encode_decode_4bit_cosine_near_one() {
    let tq = TurboQuant::new(4);
    let x = unit_norm_vec(256, 42);
    let enc = tq.encode_vector(&x);
    let dec = tq.decode_vector(&enc, 256);
    let cos = cosine_similarity(&x, &dec);
    assert!(cos > MIN_COS_4BIT, "4-bit cosine {cos:.4} < {MIN_COS_4BIT}");
}

#[test]
fn encode_decode_3bit_cosine_acceptable() {
    let tq = TurboQuant::new(3);
    let x = unit_norm_vec(256, 99);
    let enc = tq.encode_vector(&x);
    let dec = tq.decode_vector(&enc, 256);
    let cos = cosine_similarity(&x, &dec);
    assert!(cos > MIN_COS_3BIT, "3-bit cosine {cos:.4} < {MIN_COS_3BIT}");
}

#[test]
fn encode_decode_dim128_roundtrip() {
    let tq = TurboQuant::new(4);
    let x = unit_norm_vec(128, 7);
    let enc = tq.encode_vector(&x);
    let dec = tq.decode_vector(&enc, 128);
    let cos = cosine_similarity(&x, &dec);
    assert!(
        cos > MIN_COS_4BIT,
        "4-bit d=128 cosine {cos:.4} < {MIN_COS_4BIT}"
    );
}

/// d = 64 and d = 32 used to fall back to the D256 table silently
/// (cos 0.92–0.95); the sigma-scaled unit codebook must serve them
/// at full quality. Per-vector cosine variance grows as d shrinks
/// (a couple of tail coordinates dominate at d = 32), so the
/// single-fixture floor sits below the d ≥ 128 one: measured
/// 2026-07-30 at 0.9967 (d=64, seed 21) and 0.9762 (d=32, seed 5).
const MIN_COS_4BIT_SMALL_DIM: f64 = 0.975;

#[test]
fn encode_decode_small_dims_full_quality() {
    let tq = TurboQuant::new(4);
    for (dim, seed) in [(64usize, 21u64), (32, 5)] {
        let x = unit_norm_vec(dim, seed);
        let enc = tq.encode_vector(&x);
        let dec = tq.decode_vector(&enc, dim);
        let cos = cosine_similarity(&x, &dec);
        assert!(
            cos > MIN_COS_4BIT_SMALL_DIM,
            "4-bit d={dim} cosine {cos:.4} < {MIN_COS_4BIT_SMALL_DIM}"
        );
    }
}

#[test]
fn norm_approximately_preserved() {
    let tq = TurboQuant::new(4);
    let x = unit_norm_vec(256, 13);
    let norm_orig: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let enc = tq.encode_vector(&x);
    let dec = tq.decode_vector(&enc, 256);
    let norm_dec: f32 = dec.iter().map(|v| v * v).sum::<f32>().sqrt();
    let ratio = norm_dec / norm_orig;
    // The codec stores the norm explicitly; the residual shortfall is
    // quantisation error only (measured ratio ≈ 0.995 at 4-bit,
    // 2026-07-30 — the mis-scaled codebooks clipped it to ≈ 0.95).
    const NORM_RATIO_TOL: f32 = 0.02;
    assert!(
        (ratio - 1.0).abs() < NORM_RATIO_TOL,
        "norm ratio {ratio:.4} outside 1.0 ± {NORM_RATIO_TOL}"
    );
}

#[test]
fn zero_vector_roundtrip_no_panic() {
    let tq = TurboQuant::new(4);
    let x = vec![0.0f32; 256];
    let enc = tq.encode_vector(&x);
    let dec = tq.decode_vector(&enc, 256);
    // Zero vector: all decoded values should be ~0 (codec stores norm=0).
    let max_abs = dec.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-6,
        "zero vector decoded to non-zero: max_abs={max_abs}"
    );
}

#[test]
fn identical_vectors_same_encoding() {
    let tq = TurboQuant::new(4);
    let x = unit_norm_vec(256, 55);
    let enc1 = tq.encode_vector(&x);
    let enc2 = tq.encode_vector(&x);
    assert_eq!(enc1, enc2, "encoding is not deterministic");
}

// ── Encoded byte size ────────────────────────────────────────────────────

#[test]
fn bytes_per_vector_4bit_dim256() {
    let tq = TurboQuant::new(4);
    // norm (4 bytes) + 256 × 4 bits / 8 = 4 + 128 = 132
    assert_eq!(tq.bytes_per_vector(256), 132);
}

#[test]
fn bytes_per_vector_3bit_dim256() {
    let tq = TurboQuant::new(3);
    // norm (4 bytes) + ceil(256 × 3 / 8) = 4 + 96 = 100
    assert_eq!(tq.bytes_per_vector(256), 100);
}

#[test]
fn bytes_per_vector_4bit_dim128() {
    let tq = TurboQuant::new(4);
    // 4 + 128 × 4 / 8 = 4 + 64 = 68
    assert_eq!(tq.bytes_per_vector(128), 68);
}

#[test]
fn compression_ratio_vs_fp16() {
    let tq = TurboQuant::new(4);
    // FP16 per dim=256 vector: 256 × 2 = 512 bytes
    // TurboQuant 4-bit: 132 bytes
    // Ratio: 512 / 132 ≈ 3.9×
    let fp16_bytes = 256 * 2;
    let tq_bytes = tq.bytes_per_vector(256);
    let ratio = fp16_bytes as f64 / tq_bytes as f64;
    assert!(ratio > 3.5, "compression ratio {ratio:.2} < 3.5");
}

// ── Engine construction and config ────────────────────────────────────────

#[test]
fn engine_name_and_config_4bit() {
    let eng = TurboQuantEngine::new(4);
    assert_eq!(eng.name(), "turbo-quant");
    let info = eng.info();
    assert_eq!(info.config, "bits=4");
    assert!(info.backend.starts_with("cpu"));
    assert!(info.description.contains("4-bit"));
}

#[test]
fn engine_name_and_config_3bit() {
    let eng = TurboQuantEngine::new(3);
    assert_eq!(eng.info().config, "bits=3");
    assert!(eng.info().description.contains("3-bit"));
}

#[test]
fn engine_memory_zero_before_prefill() {
    let eng = TurboQuantEngine::new(4);
    assert_eq!(eng.memory_bytes(), 0);
}

#[test]
fn engine_summary_shows_bits_in_config() {
    let eng = TurboQuantEngine::new(4);
    let s = eng.info().summary();
    assert!(s.contains("turbo-quant"), "summary missing name: {s}");
    assert!(s.contains("bits=4"), "summary missing config: {s}");
}

// ── CompressedLayer memory accounting ────────────────────────────────────

#[test]
fn compressed_layer_memory_is_smaller_than_fp32() {
    use ndarray::Array2;
    let tq = TurboQuant::new(4);
    // Single K/V pair: 10 positions, kv_dim=1024 (Gemma 3 4B-like)
    let k = Array2::<f32>::from_elem((10, 1024), 0.1);
    let v = Array2::<f32>::from_elem((10, 1024), 0.2);
    let cl = CompressedLayer::compress(&(k, v), &tq);
    let fp32_bytes = 10 * 1024 * 4 * 2; // K+V, f32
    let compressed = cl.memory_bytes();
    assert!(
        compressed < fp32_bytes,
        "compressed {compressed}B should be < fp32 {fp32_bytes}B"
    );
    // Compression ratio should be ~4×
    let ratio = fp32_bytes as f64 / compressed as f64;
    assert!(ratio > 3.0, "ratio {ratio:.2} < 3.0");
}

#[test]
fn compressed_layer_roundtrip_cosine() {
    use ndarray::Array2;
    let tq = TurboQuant::new(4);
    // Use unit-norm rows matching TurboQuant's codebook distribution.
    let k_data: Vec<f32> = (0..10)
        .flat_map(|i| unit_norm_vec(256, i * 7 + 17))
        .collect();
    let v_data: Vec<f32> = (0..10)
        .flat_map(|i| unit_norm_vec(256, i * 7 + 31))
        .collect();
    let k = Array2::from_shape_vec((10, 256), k_data.clone()).unwrap();
    let v = Array2::from_shape_vec((10, 256), v_data.clone()).unwrap();
    let cl = CompressedLayer::compress(&(k, v), &tq);
    let (k_dec, v_dec) = cl.decompress(&tq);
    // Check last row cosine (most relevant for decode) on both K and V.
    let k_orig_last: Vec<f32> = k_data[9 * 256..10 * 256].to_vec();
    let k_dec_last: Vec<f32> = k_dec.row(9).to_vec();
    let k_cos = cosine_similarity(&k_orig_last, &k_dec_last);
    assert!(
        k_cos > MIN_COS_4BIT,
        "K roundtrip cosine {k_cos:.4} < {MIN_COS_4BIT}"
    );
    let v_orig_last: Vec<f32> = v_data[9 * 256..10 * 256].to_vec();
    let v_dec_last: Vec<f32> = v_dec.row(9).to_vec();
    let v_cos = cosine_similarity(&v_orig_last, &v_dec_last);
    assert!(
        v_cos > MIN_COS_4BIT,
        "V roundtrip cosine {v_cos:.4} < {MIN_COS_4BIT}"
    );
}

// ── Block-dim validation ─────────────────────────────────────────────────

#[test]
fn resolve_block_dim_accepts_power_of_two_splits() {
    assert_eq!(resolve_block_dim(1024).unwrap(), 256);
    assert_eq!(resolve_block_dim(128).unwrap(), 128);
    assert_eq!(resolve_block_dim(96).unwrap(), 32);
    assert_eq!(resolve_block_dim(64).unwrap(), 64);
}

/// kv_dim = 80 has no power-of-two head split in the supported set;
/// it used to fall through to the whole-row fallback and panic on
/// the WHT's power-of-two assert mid-prefill.
#[test]
fn resolve_block_dim_rejects_non_power_of_two() {
    let err = resolve_block_dim(80).unwrap_err();
    assert!(
        matches!(err, EngineError::InvariantViolation { ref what } if what.contains("power-of-two")),
        "expected InvariantViolation naming the power-of-two constraint, got {err:?}"
    );
}
