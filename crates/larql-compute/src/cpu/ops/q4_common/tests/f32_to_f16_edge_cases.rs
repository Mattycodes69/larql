//! f32_to_f16 edge cases

use super::*;

#[test]
fn f32_to_f16_normal_round_trip() {
    // 1.0, -1.0, 0.5: all representable exactly in f16
    for &val in &[1.0f32, -1.0, 0.5, -0.5, 2.0] {
        let bits = super::super::f32_to_f16(val);
        let back = f16_to_f32(bits);
        assert!(
            (back - val).abs() < 1e-3,
            "round-trip failed for {val}: got {back}"
        );
    }
}

#[test]
fn f32_to_f16_infinity() {
    let inf_bits = super::super::f32_to_f16(f32::INFINITY);
    let back = f16_to_f32(inf_bits);
    assert!(
        back.is_infinite() && back > 0.0,
        "expected +inf, got {back}"
    );

    let neg_inf_bits = super::super::f32_to_f16(f32::NEG_INFINITY);
    let neg_back = f16_to_f32(neg_inf_bits);
    assert!(
        neg_back.is_infinite() && neg_back < 0.0,
        "expected -inf, got {neg_back}"
    );
}

#[test]
fn f32_to_f16_large_value_clamps_to_infinity() {
    // 1e30 is beyond f16 max (~65504) → should return f16 infinity
    let bits = super::super::f32_to_f16(1e30f32);
    let back = f16_to_f32(bits);
    assert!(
        back.is_infinite(),
        "1e30 → f16 should be infinity, got {back}"
    );
}

#[test]
fn f32_to_f16_subnormal_range() {
    // 1e-10 is below f16 normal range (min normal ≈ 6.1e-5) → subnormal or zero f16
    let bits = super::super::f32_to_f16(1e-10f32);
    let back = f16_to_f32(bits);
    // Should be small (subnormal or zero), not a normal f16 value
    assert!(
        back.abs() < 1e-4,
        "1e-10 → f16 back-conversion {back} should be very small"
    );
}

#[test]
fn f32_to_f16_denormal_f32_input() {
    // f32 denormal (exp == 0) → f32_to_f16 should return signed zero
    let denormal = f32::from_bits(1u32); // smallest positive f32 denormal
    let bits = super::super::f32_to_f16(denormal);
    // exp == 0 path returns sign as u16, which for positive is 0
    assert_eq!(bits, 0, "f32 denormal should encode as f16 zero");
}

#[test]
fn q4_k_round_trip_matches_larql_models_decoder() {
    // Cross-check against the authoritative decoder in larql-models.
    // Guards against silent drift between the quantizer here and the
    // dequantizer every caller actually uses (kquant_forward.rs, vindex
    // weight load, etc.). 3 super-blocks, a mix of positive/negative.
    let data: Vec<f32> = (0..256 * 3)
        .map(|i| ((i as f32 - 383.0) / 127.0).sin())
        .collect();
    let bytes = quantize_q4_k(&data);
    assert_eq!(bytes.len(), 144 * 3);

    let decoded =
        larql_models::quant::ggml::dequantize_q4_k(&bytes, 256 * 3).expect("dequantize_q4_k");
    assert_eq!(decoded.len(), 256 * 3);

    let max_err = data
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.15,
        "cross-crate Q4_K round-trip max error {max_err} > 0.15 — \
         quantize_q4_k in larql-compute disagrees with \
         larql_models::quant::ggml::dequantize_q4_k (PR #24 llama.cpp format)"
    );
}

#[test]
fn f32_to_f16_valid_f16_subnormal() {
    // 1e-7 maps to new_exp ≈ -9 → shift = 10 → total_shift = 23 < 24
    // so it encodes as a nonzero f16 subnormal rather than clamping to zero.
    let bits = super::super::f32_to_f16(1e-7f32);
    let back = f16_to_f32(bits);
    // Must be a small positive subnormal, not zero.
    assert!(
        back > 0.0,
        "1e-7 should encode as nonzero f16 subnormal, got {back}"
    );
    assert!(
        back < 1e-4,
        "1e-7 encoded as f16 subnormal should still be small, got {back}"
    );
}

#[test]
fn quantize_q4k_all_zero_covers_d_zero_branch() {
    // All-zero data → global_max_range = 0 → d = 0 branch; global_min = 0 → dmin = 0 branch.
    // Also exercises f16_to_f32(0) in the decoder (mant==0, sign==0 path).
    let data = vec![0.0f32; 256];
    let q4k = quantize_q4_k(&data);
    assert_eq!(q4k.len(), 144);
    // Decoding should also produce all zeros.
    let decoded = dequantize_q4_k_llama(&q4k, 256);
    assert!(
        decoded.iter().all(|&v| v == 0.0),
        "all-zero encode/decode should stay zero"
    );
}

#[test]
fn quantize_q4k_all_positive_covers_dmin_zero() {
    // All-positive data → global_min = 0 → dmin = 0 branch (no negative offset needed).
    let data = vec![1.0f32; 256];
    let q4k = quantize_q4_k(&data);
    assert_eq!(q4k.len(), 144);
    // dmin bytes should encode f16 zero.
    let dmin_bits = u16::from_le_bytes([q4k[2], q4k[3]]);
    assert_eq!(
        dmin_bits, 0,
        "all-positive data should produce dmin=0 (f16 zero)"
    );
}

#[test]
fn quantize_q6k_all_zero_covers_d_zero_branch() {
    // All-zero data → d = 0 branch; all sub-block scales = 0.
    let data = vec![0.0f32; 256];
    let q6k = quantize_q6_k(&data);
    assert_eq!(q6k.len(), 210);
    // f16 super-block scale at bytes [208..210] should be zero.
    let d_bits = u16::from_le_bytes([q6k[208], q6k[209]]);
    assert_eq!(d_bits, 0, "all-zero data should produce d=0 (f16 zero)");
}

#[test]
#[should_panic(expected = "multiple of 256")]
fn quantize_q6k_rejects_non_aligned() {
    let _ = quantize_q6_k(&vec![1.0f32; 255]);
}
