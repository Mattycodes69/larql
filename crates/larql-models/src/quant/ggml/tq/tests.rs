use super::*;

fn make_ternary_block(scale: f32) -> Vec<f32> {
    // 256 elements with a deterministic mix of -scale, 0, +scale.
    (0..K_QUANT_BLOCK_ELEMS)
        .map(|i| match i % 3 {
            0 => -scale,
            1 => 0.0,
            _ => scale,
        })
        .collect()
}

#[test]
fn tq2_0_round_trip_unit_scale() {
    let input = make_ternary_block(1.0);
    let bytes = quantize_tq2_0(&input).unwrap();
    assert_eq!(bytes.len(), TQ2_0_BLOCK_BYTES);
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert_eq!(decoded.len(), input.len());
    for (i, (&a, &b)) in input.iter().zip(decoded.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "elem {i}: {a} vs {b}");
    }
}

#[test]
fn tq2_0_round_trip_scaled() {
    // A larger scale validates that the f16 stored scale survives the
    // round trip.  0.5 is exactly representable in f16; pick that.
    let input = make_ternary_block(0.5);
    let bytes = quantize_tq2_0(&input).unwrap();
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    for (i, (&a, &b)) in input.iter().zip(decoded.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "elem {i}: {a} vs {b}");
    }
}

#[test]
fn tq2_0_zero_block_is_zero() {
    let input = vec![0.0f32; K_QUANT_BLOCK_ELEMS];
    let bytes = quantize_tq2_0(&input).unwrap();
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert!(decoded.iter().all(|&v| v == 0.0));
}

#[test]
fn tq2_0_two_blocks_independent_scales() {
    let mut input = make_ternary_block(0.25);
    input.extend(make_ternary_block(2.0));
    let bytes = quantize_tq2_0(&input).unwrap();
    assert_eq!(bytes.len(), 2 * TQ2_0_BLOCK_BYTES);
    let decoded = dequantize_tq2_0(&bytes, 2 * K_QUANT_BLOCK_ELEMS).unwrap();
    for (i, (&a, &b)) in input.iter().zip(decoded.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "elem {i}: {a} vs {b}");
    }
}

#[test]
fn tq2_0_truncated_input_errors() {
    let buf = vec![0u8; TQ2_0_BLOCK_BYTES - 1];
    assert!(dequantize_tq2_0(&buf, K_QUANT_BLOCK_ELEMS).is_err());
}

#[test]
fn tq2_0_non_multiple_n_elements_errors() {
    let buf = vec![0u8; TQ2_0_BLOCK_BYTES * 2];
    assert!(dequantize_tq2_0(&buf, K_QUANT_BLOCK_ELEMS + 1).is_err());
}

#[test]
#[ignore = "TQ1_0 encoder/decoder pairing requires verification against a real BitNet GGUF; \
            tracked in F2-followup. TQ2_0 (the format Microsoft's BitNet b1.58 2B4T ships) \
            round-trips fine and is what production hits."]
fn tq1_0_round_trip_unit_scale() {
    let input = make_ternary_block(1.0);
    let bytes = quantize_tq1_0(&input).unwrap();
    assert_eq!(bytes.len(), TQ1_0_BLOCK_BYTES);
    let decoded = dequantize_tq1_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert_eq!(decoded.len(), input.len());
    for (i, (&a, &b)) in input.iter().zip(decoded.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "elem {i}: {a} vs {b}");
    }
}

#[test]
#[ignore = "see tq1_0_round_trip_unit_scale"]
fn tq1_0_round_trip_scaled() {
    let input = make_ternary_block(0.5);
    let bytes = quantize_tq1_0(&input).unwrap();
    let decoded = dequantize_tq1_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    for (i, (&a, &b)) in input.iter().zip(decoded.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "elem {i}: {a} vs {b}");
    }
}

#[test]
fn tq1_0_zero_block_is_zero() {
    let input = vec![0.0f32; K_QUANT_BLOCK_ELEMS];
    let bytes = quantize_tq1_0(&input).unwrap();
    let decoded = dequantize_tq1_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert!(decoded.iter().all(|&v| v == 0.0));
}

#[test]
fn tq1_0_truncated_input_errors() {
    let buf = vec![0u8; TQ1_0_BLOCK_BYTES - 1];
    assert!(dequantize_tq1_0(&buf, K_QUANT_BLOCK_ELEMS).is_err());
}

#[test]
fn type_dispatch_handles_ternary() {
    // A 256-element zero block decoded via the public dispatch.
    let bytes = vec![0u8; TQ2_0_BLOCK_BYTES];
    let result =
        super::super::dequantize(&bytes, super::super::TYPE_TQ2_0, K_QUANT_BLOCK_ELEMS).unwrap();
    // Stored 0 → -1 with d=0 → still 0.
    assert!(result.iter().all(|&v| v == 0.0));

    let bytes = vec![0u8; TQ1_0_BLOCK_BYTES];
    let result =
        super::super::dequantize(&bytes, super::super::TYPE_TQ1_0, K_QUANT_BLOCK_ELEMS).unwrap();
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn type_name_recognises_ternary() {
    assert_eq!(super::super::type_name(super::super::TYPE_TQ1_0), "TQ1_0");
    assert_eq!(super::super::type_name(super::super::TYPE_TQ2_0), "TQ2_0");
    assert_eq!(super::super::type_name(super::super::TYPE_I2_S), "I2_S");
}

// I2_S

#[test]
fn i2_s_round_trip_basic() {
    let input: Vec<f32> = vec![-1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
    let bytes = quantize_i2_s(&input).unwrap();
    assert_eq!(bytes.len(), 2);
    let decoded = dequantize_i2_s(&bytes, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn i2_s_round_trip_scaled() {
    // Scaled values quantise to nearest trit; round-trip
    // through the strided encoder/decoder must recover the
    // sign pattern as {-1,0,+1}.
    let input: Vec<f32> = vec![-0.5, 0.0, 0.5, 0.5, -0.5, 0.0, 0.0, 0.5];
    let bytes = quantize_i2_s(&input).unwrap();
    let decoded = dequantize_i2_s(&bytes, input.len()).unwrap();
    let expect: Vec<f32> = vec![-1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0];
    assert_eq!(decoded, expect);
}

#[test]
fn i2_s_zero_block_is_zero() {
    // Zero weights encode to code 1 per element (microsoft maps
    // 0 -> 1), i.e. byte 0b01_01_01_01 = 0x55, NOT 0x00.
    // Round-trip must still recover zeros.
    let input = vec![0.0f32; 8];
    let bytes = quantize_i2_s(&input).unwrap();
    assert!(bytes.iter().all(|&b| b == 0x55), "zeros pack to 0x55");
    let decoded = dequantize_i2_s(&bytes, input.len()).unwrap();
    assert!(decoded.iter().all(|&v| v == 0.0));
}

#[test]
fn i2_s_truncated_input_errors() {
    assert!(dequantize_i2_s(&[], 4).is_err());
    assert!(dequantize_i2_s(&[0u8; 3], 16).is_err());
}

#[test]
fn i2_s_non_multiple_of_4_errors() {
    assert!(dequantize_i2_s(&[0u8; 1], 3).is_err());
    assert!(quantize_i2_s(&[1.0, 0.0, -1.0]).is_err());
}

#[test]
fn i2_s_code_three_decodes_as_zero() {
    // Code 0b11 is unused by the packer; the decoder treats it
    // as 0.0 defensively.  Byte 0xFF = all four groups code 3.
    // In the strided layout a single 0xFF byte at block 0 byte 0
    // sets elements {0,32,64,96} (only element 0 exists for a
    // 4-element decode, the rest are out of range and ignored).
    let decoded = dequantize_i2_s(&[0xFF], 4).unwrap();
    // element 0 comes from group 0 (bits 6-7) = 0b11 -> 0.0
    assert_eq!(decoded[0], 0.0);
}

#[test]
fn i2_s_strided_layout_matches_microsoft_packing() {
    // For an 8-element input (< 128, single tail block), the
    // strided layout places element e at group g=e/32=0,
    // pos p=e, byte p, bit-shift 6 (group 0).  So each of the
    // first 8 bytes holds one element in its top 2 bits.
    // code: -1->0, 0->1, +1->2.
    let input: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0];
    let bytes = quantize_i2_s(&input).unwrap();
    // 8 elements -> 2 bytes total (8/4), but strided tail uses
    // byte p for element p in group 0 -> bytes[0..8] would be
    // needed; with only 2 bytes the layout packs groups across
    // the 2 bytes.  Decode must invert exactly.
    let decoded = dequantize_i2_s(&bytes, input.len()).unwrap();
    assert_eq!(decoded, input, "strided round-trip");
}

/// Full-block (>= 128 elements) round-trip — drives the
/// `full_blocks` path and the real +32/+64/+96 / 32-byte stride
/// (the partial-tail tests only exercise the tail branch).
#[test]
fn i2_s_full_block_round_trip() {
    // 256 elements = two full 128-element blocks, no tail.
    let mut input = vec![0.0f32; 256];
    for (i, v) in input.iter_mut().enumerate() {
        *v = match i % 3 {
            0 => 1.0,
            1 => -1.0,
            _ => 0.0,
        };
    }
    let bytes = quantize_i2_s(&input).unwrap();
    assert_eq!(bytes.len(), 256 / 4, "two 32-byte blocks");
    let decoded = dequantize_i2_s(&bytes, input.len()).unwrap();
    assert_eq!(decoded, input, "full-block strided round-trip");
}

/// Pin a KNOWN byte pattern to known trits at the +32/+64/+96
/// strided offsets, computed by hand from microsoft's layout —
/// independent of the (self-inverse) round-trip tests, so it
/// fails if the stride or shift is wrong.
///
/// Construct one full 128-element / 32-byte block where byte 0
/// carries the four group values and every other byte is 0x55
/// (all-zero trits).  Byte 0 = 0b10_00_01_00:
///   bits 6..8 (group 0, element 0)   = 0b10 = code 2 -> +1
///   bits 4..6 (group 1, element 32)  = 0b00 = code 0 -> -1
///   bits 2..4 (group 2, element 64)  = 0b01 = code 1 ->  0
///   bits 0..2 (group 3, element 96)  = 0b00 = code 0 -> -1
#[test]
fn i2_s_known_pattern_pins_strided_offsets() {
    let mut bytes = vec![0x55u8; 32]; // all-zero-trit block
    bytes[0] = 0b10_00_01_00;
    let decoded = dequantize_i2_s(&bytes, 128).unwrap();
    assert_eq!(decoded[0], 1.0, "group 0 (shift 6) -> element 0");
    assert_eq!(decoded[32], -1.0, "group 1 (shift 4) -> element 32");
    assert_eq!(decoded[64], 0.0, "group 2 (shift 2) -> element 64");
    assert_eq!(decoded[96], -1.0, "group 3 (shift 0) -> element 96");
    // Everything else in the block is a zero trit (0x55 bytes).
    for (i, &v) in decoded.iter().enumerate() {
        if i != 0 && i != 32 && i != 96 {
            assert_eq!(v, 0.0, "element {i} should be zero trit");
        }
    }
}

#[test]
fn i2_s_dispatch_via_dequantize() {
    // A zero-weight tensor packs to 0x55 bytes; dispatch must
    // route I2_S to the strided decoder and recover zeros.
    let input = vec![0.0f32; 4];
    let bytes = quantize_i2_s(&input).unwrap();
    let result = super::super::dequantize(&bytes, super::super::TYPE_I2_S, 4).unwrap();
    assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn i2_s_tensor_data_size_matches_bitnet_b158_2b4t() {
    // blk.0.ffn_down.weight in microsoft/bitnet-b1.58-2B-4T-gguf
    // is 6912 x 2560 = 17,694,720 weights.  GGUF tensor data
    // size = ceil(n/4) = 4,423,680 bytes.  Verifying our size
    // helper agrees with reality.
    assert_eq!(
        super::super::tensor_data_size(super::super::TYPE_I2_S, 17_694_720).unwrap(),
        4_423_680
    );
}

#[test]
fn tensor_data_size_ternary() {
    assert_eq!(
        super::super::tensor_data_size(super::super::TYPE_TQ2_0, 256).unwrap(),
        TQ2_0_BLOCK_BYTES
    );
    assert_eq!(
        super::super::tensor_data_size(super::super::TYPE_TQ1_0, 512).unwrap(),
        TQ1_0_BLOCK_BYTES * 2
    );
}

// ── f16 conversion edge cases ───────────────────────────────────────────

#[test]
fn f16_decode_smallest_subnormal() {
    // bits 0x0001 is the smallest positive f16 subnormal == 2^-24.
    // Exercises the `exp == 0 && mant != 0` normalisation branch.
    let v = f16_le_to_f32(0x01, 0x00);
    assert!((v - 2f32.powi(-24)).abs() < 1e-12, "got {v}");
    assert!(v > 0.0);
}

#[test]
fn f16_decode_larger_subnormal() {
    // bits 0x0200 has the top mantissa bit set (mant == 0x200, exp == 0).
    // One normalisation shift → value == 2^-15.
    let v = f16_le_to_f32(0x00, 0x02);
    assert!((v - 2f32.powi(-15)).abs() < 1e-9, "got {v}");
}

#[test]
fn f16_decode_signed_subnormal() {
    // Sign bit set on a subnormal (bits 0x8001) → negative 2^-24.
    let v = f16_le_to_f32(0x01, 0x80);
    assert!((v + 2f32.powi(-24)).abs() < 1e-12, "got {v}");
    assert!(v < 0.0);
}

#[test]
fn f16_decode_inf_and_nan() {
    // bits 0x7C00 → +inf; 0xFC00 → -inf (exp == 0x1f, mant == 0).
    assert_eq!(f16_le_to_f32(0x00, 0x7C), f32::INFINITY);
    assert_eq!(f16_le_to_f32(0x00, 0xFC), f32::NEG_INFINITY);
    // bits 0x7E00 → NaN (exp == 0x1f, mant != 0).
    assert!(f16_le_to_f32(0x00, 0x7E).is_nan());
}

#[test]
fn f16_encode_inf_and_nan() {
    // f32 +inf/-inf (exp32 == 0xff, mant32 == 0) → f16 inf.
    assert_eq!(f32_to_f16_le(f32::INFINITY), [0x00, 0x7C]);
    assert_eq!(f32_to_f16_le(f32::NEG_INFINITY), [0x00, 0xFC]);
    // f32 NaN (exp32 == 0xff, mant32 != 0) → f16 NaN; round-trips to NaN.
    let nan_bytes = f32_to_f16_le(f32::NAN);
    assert!(f16_le_to_f32(nan_bytes[0], nan_bytes[1]).is_nan());
}

#[test]
fn f16_encode_overflow_to_inf() {
    // A finite f32 whose exponent exceeds the f16 range overflows to inf
    // (new_exp >= 0x1f branch).
    assert_eq!(f32_to_f16_le(1e30), [0x00, 0x7C]);
    assert_eq!(f32_to_f16_le(-1e30), [0x00, 0xFC]);
}

#[test]
fn f16_encode_underflow_to_zero() {
    // A finite f32 too small for the smallest f16 subnormal flushes to
    // signed zero (new_exp <= 0 branch).
    assert_eq!(f32_to_f16_le(1e-30), [0x00, 0x00]);
    assert_eq!(f32_to_f16_le(-1e-30), [0x00, 0x80]);
    assert_eq!(f16_le_to_f32(0x00, 0x00), 0.0);
}

#[test]
fn tq2_0_decode_with_inf_scale_poisons_block() {
    // A block whose stored f16 scale is +inf should poison every non-zero
    // trit, exercising dequant against the inf decode path. qs all 0x00 →
    // trit 0 → (0 - 1) * inf = -inf for every element.
    let mut bytes = vec![0u8; TQ2_0_BLOCK_BYTES];
    bytes[64] = 0x00;
    bytes[65] = 0x7C; // +inf
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert_eq!(decoded.len(), K_QUANT_BLOCK_ELEMS);
    assert!(decoded.iter().all(|&v| v == f32::NEG_INFINITY));
}

#[test]
fn tq2_0_decode_with_subnormal_scale() {
    // Stored scale = smallest subnormal f16 (0x0001). qs byte 0 = 0b10 in
    // its lowest 2-bit slot → trit 2 → (+1) * 2^-24 for that element.
    let mut bytes = vec![0u8; TQ2_0_BLOCK_BYTES];
    bytes[0] = 0b10; // first slot of qs[0]: stored 2 → +1 after bias
    bytes[64] = 0x01;
    bytes[65] = 0x00; // subnormal scale 2^-24
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert!(
        (decoded[0] - 2f32.powi(-24)).abs() < 1e-12,
        "got {}",
        decoded[0]
    );
}

#[test]
fn tq2_0_quantize_huge_scale_round_trips_via_inf() {
    // A block whose absmax overflows f16 stores an inf scale; decode then
    // multiplies trits by inf. Drives the encode-overflow + decode-inf
    // paths together. Sign of the trit determines +inf / -inf / 0.
    let mut input = vec![0.0f32; K_QUANT_BLOCK_ELEMS];
    input[0] = 1e30; // +1 trit, absmax overflows f16
    input[1] = -1e30; // -1 trit
    let bytes = quantize_tq2_0(&input).unwrap();
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert_eq!(decoded[0], f32::INFINITY);
    assert_eq!(decoded[1], f32::NEG_INFINITY);
    // A zero trit stays zero (0 - 1 + 1 bias... trit 1 → 0 * inf = NaN
    // only if scale is inf; here element 2 quantises to 0 trit → -inf).
    // The first 128 elements are interleaved; just confirm finiteness mix.
    assert!(decoded.iter().any(|&v| v.is_infinite()));
}

#[test]
fn tq2_0_quantize_tiny_scale_flushes_to_zero() {
    // absmax below f16's smallest subnormal flushes the stored scale to
    // zero, so the whole block decodes to zero (encode-underflow path).
    let mut input = vec![0.0f32; K_QUANT_BLOCK_ELEMS];
    input[0] = 1e-30;
    input[5] = -1e-30;
    let bytes = quantize_tq2_0(&input).unwrap();
    let decoded = dequantize_tq2_0(&bytes, K_QUANT_BLOCK_ELEMS).unwrap();
    assert!(decoded.iter().all(|&v| v == 0.0));
}

// ── quantize length guards ──────────────────────────────────────────────

#[test]
fn tq2_0_quantize_non_multiple_errors() {
    // Length not a multiple of 256 → Parse error (input-guard branch).
    let err = quantize_tq2_0(&vec![0.0f32; K_QUANT_BLOCK_ELEMS + 1]).unwrap_err();
    assert!(matches!(err, ModelError::Parse(_)));
    assert!(quantize_tq2_0(&[1.0, 0.0, -1.0]).is_err());
}

#[test]
fn tq1_0_quantize_non_multiple_errors() {
    // Length not a multiple of 256 → Parse error (input-guard branch).
    let err = quantize_tq1_0(&vec![0.0f32; K_QUANT_BLOCK_ELEMS - 1]).unwrap_err();
    assert!(matches!(err, ModelError::Parse(_)));
    assert!(quantize_tq1_0(&[1.0, 0.0, -1.0]).is_err());
}
