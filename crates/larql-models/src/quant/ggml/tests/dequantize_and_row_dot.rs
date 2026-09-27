use super::*;

#[test]
fn q4_0_basic() {
    // Scale = 1.0, quants = 0x12 → lo=2-8=-6 (elements 0..16),
    // hi=1-8=-7 (elements 16..32) — ggml planar nibble layout.
    let mut block = vec![0x00, 0x3C]; // f16 1.0
    block.extend_from_slice(&[0x12; 16]);
    let result = dequantize_q4_0(&block, 32).unwrap();
    assert_eq!(result.len(), 32);
    assert!((result[0] - (-6.0)).abs() < 0.01);
    assert!((result[16] - (-7.0)).abs() < 0.01);
}

#[test]
fn q4_0_zero_scale() {
    let mut block = vec![0x00, 0x00]; // f16 0.0
    block.extend_from_slice(&[0xFF; 16]);
    let result = dequantize_q4_0(&block, 32).unwrap();
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn q4_0_two_blocks() {
    let mut data = vec![0x00, 0x3C]; // block 0: scale=1.0
    data.extend_from_slice(&[0x88; 16]); // quants: lo=8-8=0, hi=8-8=0
    data.extend_from_slice(&[0x00, 0x40]); // block 1: scale=2.0
    data.extend_from_slice(&[0x19; 16]); // lo=9-8=1, hi=1-8=-7
    let result = dequantize_q4_0(&data, 64).unwrap();
    assert_eq!(result.len(), 64);
    assert!((result[0] - 0.0).abs() < 0.01); // block 0
    assert!((result[32] - 2.0).abs() < 0.01); // block 1 lo plane: 1*2.0 = 2.0
    assert!((result[48] - (-14.0)).abs() < 0.01); // block 1 hi plane: -7*2.0 = -14.0
}

#[test]
fn q4_1_basic() {
    // Scale=1.0, min=0.5, quants=0x00 → lo=0*1+0.5=0.5, hi=0*1+0.5=0.5
    let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
    block.extend_from_slice(&[0x00; 16]);
    let result = dequantize_q4_1(&block, 32).unwrap();
    assert!((result[0] - 0.5).abs() < 0.01);
}

#[test]
fn q4_1_with_offset() {
    // Scale=2.0, min=-1.0, quants=0x31 → lo=1*2-1=1 (elements 0..16),
    // hi=3*2-1=5 (elements 16..32) — planar layout.
    let mut block = vec![0x00, 0x40, 0x00, 0xBC]; // scale=2.0, min=-1.0
    block.extend_from_slice(&[0x31; 16]);
    let result = dequantize_q4_1(&block, 32).unwrap();
    assert!((result[0] - 1.0).abs() < 0.01);
    assert!((result[16] - 5.0).abs() < 0.01);
}

#[test]
fn q8_0_basic() {
    let mut block = vec![0x00, 0x38]; // f16 scale = 0.5
    for _ in 0..16 {
        block.push(2u8); // +2 → 2*0.5 = 1.0
        block.push(0xFEu8); // -2 as i8 → -2*0.5 = -1.0
    }
    let result = dequantize_q8_0(&block, 32).unwrap();
    assert!((result[0] - 1.0).abs() < 0.01);
    assert!((result[1] - (-1.0)).abs() < 0.01);
}

#[test]
fn q8_0_zero_scale() {
    let mut block = vec![0x00, 0x00]; // scale = 0
    block.extend_from_slice(&[127u8; 32]); // max int8
    let result = dequantize_q8_0(&block, 32).unwrap();
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn q8_0_full_range() {
    let mut block = vec![0x00, 0x3C]; // scale = 1.0
    block.push(127); // max positive
    block.push(0x81); // -127 as i8
    block.extend_from_slice(&[0u8; 30]); // rest zeros
    let result = dequantize_q8_0(&block, 32).unwrap();
    assert!((result[0] - 127.0).abs() < 0.01);
    assert!((result[1] - (-127.0)).abs() < 0.01);
    assert!((result[2] - 0.0).abs() < 0.01);
}

#[test]
fn tensor_sizes() {
    assert_eq!(tensor_data_size(TYPE_F32, 32).unwrap(), 128);
    assert_eq!(tensor_data_size(TYPE_F16, 32).unwrap(), 64);
    assert_eq!(tensor_data_size(TYPE_Q4_0, 32).unwrap(), 18);
    assert_eq!(tensor_data_size(TYPE_Q4_1, 32).unwrap(), 20);
    assert_eq!(tensor_data_size(TYPE_Q8_0, 32).unwrap(), 34);
}

#[test]
fn type_names() {
    assert_eq!(type_name(TYPE_F32), "F32");
    assert_eq!(type_name(TYPE_Q4_0), "Q4_0");
    assert_eq!(type_name(TYPE_Q8_0), "Q8_0");
    assert_eq!(type_name(99), "unknown");
}

#[test]
fn f32_passthrough() {
    let data: Vec<u8> = [1.0f32, -2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let result = dequantize(&data, TYPE_F32, 3).unwrap();
    assert_eq!(result, vec![1.0, -2.0, 3.0]);
}

#[test]
fn q5_0_basic() {
    // scale=1.0, high_bits=0, quants=0x88 → lo4=8, hi4=8, hi1=0
    // combined=8, value=(8-16)*1.0=-8.0
    let mut block = vec![0x00, 0x3C]; // f16 1.0
    block.extend_from_slice(&[0x00; 4]); // high bits all zero
    block.extend_from_slice(&[0x88; 16]); // quants
    let result = dequantize_q5_0(&block, 32).unwrap();
    assert_eq!(result.len(), 32);
    assert!((result[0] - (-8.0)).abs() < 0.01);
    assert!((result[1] - (-8.0)).abs() < 0.01);
}

#[test]
fn q5_0_with_high_bits() {
    // scale=1.0, high_bits=0xFFFFFFFF (all 1), quants=0x00
    // lo4=0, hi1=1, combined=0|16=16, value=(16-16)*1.0=0.0
    let mut block = vec![0x00, 0x3C]; // f16 1.0
    block.extend_from_slice(&[0xFF; 4]); // high bits all one
    block.extend_from_slice(&[0x00; 16]); // quants all zero nibbles
    let result = dequantize_q5_0(&block, 32).unwrap();
    assert_eq!(result.len(), 32);
    assert!((result[0] - 0.0).abs() < 0.01);
}

#[test]
fn q5_0_mixed() {
    // scale=2.0, high_bits=0x00000001 (bit 0 set), quants[0]=0x53
    // element 0 (lo nibble, hi1=bit0=1): combined=3|16=19, (19-16)*2=6.0
    // element 16 (hi nibble, hi1=bit16=0): combined=5, (5-16)*2=-22.0
    let mut block = vec![0x00, 0x40]; // f16 2.0
    block.extend_from_slice(&0x00000001u32.to_le_bytes()); // high bits
    block.push(0x53); // quants[0]: lo=3, hi=5
    block.extend_from_slice(&[0x00; 15]); // rest zero
    let result = dequantize_q5_0(&block, 32).unwrap();
    assert!((result[0] - 6.0).abs() < 0.01);
    assert!((result[16] - (-22.0)).abs() < 0.01);
}

#[test]
fn q5_0_zero_scale() {
    let mut block = vec![0x00, 0x00]; // scale=0
    block.extend_from_slice(&[0xFF; 4]);
    block.extend_from_slice(&[0xFF; 16]);
    let result = dequantize_q5_0(&block, 32).unwrap();
    assert!(result.iter().all(|&v| v == 0.0));
}

#[test]
fn q5_1_basic() {
    // scale=1.0, min=0.5, high_bits=0, quants=0x00
    // combined=0, value=0*1.0+0.5=0.5
    let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
    block.extend_from_slice(&[0x00; 4]); // high bits
    block.extend_from_slice(&[0x00; 16]); // quants
    let result = dequantize_q5_1(&block, 32).unwrap();
    assert_eq!(result.len(), 32);
    assert!((result[0] - 0.5).abs() < 0.01);
}

#[test]
fn q5_1_with_high_bits() {
    // scale=2.0, min=1.0, high_bits=0xFFFFFFFF, quants=0xFF
    // lo4=15, hi1=1, combined=15|16=31, value=31*2.0+1.0=63.0
    let mut block = vec![0x00, 0x40, 0x00, 0x3C]; // scale=2.0, min=1.0
    block.extend_from_slice(&[0xFF; 4]); // high bits all one
    block.extend_from_slice(&[0xFF; 16]); // quants all 0xF nibbles
    let result = dequantize_q5_1(&block, 32).unwrap();
    assert!((result[0] - 63.0).abs() < 0.01);
}

#[test]
fn q5_1_via_dequantize() {
    // Verify dispatch works through the main dequantize() function
    let mut block = vec![0x00, 0x3C, 0x00, 0x00]; // scale=1.0, min=0.0
    block.extend_from_slice(&[0x00; 4]); // high bits zero
    block.extend_from_slice(&[0x33; 16]); // lo=3, hi=3, combined=3
    let result = dequantize(&block, TYPE_Q5_1, 32).unwrap();
    assert!((result[0] - 3.0).abs() < 0.01);
    assert!((result[1] - 3.0).abs() < 0.01);
}

#[test]
fn q5_0_via_dequantize() {
    // Verify dispatch works through the main dequantize() function
    let mut block = vec![0x00, 0x3C]; // scale=1.0
    block.extend_from_slice(&[0x00; 4]); // high bits zero
    block.extend_from_slice(&[0x88; 16]); // lo=8,hi=8, combined=8, value=(8-16)=-8
    let result = dequantize(&block, TYPE_Q5_0, 32).unwrap();
    assert!((result[0] - (-8.0)).abs() < 0.01);
}

#[test]
fn q6k_row_dot_neon_matches_scalar_single_block() {
    let data = synth_q6k_block(42);
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
    let scalar = q6k_row_dot_scalar(&data, &x, 1);
    let dispatched = q6k_row_dot(&data, &x).unwrap();
    // Both paths should agree to within fp accumulation noise.
    assert!(
        (scalar - dispatched).abs() < 1e-3,
        "scalar={scalar} dispatched={dispatched}"
    );
}

#[test]
fn q6k_row_dot_neon_matches_scalar_multi_block() {
    let mut data = Vec::with_capacity(210 * 8);
    for sb in 0..8 {
        data.extend_from_slice(&synth_q6k_block(1234 + sb as u32));
    }
    let x: Vec<f32> = (0..256 * 8)
        .map(|i| (((i as f32) * 0.003).cos() - 0.5) * 0.2)
        .collect();
    let scalar = q6k_row_dot_scalar(&data, &x, 8);
    let dispatched = q6k_row_dot(&data, &x).unwrap();
    let tol = (scalar.abs() + dispatched.abs()).max(1.0) * 1e-5;
    assert!(
        (scalar - dispatched).abs() < tol,
        "scalar={scalar} dispatched={dispatched} tol={tol}"
    );
}

#[test]
fn q6_k_matches_llama_cpp_ground_truth() {
    let got = dequantize_q6_k(&Q6K_GT_BYTES, 256).unwrap();
    for (i, (g, e)) in got.iter().zip(Q6K_GT_EXPECTED.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-7 + 1e-5 * e.abs(),
            "Q6_K element {i}: got {g}, expected {e}"
        );
    }
}

#[test]
fn q4_0_matches_llama_cpp_ground_truth() {
    let got = dequantize_q4_0(&Q40_GT_BYTES, 32).unwrap();
    for (i, (g, e)) in got.iter().zip(Q40_GT_EXPECTED.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-7 + 1e-5 * e.abs(),
            "Q4_0 element {i}: got {g}, expected {e}"
        );
    }
}

#[test]
fn q6k_row_dot_matches_ground_truth() {
    // dot(decode(block), x) must equal dot(ground_truth, x).
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.017).sin()).collect();
    let expected: f32 = Q6K_GT_EXPECTED.iter().zip(&x).map(|(w, xi)| w * xi).sum();
    let got = q6k_row_dot(&Q6K_GT_BYTES, &x).unwrap();
    assert!(
        (got - expected).abs() < 1e-4,
        "q6k_row_dot: got {got}, expected {expected}"
    );
}

#[test]
fn q6k_row_scaled_add_matches_ground_truth() {
    let mut out = vec![0.0f32; 256];
    q6k_row_scaled_add(&Q6K_GT_BYTES, 2.0, &mut out).unwrap();
    for (i, (g, e)) in out.iter().zip(Q6K_GT_EXPECTED.iter()).enumerate() {
        let want = 2.0 * e;
        assert!(
            (g - want).abs() <= 1e-7 + 1e-5 * want.abs(),
            "scaled_add element {i}: got {g}, expected {want}"
        );
    }
}

#[test]
fn q4_0_rejects_short_buffer() {
    // 32 elements need 18 bytes; give it 10.
    assert_short_buffer(dequantize_q4_0(&[0u8; 10], 32), "Q4_0");
}

#[test]
fn q4_1_rejects_short_buffer() {
    assert_short_buffer(dequantize(&[0u8; 4], TYPE_Q4_1, 32), "Q4_1");
}

#[test]
fn q8_0_rejects_short_buffer() {
    // 64 elements = 2 blocks × 34 bytes = 68; give 40.
    assert_short_buffer(dequantize(&[0u8; 40], TYPE_Q8_0, 64), "Q8_0");
}

#[test]
fn q5_0_rejects_short_buffer() {
    assert_short_buffer(dequantize_q5_0(&[0u8; 10], 32), "Q5_0");
}

#[test]
fn q5_1_rejects_short_buffer() {
    assert_short_buffer(dequantize_q5_1(&[0u8; 10], 32), "Q5_1");
}

#[test]
fn q4_k_rejects_short_buffer() {
    // 256 elements = 1 super-block = 144 bytes; give 100.
    assert_short_buffer(dequantize_q4_k(&[0u8; 100], 256), "Q4_K");
}

#[test]
fn q6_k_rejects_short_buffer() {
    // 256 elements = 1 super-block = 210 bytes; give 100.
    assert_short_buffer(dequantize_q6_k(&[0u8; 100], 256), "Q6_K");
}

#[test]
fn q4_0_rejects_misaligned_n_elements() {
    // 33 is not a multiple of 32.
    match dequantize_q4_0(&[0u8; 18], 33) {
        Err(ModelError::Parse(msg)) => {
            assert!(msg.contains("not a multiple of 32"), "got: {msg}");
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn q6_k_rejects_misaligned_n_elements() {
    // 300 is not a multiple of 256.
    match dequantize_q6_k(&[0u8; 210], 300) {
        Err(ModelError::Parse(msg)) => {
            assert!(msg.contains("not a multiple of 256"), "got: {msg}");
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn passthrough_f32_rejects_short_buffer() {
    // 8 elements = 32 bytes; give 20.
    match dequantize(&[0u8; 20], TYPE_F32, 8) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("F32"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn passthrough_f16_rejects_short_buffer() {
    // 8 elements = 16 bytes; give 10.
    match dequantize(&[0u8; 10], TYPE_F16, 8) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("F16"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn passthrough_bf16_rejects_short_buffer() {
    match dequantize(&[0u8; 10], TYPE_BF16, 8) {
        Err(ModelError::Parse(msg)) => assert!(msg.contains("BF16"), "got: {msg}"),
        other => panic!("expected Parse error, got {other:?}"),
    }
}

#[test]
fn empty_input_ok_when_zero_elements() {
    // Zero-element tensor should succeed with empty output across all block types.
    for &ty in &[
        TYPE_Q4_0, TYPE_Q4_1, TYPE_Q8_0, TYPE_Q5_0, TYPE_Q5_1, TYPE_Q4_K, TYPE_Q6_K,
    ] {
        let out = dequantize(&[], ty, 0).unwrap_or_else(|e| panic!("type {ty} failed: {e:?}"));
        assert!(out.is_empty(), "type {ty} produced {} elements", out.len());
    }
}

/// Max component-wise representation error for a given scale — Q4_0 maps
/// every value to the nearest multiple of `scale` in `[-8*scale, 7*scale]`,
/// so round-trip error is bounded by half a quantization step.
#[test]
fn q4_0_round_trip_preserves_within_half_step() {
    // Inputs fit the ±7*scale range cleanly.
    let vals: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.1).collect();
    let packed = quantize_q4_0(&vals);
    assert_eq!(packed.len(), 2 * 18);
    let round = dequantize_q4_0(&packed, 64).unwrap();
    let scale = 0.1 * 31.5 / 7.0; // amax / 7 per block
    let max_step = scale * 0.5 + 1e-3;
    for (i, (v, r)) in vals.iter().zip(&round).enumerate() {
        assert!(
            (v - r).abs() <= max_step,
            "idx {i}: v={v} r={r} max_step={max_step}"
        );
    }
}

#[test]
fn q4_0_round_trip_all_zero() {
    // Zero-scale corner: every value must decode to exactly 0.
    let vals = vec![0.0f32; 32];
    let packed = quantize_q4_0(&vals);
    let round = dequantize_q4_0(&packed, 32).unwrap();
    assert!(round.iter().all(|&v| v == 0.0));
}

#[test]
fn q8_0_round_trip_precise() {
    // Q8_0 has 127 steps — 2 decimal places should survive cleanly.
    let vals: Vec<f32> = (0..64).map(|i| ((i as f32 - 32.0) * 0.013).sin()).collect();
    let packed = quantize_q8_0(&vals);
    assert_eq!(packed.len(), 2 * 34);
    let round = dequantize_q8_0(&packed, 64).unwrap();
    // Per-block amax / 127 ≤ 1/127 ≈ 0.008, so round-trip error < 0.004.
    for (i, (v, r)) in vals.iter().zip(&round).enumerate() {
        assert!((v - r).abs() < 0.01, "idx {i}: v={v} r={r}");
    }
}

#[test]
fn q8_0_round_trip_edges() {
    // Values hitting the ±127/scale clamp edges. Scale is stored as f16
    // (11-bit mantissa), so allow ~1e-3 for the quantized representation
    // of ±1.0 after the f16-scale precision loss.
    let mut vals = Vec::with_capacity(32);
    for _ in 0..16 {
        vals.push(1.0);
        vals.push(-1.0);
    }
    let packed = quantize_q8_0(&vals);
    let round = dequantize_q8_0(&packed, 32).unwrap();
    for (i, (v, r)) in vals.iter().zip(&round).enumerate() {
        assert!((v - r).abs() < 1e-3, "idx {i}: v={v} r={r}");
    }
}

#[test]
fn q4_0_via_dequantize() {
    let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.05).collect();
    let packed = quantize_q4_0(&vals);
    let round = dequantize(&packed, TYPE_Q4_0, 32).unwrap();
    assert_eq!(round.len(), 32);
}

#[test]
fn q8_0_via_dequantize() {
    let vals: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    let packed = quantize_q8_0(&vals);
    let round = dequantize(&packed, TYPE_Q8_0, 32).unwrap();
    assert_eq!(round.len(), 32);
    // Matches in-module Q8_0 path exactly.
    let direct = dequantize_q8_0(&packed, 32).unwrap();
    assert_eq!(round, direct);
}

#[test]
fn q4_k_via_dequantize_roundtrips_to_known_output() {
    // Build a 144-byte Q4K block with scale 1.0, min 0.0, all sub-scales=1,
    // sub-mins=0, nibbles = low nibble index 0..7 repeated — check shape,
    // not exact values (the scale/min packing is lossy).
    let mut block = vec![0u8; 144];
    block[0] = 0x00;
    block[1] = 0x3C; // d = 1.0 (f16)
    block[2] = 0x00;
    block[3] = 0x00; // dmin = 0.0
                     // bytes 4..16: scales[0..4] = 1, mins[0..4] = 0 (low 6 bits only)
    for s in &mut block[4..8] {
        *s = 0x01;
    }
    for _m in &mut block[8..12] { /* mins lo = 0 */ }
    // Leave scales[4..8] = 0 (high nibble carrier) and quants zero.
    let out = dequantize(&block, TYPE_Q4_K, 256).unwrap();
    assert_eq!(out.len(), 256);
    // First 128 elements use scales[0..4] = 1 so decoded = 0 (nibbles zero).
    // Remaining 128 use scales[4..8] = 0 so also zero.
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn q6_k_via_dequantize() {
    // Dispatch-path check — uses the single-block synth helper.
    let block = synth_q6k_block(99);
    let direct = dequantize_q6_k(&block, 256).unwrap();
    let dispatched = dequantize(&block, TYPE_Q6_K, 256).unwrap();
    assert_eq!(direct, dispatched);
}

#[test]
fn q6k_row_dot_matches_dequantized_dot() {
    // Ground truth: dequantize_q6_k then compute the dot manually.
    let data = synth_q6k_block(7);
    let deq = dequantize_q6_k(&data, 256).unwrap();
    let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001 - 0.05).collect();
    let gold: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
    let dispatched = q6k_row_dot(&data, &x).unwrap();
    let tol = (gold.abs() + dispatched.abs()).max(1.0) * 1e-4;
    assert!(
        (gold - dispatched).abs() < tol,
        "gold={gold} dispatched={dispatched} tol={tol}"
    );
}

#[test]
fn q4k_row_dot_neon_matches_scalar_single_block() {
    use super::super::q4_k::q4k_row_dot_scalar;
    let data = synth_q4k_block(42);
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
    let scalar = q4k_row_dot_scalar(&data, &x, 1);
    let dispatched = q4k_row_dot(&data, &x).unwrap();
    assert!(
        (scalar - dispatched).abs() < 1e-3,
        "scalar={scalar} dispatched={dispatched}"
    );
}

#[test]
fn q4k_row_dot_neon_matches_scalar_multi_block() {
    use super::super::q4_k::q4k_row_dot_scalar;
    let mut data = Vec::with_capacity(144 * 8);
    for sb in 0..8u32 {
        data.extend_from_slice(&synth_q4k_block(1000 + sb));
    }
    let x: Vec<f32> = (0..256 * 8)
        .map(|i| (((i as f32) * 0.003).cos() - 0.5) * 0.2)
        .collect();
    let scalar = q4k_row_dot_scalar(&data, &x, 8);
    let dispatched = q4k_row_dot(&data, &x).unwrap();
    let tol = (scalar.abs() + dispatched.abs()).max(1.0) * 1e-5;
    assert!(
        (scalar - dispatched).abs() < tol,
        "scalar={scalar} dispatched={dispatched} tol={tol}"
    );
}

#[test]
fn q4k_row_dot_matches_dequantized_dot() {
    let data = synth_q4k_block(7);
    let deq = dequantize_q4_k(&data, 256).unwrap();
    let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001 - 0.05).collect();
    let gold: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
    let dispatched = q4k_row_dot(&data, &x).unwrap();
    let tol = (gold.abs() + dispatched.abs()).max(1.0) * 1e-4;
    assert!(
        (gold - dispatched).abs() < tol,
        "gold={gold} dispatched={dispatched} tol={tol}"
    );
}

#[test]
fn q4_k_dequantize_known_nonzero_values() {
    // d=1.0, dmin=0.0, scales[0..4]=2, scales[4..8]=0, mins all 0.
    // All quant bytes = 0x53 → lo nibble=3, hi nibble=5.
    //
    // Expected output per sub-block group:
    //   g=0: base_lo=0..32   → d*scales[0]*3 = 6.0
    //         base_hi=32..64  → d*scales[1]*5 = 10.0
    //   g=1: base_lo=64..96  → 6.0
    //         base_hi=96..128 → 10.0
    //   g=2/3: scales[4..8]=0  → 0.0
    let mut block = vec![0u8; 144];
    block[0] = 0x00;
    block[1] = 0x3C; // d = 1.0 (f16)
    block[2] = 0x00;
    block[3] = 0x00; // dmin = 0.0
                     // scales_bytes[0..4] = 0x02 → scales[0..4] = 2, mins[0..4] = 0
    block[4] = 0x02;
    block[5] = 0x02;
    block[6] = 0x02;
    block[7] = 0x02;
    // scales_bytes[4..12] = 0x00 → mins[0..4] = 0, scales[4..8] = 0
    block[8..16].fill(0x00);
    block[16..144].fill(0x53);

    let out = dequantize_q4_k(&block, 256).unwrap();
    assert_eq!(out.len(), 256);
    for (i, &v) in out.iter().enumerate().take(32) {
        assert!((v - 6.0).abs() < 1e-6, "i={i} got {v}");
    }
    for (i, &v) in out.iter().enumerate().take(64).skip(32) {
        assert!((v - 10.0).abs() < 1e-6, "i={i} got {v}");
    }
    for (i, &v) in out.iter().enumerate().take(96).skip(64) {
        assert!((v - 6.0).abs() < 1e-6, "i={i} got {v}");
    }
    for (i, &v) in out.iter().enumerate().take(128).skip(96) {
        assert!((v - 10.0).abs() < 1e-6, "i={i} got {v}");
    }
    for (i, &v) in out.iter().enumerate().skip(128) {
        assert!((v - 0.0).abs() < 1e-6, "i={i} got {v}");
    }
}
