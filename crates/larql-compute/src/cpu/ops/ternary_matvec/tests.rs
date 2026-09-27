use super::*;

/// Encode an f32 row of `{-d, 0, +d}` trits into I2_S bytes.
/// Used by tests; mirrors the bit-pattern map in the decoder.
fn encode_row(row: &[f32], d: f32) -> Vec<u8> {
    assert!(row.len().is_multiple_of(4));
    let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
    let mut out = vec![0u8; row.len() / 4];
    for (i, chunk) in row.chunks_exact(4).enumerate() {
        let mut byte: u8 = 0;
        for (slot, &v) in chunk.iter().enumerate() {
            let t = (v * inv).round().clamp(-1.0, 1.0) as i32;
            let bits: u8 = match t {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            };
            byte |= bits << (2 * slot);
        }
        out[i] = byte;
    }
    out
}

/// Naive dequant + matmul reference.  Used to verify the kernel
/// against ground truth.
fn naive_dequant_matvec(w: &BitLinearWeight, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; w.rows];
    let row_bytes = w.row_bytes();
    for (r, y_r) in y.iter_mut().enumerate() {
        let scale = w.channel_scales[r];
        for (c, &x_c) in x.iter().enumerate().take(w.cols) {
            let byte = w.i2s_bytes[r * row_bytes + c / 4];
            let bits = (byte >> (2 * (c % 4))) & 0b11;
            let trit = match bits {
                0b01 => 1.0_f32,
                0b10 => -1.0_f32,
                _ => 0.0_f32,
            };
            *y_r += trit * scale * x_c;
        }
    }
    y
}

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn synth_ternary(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bucket = ((s >> 33) % 3) as i32;
            match bucket {
                0 => 0.0,
                1 => 1.0,
                _ => -1.0,
            }
        })
        .collect()
}

#[test]
fn shape_mismatch_rejects_bad_inputs() {
    // cols not a multiple of 4
    assert!(
        BitLinearWeight::new(1, 5, vec![0; 2], vec![1.0]).is_err(),
        "cols=5 should reject"
    );
    // wrong byte count
    assert!(
        BitLinearWeight::new(2, 8, vec![0; 3], vec![1.0, 1.0]).is_err(),
        "expected 4 bytes (2*8/4), got 3"
    );
    // wrong scale count
    assert!(
        BitLinearWeight::new(2, 8, vec![0; 4], vec![1.0]).is_err(),
        "expected 2 scales"
    );
}

#[test]
fn matvec_x_dim_mismatch_errors() {
    let w = BitLinearWeight::new(1, 8, vec![0; 2], vec![1.0]).unwrap();
    let x = vec![0.0f32; 7];
    assert!(matvec_i2s_f32(&w, &x).is_err());
}

#[test]
fn matvec_y_too_small_errors() {
    let w = BitLinearWeight::new(2, 4, vec![0; 2], vec![1.0, 1.0]).unwrap();
    let x = vec![0.0f32; 4];
    let mut y = vec![0.0f32; 1];
    assert!(matvec_i2s_f32_into(&w, &x, &mut y).is_err());
}

#[test]
fn matvec_zero_weight_returns_zero() {
    // All-zero trits; result is zero regardless of x or scale.
    let w = BitLinearWeight::new(3, 16, vec![0u8; 12], vec![1.5, -2.0, 7.0]).unwrap();
    let x = synth(16, 42);
    let y = matvec_i2s_f32(&w, &x).unwrap();
    assert_eq!(y, vec![0.0, 0.0, 0.0]);
}

#[test]
fn matvec_identity_row_recovers_activation() {
    // Single row, trit 1 at position 5 only, scale 1.0.
    // Result should equal x[5] exactly.
    let mut row = vec![0.0f32; 16];
    row[5] = 1.0;
    let bytes = encode_row(&row, 1.0);
    let w = BitLinearWeight::new(1, 16, bytes, vec![1.0]).unwrap();
    let x = synth(16, 11);
    let y = matvec_i2s_f32(&w, &x).unwrap();
    assert!((y[0] - x[5]).abs() < 1e-6, "got {} expected {}", y[0], x[5]);
}

#[test]
fn matvec_negative_trit_subtracts() {
    // Row with -1 at position 3 and +1 at position 11; scale 0.5.
    // Result = (x[11] - x[3]) * 0.5
    let mut row = vec![0.0f32; 16];
    row[3] = -1.0;
    row[11] = 1.0;
    let bytes = encode_row(&row, 1.0);
    let w = BitLinearWeight::new(1, 16, bytes, vec![0.5]).unwrap();
    let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let y = matvec_i2s_f32(&w, &x).unwrap();
    let expected = (x[11] - x[3]) * 0.5;
    assert!(
        (y[0] - expected).abs() < 1e-6,
        "got {} expected {}",
        y[0],
        expected
    );
}

/// Reference equivalence: kernel result must match naive dequant
/// + matmul to within floating-point noise.
#[test]
fn matvec_matches_naive_reference_random_ternary() {
    // 32 rows x 256 cols, fully ternary weights, varied channel scales.
    let rows = 32;
    let cols = 256;
    let mut bytes = Vec::with_capacity(rows * cols / 4);
    for r in 0..rows {
        let row_trits = synth_ternary(cols, 42 + r as u64);
        bytes.extend(encode_row(&row_trits, 1.0));
    }
    let scales: Vec<f32> = (0..rows).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

    let x = synth(cols, 9999);
    let kernel = matvec_i2s_f32(&w, &x).unwrap();
    let reference = naive_dequant_matvec(&w, &x);

    for (i, (k, r)) in kernel.iter().zip(reference.iter()).enumerate() {
        // Both sum the same trits with the same scale; match
        // should be exact up to summation-order rounding.
        assert!(
            (k - r).abs() < 1e-4,
            "row {i}: kernel={k} reference={r} delta={}",
            k - r
        );
    }
}

/// The reserved 0b11 bit pattern decodes to 0, same as 0b00.
/// (Microsoft's BitNet b1.58 2 B 4 T never produces 0b11 in
/// shipped weights, but the kernel must handle it gracefully if
/// it shows up under some future toolchain.)
#[test]
fn matvec_reserved_bit_pattern_decodes_as_zero() {
    // 4 cols, 1 row, byte 0xFF (all four slots = 0b11).
    let w = BitLinearWeight::new(1, 4, vec![0xFFu8], vec![3.0]).unwrap();
    let x = vec![1.0, 1.0, 1.0, 1.0];
    let y = matvec_i2s_f32(&w, &x).unwrap();
    assert_eq!(y, vec![0.0]);
}

/// Scale flows through correctly: rescaling weights by k and
/// activations by m scales output by k*m.
#[test]
fn matvec_scale_and_activation_scale_compose() {
    let row = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
    let bytes = encode_row(&row, 1.0);
    let w_unit = BitLinearWeight::new(1, 8, bytes.clone(), vec![1.0]).unwrap();
    let w_scaled = BitLinearWeight::new(1, 8, bytes, vec![2.5]).unwrap();

    let x = vec![0.5; 8];
    let y_unit = matvec_i2s_f32(&w_unit, &x).unwrap();
    let y_scaled = matvec_i2s_f32(&w_scaled, &x).unwrap();

    let x_scaled: Vec<f32> = x.iter().map(|v| v * 4.0).collect();
    let y_act_scaled = matvec_i2s_f32(&w_unit, &x_scaled).unwrap();

    assert!((y_scaled[0] - y_unit[0] * 2.5).abs() < 1e-6);
    assert!((y_act_scaled[0] - y_unit[0] * 4.0).abs() < 1e-6);
}

/// The `_into` variant overwrites — not accumulates — its output
#[test]
fn matvec_into_overwrites_not_accumulates() {
    let rows = 4;
    let cols = 8;
    let mut bytes = Vec::new();
    for r in 0..rows {
        let row_trits = synth_ternary(cols, 100 + r as u64);
        bytes.extend(encode_row(&row_trits, 1.0));
    }
    let scales = vec![0.5_f32; rows];
    let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

    let x = synth(cols, 1);
    let mut y = vec![999.0_f32; rows]; // Pre-poisoned.
    matvec_i2s_f32_into(&w, &x, &mut y).unwrap();
    let y2 = matvec_i2s_f32(&w, &x).unwrap();
    for (a, b) in y.iter().zip(y2.iter()) {
        assert!((a - b).abs() < 1e-6, "poisoned y entry leaked: {a} vs {b}");
    }
}

/// The kernel consumes the writer's *re-packed* contiguous I2_S
/// layout (4 trits per byte, sequential per row). This is
/// deliberately NOT the microsoft GGUF strided layout that
/// `dequantize_i2_s` decodes — the keep-quant writer re-encodes
/// from the dequantised weights into this contiguous form so the
/// hot loop never handles the strided source layout (see
/// bitnet_writer.rs and BUG-infer-deadlock §5.4). Pins the kernel
/// against its own `encode_row` helper.
#[test]
fn matvec_agrees_with_contiguous_encoding() {
    let row = synth_ternary(64, 7);
    let bytes = encode_row(&row, 1.0);

    let scale: f32 = 0.7;
    let w = BitLinearWeight::new(1, 64, bytes, vec![scale]).unwrap();
    let x = synth(64, 13);
    let kernel = matvec_i2s_f32(&w, &x).unwrap();
    let reference: f32 = row.iter().zip(x.iter()).map(|(t, a)| t * a).sum::<f32>() * scale;

    assert!(
        (kernel[0] - reference).abs() < 1e-4,
        "kernel={} reference={} delta={}",
        kernel[0],
        reference,
        kernel[0] - reference
    );
}

// ── A8 (int8-activation) path ───────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na * nb)
}

#[test]
fn quantize_activation_i8_zero_input_yields_zero_scale() {
    let (q, scale) = quantize_activation_i8(&[0.0; 16]);
    assert_eq!(scale, 0.0);
    assert!(q.iter().all(|&c| c == 0));
}

#[test]
fn quantize_activation_i8_puts_max_at_127() {
    let (q, scale) = quantize_activation_i8(&[0.5, -1.0, 0.25, 0.0]);
    // amax = 1.0 → scale = 1/127; the -1.0 maps to -127.
    assert!((scale - 1.0 / 127.0).abs() < 1e-9);
    assert_eq!(q[1], -127);
    // round-trip stays within one quantum.
    for (&qc, &orig) in q.iter().zip([0.5, -1.0, 0.25, 0.0].iter()) {
        assert!((qc as f32 * scale - orig).abs() <= scale);
    }
}

#[test]
fn matvec_a8_is_exact_when_activation_is_int8_representable() {
    // Activations drawn from {0, +a, -a} all quantise to {0, ±127}
    // exactly (amax == a), so the A8 path reproduces the f32 path with
    // no quantisation error — only fp rounding.
    let (rows, cols) = (8usize, 256usize);
    let a = 0.5f32;
    let x: Vec<f32> = synth_ternary(cols, 11).iter().map(|t| t * a).collect();
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    for r in 0..rows {
        bytes.extend(encode_row(&synth_ternary(cols, 100 + r as u64), 1.0));
        scales.push(0.3 + 0.1 * r as f32);
    }
    let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

    let y_f32 = matvec_i2s_f32(&w, &x).unwrap();
    let y_a8 = matvec_i2s_a8(&w, &x).unwrap();
    for (f, q) in y_f32.iter().zip(y_a8.iter()) {
        assert!((f - q).abs() < 1e-3, "f32={f} a8={q}");
    }
}

#[test]
fn matvec_a8_matches_f32_within_int8_tolerance() {
    // Arbitrary real activations: the A8 path carries int8 activation
    // quantisation error but must stay tightly aligned with the f32 path.
    let (rows, cols) = (8usize, 512usize);
    let x = synth(cols, 19);
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    for r in 0..rows {
        bytes.extend(encode_row(&synth_ternary(cols, 200 + r as u64), 1.0));
        scales.push(0.5 + 0.05 * r as f32);
    }
    let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();

    let y_f32 = naive_dequant_matvec(&w, &x);
    let y_a8 = matvec_i2s_a8(&w, &x).unwrap();

    let cos = cosine(&y_f32, &y_a8);
    assert!(cos > 0.999, "A8 vs f32 cosine {cos} below 0.999");
    // relative L2 error from int8 activation quantisation stays small.
    let err: f32 = y_f32
        .iter()
        .zip(&y_a8)
        .map(|(f, q)| (f - q) * (f - q))
        .sum::<f32>()
        .sqrt();
    let mag: f32 = y_f32.iter().map(|f| f * f).sum::<f32>().sqrt();
    assert!(
        err / mag < 0.03,
        "A8 relative L2 error {} too high",
        err / mag
    );
}

#[test]
fn matvec_a8_zero_weight_returns_zero() {
    let w = BitLinearWeight::new(2, 8, vec![0u8; 4], vec![1.0, 1.0]).unwrap();
    let y = matvec_i2s_a8(&w, &synth(8, 5)).unwrap();
    assert_eq!(y, vec![0.0, 0.0]);
}

#[test]
fn matvec_i2s_q8_into_rejects_bad_shapes() {
    let w = BitLinearWeight::new(2, 8, vec![0u8; 4], vec![1.0, 1.0]).unwrap();
    let (x_i8, s) = quantize_activation_i8(&synth(8, 5));
    // wrong activation length
    assert!(matvec_i2s_q8_into(&w, &x_i8[..4], s, &mut [0.0; 2]).is_err());
    // output too small
    assert!(matvec_i2s_q8_into(&w, &x_i8, s, &mut [0.0; 1]).is_err());
}

#[cfg(target_arch = "aarch64")]
#[test]
fn matvec_q8_neon_is_bit_identical_to_scalar() {
    // Integer sign-select accumulation is order-independent, so the NEON
    // kernel must match the scalar A8 path bit-for-bit — across full
    // 16-wide chunks AND `cols % 16 != 0` tails (20, 36, 260).
    for &cols in &[16usize, 64, 256, 512, 20, 36, 260] {
        let rows = 6usize;
        let mut bytes = Vec::new();
        let mut scales = Vec::new();
        for r in 0..rows {
            bytes.extend(encode_row(&synth_ternary(cols, 300 + r as u64), 1.0));
            scales.push(0.4 + 0.07 * r as f32);
        }
        let w = BitLinearWeight::new(rows, cols, bytes, scales).unwrap();
        let (x_i8, x_scale) = quantize_activation_i8(&synth(cols, 77));

        let mut y_scalar = vec![0.0f32; rows];
        matvec_i2s_q8_into(&w, &x_i8, x_scale, &mut y_scalar).unwrap();
        let mut y_neon = vec![0.0f32; rows];
        matvec_i2s_q8_neon_into(&w, &x_i8, x_scale, &mut y_neon).unwrap();

        for (s, n) in y_scalar.iter().zip(&y_neon) {
            assert_eq!(s.to_bits(), n.to_bits(), "cols={cols} scalar={s} neon={n}");
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn matvec_q8_neon_rejects_bad_shapes() {
    let w = BitLinearWeight::new(2, 8, vec![0u8; 4], vec![1.0, 1.0]).unwrap();
    let (x_i8, s) = quantize_activation_i8(&synth(8, 5));
    assert!(matvec_i2s_q8_neon_into(&w, &x_i8[..4], s, &mut [0.0; 2]).is_err());
    assert!(matvec_i2s_q8_neon_into(&w, &x_i8, s, &mut [0.0; 1]).is_err());
}
