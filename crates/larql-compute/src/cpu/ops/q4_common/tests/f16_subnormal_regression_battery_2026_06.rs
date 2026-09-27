//! f16 subnormal regression battery (2026-06-12). The subnormal

use super::*;

/// Exhaustive bit-exact parity for all 65536 f16 inputs.  The fast
/// bit-manipulation `f16_to_f32` must produce the same f32 bits as
/// the powi-based reference for every finite (non-NaN) input.  NaN
/// payloads differ by design (reference collapses to canonical NaN,
/// fast path preserves payload — both are valid IEEE NaNs and the
/// distinction is unobservable in Q4_K decode because real-world
/// Q4_K headers never contain NaNs).
#[test]
fn f16_to_f32_bit_exact_for_all_inputs() {
    let mut diffs = 0usize;
    for bits in 0u16..=u16::MAX {
        let new = f16_to_f32(bits);
        let old = f16_to_f32_powi_reference(bits);
        if new.is_nan() && old.is_nan() {
            continue; // both NaN — different payloads OK
        }
        if new.to_bits() != old.to_bits() {
            if diffs < 5 {
                eprintln!(
                    "diff at bits=0x{bits:04x}: new={} ({:#x}) old={} ({:#x})",
                    new,
                    new.to_bits(),
                    old,
                    old.to_bits()
                );
            }
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "{diffs} f16 inputs decode to different f32 bits");
}

#[test]
fn f16_to_f32_subnormal_pinned_values() {
    // IEEE 754 half subnormals: value = mant × 2^-24 exactly.
    assert_eq!(
        super::super::f16_to_f32(0x0001),
        2f32.powi(-24),
        "smallest subnormal"
    );
    assert_eq!(
        super::super::f16_to_f32(0x03fe),
        1022.0 * 2f32.powi(-24),
        "the field case — the gemma3-4b L32 K-scale that exposed the 2× bug"
    );
    assert_eq!(
        super::super::f16_to_f32(0x03ff),
        1023.0 * 2f32.powi(-24),
        "largest subnormal"
    );
    assert_eq!(
        super::super::f16_to_f32(0x0400),
        2f32.powi(-14),
        "smallest normal"
    );
    assert_eq!(
        super::super::f16_to_f32(0x8001),
        -(2f32.powi(-24)),
        "negative subnormal"
    );
}

#[test]
fn f16_to_f32_strictly_monotonic_across_subnormal_boundary() {
    // The 2× bug made f16(0x03ff) ≈ 1.22e-4 > f16(0x0400) = 6.1e-5 — a
    // monotonicity violation at the subnormal/normal seam. Walk the
    // positive seam region and require strict increase.
    let mut prev = super::super::f16_to_f32(0x0000);
    for bits in 0x0001u16..=0x0410 {
        let v = super::super::f16_to_f32(bits);
        assert!(
            v > prev,
            "f16 decode must be strictly increasing: bits={bits:#06x} gives {v:e}, prev {prev:e}"
        );
        prev = v;
    }
}

/// Cross-crate seam test: same bytes, q4_common decoder vs the
/// larql-models decoder (which backs the vindex registry and the
/// staged/dequant path). These disagreed on every subnormal-scale
/// block until 2026-06-12 — same bytes, silently different weights.
#[test]
fn q4k_decode_matches_models_reference_incl_subnormal_scales() {
    for (name, magnitude) in [("normal", 1.0f32), ("subnormal-scale", 4.0e-4)] {
        let data = seeded_data(1024, magnitude, 0xA11C1);
        let bytes = quantize_q4_k(&data);
        if magnitude < 1e-3 {
            assert!(
                q4k_has_subnormal_scale(&bytes),
                "fixture drift: {name} case no longer produces subnormal f16 scales"
            );
        }
        let ours = dequantize_q4_k(&bytes, 1024);
        let reference =
            larql_models::quant::ggml::dequantize_q4_k(&bytes, 1024).expect("models decode");
        for (i, (a, b)) in ours.iter().zip(reference.iter()).enumerate() {
            let tol = 1e-5 * a.abs().max(b.abs()).max(1e-30);
            assert!(
                (a - b).abs() <= tol,
                "{name}: decoders disagree at elem {i}: q4_common {a:e} vs models {b:e}"
            );
        }
    }
}

/// Q6_K twin — its `d` is also an f16 scale, and the int8 Q6K matvec
/// reads it through the shared (previously buggy) `f16_to_f32`.
/// Reference decode comes from larql-models (independent f16 impl).
#[test]
fn q6k_int8_matvec_matches_models_reference_incl_tiny_scales() {
    use crate::cpu::ops::q4k_q8k_dot::{
        q6k_q8k_matvec_into, quantize_x_to_q8k_into, Q8KActivation,
    };
    let (rows, cols) = (2usize, 256usize);
    for (name, magnitude) in [("normal", 1.0f32), ("tiny-scale", 4.0e-4)] {
        let data = seeded_data(rows * cols, magnitude, 0xA11C2);
        let bytes = quantize_q6_k(&data);
        let x = seeded_data(cols, 1.0, 0xA11C5);
        let reference =
            larql_models::quant::ggml::dequantize_q6_k(&bytes, rows * cols).expect("models decode");
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                reference[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();
        let denom: f32 = expected.iter().map(|v| v.abs()).fold(1e-12, f32::max);
        let mut x_q8k = Q8KActivation::with_capacity(cols);
        quantize_x_to_q8k_into(&mut x_q8k, &x);
        let mut out = vec![0.0f32; rows];
        q6k_q8k_matvec_into(&mut out, &x_q8k, &bytes, rows, cols).expect("valid shape");
        for (r, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 2e-2 * denom,
                "{name}: Q6K int8 matvec row {r}: {got:e} vs models reference {want:e}"
            );
        }
    }
}

/// Both Q4_K matvec kernels against the dequant·dot reference on the
/// same bytes, including subnormal-scale blocks. Pre-fix, affected
/// blocks contributed 2× — far outside either tolerance.
#[test]
fn q4k_matvecs_match_dequant_dot_incl_subnormal_scales() {
    use crate::cpu::ops::q4k_q8k_dot::{
        q4k_q8k_matvec_into, quantize_x_to_q8k_into, Q8KActivation,
    };
    let (rows, cols) = (4usize, 256usize);
    for (name, magnitude) in [("normal", 1.0f32), ("subnormal-scale", 4.0e-4)] {
        let data = seeded_data(rows * cols, magnitude, 0xA11C3);
        let bytes = quantize_q4_k(&data);
        if magnitude < 1e-3 {
            assert!(q4k_has_subnormal_scale(&bytes), "fixture drift ({name})");
        }
        let x = seeded_data(cols, 1.0, 0xA11C4);
        let deq = dequantize_q4_k(&bytes, rows * cols);
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                deq[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();
        let denom: f32 = expected.iter().map(|v| v.abs()).fold(1e-12, f32::max);

        // f32-activation kernel: decode-identical, tight tolerance.
        let mut out_f32 = vec![0.0f32; rows];
        q4k_matvec_into(&mut out_f32, &x, &bytes, rows, cols).expect("valid shape");
        for (r, (got, want)) in out_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * denom,
                "{name}: f32-act matvec row {r}: {got:e} vs {want:e}"
            );
        }

        // int8-activation kernel: Q8_K rounding allowed, 2× is not.
        let mut x_q8k = Q8KActivation::with_capacity(cols);
        quantize_x_to_q8k_into(&mut x_q8k, &x);
        let mut out_i8 = vec![0.0f32; rows];
        q4k_q8k_matvec_into(&mut out_i8, &x_q8k, &bytes, rows, cols).expect("valid shape");
        for (r, (got, want)) in out_i8.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 2e-2 * denom,
                "{name}: int8 matvec row {r}: {got:e} vs {want:e}"
            );
        }
    }
}

#[test]
fn q8_quantize_round_trip() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let (q8, scales) = quantize_to_q8(&x);
    assert_eq!(q8.len(), 64);
    assert_eq!(scales.len(), 2); // 64 / 32
    assert!(scales.iter().all(|&s| s >= 0.0));
}

#[test]
fn q8_zero_input() {
    let x = vec![0.0f32; 32];
    let (q8, scales) = quantize_to_q8(&x);
    assert!(q8.iter().all(|&v| v == 0));
    assert!(scales[0] == 0.0);
}
