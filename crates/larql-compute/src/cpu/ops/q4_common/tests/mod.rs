use super::*;

use super::matvec_f32::dot_256_f32;
#[cfg(target_arch = "aarch64")]
use super::matvec_f32::{dot_256_f32_neon, dot_256_f32_scalar};

/// Reference implementation kept here as the correctness oracle for
/// the bit-manipulation `f16_to_f32`.  Mirrors the previous (slow)
/// version that used `2.0f32.powi(...)`.  The new fast path must
/// match this for all 65536 possible f16 inputs except canonical NaN
/// payload preservation (handled in the test).
fn f16_to_f32_powi_reference(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 {
        -val
    } else {
        val
    }
}

// branch decoded 2× too large while the exhaustive test silently
// verified a test-local `f16_to_f32` that shadowed the production fn.
// Assertions below call through `super::` so a future shadow cannot
// re-mask the production path. ──

/// Deterministic pseudo-random data at a chosen magnitude. Magnitude
/// ~4e-4 drives the per-super-block `d`/`dmin` f16 scales into the
/// subnormal range (< 2^-14), the regime the 2× bug corrupted.
fn seeded_data(n: usize, magnitude: f32, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * magnitude
        })
        .collect()
}

/// True if any Q4_K super-block in `bytes` carries a subnormal f16
/// `d` or `dmin` (exp bits zero, mantissa nonzero).
fn q4k_has_subnormal_scale(bytes: &[u8]) -> bool {
    bytes.chunks_exact(144).any(|b| {
        let d = u16::from_le_bytes([b[0], b[1]]);
        let dmin = u16::from_le_bytes([b[2], b[3]]);
        let sub = |v: u16| (v >> 10) & 0x1F == 0 && (v & 0x3FF) != 0;
        sub(d) || sub(dmin)
    })
}

/// Test alias — dispatches to the canonical module-scope implementation.
fn dequantize_q4_k_llama(data: &[u8], n_elements: usize) -> Vec<f32> {
    super::dequantize_q4_k(data, n_elements)
}

mod f16_subnormal_regression_battery_2026_06;
mod f32_to_f16_edge_cases;
mod quantize_q4_0_tests;
