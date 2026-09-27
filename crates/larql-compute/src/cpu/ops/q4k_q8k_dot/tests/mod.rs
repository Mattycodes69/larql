use super::common::BLOCK_BYTES;
#[cfg(target_arch = "x86_64")]
use super::q4k_avx2::q4k_q8k_matvec_avx2;
use super::q6k::Q6K_BLOCK_BYTES;
use super::*;
use crate::cpu::ops::q4_common::{q4k_matvec_into, quantize_q4_k, quantize_q6_k};

//
// The parallel entry is the single source for every quantised projection
// on the decode path. One fixture serves the refusal cases and the parity
// case beside them, so the same weights prove both halves of the change:
// an invalid shape refuses with the output untouched, and the unchanged
// valid shape still computes exactly what the scalar reference computes.

/// `(packed Q4_K weights, Q8_K activation, rows, cols)` for the public-path
/// tests. Five rows keeps a partial final chunk out of the picture; two
/// super-blocks per row keeps the scale/min unpack exercised twice.
fn public_path_fixture() -> (Vec<u8>, Q8KActivation, usize, usize) {
    let rows = 5;
    let cols = 512;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).sin() * 1.7).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.007).cos() * 0.6)
        .collect();
    (quantize_q4_k(&w_f32), quantize_x_to_q8k(&x), rows, cols)
}

mod fused_and_q6k_paths;
mod public_execution_path_q4k_q8k_matvec_par;
mod q8k_quantize_and_matvec;
