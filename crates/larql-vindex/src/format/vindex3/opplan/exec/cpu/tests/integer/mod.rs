//! CPU-5 instrument controls: the integer-domain kernels, before any arm
//! of the quality gate is allowed to mean anything.
//!
//! Nothing here is a quality result. These are the gates the
//! pre-registration (`bench/prompts/quality-bank-1/CPU5-Q4Q8-QUALITY.md`)
//! requires to pass FIRST, because an arm run through an unproven kernel
//! measures the kernel and not the format.
//!
//! Each kernel is judged against the format's own portable DEFINITION and
//! never against the original f32 weights. At 4.5 bits the quantiser's
//! error is large enough to hide almost any kernel bug inside a tolerance
//! chosen for it.

use super::super::integer::{quantise_activation, Bf16xQ8, Q4xQ8, Q8xQ8};
use super::super::kernels::FusedBf16;
use super::super::physical::{arithmetic_arm, ArithmeticArm};
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::fixtures::lcg_values;
use crate::format::vindex3::opplan::exec::quantise::{
    quantise_q4_for_test, quantise_q8_for_test, Q4_BLOCK, Q8_BLOCK,
};
use crate::format::vindex3::opplan::exec::weights::LoadedWeight;

/// Shapes that exercise a single block, several blocks, and the real
/// model's input width.
const SHAPES: [usize; 3] = [Q8_BLOCK, Q8_BLOCK * 3, 5120];

fn q8_parts(w: &[f32], in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    match quantise_q8_for_test(w, in_dim) {
        LoadedWeight::Q8 { codes, scales, .. } => (codes, scales),
        _ => panic!("the q8 quantiser must produce the q8 variant"),
    }
}

fn q4_parts(w: &[f32], in_dim: usize) -> (Vec<u8>, Vec<f32>) {
    match quantise_q4_for_test(w, in_dim) {
        LoadedWeight::Q4 { packed, scales } => (packed, scales),
        _ => panic!("the q4 quantiser must produce the q4 variant"),
    }
}

/// A residual-stream-shaped activation: a few channels tens of times the
/// RMS, which is the regime the real model is in (peak/rms 28-36 at
/// depth).
fn outlier_activation(n: usize, seed: u64) -> Vec<f32> {
    let mut x = lcg_values(n, seed);
    for (i, v) in x.iter_mut().enumerate() {
        if i % 173 == 0 {
            *v *= 45.0;
        }
    }
    x
}

mod integer_basics;
mod integer_basics_2;
// SDOT kernel parity: the kernels exist on aarch64 only.
#[cfg(target_arch = "aarch64")]
mod integer_basics_3;
