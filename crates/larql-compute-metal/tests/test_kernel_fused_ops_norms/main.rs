//! Correctness tests for norm, residual, and quantization Metal shaders:
//! `rms_norm` (with offset, zero offset, large vector SIMD cooperative),
//! `residual_norm` (SIMD cooperative), `residual_add`, `quantize_q8`,
//! and fused ops: `rms_norm_q8`, `residual_norm` (vs CPU), `residual_norm_q8`.
//!
//! All tests compare Metal shader output to a CPU reference implementation.

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "../common/mod.rs"]
mod common;
use common::max_diff;

mod cooperative_simd_norm_large_vector_multi;
mod fused_ops_rms_norm_q8_residual_norm_resi;
mod residual_add;
mod rms_norm_with_offset;
