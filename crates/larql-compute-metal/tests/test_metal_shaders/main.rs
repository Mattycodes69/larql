//! Per-shader correctness tests for Metal compute kernels.
//!
//! Each test runs the Metal shader and compares output against
//! a CPU reference implementation. Tests both correctness and
//! that the shader compiles and dispatches successfully.
//!
//! Run with: cargo test -p larql-compute --features gpu

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::q4;
use larql_compute::cpu::q4::quantize_q4_0;
use larql_compute::prelude::*;
use ndarray::Array2;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn get_metal() -> larql_compute_metal::MetalBackend {
    larql_compute_metal::MetalBackend::new().expect("Metal device required for these tests")
}

// Shader correctness tests — each shader vs CPU reference

// New shader kernel tests (model-agnostic compute alignment)

mod fused_attention_shader;
mod geglu;
mod new_shader_kernel_tests_model_agnostic_c;
mod q4_matvec;
mod q4kf_proj_production_single_projection_q;
mod q4kf_qkv_proj_production_fused_q_k_v_q4;
mod rope_shader;
mod shader_compilation;
mod smoke_test_full_pipeline_produces_output;
