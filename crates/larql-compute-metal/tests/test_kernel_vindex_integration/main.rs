//! End-to-end regression tests that require a real vindex on disk, plus
//! stage-level composition tests for `stages::residual` and
//! `stages::quant_matvec` encode helpers.
//!
//! The vindex test (`q4kf_proj_matches_cpu_on_real_vindex_bytes`) is
//! gated on the vindex file existing at
//! `../../output/gemma3-4b-q4k-v2.vindex` — it skips cleanly otherwise.
//!
//! Stage tests drive the `encode_post_attn`, `encode_post_ffn`, and
//! `quant_matvec::encode` helpers and compare against CPU references,
//! pinning down composition bugs that individual shader tests miss.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::prelude::*;
use ndarray::Array2;

#[path = "../common/mod.rs"]
mod common;
use common::{get_metal, max_diff};

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

// Stage-level composition tests.
//
// Each test drives a `stages::*::encode*` helper and compares the
// composed output against a CPU reference computed in the test.
// These pin down composition bugs that individual shader tests miss:
//   - wrong format dispatch inside `quant_matvec::encode`,
//   - off-by-one buffer offsets in `encode_post_attn`,
//   - pre-norm vs post-norm branching in `encode_post_ffn`,
//   - Q8 quant emission when FFN input needs Q8.

fn build_pipeline(device: &metal::Device, name: &str) -> metal::ComputePipelineState {
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&lib.get_function(name, None).unwrap())
        .unwrap()
}

fn read_f32_buf(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

/// CPU reference: RMS-norm with llama-style offset on the weight.
fn cpu_rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0f32 / (ms + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, wv)| v * inv * (offset + wv))
        .collect()
}

mod q4kf_proj_real_vindex;
mod q4kf_proj_real_vindex_more;
