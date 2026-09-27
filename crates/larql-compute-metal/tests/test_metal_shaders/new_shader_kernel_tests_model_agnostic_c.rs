//! New shader kernel tests (model-agnostic compute alignment)
//! Q6_K diagnostic: single-row, single-superblock with dequantize reference.
//! Q6_K multi-row: find the row where divergence starts.
//! Q6_K multi-superblock: the real-world failure mode.
//! f16 subnormal regression: rows with small amax (d in subnormal range)
//! Q4_K: single superblock matches CPU dequantize + gemv
//! Q4_K: multi-superblock rows, multi-row batch
//! GEGLU GELU-tanh: no NaN on gate values near the tanh-overflow threshold

use super::*;

#[test]
fn new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = [
        "silu",
        "gelu_tanh", // standalone activations
        "layer_norm",
        "layer_norm_no_bias", // LayerNorm
        "v_norm",             // V-norm
        "scale_vector",       // per-layer scalar
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

#[test]
fn silu_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.ffn.silu_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute_metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "SiLU standalone max diff {diff} exceeds 1e-5");
}

#[test]
fn gelu_tanh_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| {
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            let t = (c * (x + 0.044715 * x * x * x)).tanh();
            0.5 * x * (1.0 + t)
        })
        .collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.ffn.gelu_tanh_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute_metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(
        diff < 1e-4,
        "GELU-tanh standalone max diff {diff} exceeds 1e-4"
    );
}

#[test]
fn layer_norm_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    // CPU reference
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n)
        .map(|i| (x[i] - mean) * inv_std * (weight[i] + offset) + bias[i])
        .collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let b_buf = metal.bufs().transient_from_f32(&bias);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.layer_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&b_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_no_bias_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n)
        .map(|i| (x[i] - mean) * inv_std * (weight[i] + offset))
        .collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.layer_norm_no_bias_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(
        diff < 1e-4,
        "LayerNorm (no bias) max diff {diff} exceeds 1e-4"
    );
}

#[test]
fn v_norm_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.02).collect();
    let eps = 1e-6f32;

    // CPU reference: parameter-free RMSNorm
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| v * rms).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.v_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "V-norm max diff {diff} exceeds 1e-5");
}

#[test]
fn scale_vector_matches_cpu() {
    let metal = get_metal();
    let n = 512;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let scalar = 0.73f32;
    let expected: Vec<f32> = input.iter().map(|v| v * scalar).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.scale_vector_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &scalar as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-6, "scale_vector max diff {diff} exceeds 1e-6");
}

#[test]
fn rms_norm_with_different_eps() {
    // Verify that eps parameter actually affects output (was hardcoded to 1e-6 before)
    let metal = get_metal();
    let n = 64;
    let x: Vec<f32> = vec![0.001; n]; // tiny values where eps matters
    let weight: Vec<f32> = vec![1.0; n];
    let offset = 0.0f32;

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let n_val = n as u32;

    // Run with eps=1e-6
    let out1 = metal.bufs().output((n * 4) as u64);
    let eps1 = 1e-6f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.norms.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out1), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps1 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run with eps=0.1 (much larger)
    let out2 = metal.bufs().output((n * 4) as u64);
    let eps2 = 0.1f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.norms.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out2), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps2 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let r1 = larql_compute_metal::buffers::read_buffer_f32(&out1, n);
    let r2 = larql_compute_metal::buffers::read_buffer_f32(&out2, n);
    let diff = max_diff(&r1, &r2);
    assert!(
        diff > 0.1,
        "Different eps values should produce different outputs (diff={diff})"
    );
}

// Pin the round-trip accuracy:
//   1. Quantize a known row via `quantize_q6_k` → 210 bytes.
//   2. CPU dequant via `dequantize_q6_k` and dot with x → reference answer.
//   3. Metal `q6k_matvec` → GPU answer.
//   4. Both must agree within 0.01 on a single superblock.
#[test]
fn q6k_single_superblock_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 256usize;

    // Row with a clean monotone gradient — easy to eyeball per-element error.
    let row: Vec<f32> = (0..hidden).map(|i| (i as f32 / 255.0) - 0.5).collect();
    // One-hot probe: each x[k]=1 selects column k, making the dot product equal
    // to row[k] after dequant round-trip.
    for probe_k in [0usize, 1, 2, 15, 16, 31, 32, 127, 128, 200, 255] {
        let mut x = vec![0.0f32; hidden];
        x[probe_k] = 1.0;

        let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&row);
        assert_eq!(q6k.len(), 210, "single superblock should be 210 bytes");

        let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, hidden).unwrap();
        let cpu_ref: f32 = dequant[probe_k] * x[probe_k];

        let metal_out = metal.q6k_matvec(&q6k, &x, 1, hidden).unwrap();

        let diff = (cpu_ref - metal_out[0]).abs();
        if diff > 0.01 {
            eprintln!(
                "probe_k={probe_k} row[k]={:.4} dequant[k]={:.4} cpu={:.4} metal={:.4} diff={:.4}",
                row[probe_k], dequant[probe_k], cpu_ref, metal_out[0], diff,
            );
        }
        assert!(
            diff < 0.01,
            "Q6_K probe at k={probe_k} diverged: cpu={cpu_ref} metal={} diff={diff}",
            metal_out[0],
        );
    }
}

//
// `hidden = 256` so each row is a single superblock. `rows = 32` (matches
// the existing `q6k_matvec_matches_cpu` failure). Prints per-row diff to
// isolate whether the bug is:
//   (a) first few rows only (threadgroup indexing broken past tg_id=0), or
//   (b) every row (format/decode bug), or
//   (c) every Nth row (simdgroup assignment broken).
#[test]
fn q6k_multi_row_diagnostic() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 32usize;

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    // Reference via dequantize_q6_k + CPU gemv.
    let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, rows * hidden).unwrap();
    let mut cpu_ref = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_ref[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    let metal_out = metal.q6k_matvec(&q6k, &x, rows, hidden).unwrap();

    let mut worst_row = 0usize;
    let mut worst_diff = 0.0f32;
    for row in 0..rows {
        let diff = (cpu_ref[row] - metal_out[row]).abs();
        // Row-input stats — help spot when a bad row aligns with a pathological
        // quantization bucket (very small amax, degenerate scales).
        let row_slice = &matrix[row * hidden..(row + 1) * hidden];
        let amax = row_slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let mean = row_slice.iter().sum::<f32>() / hidden as f32;
        eprintln!(
            "row {row:2}: cpu={:+.4} metal={:+.4} diff={:+.4}  amax={:.4} mean={:+.4}",
            cpu_ref[row], metal_out[row], diff, amax, mean,
        );
        if diff > worst_diff {
            worst_diff = diff;
            worst_row = row;
        }
    }
    assert!(
        worst_diff < 0.01,
        "Worst divergence at row {worst_row}: diff={worst_diff}",
    );
}

// hidden=1536 gives `superblocks = 6`. The shader's outer loop
// `for sb = lane; sb < 6; sb += 32` means lanes 6..31 are idle and lanes
// 0..5 each handle one superblock. Tests that `simd_sum` correctly
// aggregates contributions across idle and active lanes.
#[test]
fn q6k_multi_superblock_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 1536usize; // 6 superblocks
    let rows = 1usize;

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.003).sin() * 0.5)
        .collect();
    let x: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.007).cos() * 0.5)
        .collect();

    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, rows * hidden).unwrap();
    let cpu_ref: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();

    let metal_out = metal.q6k_matvec(&q6k, &x, rows, hidden).unwrap();

    let diff = (cpu_ref - metal_out[0]).abs();
    eprintln!(
        "q6k_multi_superblock cpu={cpu_ref:.4} metal={:.4} diff={diff:.4}",
        metal_out[0]
    );
    assert!(
        diff < 0.05,
        "Q6_K multi-superblock diverged: cpu={cpu_ref} metal={} diff={diff}",
        metal_out[0]
    );
}

//
// Prior to the `as_type<half>` fix in `common.rs::decode_f16_metal`, any
// row whose `d = amax/(31*127)` fell below the f16 min normal (~6.1e-5)
// was decoded as 0 on GPU, yielding silent all-zero rows in V projections.
// This test pins one such row: amax ≈ 0.15, d ≈ 3.8e-5 (subnormal).
#[test]
fn q6k_subnormal_d_matches_cpu() {
    let metal = get_metal();
    let hidden = 256usize;

    // Row with small amplitude so `d` lands in f16 subnormal range.
    let row: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.007).sin() * 0.15)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.003).cos()).collect();
    let q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(&row);

    let dequant = larql_models::quant::ggml::dequantize_q6_k(&q6k, hidden).unwrap();
    let cpu_ref: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();
    let metal_out = metal.q6k_matvec(&q6k, &x, 1, hidden).unwrap();

    // CPU and Metal must agree within 1% of cpu_ref (or 0.01 absolute).
    let tol = (cpu_ref.abs() * 0.01).max(0.01);
    assert!(
        (cpu_ref - metal_out[0]).abs() < tol,
        "Q6_K subnormal-d regression: cpu={cpu_ref} metal={} diff={}",
        metal_out[0],
        (cpu_ref - metal_out[0]).abs()
    );
    // Belt-and-suspenders: must not be exactly zero if input is non-trivial.
    assert!(
        metal_out[0].abs() > 1e-6,
        "Metal output zeroed out (flushed subnormal d?)"
    );
}

#[test]
fn q4k_single_superblock_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 256usize;

    let row: Vec<f32> = (0..hidden).map(|i| ((i as f32) / 127.0) - 1.0).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.01).sin()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&row);
    assert_eq!(
        q4k.len(),
        144,
        "single superblock should pack into 144 bytes GGUF"
    );

    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, hidden).unwrap();
    let cpu_ref: f32 = (0..hidden).map(|k| dequant[k] * x[k]).sum();
    let metal_out = metal.q4k_matvec(&q4k, &x, 1, hidden).unwrap();

    let diff = (cpu_ref - metal_out[0]).abs();
    assert!(
        diff < 0.05,
        "Q4_K single-superblock: cpu={cpu_ref} metal={} diff={diff}",
        metal_out[0]
    );
}

#[test]
fn q4k_multi_row_matches_dequantize_reference() {
    let metal = get_metal();
    let hidden = 1536usize; // 6 superblocks (Gemma 4 E2B sliding layer)
    let rows = 32usize;

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.001).cos() * 0.5)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.007).sin()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);
    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, rows * hidden).unwrap();
    let metal_out = metal.q4k_matvec(&q4k, &x, rows, hidden).unwrap();

    let mut worst = 0.0f32;
    for row in 0..rows {
        let expected: f32 = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
        let diff = (expected - metal_out[row]).abs();
        if diff > worst {
            worst = diff;
        }
    }
    assert!(
        worst < 0.5,
        "Q4_K multi-row worst diff={worst} exceeds 0.5 (expected < 0.1 for well-conditioned input)"
    );
}

//
// Before clamping, gate values around ±10 produce tanh arguments near ±50
// and Apple Silicon's `tanh(x) ≈ (exp(2x)-1)/(exp(2x)+1)` overflows to NaN.
#[test]
fn geglu_gelu_tanh_no_nan_on_large_gate() {
    let metal = get_metal();
    let n = 256usize;
    // Range gate through [-15, +15] to stress the tanh-overflow region.
    let gate: Vec<f32> = (0..n)
        .map(|i| ((i as f32 / n as f32) * 30.0) - 15.0)
        .collect();
    let up: Vec<f32> = vec![1.0; n];

    let g_buf = metal.bufs().transient_from_f32(&gate);
    let u_buf = metal.bufs().transient_from_f32(&up);
    let out_buf = metal.bufs().output((n * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.ffn.geglu_gelu_tanh_pipeline);
    enc.set_buffer(0, Some(&g_buf), 0);
    enc.set_buffer(1, Some(&u_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n_val = n as u32;
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out = larql_compute_metal::buffers::read_buffer_f32(&out_buf, n);
    let nan_count = out.iter().filter(|v| v.is_nan()).count();
    let inf_count = out.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(
        nan_count, 0,
        "geglu_gelu_tanh emitted {nan_count} NaN values"
    );
    assert_eq!(
        inf_count, 0,
        "geglu_gelu_tanh emitted {inf_count} Inf values"
    );
}
