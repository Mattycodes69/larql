//! q4kf_proj: production single-projection Q4_K (GGUF 144-byte)
//! q4kf_proj: Gemma-3-4B Q-projection shape (hidden=2560, rows=2048).

use super::*;

//
// This is the shader that `dispatch_full_pipeline` actually dispatches for
// Q4_K gate/up/down/o projections. If this diverges from CPU dequantise
// everything downstream is wrong.
#[test]
fn q4kf_proj_matches_cpu_reference() {
    let metal = get_metal();
    // Use a shape representative of a real Q4_K projection: hidden=1536,
    // rows=512 (matches Gemma 4 sliding-layer KV dim).
    let hidden = 1536usize;
    let rows = 512usize;

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.001).cos() * 0.6)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.003).sin()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);
    assert_eq!(q4k.len(), rows * 144 * (hidden / 256));

    // CPU reference: dequantise + straightforward gemv.
    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, rows * hidden).unwrap();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_out[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    // Metal: dispatch q4kf_proj directly (not via Backend trait, which
    // routes to the legacy q4k_matvec pipeline).
    use larql_compute_metal::shaders::q4kf_qkv_proj as q4kf;
    let w_buf = metal.bufs().get_bytes(&q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4kf_proj_pipeline.state);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n = rows as u32;
    let k = hidden as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute_metal::buffers::read_buffer_f32(&out_buf, rows);
    // Also report per-bucket scale so silent scale bugs are visible.
    let met_max = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ratio = cpu_max / met_max.max(1e-9);
    eprintln!("q4kf_proj[{rows}x{hidden}]  cpu_max={cpu_max:.3e}  metal_max={met_max:.3e}  ratio_cpu/metal={ratio:.3}");
    let max_diff = metal_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 0.3,
        "q4kf_proj diverged from CPU: max_diff={max_diff} (rows={rows})"
    );
    assert!(
        metal_out.iter().all(|v| v.is_finite()),
        "q4kf_proj emitted NaN/Inf"
    );
}

//
// The 1536/512 test above uses Gemma-4-E2B dims; this variant exercises the
// `hidden % 1024 != 0` edge case (hidden=2560 → 10 superblocks) which the
// q4kf_proj inner loop handles via `for ib = ix; ib < nb; ib += 4` where
// lanes 0-1 process 3 superblocks each and lanes 2-3 process 2. Regression
// guard for divergence seen in end-to-end Gemma 3 4B Metal inference.
#[test]
fn q4kf_proj_matches_cpu_reference_gemma3_shape() {
    let metal = get_metal();
    let hidden = 2560usize; // Gemma 3 4B hidden_size
    let rows = 2048usize; // Gemma 3 4B q_dim (8 heads × 256 head_dim... wait 4*256=1024, see)

    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.0007).sin() * 0.5)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.002).cos()).collect();

    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    let dequant = larql_models::quant::ggml::dequantize_q4_k(&q4k, rows * hidden).unwrap();
    let mut cpu_out = vec![0.0f32; rows];
    for row in 0..rows {
        cpu_out[row] = (0..hidden).map(|k| dequant[row * hidden + k] * x[k]).sum();
    }

    use larql_compute_metal::shaders::q4kf_qkv_proj as q4kf;
    let w_buf = metal.bufs().get_bytes(&q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4kf_proj_pipeline.state);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    let n = rows as u32;
    let k = hidden as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    let num_tgs = (rows as u64).div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute_metal::buffers::read_buffer_f32(&out_buf, rows);
    let met_max = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_max = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let ratio = cpu_max / met_max.max(1e-9);
    eprintln!("q4kf_proj[{rows}x{hidden}]  cpu_max={cpu_max:.3e}  metal_max={met_max:.3e}  ratio={ratio:.3}");
    let max_diff = metal_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        ratio > 0.95 && ratio < 1.05,
        "q4kf_proj scale off for hidden=2560: cpu_max/metal_max={ratio:.3} (should be ~1.0)",
    );
    assert!(
        max_diff < 1.0,
        "q4kf_proj[{rows}x{hidden}] max_diff={max_diff}"
    );
}
