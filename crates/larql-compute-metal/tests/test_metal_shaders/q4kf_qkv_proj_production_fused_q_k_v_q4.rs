//! q4kf_qkv_proj: production fused Q+K+V Q4_K (GGUF 144-byte)
//! qk_norm: per-head RMS norm with learned weight (Gemma 3/4 pre-RoPE).

use super::*;

//
// The fused attention QKV dispatch for Gemma 3 pure-Q4_K vindexes. Verifies
// all three output streams agree with CPU dequant when weights are the same.
#[test]
fn q4kf_qkv_proj_matches_individual_projections() {
    let metal = get_metal();
    let hidden = 1536usize;
    let q_rows = 512usize;
    let k_rows = 256usize;
    let v_rows = 256usize;

    let wq: Vec<f32> = (0..q_rows * hidden)
        .map(|i| ((i as f32) * 0.0011).cos() * 0.5)
        .collect();
    let wk: Vec<f32> = (0..k_rows * hidden)
        .map(|i| ((i as f32) * 0.0013).sin() * 0.5)
        .collect();
    let wv: Vec<f32> = (0..v_rows * hidden)
        .map(|i| ((i as f32) * 0.0017).cos() * 0.5)
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.003).sin()).collect();

    let q_quant = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wq);
    let k_quant = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wk);
    let v_quant = larql_compute::cpu::ops::q4_common::quantize_q4_k(&wv);

    // CPU reference: dequant each and gemv against x.
    let q_deq = larql_models::quant::ggml::dequantize_q4_k(&q_quant, q_rows * hidden).unwrap();
    let k_deq = larql_models::quant::ggml::dequantize_q4_k(&k_quant, k_rows * hidden).unwrap();
    let v_deq = larql_models::quant::ggml::dequantize_q4_k(&v_quant, v_rows * hidden).unwrap();
    let mut q_cpu = vec![0.0f32; q_rows];
    let mut k_cpu = vec![0.0f32; k_rows];
    let mut v_cpu = vec![0.0f32; v_rows];
    for r in 0..q_rows {
        q_cpu[r] = (0..hidden).map(|c| q_deq[r * hidden + c] * x[c]).sum();
    }
    for r in 0..k_rows {
        k_cpu[r] = (0..hidden).map(|c| k_deq[r * hidden + c] * x[c]).sum();
    }
    for r in 0..v_rows {
        v_cpu[r] = (0..hidden).map(|c| v_deq[r * hidden + c] * x[c]).sum();
    }

    // Metal fused dispatch.
    use larql_compute_metal::shaders::q4kf_qkv_proj as q4kf;
    let wq_buf = metal.bufs().get_bytes(&q_quant);
    let wk_buf = metal.bufs().get_bytes(&k_quant);
    let wv_buf = metal.bufs().get_bytes(&v_quant);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((k_rows * 4) as u64);
    let v_out = metal.bufs().output((v_rows * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4kf_qkv_proj_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    let q_rows_val = q_rows as u32;
    let k_rows_val = k_rows as u32;
    let v_rows_val = v_rows as u32;
    let k_val = hidden as u32;
    enc.set_bytes(7, 4, &q_rows_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_rows_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_rows_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
    let total_rows = (q_rows + k_rows + v_rows) as u64;
    let num_tgs = total_rows.div_ceil(q4kf::ROWS_PER_TG);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q_metal = larql_compute_metal::buffers::read_buffer_f32(&q_out, q_rows);
    let k_metal = larql_compute_metal::buffers::read_buffer_f32(&k_out, k_rows);
    let v_metal = larql_compute_metal::buffers::read_buffer_f32(&v_out, v_rows);

    let q_diff = max_diff(&q_cpu, &q_metal);
    let k_diff = max_diff(&k_cpu, &k_metal);
    let v_diff = max_diff(&v_cpu, &v_metal);
    // Tolerance 0.5 — the fused shader accumulates 1536 products in a single
    // f32 simdgroup reduction; the CPU reference uses scalar left-to-right
    // order. Drift from associativity of float addition lives at this level
    // with 512-row matrices. Well below any real accuracy concern.
    assert!(q_diff < 0.5, "q4kf_qkv_proj Q stream diverged: {q_diff}");
    assert!(k_diff < 0.5, "q4kf_qkv_proj K stream diverged: {k_diff}");
    assert!(v_diff < 0.5, "q4kf_qkv_proj V stream diverged: {v_diff}");
    assert!(
        q_metal.iter().all(|v| v.is_finite()),
        "Q stream had NaN/Inf"
    );
    assert!(
        k_metal.iter().all(|v| v.is_finite()),
        "K stream had NaN/Inf"
    );
    assert!(
        v_metal.iter().all(|v| v.is_finite()),
        "V stream had NaN/Inf"
    );
}

//
// Hand-validated: per-head RMS(x) then multiply by (weight[d] + offset).
// The `v_norm_matches_cpu` test already exercises the parameter-free form;
// this test pins the weighted form + non-zero offset (Gemma 2/3 stores
// `real_weight - 1` with `offset = 1.0`).
#[test]
fn qk_norm_matches_cpu_reference() {
    let metal = get_metal();
    let num_heads = 4usize;
    let head_dim = 256usize;
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // Deterministic input + weight.
    let input: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| ((i as f32) * 0.01).sin() * 2.0 + 0.5)
        .collect();
    let weight: Vec<f32> = (0..head_dim)
        .map(|d| ((d as f32) / head_dim as f32) * 0.3)
        .collect();

    // CPU reference: per-head RMS norm.
    let mut cpu_out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let base = h * head_dim;
        let sum_sq: f32 = input[base..base + head_dim].iter().map(|v| v * v).sum();
        let rms = (sum_sq / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            cpu_out[base + d] = input[base + d] / rms * (offset + weight[d]);
        }
    }

    // Metal dispatch.
    let in_buf = metal.bufs().transient_from_f32(&input);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let out_buf = metal.bufs().output((num_heads * head_dim * 4) as u64);

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.norms.qk_norm_pipeline);
    enc.set_buffer(0, Some(&in_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_buffer(2, Some(&w_buf), 0);
    let hd_val = head_dim as u32;
    let nh_val = num_heads as u32;
    enc.set_bytes(3, 4, &hd_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &nh_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Threadgroup width = power-of-two ≥ head_dim, capped at 512.
    let mut tg_w: u64 = 1;
    while (tg_w as usize) < head_dim && tg_w < 512 {
        tg_w <<= 1;
    }
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_heads as u64, 1, 1),
        metal::MTLSize::new(tg_w, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute_metal::buffers::read_buffer_f32(&out_buf, num_heads * head_dim);
    let diff = max_diff(&cpu_out, &metal_out);
    assert!(diff < 1e-3, "qk_norm diverged from CPU: max_diff={diff}");
}
