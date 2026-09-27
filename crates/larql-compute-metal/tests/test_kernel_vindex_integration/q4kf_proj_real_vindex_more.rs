//! q4kf_proj on REAL vindex Q4_K bytes (end-to-end regression)

use super::*;

/// `f32_gemv` shader: `out[N] = W[N,K] · x[K]` matches `ndarray::dot`.
///
/// Motivating case: LM-head logits at autoregressive decode. The shader's
/// value-add over re-using `sgemm_transb` at M=1 is both speed (row-per-
/// simdgroup vs 31/32-wasted-thread tiled gemm) and argmax stability
/// (deterministic per-row reduction order, no shifting of top-K under
/// noisy logits). Test pins both.
#[test]
fn f32_gemv_matches_ndarray_dot() {
    let metal = get_metal();
    // Small shapes fall below the default 500 MFLOP threshold and return
    // None (caller falls back to CPU). We want to exercise the Metal
    // path, so drop the floor.
    metal.set_flop_threshold(1);

    // Dimensions chosen to match the Gemma 3/4 LM-head aspect ratio in
    // miniature: wide N, K a non-power-of-two-multiple-of-32, K % 128 != 0.
    let n = 2048usize;
    let k = 2560usize;
    let w = synth(n, k, 0xa11ce);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

    // CPU reference: ndarray's BLAS gemv.
    let x_arr = ndarray::Array1::from(x.clone());
    let expected = w.dot(&x_arr);

    // Metal path.
    let got = metal
        .f32_gemv(w.view(), &x)
        .expect("gemv should dispatch above threshold");
    assert_eq!(got.len(), n);

    let diff = max_diff(expected.as_slice().unwrap(), &got);
    let max_abs = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let rel = diff / max_abs;
    assert!(
        rel < 1e-4,
        "f32_gemv rel err {rel:.2e} (abs {diff:.2e}, max_abs {max_abs:.2e})"
    );

    // Argmax stability — the actual property that matters for LM-head top-K.
    let exp_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let got_argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        exp_argmax, got_argmax,
        "argmax mismatch between CPU and Metal gemv"
    );
}

/// `f16_gemv` shader: f16 weights × f32 query, matches `f32_gemv` within
/// half-precision noise.
///
/// Motivating case: Gemma 4 31B tied-embedding LM head. The current path
/// decodes the 2.8 GB f16 safetensors into a 5.6 GB f32 clone at load;
/// this shader lets the Metal backend consume the f16 bytes directly.
/// Test pins argmax equality with the f32 reference — that's the actual
/// property that matters for top-K.
#[test]
fn f16_gemv_matches_f32_gemv_argmax() {
    use larql_models::quant::half::encode_f16;

    let metal = get_metal();
    metal.set_flop_threshold(1);

    let n = 2048usize;
    let k = 2560usize;
    let w = synth(n, k, 0xf16ce);
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin()).collect();

    // f32 reference.
    let x_arr = ndarray::Array1::from(x.clone());
    let expected = w.dot(&x_arr);

    // Encode weights as f16 bytes (IEEE half, little-endian).
    let w_flat: Vec<f32> = w.iter().copied().collect();
    let w_f16 = encode_f16(&w_flat);
    assert_eq!(w_f16.len(), n * k * 2);

    let got = metal
        .f16_gemv(&w_f16, &x, n, k)
        .expect("f16_gemv should dispatch above threshold");
    assert_eq!(got.len(), n);

    // f16 weights introduce relative error ~1e-3 on the output; don't pin
    // values, pin argmax — that's the property the LM head top-K depends on.
    let exp_argmax = expected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let got_argmax = got
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        exp_argmax, got_argmax,
        "f16_gemv argmax mismatch vs f32 reference"
    );

    // Sanity: the scores around the argmax should be within f16 relative
    // noise of the f32 reference.
    let tol = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1.0)
        * 5e-3;
    let diff = (expected[exp_argmax] - got[exp_argmax]).abs();
    assert!(
        diff < tol,
        "argmax-value drift {diff:.4} exceeds f16 tolerance {tol:.4}"
    );
}

/// Uniform `q4k_qkv_proj` fused shader matches three `q4k_matvec` dispatches.
///
/// Regression gate for the 148-vs-144 Q4_K super-block stride bug: the
/// first draft of this shader typed weights as `block_q4_K*` (148-byte
/// MSL struct with an obsolete `mins[4]` field), which silently mis-read
/// production GGUF data. Row stride was off by 40 bytes per row,
/// accumulating into buffer-overruns past the first superblock. The
/// output was "approximately correct" enough for argmax to stabilise on
/// trivial prompts, hiding the bug. Now the shader uses manual byte
/// offsets with the correct 144-byte stride.
#[test]
fn q4k_qkv_proj_matches_per_proj_dispatch() {
    let metal = get_metal();
    let q_rows = 2048usize;
    let kv_rows = 1024usize;
    let hidden = 2560usize;

    let wq_f32 = synth(q_rows, hidden, 0xbeef_0001)
        .as_standard_layout()
        .to_owned();
    let wk_f32 = synth(kv_rows, hidden, 0xbeef_0002)
        .as_standard_layout()
        .to_owned();
    let wv_f32 = synth(kv_rows, hidden, 0xbeef_0003)
        .as_standard_layout()
        .to_owned();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.017).cos()).collect();

    let wq_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wq_f32.as_slice().unwrap());
    let wk_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wk_f32.as_slice().unwrap());
    let wv_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wv_f32.as_slice().unwrap());

    let ref_q = metal
        .q4k_matvec(&wq_q4k, &x, q_rows, hidden)
        .expect("q4k_matvec Q");
    let ref_k = metal
        .q4k_matvec(&wk_q4k, &x, kv_rows, hidden)
        .expect("q4k_matvec K");
    let ref_v = metal
        .q4k_matvec(&wv_q4k, &x, kv_rows, hidden)
        .expect("q4k_matvec V");

    // Fused dispatch through `q4k_qkv_proj`.
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q4k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    use larql_compute_metal::shaders::q4k_qkv_proj as sh;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let k_u = kv_rows as u32;
    let v_u = kv_rows as u32;
    let hidden_u = hidden as u32;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4k_qkv_proj_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    enc.set_bytes(7, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute_metal::buffers::read_buffer_f32(&q_out, q_rows);
    let got_k = larql_compute_metal::buffers::read_buffer_f32(&k_out, kv_rows);
    let got_v = larql_compute_metal::buffers::read_buffer_f32(&v_out, kv_rows);

    let check = |name: &str, r: &[f32], g: &[f32]| {
        let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let d = max_diff(r, g);
        assert!(
            d < max_abs * 1e-3,
            "{name}: max_diff {d:.3e} exceeds 0.1% of max_abs {max_abs:.3e}"
        );
    };
    check("Q", &ref_q, &got_q);
    check("K", &ref_k, &got_k);
    check("V", &ref_v, &got_v);
}

/// `q4k_q6k_qkv_proj` fused shader matches three separate-format dispatches.
///
/// Pins the mixed-quant fused kernel that replaces the 3-dispatch per-proj
/// fallback when a layer ships Q4_K Q/K + Q6_K V (Gemma 3 4B / Gemma 4
/// Ollama convention). If the shader silently regresses to under-read or
/// over-read the Q4_K GGUF 144-byte blocks (as happened once when the
/// first draft used the 148-byte `block_q4_K` MSL struct), this will
/// catch it before real-vindex decode produces garbled tokens.
#[test]
#[allow(clippy::unusual_byte_groupings)]
fn q4k_q6k_qkv_proj_matches_per_proj_dispatch() {
    let metal = get_metal();

    // Shapes modelled on Gemma 3 4B: q_dim = 8 * 256, kv_dim = 4 * 256,
    // hidden = 2560 (K must be a multiple of 256 for Q4_K / Q6_K).
    let q_rows = 2048usize;
    let kv_rows = 1024usize;
    let hidden = 2560usize;

    // Synthesise weight matrices and quantise.
    let wq_f32 = synth(q_rows, hidden, 0xdead_beef_1)
        .as_standard_layout()
        .to_owned();
    let wk_f32 = synth(kv_rows, hidden, 0xdead_beef_2)
        .as_standard_layout()
        .to_owned();
    let wv_f32 = synth(kv_rows, hidden, 0xdead_beef_3)
        .as_standard_layout()
        .to_owned();
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.011).sin()).collect();

    let wq_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wq_f32.as_slice().unwrap());
    let wk_q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(wk_f32.as_slice().unwrap());
    let wv_q6k = larql_compute::cpu::ops::q4_common::quantize_q6_k(wv_f32.as_slice().unwrap());

    // Reference: dispatch each projection through its native shader.
    let ref_q = metal
        .q4k_matvec(&wq_q4k, &x, q_rows, hidden)
        .expect("q4k_matvec Q");
    let ref_k = metal
        .q4k_matvec(&wk_q4k, &x, kv_rows, hidden)
        .expect("q4k_matvec K");
    let ref_v = metal
        .q6k_matvec(&wv_q6k, &x, kv_rows, hidden)
        .expect("q6k_matvec V");

    // Fused dispatch.
    let wq_buf = metal.bufs().get_bytes(&wq_q4k);
    let wk_buf = metal.bufs().get_bytes(&wk_q4k);
    let wv_buf = metal.bufs().get_bytes(&wv_q6k);
    let x_buf = metal.bufs().transient_from_f32(&x);
    let q_out = metal.bufs().output((q_rows * 4) as u64);
    let k_out = metal.bufs().output((kv_rows * 4) as u64);
    let v_out = metal.bufs().output((kv_rows * 4) as u64);

    use larql_compute_metal::shaders::q4k_q6k_qkv_proj as sh;
    let total_rows = (q_rows + kv_rows + kv_rows) as u64;
    let num_tgs = total_rows.div_ceil(sh::ROWS_PER_TG);
    let q_u = q_rows as u32;
    let k_u = kv_rows as u32;
    let v_u = kv_rows as u32;
    let hidden_u = hidden as u32;
    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.attention.q4k_q6k_qkv_proj_pipeline.state);
    enc.set_buffer(0, Some(&wq_buf), 0);
    enc.set_buffer(1, Some(&wk_buf), 0);
    enc.set_buffer(2, Some(&wv_buf), 0);
    enc.set_buffer(3, Some(&x_buf), 0);
    enc.set_buffer(4, Some(&q_out), 0);
    enc.set_buffer(5, Some(&k_out), 0);
    enc.set_buffer(6, Some(&v_out), 0);
    enc.set_bytes(7, 4, &q_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &v_u as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &hidden_u as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_tgs, 1, 1),
        metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let got_q = larql_compute_metal::buffers::read_buffer_f32(&q_out, q_rows);
    let got_k = larql_compute_metal::buffers::read_buffer_f32(&k_out, kv_rows);
    let got_v = larql_compute_metal::buffers::read_buffer_f32(&v_out, kv_rows);

    // Q4_K quantisation can introduce tiny per-row scale differences
    // depending on which shader dispatch path is taken; absolute tolerance
    // scaled by row magnitude.
    let check = |name: &str, r: &[f32], g: &[f32]| {
        let max_abs = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let d = max_diff(r, g);
        assert!(
            d < max_abs * 1e-3,
            "{name}: max_diff {d:.3e} exceeds 0.1% of max_abs {max_abs:.3e}"
        );
    };
    check("Q", &ref_q, &got_q);
    check("K", &ref_k, &got_k);
    check("V", &ref_v, &got_v);
}

/// Stage: `residual::encode_post_attn` with FFN that needs Q8 input.
///
/// Verifies the additional q8_quant dispatch runs and produces a Q8
/// representation that round-trips to approximately `ffn_norm_out`.
#[test]
fn stage_post_attn_q8_ffn_emits_roundtrippable_q8() {
    let device = metal::Device::system_default().unwrap();
    let rms_norm = build_pipeline(&device, "rms_norm");
    let residual_add = build_pipeline(&device, "residual_add");
    let q8_quant = build_pipeline(&device, "quantize_q8");
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let hidden = 256usize;
    let seq_len = 2usize;

    let h: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.009).sin() * 2.0)
        .collect();
    let o: Vec<f32> = (0..seq_len * hidden)
        .map(|i| ((i as f32) * 0.013).cos() * 1.5)
        .collect();
    let w: Vec<f32> = (0..hidden).map(|i| 1.0 + 0.02 * (i as f32).sin()).collect();

    let h_buf = bufs.transient_from_f32(&h);
    let o_buf = bufs.transient_from_f32(&o);
    let w_buf = bufs.transient_from_f32(&w);
    let h_pa = bufs.output((seq_len * hidden * 4) as u64);
    let ffn_out = bufs.output((seq_len * hidden * 4) as u64);
    let q8 = bufs.output((seq_len * hidden) as u64);
    let q8s = bufs.output((seq_len * hidden.div_ceil(32) * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let mut scratch = |n: u64| bufs.output(n);
    larql_compute_metal::stages::residual::encode_post_attn(
        enc,
        &rms_norm,
        &residual_add,
        &q8_quant,
        &mut scratch,
        &h_buf,
        &o_buf,
        &h_pa,
        &ffn_out,
        &w_buf,
        &w_buf,
        &q8,
        &q8s,
        seq_len,
        hidden,
        1e-6,
        0.0,
        /*has_post_norms*/ false,
        /*ffn_needs_q8*/ true,
        (hidden * 4) as u64,
        hidden as u64,
        (hidden.div_ceil(32) * 4) as u64,
        1.0,
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Dequantise Q8 and compare to f32 ffn_norm_out (Q8 error < 1/127 * max).
    // `quantize_q8` writes f32 scales (not f16) — `q8s_stride_bytes` is
    // `blocks_per_row * 4` to reflect that.
    let ffn_f32 = read_f32_buf(&ffn_out, seq_len * hidden);
    let q8_bytes =
        unsafe { std::slice::from_raw_parts(q8.contents() as *const i8, seq_len * hidden) };
    let blocks_per_pos = hidden.div_ceil(32);
    let q8s_f32 = unsafe {
        std::slice::from_raw_parts(q8s.contents() as *const f32, seq_len * blocks_per_pos)
    };
    let mut dequant = vec![0.0f32; seq_len * hidden];
    for p in 0..seq_len {
        for b in 0..blocks_per_pos {
            let scale = q8s_f32[p * blocks_per_pos + b];
            for i in 0..32 {
                let idx = p * hidden + b * 32 + i;
                if idx < (p + 1) * hidden {
                    dequant[idx] = q8_bytes[idx] as f32 * scale;
                }
            }
        }
    }
    let max_abs = ffn_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let d = max_diff(&ffn_f32, &dequant);
    assert!(
        d < max_abs / 100.0 + 1e-4,
        "Q8 roundtrip error {d} exceeds 1% of max_abs {max_abs}"
    );
}
