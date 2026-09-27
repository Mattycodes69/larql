//! Q4 matvec
//! Q4 vecmat
//! Q4 f32 matvec (for transposed down)
//! Q4 pair batch
//! Multi-layer Q4 FFN
//! Buffer cache
//! Trait dispatch
//! Q8 matvec
//! Sparse Q4 matvec
//! Residual ops

use super::*;

#[test]
fn q4_matvec_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let rows = 10240;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "q4_matvec max diff {diff} exceeds 0.01");
}

#[test]
fn q4_matvec_small_matrix() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 128;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "small q4_matvec max diff {diff}");
}

#[test]
fn f16_gemv_topk1_matches_full_argmax() {
    let metal = get_metal();
    let n = 4096usize; // vocab dim
    let k = 256usize; // hidden dim — multiple of 32 keeps the gemv kernel happy
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.011).sin()).collect();
    let w_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.0007).cos()).collect();
    let w_f16 = larql_models::quant::half::encode_f16(&w_f32);

    let topk1 = metal
        .f16_gemv_topk1(&w_f16, &x, n, k)
        .expect("metal must produce a top-1 result");

    use larql_compute::MatMul;
    let scores = metal
        .f16_gemv_force(&w_f16, &x, n, k)
        .expect("f16_gemv_force fallback for argmax reference");
    let (best_i, best_v) = scores
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        });

    assert_eq!(topk1.0 as usize, best_i, "f16 topk1 idx mismatches argmax");
    assert!(
        (topk1.1 - best_v).abs() < 1e-2,
        "f16 topk1 score {} vs argmax {}",
        topk1.1,
        best_v
    );
}

#[test]
fn f16_gemv_topk_matches_cpu_topk() {
    let metal = get_metal();
    let n = 4096usize;
    let k = 256usize;
    let top_k = 5;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();
    let w_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.00091).cos()).collect();
    let w_f16 = larql_models::quant::half::encode_f16(&w_f32);

    use larql_compute::MatMul;
    let gpu_hits = metal
        .f16_gemv_topk(&w_f16, &x, n, k, top_k)
        .expect("topk path must fire");
    let scores = metal
        .f16_gemv_force(&w_f16, &x, n, k)
        .expect("scores path must fire");

    let mut indexed: Vec<(u32, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let cpu_hits: Vec<(u32, f32)> = indexed.into_iter().take(top_k).collect();

    assert_eq!(gpu_hits.len(), top_k);
    for (g, c) in gpu_hits.iter().zip(cpu_hits.iter()) {
        assert!(
            (g.1 - c.1).abs() < 1e-2,
            "f16 topk score mismatch at rank: gpu={:?} cpu={:?}",
            g,
            c
        );
    }
    for (idx, score) in gpu_hits.iter() {
        assert!(
            (scores[*idx as usize] - *score).abs() < 1e-2,
            "f16 topk idx {} reports score {} but scores[idx] = {}",
            idx,
            score,
            scores[*idx as usize]
        );
    }
}

/// `top_k > K_TOPK` exceeds the per-TG capacity → method returns None.
/// The `lm_head_knn_backend` wiring relies on this to fall back to the
/// full-Vec sort path for unusually large top_k requests.
#[test]
fn topk_capacity_edges_return_none() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 1024usize;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);
    let w_f16 = larql_models::quant::half::encode_f16(&matrix);

    use larql_compute::QuantMatVec;
    // top_k = 0 → None (caller wants nothing)
    assert!(metal
        .q4_matvec_topk(&q4_data, &q8_x, &q8_scales, rows, hidden, 0)
        .is_none());
    assert!(metal.f16_gemv_topk(&w_f16, &x, rows, hidden, 0).is_none());

    // top_k > K_TOPK = 8 → None (per-TG capacity exceeded)
    assert!(metal
        .q4_matvec_topk(&q4_data, &q8_x, &q8_scales, rows, hidden, 9)
        .is_none());
    assert!(metal.f16_gemv_topk(&w_f16, &x, rows, hidden, 9).is_none());
}

#[test]
fn q4_matvec_topk_matches_cpu_topk() {
    let metal = get_metal();
    let hidden = 2560;
    let rows = 10240;
    let top_k = 5;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    use larql_compute::QuantMatVec;
    let gpu_hits = metal
        .q4_matvec_topk(&q4_data, &q8_x, &q8_scales, rows, hidden, top_k)
        .expect("topk path must fire");

    let scores = metal
        .q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden)
        .expect("scores path must fire");
    let mut indexed: Vec<(u32, f32)> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let cpu_hits: Vec<(u32, f32)> = indexed.into_iter().take(top_k).collect();

    assert_eq!(gpu_hits.len(), top_k);
    // Score positions must match (Q4 quantization ties are real but the
    // sorted-descending ordering is deterministic). GPU and CPU may pick
    // different indices on ties — so compare scores by position only.
    for (g, c) in gpu_hits.iter().zip(cpu_hits.iter()) {
        assert!(
            (g.1 - c.1).abs() < 1e-3,
            "topk score mismatch at rank: gpu={:?} cpu={:?}",
            g,
            c
        );
    }
    // Each returned idx must point at a score equal to what we returned
    // (proving the GPU index is one of the legitimate top-K, not stale).
    for (idx, score) in gpu_hits.iter() {
        assert!(
            (scores[*idx as usize] - *score).abs() < 1e-3,
            "topk idx {} reports score {} but scores[idx] = {}",
            idx,
            score,
            scores[*idx as usize]
        );
    }
}

#[test]
fn q4_matvec_topk1_matches_full_argmax() {
    let metal = get_metal();
    let hidden = 2560;
    let rows = 10240;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    use larql_compute::QuantMatVec;
    let topk1 = metal
        .q4_matvec_topk1(&q4_data, &q8_x, &q8_scales, rows, hidden)
        .expect("metal must produce a top-1 result");

    let scores = metal
        .q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden)
        .expect("metal must produce scores");
    let (best_i, best_v) = scores
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        });

    assert_eq!(topk1.0 as usize, best_i, "topk1 idx mismatches argmax");
    assert!(
        (topk1.1 - best_v).abs() < 1e-3,
        "topk1 score {} vs argmax {}",
        topk1.1,
        best_v
    );
}

#[test]
fn q4_matvec_zero_input() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let x = vec![0.0f32; hidden];
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);
    assert!(
        result.iter().all(|&v| v.abs() < 0.01),
        "zero input should produce near-zero output"
    );
}

#[test]
fn q4_vecmat_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter)
        .map(|i| {
            if i % 5 == 0 {
                (i as f32 * 0.01).sin()
            } else {
                0.0
            }
        })
        .collect();
    let matrix: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);

    let cpu_result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    let metal_result = metal.q4_vecmat_direct(&activation, &q4_data, inter, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.1, "q4_vecmat max diff {diff} exceeds 0.1");
}

#[test]
fn q4_f32_matvec_nonzero() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut down_t: Vec<f32> = vec![0.0; hidden * inter];
    for r in 0..inter {
        for c in 0..hidden {
            down_t[c * inter + r] = ((r * hidden + c) as f32 * 0.0001).cos();
        }
    }
    let q4_data = quantize_q4_0(&down_t);

    let result = metal.q4_f32_matvec_direct(&q4_data, &activation, hidden, inter);
    assert_eq!(result.len(), hidden);
    assert!(
        result.iter().any(|&v| v.abs() > 0.01),
        "should produce nonzero output"
    );
}

#[test]
fn q4_pair_batch_matches_individual() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 1024; // smaller for test speed
    let seq = 2;

    let gate_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();
    let up_f32: Vec<f32> = (0..inter * hidden)
        .map(|i| (i as f32 * 0.0002).sin())
        .collect();
    let gate_q4 = quantize_q4_0(&gate_f32);
    let up_q4 = quantize_q4_0(&up_f32);
    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    // Individual calls
    let mut indiv_gate = Vec::new();
    let mut indiv_up = Vec::new();
    for s in 0..seq {
        let slice = &x[s * hidden..(s + 1) * hidden];
        let (q8, sc) = q4::quantize_to_q8(slice);
        indiv_gate.push(metal.q4_matvec_direct(&gate_q4, &q8, &sc, inter, hidden));
        indiv_up.push(metal.q4_matvec_direct(&up_q4, &q8, &sc, inter, hidden));
    }

    // Batched call
    let (batch_gate, batch_up) =
        metal.q4_matvec_pair_batch_direct(&gate_q4, &up_q4, &x, seq, inter, hidden);

    // Compare
    for s in 0..seq {
        let diff_g = max_diff(&indiv_gate[s], &batch_gate[s]);
        let diff_u = max_diff(&indiv_up[s], &batch_up[s]);
        assert!(diff_g < 0.001, "pair_batch gate diff {diff_g} at seq {s}");
        assert!(diff_u < 0.001, "pair_batch up diff {diff_u} at seq {s}");
    }
}

#[test]
fn multi_layer_q4_produces_output() {
    let metal = get_metal();
    let hidden = 256; // small for test speed
    let inter = 512;
    let layers = 3;

    let mut layers_q4 = Vec::new();
    for l in 0..layers {
        let g: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i + l * 1000) as f32 * 0.001).cos())
            .collect();
        let u: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i + l * 2000) as f32 * 0.002).sin())
            .collect();
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter {
            for c in 0..hidden {
                dt[c * inter + r] = ((r * hidden + c + l * 3000) as f32 * 0.003).cos();
            }
        }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let layers_refs: Vec<(&[u8], &[u8], &[u8])> = layers_q4
        .iter()
        .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice()))
        .collect();
    let result = metal.multi_layer_q4_ffn(&layers_refs, &x, inter, hidden);

    assert_eq!(result.len(), hidden);
    assert!(
        result.iter().any(|&v| v.abs() > 0.001),
        "multi-layer should produce nonzero output"
    );
}

#[test]
fn buffer_cache_reuses_same_pointer() {
    let metal = get_metal();
    let data = vec![1.0f32; 1024];
    let q4 = quantize_q4_0(&data);
    let (q8, sc) = q4::quantize_to_q8(&data[..256]);

    // Call twice with same data — buffer should be cached
    let r1 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);
    let r2 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);

    let diff = max_diff(&r1, &r2);
    assert!(
        diff < 1e-6,
        "cached buffer should produce identical results, diff: {diff}"
    );
}

#[test]
fn metal_backend_implements_trait() {
    let metal = get_metal();

    assert!(metal.supports_quant(::larql_compute::QuantFormat::Q4_K));
    assert!(metal.name().contains("metal"));

    let a = synth(2, 64, 42);
    let b = synth(32, 64, 43);
    let result = metal.matmul_transb(a.view(), b.view());
    assert_eq!(result.shape(), &[2, 32]);
}

#[test]
fn q8_matvec_metal_nonzero() {
    let _metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let weights: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let (w_q8, w_scales) =
        larql_compute::cpu::ops::q8_matvec::quantize_weights_q8(&weights, rows, hidden);
    let (x_q8, x_scales) = larql_compute::cpu::ops::q4_common::quantize_to_q8(&x);

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::q8_matvec::dispatch(
        &w_q8, &w_scales, &x_q8, &x_scales, rows, hidden,
    );
    assert!(
        cpu_result.iter().any(|&v| v.abs() > 0.01),
        "Q8 CPU should produce nonzero"
    );
}

#[test]
fn sparse_matvec_matches_dense() {
    let metal = get_metal();
    let hidden = 256;
    let n_rows = 64;
    let k_selected = 16;

    let matrix: Vec<f32> = (0..n_rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    // Dense: score all rows
    let dense_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, n_rows, hidden);

    // Sparse: score selected rows [0, 4, 8, 12, ...]
    let indices: Vec<u32> = (0..k_selected as u32).map(|i| i * 4).collect();

    // Use the sparse shader via raw Metal dispatch
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function("q4_sparse_matvec", None).unwrap(),
        )
        .unwrap();

    let bufs = &larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();
    let buf_q4 = bufs.get_bytes(&q4_data);
    let buf_q8 = bufs.transient_from_i8(&q8_x);
    let buf_sc = bufs.transient_from_f32(&q8_scales);
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let buf_idx = bufs.transient_from_f32(unsafe {
        std::slice::from_raw_parts(idx_bytes.as_ptr() as *const f32, indices.len())
    });
    let buf_out = bufs.output((k_selected * 4) as u64);

    let k_val = k_selected as u32;
    let h_val = hidden as u32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q4), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_sc), 0);
    enc.set_buffer(3, Some(&buf_idx), 0);
    enc.set_buffer(4, Some(&buf_out), 0);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &h_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(k_selected as u64, 1, 1),
        metal::MTLSize::new(k_selected as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let sparse_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, k_selected).to_vec() };

    // Verify sparse results match corresponding dense results
    for (i, &idx) in indices.iter().enumerate() {
        let diff = (sparse_result[i] - dense_result[idx as usize]).abs();
        assert!(diff < 0.01, "sparse[{i}] (row {idx}) diff {diff}");
    }
}

#[test]
fn residual_add_correct() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("residual_add", None).unwrap())
        .unwrap();

    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output(16);
    let len = 4u32;

    // `residual_add` shader now requires `b_scale` at buffer(4) for
    // Granite-family residual_multiplier. 1.0 recovers the legacy
    // `a + b` semantics this test was written against.
    let b_scale: f32 = 1.0;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &b_scale as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(4, 1, 1), metal::MTLSize::new(4, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, 4).to_vec() };
    assert!((result[0] - 11.0).abs() < 1e-5);
    assert!((result[1] - 22.0).abs() < 1e-5);
    assert!((result[2] - 33.0).abs() < 1e-5);
    assert!((result[3] - 44.0).abs() < 1e-5);
}
