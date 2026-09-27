use super::*;

/// `f32_topk_partial` correctness against synthetic scores. Exercises:
///   - the partial last TG (vocab not divisible by 256), which is the
///     case that broke `q4_matvec_topk` parity in development.
///   - vocab smaller than one TG (single partial TG only).
///
/// The Q4/f16 integration tests cover the typical "full TGs" path; this
/// pins the boundary cases that those don't reach.
/// `wire_resident` is the residency bootstrap that fixed the wired-
/// collector wall (a >45 GB working set decoding ~10x slow). It had
/// no test anywhere in the crate despite being load-bearing at
/// startup, and it is all refusal paths and side effects — it
/// returns nothing, so the only observable contract is that it
/// refuses the degenerate inputs without panicking and survives a
/// real one.
#[test]
fn wire_resident_refuses_degenerate_input_and_survives_a_real_one() {
    let Some(metal) = MetalBackend::new() else {
        return; // not on Metal-capable hardware
    };
    // No buffers at all: nothing to wire, must not touch the queue.
    metal.wire_resident(&[]);
    // A first buffer too short for the 1x1 gemv's two bytes. The
    // guard exists because the encoder needs real work to trigger
    // residency, and a sub-2-byte read would be out of bounds.
    metal.wire_resident(&[&[7u8]]);
    metal.wire_resident(&[&[]]);
    // A real call: several page-sized buffers, as the bootstrap sees
    // them. Passing is "did not panic and did not hang" — there is
    // no return value, and asserting on timing here would be
    // asserting on the machine rather than the code.
    let a = vec![0u8; 4096];
    let b = vec![1u8; 8192];
    let c = vec![2u8; 4096];
    metal.wire_resident(&[&a, &b, &c]);
}

/// The multi-matrix gemv paths share one command buffer across
/// several weight matrices, and their tail — collecting each output,
/// then recycling every buffer after the wait — is distinct from the
/// single-matrix wrappers the other tests exercise. A wrong output
/// order here would be invisible to a single-matrix test.
#[test]
fn f16_gemv_multi_returns_each_matrix_in_order() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let k = 64usize;
    let x: Vec<f32> = (0..k).map(|i| (i % 5) as f32 - 2.0).collect();
    // Two matrices with deliberately DIFFERENT row counts and
    // different content, so a swapped or duplicated result cannot
    // pass: matrix 0 is all ones (dot = sum of x), matrix 1 is all
    // twos (dot = 2 * sum of x).
    let f16_ones = larql_models::quant::half::encode_f16(&vec![1.0f32; 2 * k]);
    let f16_twos = larql_models::quant::half::encode_f16(&vec![2.0f32; 3 * k]);
    let Some(out) = metal.f16_gemv_multi(&[(&f16_ones, 2, k), (&f16_twos, 3, k)], &x) else {
        return; // shape refused on this device
    };
    assert_eq!(out.len(), 2, "one result per matrix");
    assert_eq!(out[0].len(), 2);
    assert_eq!(out[1].len(), 3);
    let sum: f32 = x.iter().sum();
    for v in &out[0] {
        assert!((v - sum).abs() < 1e-2, "matrix 0 row {v} vs {sum}");
    }
    for v in &out[1] {
        assert!(
            (v - 2.0 * sum).abs() < 1e-2,
            "matrix 1 row {v} vs {}",
            2.0 * sum
        );
    }
}

#[test]
fn topk_partial_handles_partial_last_tg() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );

    // 4 full TGs + 1 partial (1024 + 100 = 1124). Plant maxima at 700
    // (full TG) and 1100 (partial last TG) so both must be picked.
    let n = 1124usize;
    let mut scores = vec![0.0f32; n];
    for (i, s) in scores.iter_mut().enumerate() {
        *s = (i as f32) * 0.001;
    }
    scores[700] = 999.0;
    scores[1100] = 998.0;

    let scores_buf = metal.bufs.transient_from_f32(&scores);
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let (vals, idxs, num_tgs) = metal.encode_topk_partial(enc, &scores_buf, n);
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        cmd,
        "crates/larql-compute-metal/src/trait_impl/matmul.rs:570",
    )
    .expect("command buffer completed");

    let hits = MetalBackend::reduce_topk_partial(&vals, &idxs, num_tgs, 5);
    assert_eq!(hits.len(), 5);
    let top_idxs: Vec<u32> = hits.iter().map(|(i, _)| *i).collect();
    assert!(
        top_idxs.contains(&700),
        "missing planted argmax 700: {:?}",
        top_idxs
    );
    assert!(
        top_idxs.contains(&1100),
        "missing planted second-max 1100 (in partial TG): {:?}",
        top_idxs
    );

    // vocab smaller than one TG (200 elements, single partial TG).
    let n = 200usize;
    let mut scores = vec![0.0f32; n];
    for (i, s) in scores.iter_mut().enumerate() {
        *s = -(i as f32);
    }
    scores[42] = 5.0;
    scores[99] = 4.0;
    let scores_buf = metal.bufs.transient_from_f32(&scores);
    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    let (vals, idxs, num_tgs) = metal.encode_topk_partial(enc, &scores_buf, n);
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        cmd,
        "crates/larql-compute-metal/src/trait_impl/matmul.rs:600",
    )
    .expect("command buffer completed");
    let hits = MetalBackend::reduce_topk_partial(&vals, &idxs, num_tgs, 2);
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].0, 42);
    assert_eq!(hits[1].0, 99);
}

/// `top_k > K_TOPK` is rejected at the public method (returns `None`)
/// so the reducer is never called with mismatched K. Sanity-check the
/// public-facing wrappers honour the `K_TOPK = 8` ceiling.
#[test]
fn topk_capacity_ceiling_enforced() {
    let metal = MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let n = 512;
    let k = 256;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001).cos()).collect();
    let w_f16 = larql_models::quant::half::encode_f16(&vec![0.5f32; n * k]);
    // top_k = 0 and top_k > K_TOPK both yield None — caller falls back.
    assert!(metal.f16_gemv_topk(&w_f16, &x, n, k, 0).is_none());
    assert!(metal.f16_gemv_topk(&w_f16, &x, n, k, 9).is_none());
    // top_k within range produces a result.
    let hits = metal
        .f16_gemv_topk(&w_f16, &x, n, k, 8)
        .expect("top_k=8 is exactly K_TOPK and must be accepted");
    assert_eq!(hits.len(), 8);
}

// ─── End-to-end trait-method coverage for the gemv family ───

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

/// Width sized above the calibration FLOP threshold so the
/// `2*n*k < flop_threshold` guard doesn't short-circuit the
/// dispatch. Apple Silicon calibrates to ~5K-50K depending on
/// device, so 256×256 = 131K flops is safe.
fn gemv_shapes() -> (usize, usize) {
    (256, 256)
}

/// `f32_gemv` returns `None` when the input vector length doesn't
/// match the matrix `k` dim (line 34 early-out).
#[test]
fn f32_gemv_rejects_mismatched_x_length() {
    let m = backend();
    let w = ndarray::Array2::<f32>::zeros((16, 32));
    let x = vec![0.0f32; 31]; // wrong length
    assert!(m.f32_gemv(w.view(), &x).is_none());
}

/// `f32_gemv` returns `None` when the operation falls below the
/// FLOP threshold (line 39 — CPU fallback, dispatch overhead would
/// dominate).
#[test]
fn f32_gemv_falls_back_below_flop_threshold() {
    let m = backend();
    // 2×2 gemv → 8 FLOPs, well under any sane threshold.
    let w = ndarray::Array2::<f32>::zeros((2, 2));
    let x = vec![0.0f32; 2];
    assert!(m.f32_gemv(w.view(), &x).is_none());
}

/// `f32_gemv` runs the GPU path on large-enough shapes.  Force the
/// threshold low so the test doesn't depend on the calibration
/// number on whatever host is running.
#[test]
fn f32_gemv_dispatches_above_threshold() {
    let m = backend();
    m.set_flop_threshold(1);
    let (n, k) = gemv_shapes();
    let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.0001).collect();
    let w = ndarray::Array2::from_shape_vec((n, k), w_data).unwrap();
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
    let out = m
        .f32_gemv(w.view(), &x)
        .expect("f32_gemv above threshold returns Some");
    assert_eq!(out.len(), n);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `f32_gemv_force` bypasses the threshold guard and always
/// dispatches when shapes agree (covers lines 44-50).
#[test]
fn f32_gemv_force_bypasses_threshold() {
    let m = backend();
    // Tiny shape that f32_gemv would short-circuit on.
    let w = ndarray::Array2::<f32>::from_shape_vec((4, 4), vec![1.0f32; 16]).unwrap();
    let x = vec![1.0f32; 4];
    let out = m
        .f32_gemv_force(w.view(), &x)
        .expect("f32_gemv_force dispatches regardless of threshold");
    assert_eq!(out.len(), 4);
}

/// `f32_gemv_force` still rejects shape mismatches.
#[test]
fn f32_gemv_force_rejects_mismatched_x_length() {
    let m = backend();
    let w = ndarray::Array2::<f32>::zeros((4, 8));
    let x = vec![0.0f32; 7];
    assert!(m.f32_gemv_force(w.view(), &x).is_none());
}

/// `encode_f32_gemv` handles non-contiguous `ArrayView2` inputs by
/// materialising a standard-layout copy (lines 113-114).
#[test]
fn f32_gemv_force_handles_non_contiguous_input() {
    let m = backend();
    // Build a column-major view (non-standard layout) by transposing.
    let raw: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let w_t = ndarray::Array2::from_shape_vec((4, 4), raw)
        .unwrap()
        .reversed_axes();
    assert!(w_t.as_slice().is_none(), "test setup: non-standard layout");
    let x = vec![1.0f32; 4];
    let out = m
        .f32_gemv_force(w_t.view(), &x)
        .expect("non-contiguous gemv still dispatches");
    assert_eq!(out.len(), 4);
}

/// `f16_gemv` shape-mismatch early-outs (lines 53-54, 64).
#[test]
fn f16_gemv_rejects_invalid_shapes() {
    let m = backend();
    let w = vec![0u8; 7]; // too short for n*k*2
    let x = vec![0.0f32; 4];
    assert!(m.f16_gemv(&w, &x, 4, 4).is_none());
    assert!(m.f16_gemv_force(&w, &x, 4, 4).is_none());
    // x.len() != k — even on `force`.
    let w = larql_models::quant::half::encode_f16(&[0.0f32; 16]);
    let x = vec![0.0f32; 3];
    assert!(m.f16_gemv_force(&w, &x, 4, 4).is_none());
}

/// `f16_gemv` falls back below the FLOP threshold (line 57).
#[test]
fn f16_gemv_falls_back_below_flop_threshold() {
    let m = backend();
    let w_f16 = larql_models::quant::half::encode_f16(&[0.5f32; 4]);
    let x = vec![1.0f32; 2];
    assert!(m.f16_gemv(&w_f16, &x, 2, 2).is_none());
}

/// `f32_gemv_topk1` and `f16_gemv_topk1` wrap the inherent helpers
/// — they're the trait surface (lines 69-77, 85).
#[test]
fn gemv_topk1_trait_wrappers_route_to_inherent_helpers() {
    let m = backend();
    let (n, k) = gemv_shapes();
    let w_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.0001).collect();
    let w = ndarray::Array2::from_shape_vec((n, k), w_data.clone()).unwrap();
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

    let (idx, val) = m
        .f32_gemv_topk1(w.view(), &x)
        .expect("topk1 returns Some on valid shape");
    assert!((idx as usize) < n);
    assert!(val.is_finite());

    let w_f16 = larql_models::quant::half::encode_f16(&w_data);
    let (idx2, val2) = m
        .f16_gemv_topk1(&w_f16, &x, n, k)
        .expect("f16 topk1 returns Some on valid shape");
    assert!((idx2 as usize) < n);
    assert!(val2.is_finite());

    // f16_gemv_topk trait wrapper round-trips top_k=4.
    let hits = m
        .f16_gemv_topk(&w_f16, &x, n, k, 4)
        .expect("top_k=4 returns Some");
    assert_eq!(hits.len(), 4);
}

/// `f32_gemv_topk1` rejects shape mismatch + n=0 (lines 159-160).
#[test]
fn f32_gemv_topk1_rejects_invalid_shapes() {
    let m = backend();
    let w = ndarray::Array2::<f32>::zeros((4, 8));
    let x = vec![0.0f32; 7]; // wrong length
    assert!(m.f32_gemv_topk1(w.view(), &x).is_none());
    // n = 0
    let w_empty = ndarray::Array2::<f32>::zeros((0, 8));
    let x_ok = vec![0.0f32; 8];
    assert!(m.f32_gemv_topk1(w_empty.view(), &x_ok).is_none());
}

/// `matmul_batch` dispatches each op through `matmul` or
/// `matmul_transb` based on `transpose_b` (lines 88-98).
#[test]
fn matmul_batch_dispatches_per_op_transpose() {
    let m = backend();
    let a = ndarray::Array2::from_shape_vec((2, 3), vec![1.0f32; 6]).unwrap();
    let b = ndarray::Array2::from_shape_vec((3, 4), vec![0.5f32; 12]).unwrap();
    let b_t = ndarray::Array2::from_shape_vec((4, 3), vec![0.5f32; 12]).unwrap();
    let ops = vec![
        MatMulOp {
            a: a.clone(),
            b: b.clone(),
            transpose_b: false,
        },
        MatMulOp {
            a: a.clone(),
            b: b_t.clone(),
            transpose_b: true,
        },
    ];
    let outs = m.matmul_batch(&ops);
    assert_eq!(outs.len(), 2);
    assert_eq!(outs[0].shape(), &[2, 4]);
    assert_eq!(outs[1].shape(), &[2, 4]);
}

/// `reduce_argmax_partial` returns `None` when every partial is
/// `NaN`/`INF` (line 297-298 early-out).
#[test]
fn reduce_argmax_partial_returns_none_for_all_non_finite() {
    let m = backend();
    // Build vals = [NaN, NaN, ...] and idxs = [0, 1, ...].
    let vals: Vec<f32> = (0..4).map(|_| f32::NAN).collect();
    let idxs: Vec<u32> = (0..4u32).collect();
    let vals_buf = m.bufs.transient_from_f32(&vals);
    let idxs_buf = m.bufs.transient_from_bytes(unsafe {
        std::slice::from_raw_parts(idxs.as_ptr() as *const u8, idxs.len() * 4)
    });
    let out = MetalBackend::reduce_argmax_partial(&vals_buf, &idxs_buf, vals.len());
    assert!(out.is_none());
}

/// `reduce_topk_partial` with `k=0` returns an empty vec (line 351-352).
/// Note: the reducer reads `num_tgs * K_TOPK` floats from the buffer
/// before clamping `k`, so we still need to provide a correctly-sized
/// vals/idxs buffer.
#[test]
fn reduce_topk_partial_zero_k_returns_empty() {
    let m = backend();
    let k_topk = crate::shaders::f32_gemv::K_TOPK;
    let vals = vec![1.0f32; k_topk];
    let idxs = vec![0u32; k_topk];
    let vals_buf = m.bufs.transient_from_f32(&vals);
    let idxs_bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idxs_buf = m.bufs.transient_from_bytes(&idxs_bytes);
    let out = MetalBackend::reduce_topk_partial(&vals_buf, &idxs_buf, 1, 0);
    assert!(out.is_empty());
}

/// `reduce_topk_partial` skips non-finite vals + `u32::MAX`
/// sentinel indices (lines 378-385).
#[test]
fn reduce_topk_partial_skips_invalid_entries() {
    let m = backend();
    let k_topk = crate::shaders::f32_gemv::K_TOPK;
    // Two TGs worth of partials. Plant valid (idx=5, val=10) in
    // the first slot, and fill the rest with non-finite + sentinel
    // garbage.  reduce_topk_partial should pick only the valid one.
    let mut vals: Vec<f32> = Vec::with_capacity(2 * k_topk);
    let mut idxs: Vec<u32> = Vec::with_capacity(2 * k_topk);
    // First slot: valid.
    vals.push(10.0);
    idxs.push(5);
    // Remainder of first TG: non-finite + sentinel.
    for _ in 1..k_topk {
        vals.push(f32::NAN);
        idxs.push(u32::MAX);
    }
    // Second TG: all sentinels.
    for _ in 0..k_topk {
        vals.push(0.0); // finite but idx is sentinel
        idxs.push(u32::MAX);
    }

    let vals_buf = m.bufs.transient_from_f32(&vals);
    let idxs_bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idxs_buf = m.bufs.transient_from_bytes(&idxs_bytes);
    let out = MetalBackend::reduce_topk_partial(&vals_buf, &idxs_buf, 2, 3);
    // Only 1 valid candidate; reducer returns a 1-element vec.
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, 5);
    assert_eq!(out[0].1, 10.0);
}

/// `reduce_topk_partial` with `k > total` clamps k to total —
/// covers the final-heapify branch (lines 398-400) when the heap
/// fills incompletely.
#[test]
fn reduce_topk_partial_clamps_k_to_total_and_finalizes_heap() {
    let m = backend();
    let k_topk = crate::shaders::f32_gemv::K_TOPK;
    // One TG of partials. Plant 4 valid scores: (idx=0, val=1.0),
    // (idx=1, val=3.0), (idx=2, val=2.0), (idx=3, val=4.0).
    let mut vals: Vec<f32> = vec![1.0, 3.0, 2.0, 4.0];
    let mut idxs: Vec<u32> = vec![0, 1, 2, 3];
    while vals.len() < k_topk {
        vals.push(0.0);
        idxs.push(u32::MAX);
    }
    let vals_buf = m.bufs.transient_from_f32(&vals);
    let idxs_bytes: Vec<u8> = idxs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idxs_buf = m.bufs.transient_from_bytes(&idxs_bytes);

    // Request top-100, far more than the 4 finite entries available.
    // Reducer should clamp and return 4 sorted descending.
    let out = MetalBackend::reduce_topk_partial(&vals_buf, &idxs_buf, 1, 100);
    // total = num_tgs * K_TOPK = 8; clamped to 4 valid entries; sorted desc.
    // (Note: k is clamped to `total`, not to `valid_count`, so the
    // result may include up to `total` entries; here only 4 are valid.)
    assert!(out.len() <= 8 && !out.is_empty());
    // Top entry is (3, 4.0).
    assert_eq!(out[0], (3, 4.0));
}
