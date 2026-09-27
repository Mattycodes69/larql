use super::*;

/// Fused gate+up must produce bit-exact outputs equal to two separate
/// matvec calls — both compile down to the same i32 dot math; only the
/// instruction interleaving differs.
#[test]
fn q8k_gate_up_fused_matches_separate_matvecs() {
    let cols = 1024;
    let rows = 11;
    let x: Vec<f32> = (0..cols)
        .map(|i| (i as f32 * 0.0151).sin() * 1.4 + (i as f32 * 0.029).cos() * 0.7)
        .collect();
    let g_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.011).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
        .collect();
    let u_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.013).sin() * 0.3 + (i as f32 * 0.041).cos() * 0.5)
        .collect();
    let g_w = quantize_q4_k(&g_f32);
    let u_w = quantize_q4_k(&u_f32);
    let q8 = quantize_x_to_q8k(&x);

    let mut g_sep = vec![0.0f32; rows];
    let mut u_sep = vec![0.0f32; rows];
    q4k_q8k_matvec_into(&mut g_sep, &q8, &g_w, rows, cols).expect("valid shape");
    q4k_q8k_matvec_into(&mut u_sep, &q8, &u_w, rows, cols).expect("valid shape");

    let mut g_fused = vec![0.0f32; rows];
    let mut u_fused = vec![0.0f32; rows];
    q4k_q8k_gate_up_into(&mut g_fused, &mut u_fused, &q8, &g_w, &u_w, rows, cols)
        .expect("valid shape");

    for r in 0..rows {
        assert_eq!(
            g_sep[r].to_bits(),
            g_fused[r].to_bits(),
            "gate row {r}: sep={} fused={}",
            g_sep[r],
            g_fused[r]
        );
        assert_eq!(
            u_sep[r].to_bits(),
            u_fused[r].to_bits(),
            "up row {r}: sep={} fused={}",
            u_sep[r],
            u_fused[r]
        );
    }
}

/// Empty / degenerate dims should produce zeros without panic.
#[test]
fn q8k_matvec_zero_dims_returns_zero() {
    let q = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_scalar(&mut out, &q, &[], 4, 0).expect("valid shape");
    assert!(out.iter().all(|&v| v == 0.0));
}

/// A weight slab shorter than `rows` packed rows is refused by name, and
/// the output is left exactly as the caller handed it in — never zeros,
/// which a sampler would accept as logits.
#[test]
fn q8k_matvec_short_weight_buffer_is_refused_untouched() {
    let cols = 256;
    let rows = 2;
    let x = vec![0.5f32; cols];
    let q = quantize_x_to_q8k(&x);
    let w = vec![0u8; 144]; // only enough for 1 row, but rows=2
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_scalar(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_scalar");
    assert_eq!((err.weight_bytes, err.needed_bytes), (144, 288));
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

#[test]
fn q6k_q8k_matvec_matches_q6k_f32_dispatch_within_noise() {
    let cols = 512;
    let rows = 5;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.006).cos() * 0.7)
        .collect();
    let w_q6 = quantize_q6_k(&w_f32);

    let f32_path = crate::cpu::ops::q6k_matvec::dispatch(&w_q6, &x, rows, cols);
    let q8 = quantize_x_to_q8k(&x);
    let mut q8_path = vec![0.0f32; rows];
    q6k_q8k_matvec_scalar(&mut q8_path, &q8, &w_q6, rows, cols).expect("valid shape");

    for r in 0..rows {
        let diff = (f32_path[r] - q8_path[r]).abs();
        assert!(
            diff < 1.2e-1,
            "row {r}: f32={} q8={} diff={diff}",
            f32_path[r],
            q8_path[r]
        );
    }
}

#[test]
fn q6k_q8k_public_entrypoint_matches_scalar() {
    let cols = 256;
    let rows = 3;
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.031).cos()).collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.011).sin() * 0.4)
        .collect();
    let w_q6 = quantize_q6_k(&w_f32);
    let q8 = quantize_x_to_q8k(&x);
    let mut scalar = vec![0.0f32; rows];
    let mut dispatched = vec![0.0f32; rows];

    q6k_q8k_matvec_scalar(&mut scalar, &q8, &w_q6, rows, cols).expect("valid shape");
    q6k_q8k_matvec_into(&mut dispatched, &q8, &w_q6, rows, cols).expect("valid shape");

    for (a, b) in scalar.iter().zip(dispatched.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// `q6k_q8k_matvec_scalar`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[test]
fn q6k_q8k_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q6k_q8k_matvec_scalar(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; Q6K_BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q6k_q8k_matvec_scalar(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q6k_q8k_matvec_scalar");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (Q6K_BLOCK_BYTES, 2 * Q6K_BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// AVX2 must produce bit-identical output to the scalar reference.
#[cfg(target_arch = "x86_64")]
#[test]
fn q8k_matvec_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip on hardware without AVX2.
    }
    let cols = 1024;
    let rows = 7;
    let x: Vec<f32> = (0..cols)
        .map(|i| {
            let f = i as f32;
            ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
        })
        .collect();
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let f = i as f32;
            ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
        })
        .collect();
    let w_q4 = quantize_q4_k(&w_f32);
    let q8 = quantize_x_to_q8k(&x);

    let mut out_scalar = vec![0.0f32; rows];
    let mut out_avx2 = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols).expect("valid shape");
    unsafe { q4k_q8k_matvec_avx2(&mut out_avx2, &q8, &w_q4, rows, cols) }.expect("valid shape");

    for r in 0..rows {
        assert_eq!(
            out_scalar[r].to_bits(),
            out_avx2[r].to_bits(),
            "row {r}: scalar={} avx2={} diff={}",
            out_scalar[r],
            out_avx2[r],
            (out_scalar[r] - out_avx2[r]).abs()
        );
    }
}

/// Unknown-format contract (`quant_route`): the kernel entry point must
/// panic on a tag with no route, never leave `out` silently zeroed —
/// a plausible-but-wrong logit vector is the dec-readiness review's §1
/// failure class.
#[test]
#[should_panic(expected = "unknown quant format tag")]
fn matvec_parallel_panics_on_unknown_format_tag_instead_of_zero_filling() {
    let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let q8k_x = quantize_x_to_q8k(&x);
    let bytes = quantize_q4_k(&x);
    let mut out = vec![0.0f32; 1];
    q4k_q8k_matvec_parallel(&mut out, &q8k_x, &bytes, 1, 256, "MXFP9").expect("valid shape");
}

/// Same contract for a format that parses but has no Q8K matvec kernel.
#[test]
#[should_panic(expected = "no Q8K matvec kernel")]
fn matvec_parallel_panics_on_format_without_q8k_kernel() {
    let x: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
    let q8k_x = quantize_x_to_q8k(&x);
    let mut out = vec![0.0f32; 1];
    // BF16 parses via `from_registry_tag` but has no block-stream kernel.
    let bytes = vec![0u8; 512];
    q4k_q8k_matvec_parallel(&mut out, &q8k_x, &bytes, 1, 256, "BF16").expect("valid shape");
}

/// The NEON-intrinsic fused gate+up kernel must be bit-exact with two
/// independent scalar matvecs — same discipline as the asm form. The
/// default `q4k_q8k_gate_up_into` dispatch takes the asm path, so the
/// intrinsic twin needs its own direct exercise (per-file coverage floor).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_gate_up_neon_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let g_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let u_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.019).sin() * 0.5 + (f * 0.031).cos() * 0.3) * 0.7
            })
            .collect();
        let g_q4 = quantize_q4_k(&g_f32);
        let u_q4 = quantize_q4_k(&u_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut g_scalar = vec![0.0f32; rows];
        let mut u_scalar = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut g_scalar, &q8, &g_q4, rows, cols).expect("valid shape");
        q4k_q8k_matvec_scalar(&mut u_scalar, &q8, &u_q4, rows, cols).expect("valid shape");

        let mut g_neon = vec![0.0f32; rows];
        let mut u_neon = vec![0.0f32; rows];
        q4k_q8k_gate_up_neon(&mut g_neon, &mut u_neon, &q8, &g_q4, &u_q4, rows, cols)
            .expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                g_scalar[r].to_bits(),
                g_neon[r].to_bits(),
                "gate rows={rows} cols={cols} row {r}: scalar={} neon={}",
                g_scalar[r],
                g_neon[r],
            );
            assert_eq!(
                u_scalar[r].to_bits(),
                u_neon[r].to_bits(),
                "up rows={rows} cols={cols} row {r}: scalar={} neon={}",
                u_scalar[r],
                u_neon[r],
            );
        }
    }
}

/// `q4k_q8k_gate_up_neon`: zero dims zero BOTH outputs, and a short gate slab is
/// refused naming the gate half, with both outputs left untouched.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q8k_gate_up_neon_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut g = vec![1.0f32; 4];
    let mut u = vec![1.0f32; 4];
    q4k_q8k_gate_up_neon(&mut g, &mut u, &empty, &[], &[], 4, 0)
        .expect("zero dims are the empty product");
    assert!(g.iter().chain(u.iter()).all(|&v| v == 0.0));

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w_short = vec![0u8; BLOCK_BYTES]; // one row's worth, rows == 2
    let w_full = vec![0u8; 2 * BLOCK_BYTES];
    let mut g = vec![1.0f32; rows];
    let mut u = vec![1.0f32; rows];
    let err = q4k_q8k_gate_up_neon(&mut g, &mut u, &q, &w_short, &w_full, rows, cols)
        .expect_err("a short gate slab must be refused");
    assert_eq!(
        (g, u),
        (vec![1.0f32; rows], vec![1.0f32; rows]),
        "refusal must not touch either output"
    );
    assert_eq!(err.kernel, "q4k_q8k_gate_up_neon (gate)");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (BLOCK_BYTES, 2 * BLOCK_BYTES)
    );

    // The up half is checked independently: a short UP slab names it.
    let mut g = vec![1.0f32; rows];
    let mut u = vec![1.0f32; rows];
    let err = q4k_q8k_gate_up_neon(&mut g, &mut u, &q, &w_full, &w_short, rows, cols)
        .expect_err("a short up slab must be refused");
    assert_eq!(err.kernel, "q4k_q8k_gate_up_neon (up)");
}

/// The Q6_K NEON-intrinsic kernel must be bit-exact with the scalar
/// reference — the default `q6k_q8k_matvec_into` dispatch takes the asm
/// path, so the intrinsic twin needs its own direct exercise.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q6k_matvec_neon_matches_scalar_bit_exact() {
    for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_neon = vec![0.0f32; rows];
        q6k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q6, rows, cols).expect("valid shape");
        q6k_q8k_matvec_neon(&mut out_neon, &q8, &w_q6, rows, cols).expect("valid shape");

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_neon[r].to_bits(),
                "rows={rows} cols={cols} row {r}: scalar={} neon={}",
                out_scalar[r],
                out_neon[r],
            );
        }
    }
}

/// `q6k_q8k_matvec_neon`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q6k_matvec_neon_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q6k_q8k_matvec_neon(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; Q6K_BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q6k_q8k_matvec_neon(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q6k_q8k_matvec_neon");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (Q6K_BLOCK_BYTES, 2 * Q6K_BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// `q4k_q8k_matvec_neon`: zero dims are the empty product (zeros written), and a
/// weight slab shorter than `rows` packed rows is refused by name with the
/// output left exactly as handed in — never silently zeroed.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q4k_matvec_neon_zero_dims_are_empty_and_short_weights_are_refused() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_neon(&mut out, &empty, &[], 4, 0).expect("zero dims are the empty product");
    assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

    let cols = 256;
    let rows = 2;
    let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
    let w = vec![0u8; BLOCK_BYTES]; // one row's worth, but rows == 2
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_neon(&mut out, &q, &w, rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_neon");
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (BLOCK_BYTES, 2 * BLOCK_BYTES)
    );
    assert_eq!(
        (err.out_len, err.rows, err.x_len, err.cols),
        (rows, rows, cols, cols)
    );
}

/// Q4_K NEON 2-row kernel: guards zero the output, and an ODD row count
/// exercises the single-row tail fallback documented on the kernel.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[test]
fn q4k_matvec_neon_2row_guards_and_odd_row_tail() {
    let empty = Q8KActivation {
        qs: vec![],
        d: vec![],
        sums: vec![],
    };
    let mut out = vec![1.0f32; 4];
    q4k_q8k_matvec_neon_2row(&mut out, &empty, &[], 4, 0).expect("valid shape");
    assert!(out.iter().all(|&v| v == 0.0));

    let cols = 512;
    let rows = 3; // odd → last row via the single-row tail
    let q8 = quantize_x_to_q8k(
        &(0..cols)
            .map(|i| ((i as f32) * 0.017).sin())
            .collect::<Vec<_>>(),
    );
    let w_f32: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.011).cos() * 0.5)
        .collect();
    let w = quantize_q4_k(&w_f32);

    // short-weight guard: one row's bytes for rows == 3 is refused,
    // output untouched.
    let mut out = vec![1.0f32; rows];
    let err = q4k_q8k_matvec_neon_2row(&mut out, &q8, &w[..2 * BLOCK_BYTES], rows, cols)
        .expect_err("a short weight slab must be refused");
    assert_eq!(out, vec![1.0f32; rows]);
    assert_eq!(err.kernel, "q4k_q8k_matvec_neon_2row");

    let mut out_single = vec![0.0f32; rows];
    let mut out_2row = vec![0.0f32; rows];
    q4k_q8k_matvec_neon(&mut out_single, &q8, &w, rows, cols).expect("valid shape");
    q4k_q8k_matvec_neon_2row(&mut out_2row, &q8, &w, rows, cols).expect("valid shape");
    for r in 0..rows {
        assert_eq!(
            out_single[r].to_bits(),
            out_2row[r].to_bits(),
            "row {r}: single={} 2row={}",
            out_single[r],
            out_2row[r],
        );
    }
}
