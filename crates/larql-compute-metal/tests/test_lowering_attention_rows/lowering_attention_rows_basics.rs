use super::*;

#[test]
fn serial_block_is_one_rows_dispatch_at_parity_with_each_position() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    for (window, variant) in [(None, Variant::Full), (Some(WINDOW), Variant::Plain)] {
        check(
            &gpu,
            Case {
                geom: UNMEASURED,
                base: SHORT_BASE,
                window,
                variant,
                splitk_rows: None,
            },
            Witness {
                serial: ONE_OP,
                ..Witness::default()
            },
            Witness {
                serial: PER_ROW,
                ..Witness::default()
            },
        );
    }
}

#[test]
fn seqpar_block_is_one_rows_dispatch_at_parity_with_each_position() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    for variant in [Variant::Full, Variant::Plain] {
        check(
            &gpu,
            Case {
                geom: GEMMA,
                base: SHORT_BASE,
                window: None,
                variant,
                splitk_rows: None,
            },
            Witness {
                seqpar: ONE_OP,
                ..Witness::default()
            },
            Witness {
                seqpar: PER_ROW,
                ..Witness::default()
            },
        );
    }
}

#[test]
fn splitk_block_is_one_splitk_op_at_parity_with_each_position() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Full carries sinks (merge pass) and softcap (partial pass).
    for variant in [Variant::Full, Variant::Plain] {
        check(
            &gpu,
            Case {
                geom: GEMMA,
                base: SPLITK_BASE,
                window: None,
                variant,
                splitk_rows: Some(ROWS),
            },
            Witness {
                splitk: ONE_OP,
                ..Witness::default()
            },
            Witness {
                splitk: PER_ROW,
                ..Witness::default()
            },
        );
    }
}

#[test]
fn splitk_scratch_too_small_for_the_block_falls_back_to_seqpar_rows() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Scratch for ONE position: every single-position op still splits,
    // the block does not fit and must take the seqpar rows kernel rather
    // than overrun the partial buffers.
    check(
        &gpu,
        Case {
            geom: GEMMA,
            base: SPLITK_BASE,
            window: None,
            variant: Variant::Full,
            splitk_rows: Some(1),
        },
        Witness {
            seqpar: ONE_OP,
            ..Witness::default()
        },
        Witness {
            splitk: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn splitk_block_straddling_a_chunk_tier_falls_back_to_seqpar_rows() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // Rows below span 512 split into 8 chunks, rows from 512 into 16: one
    // split-K op cannot carry both, so the block takes seqpar rows while
    // each position still splits at its own chunk count.
    check(
        &gpu,
        Case {
            geom: GEMMA,
            base: TIER_512_BASE,
            window: None,
            variant: Variant::Plain,
            splitk_rows: Some(ROWS),
        },
        Witness {
            seqpar: ONE_OP,
            ..Witness::default()
        },
        Witness {
            splitk: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn block_past_the_short_span_runs_the_long_kernel_per_row() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // No rows kernel holds more than the short kernel's scores, so the
    // block loops the per-position dispatch — the same kernel `step` runs.
    check(
        &gpu,
        Case {
            geom: GEMMA,
            base: LONG_BASE,
            window: None,
            variant: Variant::Plain,
            splitk_rows: None,
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn block_straddling_a_seqpar_slice_tier_runs_per_row() {
    let _lock = WITNESS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let gpu = backend();
    // gpt-oss's rows below span 512 take 8 slices and from 512 take 12;
    // one rows dispatch carries one threadgroup width, so it must loop.
    check(
        &gpu,
        Case {
            geom: GPT_OSS,
            base: TIER_512_BASE,
            window: None,
            variant: Variant::Plain,
            splitk_rows: None,
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
        Witness {
            seqpar: PER_ROW,
            ..Witness::default()
        },
    );
}

#[test]
fn attention_rows_refuses_the_output_gate_by_name() {
    let gpu = backend();
    let g = UNMEASURED;
    // Leaked for the same reason as `run_case`'s operands.
    let m: &'static nvfp4::Nvfp4Matrix = Box::leak(Box::new(
        nvfp4::quantize(
            &det(g.q_rows() * HIDDEN, 1, WEIGHT_AMPLITUDE),
            g.q_rows(),
            HIDDEN,
        )
        .expect("quantise"),
    ));
    let (packed, scales) = (
        gpu.lowering_weight(&m.packed),
        gpu.lowering_weight(&m.scales),
    );
    let mat = || LoweredMatrix::Nvfp4 {
        packed: &packed,
        packed_offset: 0,
        scales: &scales,
        scales_offset: 0,
        tensor_scale: m.tensor_scale,
    };
    let buf = gpu.lowering_scratch(ROWS * g.q_rows().max(HIDDEN));
    let w = AttnWeights {
        q: mat(),
        k: mat(),
        v: mat(),
        o: mat(),
        gate: Some(mat()),
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        sinks: None,
        qk_norm: None,
        norm_weight: &buf,
        post_norm: None,
    };
    let s = AttnScratch {
        normed: &buf,
        q: &buf,
        k_cache: &buf,
        v_cache: &buf,
        gate: &buf,
        concat: &buf,
        gated: &buf,
        attn_out: &buf,
        inv_freq: &buf,
        splitk: None,
    };
    let c = Case {
        geom: g,
        base: SHORT_BASE,
        window: None,
        variant: Variant::Plain,
        splitk_rows: None,
    };
    let mut err = None;
    run(&gpu, |enc| {
        err = gpu
            .encode_attention_rows(
                &mut SingleEncoder(enc),
                &buf,
                &buf,
                &w,
                &s,
                &buf,
                &shape(&c, c.base),
                ROWS,
            )
            .err();
    });
    let err = err.expect("a gated attention op has no multi-position lowering");
    assert!(err.contains("output-gate"), "{err}");
}

#[test]
fn splitk_scratch_sizes_and_fits_the_ops_it_was_sized_for() {
    let gpu = backend();
    let g = GEMMA;
    let (o_len, ml_len) = SplitKScratch::lens(ROWS, g.num_q, g.q_rows());
    // Every chunk of every head of every row gets its own partial.
    assert_eq!(o_len, ROWS * g.q_rows() * SPLITK_MAX_CHUNKS);
    assert_eq!(ml_len, ml_part_len(ROWS, g.num_q, SPLITK_MAX_CHUNKS));
    let (o_part, ml_part) = (gpu.lowering_scratch(o_len), gpu.lowering_scratch(ml_len));
    let s = SplitKScratch {
        o_part: &o_part,
        ml_part: &ml_part,
        rows: ROWS,
        max_q_heads: g.num_q,
        max_q_rows: g.q_rows(),
    };
    assert!(s.fits(ROWS, g.num_q, g.q_rows()), "exactly the sized op");
    assert!(s.fits(1, g.num_q / 2, g.q_rows() / 2), "a smaller op");
    assert!(!s.fits(ROWS + 1, g.num_q, g.q_rows()), "one row too many");
    assert!(!s.fits(ROWS, g.num_q + 1, g.q_rows()), "one head too many");
    assert!(
        !s.fits(ROWS, g.num_q, g.q_rows() + 1),
        "one float too many per row"
    );
}
