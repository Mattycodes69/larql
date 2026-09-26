use super::*;

/// The whole step, and the state it leaves behind, across several
/// tokens — a recurrence that ignored its state, or advanced it
/// differently, diverges here and not at token one.
#[test]
fn the_device_step_matches_the_scalar_reference_across_tokens() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let device = KdaDeviceState::zeros(&m, shape);
    let mut host = RefState::zeros();

    for t in 0..6 {
        let x = synth(HIDDEN, 20.0 + t as f32);
        let want = reference_step(&w, &mut host, &x);
        let (got, gpu) = m
            .kda_attention_step(w.device(), shape, &device, &x)
            .expect("device step");
        assert!(gpu >= 0.0, "the GPU window must be reported");
        assert!(
            max_abs(&got, &want) < TOLERANCE,
            "token {t}: max|Δ| {:e}",
            max_abs(&got, &want)
        );
        let (rec, conv) = device.read_back();
        assert!(
            max_abs(&rec, &host.recurrent) < TOLERANCE,
            "token {t} recurrent state: max|Δ| {:e}",
            max_abs(&rec, &host.recurrent)
        );
        for (i, (got, want)) in conv.iter().zip(&host.conv).enumerate() {
            assert!(max_abs(got, want) < TOLERANCE, "token {t} conv window {i}");
        }
    }
}

/// The traced variant must report the same output as the plain one, and
/// every plane it names must be the right length — a trace whose planes
/// were mis-sized would be read as a stage disagreement.
#[test]
fn the_traced_step_agrees_with_the_plain_one() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let x = synth(HIDDEN, 3.0);

    let plain = KdaDeviceState::zeros(&m, shape);
    let (out, _) = m.kda_attention_step(w.device(), shape, &plain, &x).unwrap();
    let traced_state = KdaDeviceState::zeros(&m, shape);
    let p = m
        .kda_attention_step_traced(w.device(), shape, &traced_state, &x)
        .expect("traced step");

    assert_eq!(p.output, out, "tracing must not change the answer");
    for (name, len) in [
        ("q_proj", p.q_proj.len()),
        ("k_proj", p.k_proj.len()),
        ("v_proj", p.v_proj.len()),
        ("q_conv", p.q_conv.len()),
        ("k_conv", p.k_conv.len()),
        ("v_conv", p.v_conv.len()),
        ("q_norm", p.q_norm.len()),
        ("k_norm", p.k_norm.len()),
        ("f_lowrank", p.f_lowrank.len()),
        ("g_decay", p.g_decay.len()),
        ("recurrent_out", p.recurrent_out.len()),
        ("o_gate", p.o_gate.len()),
        ("o_norm", p.o_norm.len()),
    ] {
        assert_eq!(len, WIDTH, "{name} should be width-long");
    }
    assert_eq!(p.beta.len(), HEADS);
    assert_eq!(p.output.len(), HIDDEN);

    // The q/k planes must be L2-normalised per head, which is the one
    // property a wrong reduction would quietly break.
    for h in 0..HEADS {
        for plane in [&p.q_norm, &p.k_norm] {
            let n = plane[h * DIM..(h + 1) * DIM]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            assert!((n - 1.0).abs() < 1e-5, "head {h} norm {n}");
        }
    }
}

/// Shape faults refuse rather than reading out of bounds, and refuse
/// before the encoder opens — Metal aborts the process if a compute
/// encoder is dropped without `end_encoding`.
#[test]
fn shape_faults_are_refused_before_the_encoder_opens() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let state = KdaDeviceState::zeros(&m, shape);

    let short_x = synth(HIDDEN - 1, 0.0);
    assert!(matches!(
        m.kda_attention_step(w.device(), shape, &state, &short_x),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    let mut truncated = w.device();
    let half = &w.qkv_bank[..w.qkv_bank.len() / 2];
    truncated.qkv_bank = half;
    assert!(matches!(
        m.kda_attention_step(truncated, shape, &state, &synth(HIDDEN, 0.0)),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    let other = KdaShape {
        head_dim: DIM * 2,
        ..shape
    };
    assert!(matches!(
        m.kda_attention_step(w.device(), other, &state, &synth(HIDDEN, 0.0)),
        Err(GroupedError::SlotCountMismatch { .. })
    ));

    // Still usable, which is the real assertion.
    assert!(m
        .kda_attention_step(w.device(), shape, &state, &synth(HIDDEN, 1.0))
        .is_ok());
}

/// A `head_dim` the recurrence's one-threadgroup-per-head mapping cannot
/// cover is refused before encoding. Without the refusal the step
/// completes normally and value columns past the threadgroup are never
/// written — half the state frozen, no fault.
#[test]
fn a_head_dim_past_the_recurrence_threadgroup_is_refused() {
    let m = backend();
    let w = weights();
    let max = kda_shader::RECURRENCE_THREADS_PER_TG as usize;
    let wide = KdaShape {
        head_dim: max + 1,
        ..shape()
    };
    let state = KdaDeviceState::zeros(&m, wide);
    assert_eq!(
        m.kda_attention_step(w.device(), wide, &state, &synth(HIDDEN, 0.0))
            .map(|_| ()),
        Err(GroupedError::KdaGeometryUnsupported {
            field: "head_dim",
            value: max + 1,
            min: 1,
            max,
        })
    );
    // The boundary itself is admitted by the geometry check.
    assert!(MetalBackend::validate_kda_geometry(KdaShape {
        head_dim: max,
        ..shape()
    })
    .is_ok());
}

/// Every degenerate extent refuses by name. `conv_kernel = 0` is the one
/// that matters most: the short-conv kernel computes `kernel - 1` in
/// `uint` and would walk ~4G history entries.
#[test]
fn degenerate_kda_extents_are_refused_by_name() {
    for (field, bad) in [
        (
            "conv_kernel",
            KdaShape {
                conv_kernel: 0,
                ..shape()
            },
        ),
        (
            "head_dim",
            KdaShape {
                head_dim: 0,
                ..shape()
            },
        ),
        (
            "num_heads",
            KdaShape {
                num_heads: 0,
                ..shape()
            },
        ),
        (
            "hidden",
            KdaShape {
                hidden: 0,
                ..shape()
            },
        ),
    ] {
        match MetalBackend::validate_kda_geometry(bad) {
            Err(GroupedError::KdaGeometryUnsupported {
                field: got,
                value: 0,
                ..
            }) => {
                assert_eq!(got, field)
            }
            other => panic!("{field}=0 must be refused by name, got {other:?}"),
        }
    }
    assert!(MetalBackend::validate_kda_geometry(shape()).is_ok());
}

/// Each small operand one element short refuses NAMING that operand,
/// before the encoder opens. The wide projections are covered by
/// `shape_faults_are_refused_before_the_encoder_opens`.
#[test]
fn every_short_kda_operand_is_refused_by_name() {
    let m = backend();
    let w = weights();
    let shape = shape();
    let state = KdaDeviceState::zeros(&m, shape);
    let x = synth(HIDDEN, 0.0);
    let short = |v: &[f32]| v.len() - 1;
    let base = w.device();
    let cases = [
        (
            "q_conv1d",
            KdaDeviceWeights {
                q_conv1d: &w.conv[0][..short(&w.conv[0])],
                ..base
            },
        ),
        (
            "k_conv1d",
            KdaDeviceWeights {
                k_conv1d: &w.conv[1][..short(&w.conv[1])],
                ..base
            },
        ),
        (
            "v_conv1d",
            KdaDeviceWeights {
                v_conv1d: &w.conv[2][..short(&w.conv[2])],
                ..base
            },
        ),
        (
            "f_a_proj",
            KdaDeviceWeights {
                f_a_proj: SmallMatrix::F32(&w.fa[..short(&w.fa)]),
                ..base
            },
        ),
        (
            "f_b_proj",
            KdaDeviceWeights {
                f_b_proj: SmallMatrix::F32(&w.fb[..short(&w.fb)]),
                ..base
            },
        ),
        (
            "g_a_proj",
            KdaDeviceWeights {
                g_a_proj: SmallMatrix::F32(&w.ga[..short(&w.ga)]),
                ..base
            },
        ),
        (
            "g_b_proj",
            KdaDeviceWeights {
                g_b_proj: SmallMatrix::F32(&w.gb[..short(&w.gb)]),
                ..base
            },
        ),
        (
            "b_proj",
            KdaDeviceWeights {
                b_proj: SmallMatrix::F32(&w.bp[..short(&w.bp)]),
                ..base
            },
        ),
        (
            "a_log",
            KdaDeviceWeights {
                a_log: &w.a_log[..short(&w.a_log)],
                ..base
            },
        ),
        (
            "dt_bias",
            KdaDeviceWeights {
                dt_bias: &w.dt[..short(&w.dt)],
                ..base
            },
        ),
        (
            "o_norm",
            KdaDeviceWeights {
                o_norm: &w.o_norm[..short(&w.o_norm)],
                ..base
            },
        ),
    ];
    for (operand, d) in cases {
        match m.kda_attention_step(d, shape, &state, &x) {
            Err(GroupedError::KdaOperandShape { operand: got, .. }) => {
                assert_eq!(got, operand)
            }
            other => panic!(
                "{operand} one short must be refused by name, got {:?}",
                other.map(|_| ())
            ),
        }
    }
    // A bf16 matrix that is not a whole number of codes is refused too,
    // rather than rounded down to a length that happens to match.
    let fa_bytes: Vec<u8> = w.fa.iter().flat_map(|v| narrow(*v).to_le_bytes()).collect();
    let mut d = w.device();
    let mut odd = fa_bytes.clone();
    odd.push(0);
    d.f_a_proj = SmallMatrix::Bf16(&odd);
    assert!(matches!(
        m.kda_attention_step(d, shape, &state, &x),
        Err(GroupedError::KdaOperandShape {
            operand: "f_a_proj",
            ..
        })
    ));
    // Still usable, which is the real assertion.
    assert!(m.kda_attention_step(w.device(), shape, &state, &x).is_ok());
}

/// `KdaShape`'s derived quantities, and that `zeroed` really is zero —
/// a recurrent state starting from a recycled buffer's leftovers would
/// produce a plausible wrong answer on token one.
#[test]
fn the_state_starts_at_zero_and_the_shape_derives_its_widths() {
    let m = backend();
    let shape = shape();
    assert_eq!(shape.width(), WIDTH);
    let state = KdaDeviceState::zeros(&m, shape);
    let (rec, conv) = state.read_back();
    assert_eq!(rec.len(), HEADS * DIM * DIM);
    assert!(
        rec.iter().all(|v| *v == 0.0),
        "recurrent state must start zero"
    );
    for c in &conv {
        assert_eq!(c.len(), WIDTH * (KERNEL - 1));
        assert!(c.iter().all(|v| *v == 0.0), "conv window must start zero");
    }
}
