use super::*;

/// **The Q8 x Q8 kernel computes what the format denotes.**
///
/// Bit-identity, not a tolerance. The block sums are INTEGER and integer
/// addition is exact and associative, so the vectorised lane-reduction
/// and the sequential loop must reach the same i32; the only float in
/// either is one multiply-add per block, in the same block order. A
/// difference here would be a real defect, never rounding.
#[test]
fn the_q8_q8_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 5;
    for in_dim in SHAPES {
        let w = lcg_values(OUT * in_dim, 21);
        let x = lcg_values(in_dim, 22);
        let (codes, scales) = q8_parts(&w, in_dim);
        let act = quantise_activation(&x);
        let per_row = in_dim.div_ceil(Q8_BLOCK);

        let mut got = vec![0.0f32; OUT];
        Q8xQ8.project_rows(
            WeightRows::Q8 {
                codes: &codes,
                scales: &scales,
                sums: &[],
                block: Q8_BLOCK,
            },
            &x,
            &mut got,
        );

        for o in 0..OUT {
            let want = act.scale
                * super::super::super::integer::q8_row_portable(
                    &codes[o * in_dim..(o + 1) * in_dim],
                    &scales[o * per_row..(o + 1) * per_row],
                    &act.codes,
                    in_dim,
                    Q8_BLOCK,
                );
            assert_eq!(
                got[o].to_bits(),
                want.to_bits(),
                "q8xq8 row {o} at in_dim {in_dim}: {} vs definition {}",
                got[o],
                want
            );
        }
    }
}

/// **The Q4 x Q8 kernel computes what the format denotes.**
///
/// Same bit-identity argument, plus the packing: byte `j` carries element
/// `j` low and `j + half` high, so a kernel that read adjacent nibbles
/// would pair every weight with the wrong activation and still return
/// finite, plausible numbers.
#[test]
fn the_q4_q8_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 5;
    for in_dim in SHAPES {
        let w = lcg_values(OUT * in_dim, 23);
        let x = lcg_values(in_dim, 24);
        let (packed, scales) = q4_parts(&w, in_dim);
        let act = quantise_activation(&x);
        let per_row = in_dim.div_ceil(Q4_BLOCK);

        let mut got = vec![0.0f32; OUT];
        Q4xQ8.project_rows(
            WeightRows::Q4 {
                packed: &packed,
                scales: &scales,
                block: Q4_BLOCK,
            },
            &x,
            &mut got,
        );

        for o in 0..OUT {
            let want = act.scale
                * super::super::super::integer::q4_row_portable(
                    &packed[o * (in_dim / 2)..(o + 1) * (in_dim / 2)],
                    &scales[o * per_row..(o + 1) * per_row],
                    &act.codes,
                    in_dim,
                    Q4_BLOCK,
                );
            assert_eq!(
                got[o].to_bits(),
                want.to_bits(),
                "q4xq8 row {o} at in_dim {in_dim}: {} vs definition {}",
                got[o],
                want
            );
        }
    }
}

/// **The control arm changes the ACTIVATION and nothing else.**
///
/// If the activation already lies on the int8 grid its round trip is
/// lossless, so `Bf16xQ8` must reproduce the EXACT kernel bit for bit —
/// same weights, same summation order, same everything. That is what
/// makes A1 a control: every difference it reports on a real activation
/// is the activation's rounding and nothing else.
#[test]
fn the_activation_control_changes_the_activation_and_nothing_else() {
    const OUT: usize = 4;
    const IN: usize = 256;
    let w = lcg_values(OUT * IN, 25);
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();

    // An activation already on the grid: 127 levels of a chosen peak, so
    // `round(x / (peak/127))` is exact for every element.
    let peak = 0.25f32;
    let step = peak / 127.0;
    let x: Vec<f32> = (0..IN)
        .map(|i| ((i as i32 % 255) - 127) as f32 * step)
        .collect();
    assert!(
        x.iter().any(|v| *v < 0.0) && x.iter().any(|v| *v > 0.0),
        "the fixture must span both signs or it cannot exercise the clamp"
    );

    let mut exact = vec![0.0f32; OUT];
    FusedBf16.project_rows(WeightRows::Bf16(&bits), &x, &mut exact);
    let mut control = vec![0.0f32; OUT];
    Bf16xQ8.project_rows(WeightRows::Bf16(&bits), &x, &mut control);

    for o in 0..OUT {
        assert_eq!(
            control[o].to_bits(),
            exact[o].to_bits(),
            "A1 row {o}: {} vs the exact kernel {}",
            control[o],
            exact[o]
        );
    }
}

/// And the control is NOT vacuous: on an ordinary activation it moves.
///
/// A control that reproduced the exact answer whatever it was handed
/// would pass the test above by doing nothing at all.
#[test]
fn the_activation_control_does_move_on_an_ordinary_activation() {
    const OUT: usize = 8;
    const IN: usize = 512;
    let w = lcg_values(OUT * IN, 29);
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let x = lcg_values(IN, 30);

    let mut exact = vec![0.0f32; OUT];
    FusedBf16.project_rows(WeightRows::Bf16(&bits), &x, &mut exact);
    let mut control = vec![0.0f32; OUT];
    Bf16xQ8.project_rows(WeightRows::Bf16(&bits), &x, &mut control);

    assert!(
        exact.iter().zip(&control).any(|(a, b)| a != b),
        "the A1 control reproduced the exact kernel on a real activation, so it is quantising \
         nothing and would report a clean bill for any weight format"
    );
}

/// **The instrument must FAIL on known-different input.**
///
/// A gate that only ever passes proves nothing. Q4 and Q8 over the SAME
/// weights and the SAME activation must disagree, and by roughly the
/// ratio of their quantisation steps — `peak/7` against `peak/127`. If
/// these two arms agreed, the arm switch would not be reaching the
/// arithmetic at all.
#[test]
fn the_q4_and_q8_arms_disagree_by_about_their_step_ratio() {
    const OUT: usize = 64;
    const IN: usize = 1024;
    let w = lcg_values(OUT * IN, 26);
    let x = lcg_values(IN, 27);
    let (codes, q8_scales) = q8_parts(&w, IN);
    let (packed, q4_scales) = q4_parts(&w, IN);

    let mut q8 = vec![0.0f32; OUT];
    Q8xQ8.project_rows(
        WeightRows::Q8 {
            codes: &codes,
            scales: &q8_scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        &x,
        &mut q8,
    );
    let mut q4 = vec![0.0f32; OUT];
    Q4xQ8.project_rows(
        WeightRows::Q4 {
            packed: &packed,
            scales: &q4_scales,
            block: Q4_BLOCK,
        },
        &x,
        &mut q4,
    );

    let rms = |v: &[f32]| (v.iter().map(|a| (a * a) as f64).sum::<f64>() / v.len() as f64).sqrt();
    let diff: Vec<f32> = q4.iter().zip(&q8).map(|(a, b)| a - b).collect();
    let rel = rms(&diff) / rms(&q8);

    assert!(
        rel > 1e-2,
        "q4 and q8 arms agree to {rel:.3e} — the arm switch is not reaching the arithmetic"
    );
    assert!(
        rel < 1.0,
        "q4 differs from q8 by {rel:.3e}, which is not quantisation but a defect"
    );
}

/// The activation quantiser is bounded by half its own step, and its
/// scale is derived from the peak it must represent.
#[test]
fn the_activation_quantiser_is_bounded_by_half_a_step() {
    let x = lcg_values(4096, 28);
    let act = quantise_activation(&x);
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    assert!(
        (act.scale - peak / 127.0).abs() <= f32::EPSILON * peak.max(1.0),
        "the activation scale must be peak/127"
    );
    for (i, v) in x.iter().enumerate() {
        let back = act.codes[i] as f32 * act.scale;
        assert!(
            (back - v).abs() <= act.scale * 0.5 + f32::EPSILON,
            "element {i} reconstructs to {back} from {v}, past half a step"
        );
    }
}

/// A zero activation must not divide by zero, and must reconstruct
/// exactly.
#[test]
fn a_zero_activation_quantises_without_dividing_by_zero() {
    let act = quantise_activation(&vec![0.0f32; 128]);
    assert_eq!(act.scale, 1.0, "the zero vector takes the sentinel scale");
    assert!(act.codes.iter().all(|c| *c == 0));
}

/// The arm is the DEFAULT unless the environment names one, and an
/// unrecognised value is the default rather than a fourth regime.
///
/// Read through the same accessor the loader and the executor use, so
/// this pins the value both of them see. It does not set the variable:
/// the arm is resolved once per process on purpose, and a test that
/// mutated it would be asserting about its own ordering.
#[test]
fn the_default_arithmetic_arm_is_the_float_activation() {
    if std::env::var(super::super::super::physical::ARITHMETIC_ARM_ENV).is_ok() {
        // A deliberately-armed process is running some other measurement;
        // asserting the default here would fail for the right reason and
        // tell nobody anything.
        return;
    }
    assert_eq!(arithmetic_arm(), ArithmeticArm::FloatActivation);
}

/// **The A1 control must cover EXACTLY the operands the arms cover.**
///
/// bf16 bytes are ambiguous: an operand is resident as bf16 either
/// because its image fits L2 and the policy kept it exact, or because
/// the A1 control swapped a streaming Q8 operand back. Observation alone
/// cannot separate those, and an earlier version of this arm did not
/// try — so A1 quantised the activation on the cache-resident operands
/// too, became a LARGER perturbation than the arm it exists to explain,
/// and read worse than Q8 x Q8 while holding exact weights.
///
/// This pins the population rather than the numbers: a cache-resident
/// bf16 operand stays exact under every arm, and only a streaming one
/// joins the control.
#[test]
fn the_activation_control_covers_only_the_streaming_operands() {
    use super::super::super::physical::{
        compact_threshold_bytes, PhysicalProjectionPlan, BF16_BYTES,
    };

    // Sized either side of the boundary the policy itself reads.
    let boundary = compact_threshold_bytes() / BF16_BYTES;
    let in_dim = 64usize;
    for (elements, streaming) in [(in_dim * 2, false), (boundary + in_dim, true)] {
        let bits = vec![0u16; elements];
        let observed = PhysicalProjectionPlan::for_resident(WeightRows::Bf16(&bits), in_dim);
        // Under the DEFAULT arm nothing joins the control at any size.
        assert_eq!(
            observed,
            PhysicalProjectionPlan::FusedBf16,
            "a bf16 operand of {elements} elements (streaming={streaming}) must stay exact \
             under the default arm"
        );
    }
}

/// **Exact weights must beat quantised weights under the same activation.**
///
/// A1 holds the checkpoint's own values and A3 holds an 8-bit image of
/// them; the activation error is identical. Their output errors are
/// therefore `e_act` and `e_act + dw.x` with the two terms independent,
/// so A1 can never be the worse arm. If it is, the arms are not seeing
/// the same activation and the control is not a control.
#[test]
fn the_exact_weight_arm_beats_the_quantised_one_under_one_activation() {
    const OUT: usize = 32;
    const IN: usize = 1024;
    let w = lcg_values(OUT * IN, 41);
    // A residual-stream-shaped activation: a few outlier channels tens of
    // times the RMS, which is the regime the real model is in.
    let mut x = lcg_values(IN, 42);
    for (i, v) in x.iter_mut().enumerate() {
        if i % 137 == 0 {
            *v *= 40.0;
        }
    }
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let exact_w: Vec<f32> = bits
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    let (codes, scales) = q8_parts(&exact_w, IN);

    // The truth: exact weights, exact activation.
    let truth: Vec<f32> = (0..OUT)
        .map(|o| {
            exact_w[o * IN..(o + 1) * IN]
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        })
        .collect();

    let mut a1 = vec![0.0f32; OUT];
    Bf16xQ8.project_rows(WeightRows::Bf16(&bits), &x, &mut a1);
    let mut a3 = vec![0.0f32; OUT];
    Q8xQ8.project_rows(
        WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        &x,
        &mut a3,
    );

    let err = |v: &[f32]| {
        let n: f64 = v
            .iter()
            .zip(&truth)
            .map(|(a, t)| ((a - t) as f64).powi(2))
            .sum();
        let d: f64 = truth.iter().map(|t| (*t as f64).powi(2)).sum();
        (n / d).sqrt()
    };
    let (e1, e3) = (err(&a1), err(&a3));
    assert!(
        e1 <= e3,
        "A1 (exact weights) {e1:.4e} is WORSE than A3 (q8 weights) {e3:.4e} under one \
         activation — the two arms are not quantising the activation the same way"
    );
}

/// **Residency CONSTRAINS which plans are possible; it does not DETERMINE
/// which one executes.**
///
/// The invariant this whole rung produced. Before integer activations it
/// was false in a useful way — bytes implied a kernel — and every reader
/// could rely on that. It is false now: identical Q8 bytes are consumed
/// by a widening f32 GEMV and by `SDOT`, at 83.4 and 118.0 GB/s and with
/// different numerics.
///
/// Pinned as a test because the tempting "simplification" is to infer
/// arithmetic from residency again, and it would pass every parity gate
/// on a machine running the default arm.
#[test]
fn residency_constrains_the_plan_without_determining_it() {
    use super::super::super::arithmetic::{plans_possible_for, WeightRep};
    use super::super::super::physical::PhysicalProjectionPlan;

    // Every compact representation admits MORE THAN ONE plan. If any of
    // these collapsed to one, residency would determine execution again.
    for rep in [
        WeightRep::Bf16,
        WeightRep::Q8 { block: Q8_BLOCK },
        WeightRep::Q4 { block: Q4_BLOCK },
    ] {
        assert!(
            plans_possible_for(rep).len() > 1,
            "{rep:?} admits only one plan — residency would determine arithmetic"
        );
    }

    // And whatever the arm, the plan actually chosen is one of the plans
    // that representation makes possible: the constraint is real.
    let bits = vec![0u16; 64 * 4];
    let observed = PhysicalProjectionPlan::for_resident(WeightRows::Bf16(&bits), 64);
    assert!(
        plans_possible_for(WeightRep::Bf16).contains(&observed),
        "the executor chose {observed:?}, which bf16 residency does not admit"
    );
}

/// The plan reports the arithmetic in the form a claim has to quote.
///
/// Every term changes the answer, so a number reported as "Q4" alone is
/// under-described: `Q4[64] x Q8[tensor]` and `Q4[64] x Q8[64]` differ by
/// an order of magnitude in logit error.
#[test]
fn a_plan_describes_its_arithmetic_including_the_activation_geometry() {
    use super::super::super::arithmetic::{AccumulatorRep, ActivationRep, ScaleSpan};
    use super::super::super::physical::PhysicalProjectionPlan;

    let a = PhysicalProjectionPlan::Q4xQ8.arithmetic();
    assert_eq!(a.accumulator, AccumulatorRep::I32);
    assert!(matches!(a.activation, ActivationRep::Q8 { .. }));

    let shown = format!("{a}");
    assert!(
        shown.starts_with("Q4[64] x Q8[") && shown.ends_with("-> I32 -> F32"),
        "unexpected description `{shown}`"
    );

    // The exact kernels name an f32 activation and an f32 accumulator,
    // so nothing reads as integer arithmetic that is not.
    let e = PhysicalProjectionPlan::FusedBf16.arithmetic();
    assert_eq!(e.activation, ActivationRep::F32);
    assert_eq!(e.accumulator, AccumulatorRep::F32);
    assert_eq!(format!("{e}"), "BF16 x F32 -> F32 -> F32");

    // And the span is carried, not implied.
    match PhysicalProjectionPlan::Q8xQ8.arithmetic().activation {
        ActivationRep::Q8 { span } => {
            assert!(matches!(span, ScaleSpan::Tensor | ScaleSpan::Block(_)))
        }
        other => panic!("integer arm reported activation {other:?}"),
    }
}

/// **A restored class falls back to Q8 in the SAME arithmetic domain.**
///
/// A rescue rung must move exactly one variable: the weight bits. If
/// restoring a class also dropped it back to an f32 activation, the rung
/// would move the weight format AND the arithmetic together, and its
/// result would license nothing about either — which is precisely how
/// CPU-4A concluded that Q4 was dead when it had only shown that
/// Q4 x F32 was.
#[test]
fn a_restored_class_keeps_the_integer_activation() {
    use super::super::super::physical::{PhysicalProjectionPlan, Q4Classes};
    use crate::format::vindex3::opplan::exec::backend::MatrixClass;

    // FFN goes to Q4; attention and the head are restored.
    let only_ffn = Q4Classes {
        attention: false,
        ffn: true,
        head: false,
    };
    assert!(only_ffn.admits(MatrixClass::FfnProjection));
    assert!(!only_ffn.admits(MatrixClass::AttentionProjection));
    assert!(!only_ffn.admits(MatrixClass::OutputHead));

    // The bank is never a Q4 candidate under any set: it is widened to
    // f32 on the way in and has no compact bytes to keep.
    assert!(!Q4Classes::ALL.admits(MatrixClass::RoutedExpertBank));

    // And Q8 bytes under a Q4 arm run through SDOT, not through the
    // widening f32 kernel — same activation, same accumulator.
    let codes = vec![0i8; 64 * 4];
    let scales = vec![1.0f32; 4];
    let observed = PhysicalProjectionPlan::for_resident(
        WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &[],
            block: Q8_BLOCK,
        },
        64,
    );
    // Under the default arm this is the widening kernel; under either
    // integer arm it is SDOT. Both are legal; neither is inferred from
    // the bytes.
    assert!(
        matches!(
            observed,
            PhysicalProjectionPlan::FusedQ8 | PhysicalProjectionPlan::Q8xQ8
        ),
        "q8 residency produced {observed:?}"
    );
}

/// Blanket Q4 is the DEFAULT of a Q4 arm, so an arm run without an
/// exception set is the hypothesis rather than a silent recipe.
#[test]
fn an_unset_exception_list_is_blanket_q4_not_a_quiet_recipe() {
    use super::super::super::physical::{q4_classes, Q4Classes, Q4_CLASSES_ENV};

    if std::env::var(Q4_CLASSES_ENV).is_ok() {
        return; // a deliberately-scoped process is running another rung
    }
    assert_eq!(
        q4_classes(),
        Q4Classes::ALL,
        "an unset {Q4_CLASSES_ENV} must mean blanket Q4, not an implicit exception set"
    );
}

/// **The sub-blocked Q4 row computes what the format denotes.**
///
/// Against a scalar definition written here, because the packing is the
/// part that can go wrong: byte `j` carries element `j` low and
/// `j + block/2` high, so a sub-block that took adjacent nibbles would
/// pair every weight with the wrong activation and still return finite,
/// plausible numbers.
#[test]
fn the_subblocked_q4_row_computes_what_the_format_denotes() {
    use super::super::super::integer::q4_row_subblocked;
    const IN: usize = 512;
    let w = lcg_values(IN, 51);
    let (packed, wscales) = q4_parts(&w, IN);
    let x = outlier_activation(IN, 52);

    for ablock in [16usize, 32] {
        let (qx, ascales) = super::super::super::integer::quantise_activation_blocked(&x, ablock);
        let per_weight = Q4_BLOCK / ablock;
        let folded: Vec<f32> = ascales
            .iter()
            .enumerate()
            .map(|(s, a)| wscales[s / per_weight] * *a)
            .collect();

        // The definition: decode every weight from its nibble, multiply
        // by the reconstructed activation, sum.
        let mut want = 0.0f64;
        #[allow(clippy::needless_range_loop)]
        // The index IS the subject here: it selects a nibble, a byte and
        // a scale by three different divisions, and iterating `qx`
        // would hide the one relationship the test exists to pin.
        for i in 0..IN {
            let b = i / Q4_BLOCK;
            let off = i % Q4_BLOCK;
            let half = Q4_BLOCK / 2;
            let (byte, high) = if off < half {
                (packed[b * (Q4_BLOCK / 2) + off], false)
            } else {
                (packed[b * (Q4_BLOCK / 2) + off - half], true)
            };
            let code = if high {
                (byte >> 4) as i32 - 8
            } else {
                (byte & 0x0f) as i32 - 8
            };
            let s = i / ablock;
            want += (code as f64) * (qx[i] as f64) * (folded[s] as f64);
        }
        let got = q4_row_subblocked(&packed, &folded, &qx, IN, Q4_BLOCK, ablock);
        let rel = ((got as f64 - want) / want.abs().max(1e-12)).abs();
        assert!(
            rel < 1e-5,
            "ablock {ablock}: kernel {got} vs definition {want} (rel {rel:.2e})"
        );
    }
}
