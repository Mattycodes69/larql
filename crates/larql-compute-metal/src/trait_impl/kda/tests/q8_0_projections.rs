//! Q8_0 projections

use super::*;

/// Q8_0 projections through the real quantised grouped kernel track the
/// bf16 step across a multi-token run, and the delta is NOT vacuous —
/// the roundtrip genuinely changed the weights, so an exactly-equal
/// output would mean the Q8_0 arm silently ran the bf16 kernel.
///
/// Multi-token matters here for the same reason it did for the delta
/// rule: the recurrence carries the perturbation forward, so a token-0
/// agreement alone would not show the state stays coherent under a
/// quantised projection feeding it.
#[test]
fn q8_projections_track_the_bf16_step_across_tokens() {
    let m = backend();
    let banks = dual_banks();
    let shape = q8_shape();
    let s_bf16 = KdaDeviceState::zeros(&m, shape);
    let s_q8 = KdaDeviceState::zeros(&m, shape);
    let mut max_rel = 0.0f32;
    for t in 0..4 {
        let x = synth(Q8_HIDDEN, 0.3 + t as f32);
        let (out_b, _) = m
            .kda_attention_step(banks.device(ExpertEncoding::Bf16), shape, &s_bf16, &x)
            .expect("bf16 arm runs");
        let (out_q, _) = m
            .kda_attention_step(banks.device(ExpertEncoding::Q80), shape, &s_q8, &x)
            .expect("q8 arm runs");
        let rms: f32 = (out_b.iter().map(|v| v * v).sum::<f32>() / out_b.len() as f32).sqrt();
        let d_rms: f32 = (out_b
            .iter()
            .zip(&out_q)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / out_b.len() as f32)
            .sqrt();
        let rel = d_rms / rms.max(f32::EPSILON);
        max_rel = max_rel.max(rel);
        assert!(
            rel < 5e-2,
            "token {t}: Q8_0 projections displaced the output by rel {rel} — that is \
             quantisation of two projections, not a decode fault, and it should be \
             orders under this bound"
        );
    }
    assert!(
        max_rel > 1e-6,
        "the arms never separated: the Q8_0 dispatch is not actually reading \
         quantised bytes"
    );
}

/// Bounds are enforced at the ENCODING's own stride: a Q8_0 bank one
/// byte short of three slots is refused by name, where the bf16
/// validator's larger stride would have mis-blamed a healthy bank.
#[test]
fn q8_bank_bounds_are_checked_at_the_q8_stride() {
    let m = backend();
    let banks = dual_banks();
    let shape = q8_shape();
    let state = KdaDeviceState::zeros(&m, shape);
    let mut w = banks.device(ExpertEncoding::Q80);
    let truncated = &banks.q8_qkv[..banks.q8_qkv.len() - 1];
    w.qkv_bank = truncated;
    assert!(
        matches!(
            m.kda_attention_step(w, shape, &state, &synth(Q8_HIDDEN, 0.0)),
            Err(GroupedError::OffsetOutOfRange { .. })
        ),
        "a truncated Q8_0 bank must be refused before the encoder opens"
    );
}

/// A reduction axis that is not a whole number of Q8_0 blocks cannot be
/// encoded, and the step says so rather than reading garbage: the
/// file's own HIDDEN = 6 geometry is exactly such a shape.
#[test]
fn a_misaligned_reduction_axis_refuses_q8_by_name() {
    let m = backend();
    let w = weights();
    let state = KdaDeviceState::zeros(&m, shape());
    let device = KdaDeviceWeights {
        projection_encoding: ExpertEncoding::Q80,
        ..w.device()
    };
    assert!(
        matches!(
            m.kda_attention_step(device, shape(), &state, &synth(HIDDEN, 0.0)),
            Err(GroupedError::KNotSuperblockAligned { k: HIDDEN })
        ),
        "k = {HIDDEN} is not a whole number of 32-wide blocks"
    );
}
