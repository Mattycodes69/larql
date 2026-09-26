use super::*;

/// The whole layer, against a reference that composes the already-gated
/// attention and expert paths with a host router and combine.
#[test]
fn the_layer_matches_a_reference_composed_of_its_gated_parts() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let got = m
        .kimi_decoder_layer_traced(f.layer(&state), &f.x)
        .expect("device layer");

    // The reference, stage by stage, on a state that advances the same way.
    let ref_state = KdaDeviceState::zeros(&m, shape());
    let normed = rms_norm(&f.x, &f.input_norm, EPS);
    let (attn, _) = m
        .kda_attention_step(f.kda(), shape(), &ref_state, &normed)
        .expect("attention");
    let after: Vec<f32> = f.x.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let post = rms_norm(&after, &f.post_norm, EPS);
    let (ids, weights) = route(&post, &f.router_weight, &f.router_bias);

    assert_eq!(got.input_normed, normed, "input norm");
    assert!(max_abs(&got.attention, &attn) < TOLERANCE, "attention");
    assert!(
        max_abs(&got.after_attention, &after) < TOLERANCE,
        "residual"
    );
    assert!(
        max_abs(&got.post_attention_normed, &post) < TOLERANCE,
        "post norm"
    );
    assert_eq!(
        got.selected_ids,
        ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        "the device chose other experts"
    );
    assert!(
        max_abs(&got.combine_weights[..TOP_K], &weights) < TOLERANCE,
        "routed weights"
    );
    assert_eq!(
        got.combine_weights[TOP_K], 1.0,
        "the shared branch is unscaled"
    );
    assert_eq!(
        got.expert_offsets,
        ids.iter().map(|&i| f.residency[i]).collect::<Vec<_>>(),
        "the GPU-written ROUTED offset table"
    );

    // The routed experts, through the already-gated block path — routed
    // slots only, because the shared branch no longer lives in the
    // routed bank.
    let offsets: Vec<ExpertOffset> = ids.iter().map(|&i| ExpertOffset(f.residency[i])).collect();
    let (outs, _) = m
        .bf16_moe_ffn_blocks(
            &[MoeBlockCall {
                banks: MoeFfnBanks {
                    gate: bank(&f.bank_gate, &offsets),
                    up: bank(&f.bank_up, &offsets),
                    down: bank(&f.bank_down, &offsets),
                    hidden: HIDDEN,
                    inter: INTER,
                },
                x: &post,
            }],
            BlockLowering::Separate,
        )
        .expect("experts");
    // The shared branch: the same gated block path over its own
    // regions, one slot at offset zero.
    let zero = [ExpertOffset(0)];
    let (shared_outs, _) = m
        .bf16_moe_ffn_blocks(
            &[MoeBlockCall {
                banks: MoeFfnBanks {
                    gate: bank(&f.shared_gate, &zero),
                    up: bank(&f.shared_up, &zero),
                    down: bank(&f.shared_down, &zero),
                    hidden: HIDDEN,
                    inter: INTER,
                },
                x: &post,
            }],
            BlockLowering::Separate,
        )
        .expect("shared expert");
    let all_outs: Vec<f32> = outs[0].iter().chain(&shared_outs[0]).copied().collect();
    assert!(
        max_abs(&got.expert_outputs, &all_outs) < TOLERANCE,
        "per-slot expert outputs"
    );

    let mut want = after.clone();
    for (slot, w) in got.combine_weights.iter().enumerate() {
        for (j, o) in want.iter_mut().enumerate() {
            *o += w * all_outs[slot * HIDDEN + j];
        }
    }
    assert!(max_abs(&got.output, &want) < TOLERANCE, "layer output");
}

/// A route naming a non-resident expert is refused, not served.
#[test]
fn a_non_resident_selection_is_refused() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let mut evicted = f.residency.clone();
    // Evict every resident expert but one, so the route must leave the
    // bank.
    for slot in evicted.iter_mut().take(RESIDENT).skip(1) {
        *slot = layer_shader::NOT_RESIDENT;
    }
    let mut w = f.layer(&state);
    {
        let m = moe_mut(&mut w);
        let a = ExpertAddressing::Table(&evicted);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }
    assert!(matches!(
        m.kimi_decoder_layer(w, &f.x),
        Err(GroupedError::LayerRouteNotResident { layer: 0, .. })
    ));
    // Still usable — a refusal must not have left the backend wedged.
    assert!(m.kimi_decoder_layer(f.layer(&state), &f.x).is_ok());
}

/// An identity bank whose offsets pass 32 bits is REFUSED, not wrapped.
///
/// `expert * stride` used to be formed in `u32`: past 4 GiB it wrapped
/// (release) to an in-bounds offset — another expert's weights — and the
/// host validator, doing the same arithmetic, agreed with the wrapped
/// value. Here expert 1 sits just inside `u32` and genuinely inside the
/// bank, so the ONLY fault is expert 2's width: a refusal of any other
/// kind would mean the width was never checked.
///
/// The bank is a zeroed allocation that validation measures and never
/// reads, so it stays virtual.
#[test]
fn an_identity_bank_past_32_bit_offsets_is_refused_not_wrapped() {
    const { assert!(EXPERTS > 2, "expert 2 must exist to overflow") };
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let stride = u32::MAX / 2 + 1;
    let per = INTER * HIDDEN * std::mem::size_of::<u16>();
    let huge = vec![0u8; stride as usize + per];
    let identity = ExpertAddressing::Identity {
        experts: EXPERTS,
        stride,
    };

    let mut w = f.layer(&state);
    {
        let moe = moe_mut(&mut w);
        for bank in [&mut moe.gate, &mut moe.up, &mut moe.down] {
            bank.routed.bytes = &huge;
            bank.addressing = identity;
        }
    }
    let overflow = 2 * u64::from(stride);
    assert_eq!(
        m.kimi_decoder_layer(w, &f.x).map(|(o, _)| o),
        Err(GroupedError::OffsetExceedsAddressWidth {
            slot: 0,
            offset: overflow,
        })
    );
    assert_eq!(
        identity.offset_of(2),
        Some(overflow),
        "the host offset is formed in 64 bits, not wrapped to {}",
        overflow as u32
    );
}

/// Host-side shape faults refuse before anything is encoded.
#[test]
fn shape_faults_are_refused() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());

    let mut no_experts = f.layer(&state);
    moe_mut(&mut no_experts).top_k = 0;
    assert_eq!(
        m.kimi_decoder_layer(no_experts, &f.x).map(|(o, _)| o),
        Err(GroupedError::NoExpertsSelected)
    );

    let short = vec![0u32; EXPERTS - 1];
    let mut bad_residency = f.layer(&state);
    {
        let m = moe_mut(&mut bad_residency);
        let a = ExpertAddressing::Table(&short);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }
    assert!(matches!(
        m.kimi_decoder_layer(bad_residency, &f.x),
        Err(GroupedError::SlotCountMismatch { .. })
    ));

    let mut truncated = f.layer(&state);
    let half = &f.bank_down[..f.bank_down.len() / 2];
    moe_mut(&mut truncated).down.routed.bytes = half;
    assert!(matches!(
        m.kimi_decoder_layer(truncated, &f.x),
        Err(GroupedError::OffsetOutOfRange { .. })
    ));

    let mut too_many = f.layer(&state);
    let wide = vec![0u32; layer_shader::MAX_EXPERTS + 1];
    let wide_w = vec![0.0f32; (layer_shader::MAX_EXPERTS + 1) * HIDDEN];
    let wide_b = vec![0.0f32; layer_shader::MAX_EXPERTS + 1];
    {
        let m = moe_mut(&mut too_many);
        let a = ExpertAddressing::Table(&wide);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }
    moe_mut(&mut too_many).router_weight = &wide_w;
    moe_mut(&mut too_many).router_bias = &wide_b;
    assert!(matches!(
        m.kimi_decoder_layer(too_many, &f.x),
        Err(GroupedError::SlotCountMismatch { .. })
    ));
}

/// The plain entry point agrees with the traced one, and the traced
/// planes are the lengths their names imply.
#[test]
fn the_traced_layer_agrees_with_the_plain_one() {
    let m = backend();
    let f = fixture();
    let a = KdaDeviceState::zeros(&m, shape());
    let (plain, gpu) = m.kimi_decoder_layer(f.layer(&a), &f.x).unwrap();
    let b = KdaDeviceState::zeros(&m, shape());
    let traced = m.kimi_decoder_layer_traced(f.layer(&b), &f.x).unwrap();

    assert_eq!(traced.output, plain, "tracing must not change the answer");
    assert!(gpu >= 0.0 && traced.gpu_ms >= 0.0);
    assert_eq!(traced.router_logits.len(), EXPERTS);
    assert_eq!(traced.router_scores.len(), EXPERTS);
    assert_eq!(traced.router_selection_scores.len(), EXPERTS);
    assert_eq!(traced.selected_ids.len(), TOP_K);
    assert_eq!(traced.combine_weights.len(), TOP_K + 1);
    // Routed offsets only: the shared branch resolves no address.
    assert_eq!(traced.expert_offsets.len(), TOP_K);
    assert_eq!(traced.expert_outputs.len(), (TOP_K + 1) * HIDDEN);
    assert_eq!(traced.input_normed.len(), HIDDEN);
    assert_eq!(traced.attention.len(), HIDDEN);
}
