//! inline_moe_preconditions: the single authority both arms consume

use super::*;

#[test]
fn admits_a_servable_inline_moe_layer() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    if let Err(e) = MetalBackend::inline_moe_preconditions(&layer, &precondition_ctx(), &scratch) {
        panic!(
            "the fixture must be the servable shape, else every refusal \
             assertion below is vacuous; refused with: {e}"
        );
    }
}

#[test]
fn refuses_a_layer_without_moe_weights() {
    let m = backend();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer::default();
    assert_refuses(&layer, &precondition_ctx(), &scratch, "no MoE weights");
}

#[test]
fn refuses_a_remote_ffn_layer() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ffn_is_remote: true,
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "ffn_is_remote");
}

/// The two context flags that mean "another arm owns this command buffer".
/// `stage_timing_split` in particular is why `LARQL_PROFILE_SPLIT=1` must
/// not be used to diagnose the merged-CB path — it disables it.
#[test]
fn refuses_when_another_arm_owns_the_command_buffer() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };

    let mut ctx = precondition_ctx();
    ctx.defer_ffn_for_split = true;
    assert_refuses(&layer, &ctx, &scratch, "defer_ffn_for_split");

    let mut ctx = precondition_ctx();
    ctx.stage_timing_split = true;
    assert_refuses(&layer, &ctx, &scratch, "stage_timing_split");
}

/// A capture hook needs the intermediate values the merged CB never
/// materialises on the host, so the two are mutually exclusive.
#[test]
fn refuses_while_a_capture_hook_is_active() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    let snapshot = vec![0.0f32; P_HIDDEN];

    let mut ctx = precondition_ctx();
    ctx.layer_in_snapshot = Some(&snapshot);
    assert_refuses(&layer, &ctx, &scratch, "layer_in_snapshot");

    let mut ctx = precondition_ctx();
    ctx.dump_l0_dir = Some("/tmp/does-not-need-to-exist");
    assert_refuses(&layer, &ctx, &scratch, "dump_l0_dir");
}

/// The identity-combine class: anything that post-processes the combined
/// output is a different shape than the merged CB encodes.
#[test]
fn refuses_layers_outside_the_identity_combine_class() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let ctx = precondition_ctx();

    let mut moe = f.moe();
    moe.routing_policy.post_expert_norm = larql_compute::MoePostExpertNormPolicy::RmsNorm;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "post_expert_norm");

    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        moe_combined_output_norm: true,
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "moe_combined_output_norm");

    // A layer scalar of 0 or 1 is absorbed; anything else must be applied
    // to the whole layer output, which this path does not do.
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        layer_scalar: 0.5,
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "layer_scalar");
}

/// A `Gated` layer carrying expert biases has no kernel — `ClampedGlu` is
/// the biased shape that does. Both directions are asserted so the check
/// cannot be satisfied by refusing every biased layer.
#[test]
fn refuses_a_gated_layer_with_expert_biases_but_admits_clamped_glu() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let ctx = precondition_ctx();
    let gu_bias = vec![0.1f32; f.gate_up.len() * 2 * P_INTER];

    let mut gated = f.moe();
    gated.gate_rule = MoeGateRule::Gated(Activation::GeluTanh);
    gated.experts_gate_up_bias = &gu_bias;
    let layer = FullPipelineLayer {
        moe: Some(gated),
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "no kernel");

    let mut clamped = f.moe();
    clamped.gate_rule = MoeGateRule::ClampedGlu {
        limit: 7.0,
        alpha: 1.702,
    };
    clamped.experts_gate_up_bias = &gu_bias;
    let layer = FullPipelineLayer {
        moe: Some(clamped),
        ..Default::default()
    };
    assert!(
        MetalBackend::inline_moe_preconditions(&layer, &ctx, &scratch).is_ok(),
        "ClampedGlu IS the biased shape with a kernel"
    );
}

/// Every dimension the scratch was allocated against. These are the checks
/// that stop a layer writing into slots sized for a different shape, and
/// each names the mismatched pair so the diagnostic is actionable.
#[test]
fn refuses_each_shape_that_disagrees_with_the_scratch() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);

    let mut moe = f.moe();
    moe.top_k = P_TOP_K + 1;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "top_k");

    let mut moe = f.moe();
    moe.intermediate_size = P_INTER * 2;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "intermediate_size");

    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    let mut ctx = precondition_ctx();
    ctx.hidden = P_HIDDEN * 2;
    assert_refuses(&layer, &ctx, &scratch, "hidden");

    let mut moe = f.moe();
    moe.expert_data_format = QuantFormat::Q6_K;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "expert_data_format");
}

/// The stored row width the scratch was sized for. A writer-padded bank
/// (gpt-oss stores 2880-wide rows at 3072) must match the scratch's
/// `weight_cols`, or every expert row is read at the wrong stride — the
/// one precondition here whose mismatch is silent rather than loud.
#[test]
fn refuses_a_stored_row_width_that_disagrees_with_the_scratch() {
    let m = backend();
    let f = precondition_fixture();
    // Scratch sized for a padded store; the fixture's bank is unpadded.
    let padded = MoeScratch::new_public_with_format(
        &m,
        P_TOP_K,
        P_HIDDEN,
        P_INTER,
        QuantFormat::Q4_K,
        P_HIDDEN + 256,
    );
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &padded, "gate_up_cols");
}
