use super::*;

/// A routed block with no bottleneck, for the serialisation arms
/// below. Every other field is arbitrary and unread by them.
fn uniform_moe() -> MoeExecution {
    MoeExecution {
        branch_scale: None,
        dense_prefix_layers: None,
        experts: 8,
        top_k: 2,
        expert_intermediate_size: 64,
        router_kind: crate::config::MoeRouterKind::TopKSoftmax,
        routing_policy: crate::config::ExpertRoutingPolicy::NormalisedOverSelected,
        router_bias: false,
        expert_format: crate::config::ExpertFormat::PerExpert,
        gate_up_layout: None,
        shared_experts: 0,
        shared_expert_intermediate_size: None,
        shared_expert_gate: None,
        hybrid: false,
        routed_expert_form: crate::config::RoutedExpertForm::Uniform,
    }
}

/// **A routed block that declares no bottleneck serialises exactly
/// as it did before the latent form existed.**
///
/// The skip predicate is the whole reason for this: `Uniform` is the
/// overwhelming default, and a container that started carrying
/// `routed_expert_form: "Uniform"` on every MoE row would make every
/// stored plan differ from its predecessor for a fact none of them
/// declares. Measured on the conformance corpus, this is what keeps
/// 108 of 109 rows byte-identical apart from the semantics stamp.
///
/// The latent arm beside it is what stops this from having been
/// implemented as "never write the field": a declared bottleneck
/// must appear, or the container would silently lose it.
#[test]
fn a_routed_block_without_a_bottleneck_serialises_as_it_always_did() {
    let uniform = serde_json::to_value(uniform_moe()).expect("serialises");
    assert!(
        uniform.get("routed_expert_form").is_none(),
        "the uniform form must add nothing to the container: {uniform}"
    );

    let mut latent = uniform_moe();
    latent.routed_expert_form = crate::config::RoutedExpertForm::Latent {
        width: 3584,
        norm: Some(crate::config::LatentNormSpec { eps: 1e-5 }),
    };
    let encoded = serde_json::to_value(latent).expect("serialises");
    assert!(
        encoded.get("routed_expert_form").is_some(),
        "a declared bottleneck must reach the container: {encoded}"
    );
    assert_eq!(
        serde_json::from_value::<MoeExecution>(encoded).expect("reads back"),
        latent,
        "and must round-trip unchanged"
    );

    // The absent field reads back as the uniform form, so a
    // container written before this rung is not a container that
    // declares something unknown.
    assert_eq!(
        serde_json::from_value::<MoeExecution>(uniform).expect("reads back"),
        uniform_moe()
    );
}

/// **The recurrent state dtype round-trips, and refuses what it
/// does not represent.**
///
/// `None` means undeclared or spelled in a way this build cannot
/// represent — never "float32 by default". Qwen3.8 declares
/// `float32` against a bf16 model, so a defaulted answer would put
/// the recurrence at the model's precision and quietly change the
/// operator.
#[test]
fn the_recurrent_state_dtype_round_trips_and_refuses_the_unrepresented() {
    for spelling in ["float32", "f32"] {
        assert_eq!(
            RecurrentStateDtype::from_declared(spelling),
            Some(RecurrentStateDtype::Float32),
            "`{spelling}` is a spelling of the same dtype"
        );
    }
    assert_eq!(
        RecurrentStateDtype::Float32.declared_name(),
        "float32",
        "the canonical spelling is what a container records"
    );
    assert_eq!(
        RecurrentStateDtype::from_declared(RecurrentStateDtype::Float32.declared_name()),
        Some(RecurrentStateDtype::Float32),
        "the canonical spelling must parse back"
    );
    for unknown in ["bfloat16", "float16", "fp32", "", "FLOAT32"] {
        assert_eq!(
            RecurrentStateDtype::from_declared(unknown),
            None,
            "`{unknown}` is not represented, so it must be refused rather than \
             approximated by the one variant that exists"
        );
    }
}

/// **The linear-attention widths are DERIVED, so they cannot drift
/// from the head counts they come from.**
///
/// Checked at Qwen3.8's real geometry, where the key and value sides
/// genuinely differ: `2·16·128 + 48·128 = 10240`, the observed
/// `in_proj_qkv` row count. A build that folded the two sides into
/// one head count would have to pick one, and either choice misses.
#[test]
fn the_linear_attention_widths_are_derived_from_both_sides() {
    let qwen38 = LinearAttentionTopology {
        key_heads: 16,
        key_head_dim: 128,
        value_heads: 48,
        value_head_dim: 128,
        conv_kernel: 4,
        state_dtype: Some(RecurrentStateDtype::Float32),
    };
    assert_eq!(
        qwen38.qkv_channels(),
        10240,
        "q and k at the KEY geometry, v at the value's"
    );
    assert_eq!(qwen38.value_width(), 6144);
    // Folding the sides would give a different number either way,
    // which is why they stay separate.
    assert_ne!(
        qwen38.qkv_channels(),
        3 * qwen38.key_heads * qwen38.key_head_dim
    );
    assert_ne!(qwen38.qkv_channels(), 3 * qwen38.value_width());
}

#[test]
fn key_status_serialises_lowercase() {
    assert_eq!(
        serde_json::to_string(&KeyStatus::Unconsumed).unwrap(),
        "\"unconsumed\""
    );
    assert_eq!(
        serde_json::to_string(&KeyStatus::Consumed).unwrap(),
        "\"consumed\""
    );
    assert_eq!(
        serde_json::to_string(&KeyStatus::Metadata).unwrap(),
        "\"metadata\""
    );
}

#[test]
fn key_status_round_trips() {
    for status in [
        KeyStatus::Consumed,
        KeyStatus::Metadata,
        KeyStatus::Unconsumed,
    ] {
        let json = serde_json::to_string(&status).unwrap();
        let back: KeyStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, status);
    }
}
