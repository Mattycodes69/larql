//! Qwen — attention bias, QK norm keys

use super::*;

#[test]
fn qwen_detection() {
    let arch = qwen_arch();
    assert_eq!(arch.family(), "qwen2");
    assert_eq!(arch.config().num_layers, 24);
}

#[test]
fn qwen_attention_bias_keys() {
    let arch = qwen_arch();
    assert_eq!(
        arch.attn_q_bias_key(3).unwrap(),
        "layers.3.self_attn.q_proj.bias"
    );
    assert_eq!(
        arch.attn_k_bias_key(3).unwrap(),
        "layers.3.self_attn.k_proj.bias"
    );
    assert_eq!(
        arch.attn_v_bias_key(3).unwrap(),
        "layers.3.self_attn.v_proj.bias"
    );
}

#[test]
fn qwen_qk_norm_keys() {
    let arch = qwen_arch();
    assert_eq!(
        arch.attn_q_norm_key(0).unwrap(),
        "layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.attn_k_norm_key(0).unwrap(),
        "layers.0.self_attn.k_norm.weight"
    );
}

#[test]
fn qwen_classic_norm_weights_are_not_offsets() {
    // Only qwen3_5/qwen3_next store norm weights as offsets from one;
    // the rest of the family stores them plain.
    let arch = qwen_arch();
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.qk_norm_weight_offset(), 0.0);
}

#[test]
fn qwen35_declared_output_gate_resolves_to_the_hf_sigmoid_spec() {
    use larql_models::config::{GateActivation, GateCombine, GatePlacement, GateSource};
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2048, "num_hidden_layers": 4, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "attn_output_gate": true
    }));
    let spec = arch.attention_output_gate().expect("gate declared true");
    assert_eq!(spec.source, GateSource::FusedQueryProjection);
    // Sigmoid, NOT the silu the config's `output_gate_type: "swish"`
    // spelling suggests — HF computes `x · sigmoid(g)`.
    assert_eq!(spec.activation, GateActivation::Sigmoid);
    assert_eq!(spec.combine, GateCombine::ElementwiseMultiply);
    assert_eq!(
        spec.placement,
        GatePlacement::AfterAggregationBeforeOutputProjection
    );

    // Declared false (or absent) is no gate, not a disabled gate.
    let ungated = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_5",
        "hidden_size": 2048, "num_hidden_layers": 4, "intermediate_size": 5504,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "attn_output_gate": false
    }));
    assert!(ungated.attention_output_gate().is_none());
}

#[test]
fn qwen_dense_returns_none_for_moe_keys() {
    let arch = qwen_arch();
    assert!(arch.moe_router_key(0).is_none());
    assert!(arch.expert_ffn_gate_key(0, 0).is_none());
    assert!(arch.expert_ffn_up_key(0, 0).is_none());
    assert!(arch.expert_ffn_down_key(0, 0).is_none());
}
