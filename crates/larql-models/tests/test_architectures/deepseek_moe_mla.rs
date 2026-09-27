//! DeepSeek — MoE + MLA
//! Granite — scaling multipliers

use super::*;

#[test]
fn deepseek_detection() {
    let arch = deepseek_arch();
    assert_eq!(arch.family(), "deepseek");
    assert_eq!(arch.config().num_layers, 60);
}

#[test]
fn deepseek_moe() {
    let arch = deepseek_arch();
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 160);
    assert_eq!(arch.num_experts_per_token(), 6);
    assert_eq!(arch.num_shared_experts(), 2);
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
}

#[test]
fn deepseek_expert_keys() {
    let arch = deepseek_arch();
    assert_eq!(arch.moe_router_key(0).unwrap(), "layers.0.mlp.gate.weight");
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.gate_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.up_proj.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).unwrap(),
        "layers.0.mlp.experts.5.down_proj.weight"
    );
}

#[test]
fn deepseek_shared_expert_keys() {
    let arch = deepseek_arch();
    assert_eq!(
        arch.shared_expert_gate_key(0).unwrap(),
        "layers.0.mlp.shared_experts.gate_proj.weight"
    );
    assert_eq!(
        arch.shared_expert_up_key(0).unwrap(),
        "layers.0.mlp.shared_experts.up_proj.weight"
    );
    assert_eq!(
        arch.shared_expert_down_key(0).unwrap(),
        "layers.0.mlp.shared_experts.down_proj.weight"
    );
}

#[test]
fn deepseek_mla() {
    let arch = deepseek_arch();
    assert!(arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 512);
    assert_eq!(arch.q_lora_rank(), 1536);
    assert_eq!(
        arch.mla_kv_a_key(0).unwrap(),
        "layers.0.self_attn.kv_a_proj_with_mqa.weight"
    );
    assert_eq!(
        arch.mla_kv_b_key(0).unwrap(),
        "layers.0.self_attn.kv_b_proj.weight"
    );
    assert_eq!(
        arch.mla_q_a_key(0).unwrap(),
        "layers.0.self_attn.q_a_proj.weight"
    );
    assert_eq!(
        arch.mla_q_b_key(0).unwrap(),
        "layers.0.self_attn.q_b_proj.weight"
    );
}

#[test]
fn deepseek_rope_scaling() {
    let arch = deepseek_arch();
    assert_eq!(arch.rope_scaling_type(), Some("yarn"));
    assert_eq!(arch.rope_scaling_factor(), 40.0);
}

#[test]
fn granite_detection() {
    let arch = granite_arch();
    assert_eq!(arch.family(), "granite");
    assert_eq!(arch.config().num_layers, 40);
}

#[test]
fn granite_scaling_multipliers() {
    let arch = granite_arch();
    assert_eq!(arch.embed_scale(), Some(12.0));
    assert_eq!(arch.residual_multiplier(), 0.22);
    assert_eq!(arch.attention_multiplier(), 0.22);
    assert_eq!(arch.logits_scaling(), 0.13);
}

#[test]
fn granite_uses_llama_defaults() {
    let arch = granite_arch();
    // Same keys, norm, activation as Llama
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert!(!arch.is_moe());
}
