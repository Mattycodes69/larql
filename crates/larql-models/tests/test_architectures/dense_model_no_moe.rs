//! Dense model — no MoE
//! `output_head_reuses_embedding` — tied-embedding synthesis

use super::*;

#[test]
fn llama_not_moe() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }));
    assert!(!arch.is_moe());
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert); // default
    assert_eq!(arch.num_experts(), 0);
}

#[test]
fn generic_architecture_exercises_default_trait_contract() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "unknown_model",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "sliding_window": 128,
        "rope_theta": 20000.0,
        "rope_scaling": {"type": "linear", "factor": 2.0}
    }));

    assert_eq!(arch.family(), "generic");
    assert_eq!(arch.layer_prefix(7), "layers.7.");
    assert_eq!(
        arch.key_prefixes_to_strip(),
        &["language_model.model.", "model."]
    );
    assert_eq!(arch.embed_key(), "embed_tokens.weight");
    assert_eq!(arch.final_norm_key(), "norm.weight");
    assert_eq!(arch.attn_q_key(1), "layers.1.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(1), "layers.1.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(1), "layers.1.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(1), "layers.1.self_attn.o_proj.weight");
    assert_eq!(arch.ffn_gate_key(1), "layers.1.mlp.gate_proj.weight");
    assert_eq!(arch.ffn_up_key(1), "layers.1.mlp.up_proj.weight");
    assert_eq!(arch.ffn_down_key(1), "layers.1.mlp.down_proj.weight");
    assert_eq!(
        arch.input_layernorm_key(1),
        "layers.1.input_layernorm.weight"
    );
    assert_eq!(
        arch.post_attention_layernorm_key(1),
        "layers.1.post_attention_layernorm.weight"
    );
    assert_eq!(
        arch.pre_feedforward_layernorm_key(1),
        Some("layers.1.pre_feedforward_layernorm.weight".to_string())
    );
    assert_eq!(
        arch.post_feedforward_layernorm_key(1),
        Some("layers.1.post_feedforward_layernorm.weight".to_string())
    );

    assert_eq!(arch.attn_o_bias_key(1), None);
    assert_eq!(arch.attn_q_bias_key(1), None);
    assert_eq!(arch.attn_k_bias_key(1), None);
    assert_eq!(arch.attn_v_bias_key(1), None);
    assert_eq!(arch.attn_q_norm_key(1), None);
    assert_eq!(arch.attn_k_norm_key(1), None);
    assert_eq!(arch.ffn_up_bias_key(1), None);
    assert_eq!(arch.ffn_down_bias_key(1), None);

    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.norm_weight_offset(), 0.0);
    assert_eq!(arch.qk_norm_weight_offset(), 0.0);
    // No embedding-scale operation declared — `None`, not `Some(1.0)`.
    assert_eq!(arch.embed_scale(), None);
    assert_eq!(arch.bos_token_id(), None);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Gated);
    assert!(!arch.has_post_norms());
    assert!(!arch.is_sliding_window_layer(1));
    assert_eq!(arch.sliding_window_size(), Some(128));
    assert_eq!(arch.rope_base_for_layer(1), 20000.0);
    assert_eq!(arch.head_dim_for_layer(1), 4);
    assert_eq!(arch.num_q_heads_for_layer(1), 4);
    assert_eq!(arch.num_kv_heads_for_layer(1), 2);
    assert_eq!(arch.rotary_fraction_for_layer(1), 1.0);
    assert!(!arch.v_shares_k(1));
    assert!(!arch.has_v_norm());
    assert_eq!(arch.layer_scalar_key(1), None);
    assert_eq!(arch.attention_scale(), 0.5);
    assert_eq!(arch.attention_scale_for_layer(1), 0.5);
    assert_eq!(arch.kv_shared_source_layer(1), None);

    assert!(!arch.has_per_layer_embeddings());
    assert_eq!(arch.per_layer_embed_dim(), 0);
    assert_eq!(arch.per_layer_embed_key(), None);
    assert_eq!(arch.per_layer_input_gate_key(1), None);
    assert_eq!(arch.per_layer_projection_key(1), None);
    assert_eq!(arch.post_per_layer_input_norm_key(1), None);
    assert_eq!(arch.attn_logit_softcapping(), None);
    assert_eq!(arch.final_logit_softcapping(), None);
    assert_eq!(arch.residual_multiplier(), 1.0);
    assert_eq!(arch.attention_multiplier(), 1.0);
    assert_eq!(arch.logits_scaling(), 1.0);

    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
    assert!(!arch.is_moe());
    assert_eq!(arch.num_experts(), 0);
    assert_eq!(arch.num_experts_per_token(), 0);
    assert_eq!(arch.num_shared_experts(), 0);
    assert_eq!(arch.moe_router_key(1), None);
    assert_eq!(arch.moe_router_type(), "top_k_softmax");
    assert_eq!(arch.expert_ffn_gate_key(1, 0), None);
    assert_eq!(arch.expert_ffn_up_key(1, 0), None);
    assert_eq!(arch.expert_ffn_down_key(1, 0), None);
    assert_eq!(arch.packed_gate_up_blocks_key(1), None);
    assert_eq!(arch.packed_gate_up_scales_key(1), None);
    assert_eq!(arch.packed_down_blocks_key(1), None);
    assert_eq!(arch.packed_down_scales_key(1), None);
    assert_eq!(arch.shared_expert_gate_key(1), None);
    assert_eq!(arch.shared_expert_up_key(1), None);
    assert_eq!(arch.shared_expert_down_key(1), None);

    assert!(!arch.is_hybrid_moe());
    assert_eq!(arch.moe_intermediate_size(), 0);
    assert_eq!(arch.packed_experts_gate_up_key(1), None);
    assert_eq!(arch.packed_experts_down_key(1), None);
    assert_eq!(arch.moe_router_scale_key(1), None);
    assert_eq!(arch.moe_router_per_expert_scale_key(1), None);
    assert_eq!(arch.moe_router_norm_key(1), None);
    assert!(!arch.moe_router_norm_parameter_free());
    assert_eq!(arch.moe_router_input_scalar(), None);
    assert_eq!(arch.moe_post_outer_norm_key(1), None);
    assert_eq!(arch.moe_post_ffn1_norm_key(1), None);
    assert_eq!(arch.moe_pre_experts_norm_key(1), None);
    assert_eq!(arch.moe_post_experts_norm_key(1), None);
    assert!(!arch.moe_has_combined_output_norm());

    assert!(!arch.uses_mla());
    assert_eq!(arch.kv_lora_rank(), 0);
    assert_eq!(arch.q_lora_rank(), 0);
    assert_eq!(arch.mla_kv_a_key(1), None);
    assert_eq!(arch.mla_kv_b_key(1), None);
    assert_eq!(arch.mla_q_a_key(1), None);
    assert_eq!(arch.mla_q_b_key(1), None);
    assert_eq!(arch.rope_scaling_type(), Some("linear"));
    assert_eq!(arch.rope_scaling_factor(), 2.0);
    assert_eq!(arch.norm_eps(), 1e-6);
}

#[test]
fn output_head_reuses_embedding_when_tie_word_embeddings_absent() {
    // H5a's own rule: absence is not a claim either way, and every
    // existing tied-embedding checkpoint (Gemma 2/3 included) relies on
    // exactly this — they never bother re-declaring the HF class default.
    let arch = text_lm_arch_with_tie(None);
    assert!(arch.output_head_reuses_embedding());
}

#[test]
fn output_head_reuses_embedding_when_tie_word_embeddings_true() {
    let arch = text_lm_arch_with_tie(Some(true));
    assert!(arch.output_head_reuses_embedding());
}

#[test]
fn output_head_does_not_reuse_embedding_when_explicitly_untied() {
    // GPT-OSS/OLMoE shape: `tie_word_embeddings: false` must block the
    // reuse fallback, so a checkpoint that lost its `lm_head` tensor fails
    // loudly instead of silently serving the embedding as the projection.
    let arch = text_lm_arch_with_tie(Some(false));
    assert!(!arch.output_head_reuses_embedding());
}

#[test]
fn gemma2_output_head_reuses_embedding_by_default() {
    // Gemma 2 checkpoints ship no `lm_head.weight` and never declare
    // `tie_word_embeddings` — the trait default must resolve this to
    // "tied", not "unjudged".
    let arch = gemma2_arch();
    assert!(arch.has_lm_head());
    assert!(arch.output_head_reuses_embedding());
}
