//! StarCoder2 — LayerNorm, GELU, bias, non-gated FFN, c_fc/c_proj keys
//! Generic fallback — safe defaults for unknown models
//! Cross-architecture: default multipliers are 1.0 for non-Granite
//! Q4 round-trip: quantize then dequantize

use super::*;

#[test]
fn starcoder2_detection() {
    let arch = starcoder2_arch();
    assert_eq!(arch.family(), "starcoder2");
    assert_eq!(arch.config().num_layers, 30);
}

/// LayerNorm is `γ·x̂ + β`, so a LayerNorm architecture must name its `β`.
///
/// Nothing named a norm bias until 2026-07-31: extraction never wrote one and
/// `build_pipeline_layers` hardcoded `input_norm_bias: None`, so every
/// vindex-backed and Metal path silently dropped the shift term for GPT-2 and
/// StarCoder2 — while the Metal `layer_norm` shader implemented `+ bias` and
/// always selected its no-bias variant.
#[test]
fn starcoder2_names_its_layernorm_biases() {
    let arch = starcoder2_arch();
    assert_eq!(
        arch.input_layernorm_bias_key(3).as_deref(),
        Some("layers.3.input_layernorm.bias")
    );
    assert_eq!(
        arch.post_attention_layernorm_bias_key(3).as_deref(),
        Some("layers.3.post_attention_layernorm.bias")
    );
    assert_eq!(arch.final_norm_bias_key().as_deref(), Some("norm.bias"));
}

/// An RMSNorm architecture has no `β` at all, so it must claim none —
/// otherwise the coverage audit would treat a nonexistent tensor as expected
/// and extraction would ask for a key that can never resolve.
#[test]
fn an_rmsnorm_architecture_claims_no_layernorm_bias() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 32
    }));
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert!(arch.input_layernorm_bias_key(0).is_none());
    assert!(arch.post_attention_layernorm_bias_key(0).is_none());
    assert!(arch.final_norm_bias_key().is_none());
}

/// The bias key is derived from the weight key, so it tracks any architecture
/// that overrides its norm naming instead of drifting from it.
#[test]
fn the_norm_bias_key_tracks_the_weight_key() {
    let arch = starcoder2_arch();
    for layer in [0usize, 7] {
        let weight = arch.input_layernorm_key(layer);
        let bias = arch.input_layernorm_bias_key(layer).unwrap();
        assert_eq!(bias, weight.replace(".weight", ".bias"));
        assert!(bias.ends_with(".bias"));
    }
}

#[test]
fn starcoder2_norm_and_activation() {
    let arch = starcoder2_arch();
    assert_eq!(arch.norm_type(), larql_models::NormType::LayerNorm);
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Standard);
}

#[test]
fn starcoder2_ffn_keys() {
    let arch = starcoder2_arch();
    // Uses c_fc/c_proj naming
    assert_eq!(arch.ffn_up_key(0), "layers.0.mlp.c_fc.weight");
    assert_eq!(arch.ffn_down_key(0), "layers.0.mlp.c_proj.weight");
}

#[test]
fn starcoder2_bias_keys() {
    let arch = starcoder2_arch();
    // FFN biases
    assert_eq!(arch.ffn_up_bias_key(0).unwrap(), "layers.0.mlp.c_fc.bias");
    assert_eq!(
        arch.ffn_down_bias_key(0).unwrap(),
        "layers.0.mlp.c_proj.bias"
    );
    // Attention biases (including O)
    assert_eq!(
        arch.attn_q_bias_key(0).unwrap(),
        "layers.0.self_attn.q_proj.bias"
    );
    assert_eq!(
        arch.attn_k_bias_key(0).unwrap(),
        "layers.0.self_attn.k_proj.bias"
    );
    assert_eq!(
        arch.attn_v_bias_key(0).unwrap(),
        "layers.0.self_attn.v_proj.bias"
    );
    assert_eq!(
        arch.attn_o_bias_key(0).unwrap(),
        "layers.0.self_attn.o_proj.bias"
    );
}

#[test]
fn generic_fallback() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_future_model",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008,
        "num_attention_heads": 32, "num_key_value_heads": 32
    }));
    assert_eq!(arch.family(), "generic");
    // All safe defaults
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Gated);
    assert_eq!(arch.norm_weight_offset(), 0.0);
    // No embedding-scale operation declared — `None`, not `Some(1.0)`.
    assert_eq!(arch.embed_scale(), None);
    assert!(!arch.has_post_norms());
    assert!(!arch.is_moe());
    assert!(!arch.uses_mla());
    assert!(arch.attn_q_norm_key(0).is_none());
    assert!(arch.attn_logit_softcapping().is_none());
    assert!(arch.attn_q_bias_key(0).is_none());
    assert!(arch.ffn_up_bias_key(0).is_none());
    // Standard keys still work
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
}

#[test]
fn non_granite_multipliers_are_one() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504, "num_attention_heads": 16, "num_key_value_heads": 2}),
    ];
    for config in &configs {
        let arch = detect_from_json(config);
        assert_eq!(
            arch.residual_multiplier(),
            1.0,
            "{} should have residual_multiplier=1.0",
            arch.family()
        );
        assert_eq!(
            arch.attention_multiplier(),
            1.0,
            "{} should have attention_multiplier=1.0",
            arch.family()
        );
        assert_eq!(
            arch.logits_scaling(),
            1.0,
            "{} should have logits_scaling=1.0",
            arch.family()
        );
    }
}

#[test]
fn q4_0_round_trip() {
    use larql_models::quant::ggml;

    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let q4 = ggml::quantize_q4_0(&data);
    let decoded = ggml::dequantize_q4_0(&q4, 64).unwrap();

    assert_eq!(decoded.len(), 64);
    let max_err: f32 = data
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q4 is lossy but should be within ~2x the quantization step
    assert!(
        max_err < 2.0,
        "Q4 round-trip max error {max_err} exceeds 2.0"
    );
}

#[test]
fn q8_0_round_trip() {
    use larql_models::quant::ggml;

    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let q8 = ggml::quantize_q8_0(&data);
    let decoded = ggml::dequantize(&q8, ggml::TYPE_Q8_0, 32).unwrap();

    assert_eq!(decoded.len(), 32);
    let max_err: f32 = data
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q8 should be much more accurate than Q4
    assert!(
        max_err < 0.02,
        "Q8 round-trip max error {max_err} exceeds 0.02"
    );
}
