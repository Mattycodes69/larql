//! Config validation
//! Cross-architecture key comparison

use super::*;

#[test]
fn validation_accepts_known_architecture_configs() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880, "num_attention_heads": 64, "num_key_value_heads": 8, "num_local_experts": 128, "num_experts_per_tok": 4, "head_dim": 64}),
        serde_json::json!({"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48, "intermediate_size": 6144, "moe_intermediate_size": 768, "num_attention_heads": 32, "num_key_value_heads": 4, "num_experts": 128, "num_experts_per_tok": 8}),
        serde_json::json!({"model_type": "gemma4", "text_config": {"model_type": "gemma4_text", "hidden_size": 1536, "intermediate_size": 6144, "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512, "num_global_key_value_heads": 1, "sliding_window_pattern": 2, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"], "num_kv_shared_layers": 1, "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}, "sliding_attention": {"rope_theta": 10000.0}}}}),
    ];

    for config in &configs {
        let arch = detect_from_json(config);
        assert!(
            arch.validate().is_ok(),
            "{} failed validation: {:?}",
            arch.family(),
            arch.validate().err()
        );
    }
}

#[test]
fn validation_rejects_zero_core_dimensions() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 0,
        "num_hidden_layers": 0,
        "intermediate_size": 0,
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
        "head_dim": 0,
        "rope_theta": 0.0,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_LAYERS));
    assert!(fields.contains(&FIELD_HIDDEN_SIZE));
    assert!(fields.contains(&FIELD_INTERMEDIATE_SIZE));
    assert!(fields.contains(&FIELD_NUM_Q_HEADS));
    assert!(fields.contains(&FIELD_NUM_KV_HEADS));
    assert!(fields.contains(&FIELD_HEAD_DIM));
    assert!(fields.contains(&FIELD_ROPE_BASE));
}

#[test]
fn detect_from_json_validated_returns_validation_error() {
    let result = detect_from_json_validated(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 0,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
    }));

    assert!(result.is_err());
}

#[test]
fn validation_rejects_invalid_attention_geometry() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4100,
        "num_hidden_layers": 2,
        "intermediate_size": 8192,
        "num_attention_heads": 10,
        "num_key_value_heads": 3,
        "head_dim": 128,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_Q_HEADS));
}

#[test]
fn validation_rejects_invalid_rope_values() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 2,
        "intermediate_size": 8192,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "partial_rotary_factor": 1.5,
        "rope_scaling": {"type": "", "factor": -1.0},
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_PARTIAL_ROTARY_FACTOR));
    assert!(fields.contains(&FIELD_ROPE_SCALING_TYPE));
    assert!(fields.contains(&FIELD_ROPE_SCALING_FACTOR));
}

#[test]
fn validation_rejects_layer_metadata_mismatch() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "layer_types": ["sliding_attention", ""],
            "num_kv_shared_layers": 4,
        }
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_LAYER_TYPES));
    assert!(fields.contains(&FIELD_NUM_KV_SHARED_LAYERS));
}

#[test]
fn validation_rejects_moe_without_routing_width() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 16,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_EXPERTS_PER_TOKEN));
}

#[test]
fn validation_rejects_moe_top_k_greater_than_experts() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 8,
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_NUM_EXPERTS_PER_TOKEN));
}

#[test]
fn validation_rejects_hybrid_moe_without_expert_hidden_size() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "enable_moe_block": true,
            "num_experts": 4,
            "top_k_experts": 1,
        }
    }));
    let fields = validation_fields(arch.as_ref());

    assert!(fields.contains(&FIELD_MOE_INTERMEDIATE_SIZE));
}

#[test]
fn all_architectures_have_attn_keys() {
    let configs = [
        serde_json::json!({"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "hidden_size": 2560, "num_hidden_layers": 34, "intermediate_size": 10240, "num_attention_heads": 8, "num_key_value_heads": 4}}),
        serde_json::json!({"model_type": "mistral", "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8}),
        serde_json::json!({"model_type": "qwen2", "hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 5504, "num_attention_heads": 16, "num_key_value_heads": 2}),
        serde_json::json!({"model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 36, "intermediate_size": 2880, "num_attention_heads": 64, "num_key_value_heads": 8, "num_local_experts": 128, "num_experts_per_tok": 4}),
    ];

    for config in &configs {
        let arch = detect_from_json(config);
        // All architectures must produce non-empty attention keys
        assert!(
            !arch.attn_q_key(0).is_empty(),
            "{} has empty Q key",
            arch.family()
        );
        assert!(
            !arch.attn_k_key(0).is_empty(),
            "{} has empty K key",
            arch.family()
        );
        assert!(
            !arch.attn_v_key(0).is_empty(),
            "{} has empty V key",
            arch.family()
        );
        assert!(
            !arch.attn_o_key(0).is_empty(),
            "{} has empty O key",
            arch.family()
        );
    }
}
