//! ModelWeights: drop_ffn_weights

use super::*;

#[test]
fn drop_ffn_weights_removes_ffn_tensors() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2
    }));

    let small = WeightArray::zeros((2, 4));

    let mut tensors = HashMap::new();
    // FFN tensors (should be removed)
    tensors.insert("layers.0.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.down_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.gate_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.up_proj.weight".into(), small.clone());
    tensors.insert("layers.1.mlp.down_proj.weight".into(), small.clone());
    // Attention tensors (should be kept)
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.k_proj.weight".into(), small.clone());
    // Norm (should be kept)
    tensors.insert("layers.0.input_layernorm.weight".into(), small.clone());

    let mut weights = ModelWeights {
        tensors,
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    assert_eq!(weights.tensors.len(), 9);
    let freed = weights.drop_ffn_weights();

    // 6 FFN tensors removed (2 layers × gate/up/down)
    assert_eq!(weights.tensors.len(), 3, "should keep attn + norm only");
    assert!(freed > 0, "should report freed bytes");

    // Verify correct tensors remain
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.k_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.input_layernorm.weight"));

    // Verify FFN tensors are gone
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(!weights
        .tensors
        .contains_key("layers.1.mlp.down_proj.weight"));
}

#[test]
fn drop_ffn_weights_removes_moe_experts() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2
    }));

    let small = WeightArray::zeros((2, 4));
    let mut tensors = HashMap::new();
    // MoE expert tensors
    tensors.insert(
        "layers.0.block_sparse_moe.experts.0.w1.weight".into(),
        small.clone(),
    );
    tensors.insert(
        "layers.0.block_sparse_moe.experts.0.w2.weight".into(),
        small.clone(),
    );
    tensors.insert(
        "layers.0.block_sparse_moe.experts.0.w3.weight".into(),
        small.clone(),
    );
    // Attention (keep)
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());

    let mut weights = ModelWeights {
        tensors,
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    weights.drop_ffn_weights();
    // mlp.experts matches the "mlp.experts" pattern
    assert_eq!(weights.tensors.len(), 1, "should only keep attn");
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn drop_ffn_weights_removes_mmap_backed_packed_experts() {
    let mut weights = minimal_weights();
    weights.packed_byte_ranges.insert(
        "layers.0.experts.gate_up_proj".into(),
        ("experts.safetensors".into(), 128, 16),
    );
    weights.packed_byte_ranges.insert(
        "layers.0.experts.down_proj".into(),
        ("experts.safetensors".into(), 256, 8),
    );

    let freed = weights.drop_ffn_weights();

    assert!(freed >= 24);
    assert!(weights.packed_byte_ranges.is_empty());
}

#[test]
fn drop_ffn_weights_removes_starcoder2_ffn_tensors_and_biases() {
    use larql_models::{ModelWeights, WeightArray};
    use std::collections::HashMap;

    let arch = detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 2
    }));

    let small = WeightArray::zeros((2, 4));
    let mut tensors = HashMap::new();
    tensors.insert("layers.0.mlp.c_fc.weight".into(), small.clone());
    tensors.insert("layers.0.mlp.c_proj.weight".into(), small.clone());
    tensors.insert("layers.0.self_attn.q_proj.weight".into(), small.clone());

    let mut vectors = HashMap::new();
    vectors.insert("layers.0.mlp.c_fc.bias".into(), vec![0.0; 8]);
    vectors.insert("layers.0.mlp.c_proj.bias".into(), vec![0.0; 4]);
    vectors.insert("layers.0.input_layernorm.weight".into(), vec![1.0; 4]);

    let mut weights = ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: small.clone(),
        lm_head: small.clone(),
        position_embed: None,
        arch,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 100,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
        rope_base: 10000.0,
    };

    let freed = weights.drop_ffn_weights();
    assert!(freed > 0);
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_fc.weight"));
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_proj.weight"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_fc.bias"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_proj.bias"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(weights
        .vectors
        .contains_key("layers.0.input_layernorm.weight"));
}
