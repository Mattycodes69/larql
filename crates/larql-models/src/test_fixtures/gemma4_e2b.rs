//! Gemma 4 E2B-like fixture (per-layer embeddings).

use crate::{detect_from_json, ModelWeights, WeightArray};

use super::*;

/// Tiny synthetic Gemma-4-E2B-shaped arch with PLE + KV sharing.
///
/// Same shape as `crates/larql-models/tests/test_architectures.rs::gemma4_e2b_arch`
/// but smaller (4 layers, hidden=8) so weights fit in-memory cheaply.
/// Shared with `layer_graph::pipeline_layer::tests` and the `forward::ple::tests`
/// module — both need `has_per_layer_embeddings()=true` AND valid PLE tensor
/// keys populated in `weights.tensors` / `weights.vectors`.
pub fn synthetic_e2b_like_arch_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 8,
            "vocab_size": 32,
            "sliding_window": 4,
            "hidden_size_per_layer_input": 4,
            "num_kv_shared_layers": 2,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                },
                "sliding_attention": {"rope_theta": 10000.0}
            },
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ]
        }
    })
}

/// Weight amplitude for the synthetic E2B-like fixture's LCG-seeded tensors.
pub const E2B_TEST_WEIGHT_SCALE: f32 = 0.05;
/// Per-layer `layer_scalar` value for the synthetic E2B-like fixture.
/// Deliberately neither 0.0 (skipped as absent) nor 1.0 (skipped as
/// identity) so `apply_layer_scalar` visibly changes the hidden state —
/// a forward path that drops the scalar diverges instead of matching.
pub const E2B_TEST_LAYER_SCALAR: f32 = 0.75;

/// Build minimal `ModelWeights` matching the synthetic E2B-like arch.
/// Tensors carry deterministic non-zero LCG values (not zeros) so the
/// PLE and `layer_scalar` contributions are non-trivial: parity tests
/// between forward paths fail if either step is dropped, not just when
/// the tensor keys are missing.
pub fn make_synthetic_e2b_like_weights() -> ModelWeights {
    let arch = detect_from_json(&synthetic_e2b_like_arch_json());
    let num_layers = 4;
    let hidden = 8;
    let intermediate = 16;
    let head_dim = 4;
    let global_head_dim = 8;
    let num_q_heads = 2;
    let num_kv_heads = 1;
    let vocab_size = 32;
    let ple_dim = 4;

    let mut tensors: std::collections::HashMap<String, WeightArray> =
        std::collections::HashMap::new();
    let mut vectors: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();

    let mut seed = 0xe2b_0000_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let scale = E2B_TEST_WEIGHT_SCALE;

    let embed = rand_mat_seeded(vocab_size, hidden, scale, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    if let Some(k) = arch.per_layer_model_projection_key() {
        tensors.insert(
            k,
            rand_mat_seeded(num_layers * ple_dim, hidden, scale, next_seed()),
        );
    }
    if let Some(k) = arch.per_layer_embed_key() {
        tensors.insert(
            k,
            rand_mat_seeded(vocab_size, num_layers * ple_dim, scale, next_seed()),
        );
    }
    if let Some(k) = arch.per_layer_projection_norm_key() {
        vectors.insert(k, vec![1.0; ple_dim]);
    }

    for layer in 0..num_layers {
        let layer_head_dim = if arch.is_sliding_window_layer(layer) {
            head_dim
        } else {
            global_head_dim
        };
        let q_dim = num_q_heads * layer_head_dim;
        let kv_dim = num_kv_heads * layer_head_dim;
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(hidden, q_dim, scale, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(intermediate, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(intermediate, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, intermediate, scale, next_seed()),
        );
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; hidden]);
        if let Some(k) = arch.per_layer_input_gate_key(layer) {
            tensors.insert(k, rand_mat_seeded(ple_dim, hidden, scale, next_seed()));
        }
        if let Some(k) = arch.per_layer_projection_key(layer) {
            tensors.insert(k, rand_mat_seeded(hidden, ple_dim, scale, next_seed()));
        }
        if let Some(k) = arch.post_per_layer_input_norm_key(layer) {
            vectors.insert(k, vec![1.0; hidden]);
        }
        if let Some(k) = arch.layer_scalar_key(layer) {
            vectors.insert(k, vec![E2B_TEST_LAYER_SCALAR]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        packed_mmaps: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base: 10_000.0,
    }
}
