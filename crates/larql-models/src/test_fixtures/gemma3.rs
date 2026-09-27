//! Gemma 3 fixtures: plain, RoPE-scaled and narrow-window.

use crate::{detect_from_json, ModelWeights, WeightArray};
use std::collections::HashMap;

use super::*;

/// Build a synthetic `ModelWeights` configured as a Gemma 3-style arch.
///
/// Enables the dormant branches in `attention/{block, gpu}.rs` and
/// `forward/layer.rs` that tinymodel never reaches:
/// - **QK norm** — `attn_q_norm_key` / `attn_k_norm_key` return Some
/// - **post norms** — `has_post_norms()` is true; pre/post FFN norm keys
///   are populated, the FFN dispatch routes through the post-norm arm
/// - **GeluTanh activation** — `activation()` is `GeluTanh`
/// - **`embed_scale = sqrt(hidden)`** — non-1.0 embed scaling
/// - **`norm_weight_offset = 1.0`** — non-zero offset added to every
///   norm weight at runtime
pub fn make_gemma3_test_weights() -> ModelWeights {
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const VOCAB: usize = 32;
    const NUM_LAYERS: usize = 2;
    gemma3_test_weights_inner(
        serde_json::json!({
            "model_type": "gemma3",
            "hidden_size": HIDDEN,
            "num_hidden_layers": NUM_LAYERS,
            "intermediate_size": INTER,
            "head_dim": HEAD_DIM,
            "num_attention_heads": NUM_Q,
            "num_key_value_heads": NUM_KV,
            "vocab_size": VOCAB,
            "rope_theta": 10000.0,
            "residual_multiplier": 0.5,
        }),
        NUM_LAYERS,
    )
}

/// Like [`make_gemma3_test_weights`] but with the structured per-layer-type
/// `rope_scaling` form (linear factor 8 on full-attention layers, default on
/// sliding) and **6 layers**, so layer 5 is a global/full-attention layer with
/// `position_divisor = 8`. Used to exercise the scaled-RoPE path — the engine
/// prefill and full-recompute attention must agree there (regression guard for
/// the 2026-05-28 prefill-RoPE divergence).
pub fn make_gemma3_rope_scaled_test_weights() -> ModelWeights {
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const VOCAB: usize = 32;
    const NUM_LAYERS: usize = 6;
    gemma3_test_weights_inner(
        serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": HIDDEN,
                "num_hidden_layers": NUM_LAYERS,
                "intermediate_size": INTER,
                "head_dim": HEAD_DIM,
                "num_attention_heads": NUM_Q,
                "num_key_value_heads": NUM_KV,
                "vocab_size": VOCAB,
                "rope_theta": 10000.0,
                "residual_multiplier": 0.5,
                "sliding_window": 1024,
                "rope_scaling": {
                    "full_attention": {"rope_type": "linear", "factor": 8.0},
                    "sliding_attention": {"rope_type": "default"},
                },
            },
        }),
        NUM_LAYERS,
    )
}

/// Gemma-3 shaped with a **narrow** sliding window (4 tokens) over 6
/// layers, so layers 0–4 slide and layer 5 is global.
///
/// The narrow window is the point: with a realistic 1024-token window a
/// short test prompt keeps every position inside every sliding layer's
/// view, so a global-only exclusion is correctly *refused* and the
/// heterogeneous-row-count case never arises. Four tokens puts a planted
/// span outside the sliding layers while keeping the fixture small
/// enough to prefill in a unit test.
pub fn make_gemma3_narrow_window_test_weights() -> ModelWeights {
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const VOCAB: usize = 32;
    const NUM_LAYERS: usize = 6;
    gemma3_test_weights_inner(
        serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": HIDDEN,
                "num_hidden_layers": NUM_LAYERS,
                "intermediate_size": INTER,
                "head_dim": HEAD_DIM,
                "num_attention_heads": NUM_Q,
                "num_key_value_heads": NUM_KV,
                "vocab_size": VOCAB,
                "rope_theta": 10000.0,
                "residual_multiplier": 0.5,
                "sliding_window": 4,
            },
        }),
        NUM_LAYERS,
    )
}

pub(super) fn gemma3_test_weights_inner(
    arch_json: serde_json::Value,
    num_layers: usize,
) -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x9e3779b9);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0xa1b2c3d4);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Gemma 3: norm_weight_offset=1.0; saved weight is delta off identity.
    vectors.insert(arch.final_norm_key().to_string(), vec![0.0; HIDDEN]);

    let mut seed_counter: u64 = 0xdeadbeef;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..num_layers {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );

        // Layer norms — input + post-attention. Gemma 3 norm_weight_offset=1.0
        // means saved weights are deltas; zeros → identity at runtime.
        vectors.insert(arch.input_layernorm_key(layer), vec![0.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.0; HIDDEN]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }

        // QK norm — per-head dim weights.
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}
