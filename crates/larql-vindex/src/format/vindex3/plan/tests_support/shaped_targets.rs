//! Gemma-4-, Glimmer-, Gemma-3- and drafter-shaped plan fixtures.

use larql_models::inventory::ArchitectureInventory;
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// The same fixture with `mutate_config` applied to its config and
/// `mutate_tensors` to its `(name, shape)` list before writing.
pub fn gemma4_shaped_target_with(
    dir: &Path,
    mutate_config: impl FnOnce(&mut serde_json::Value),
    mutate_tensors: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
) -> ArchitectureInventory {
    let layer_types: Vec<&str> = (0..GEMMA4_FIXTURE_LAYERS)
        .map(|i| {
            if i == GEMMA4_FULL_LAYER {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    // Built in three pieces: one `json!` of this depth trips the macro's
    // recursion limit.
    let text_config = serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": GEMMA4_HIDDEN,
            "num_hidden_layers": GEMMA4_FIXTURE_LAYERS,
            "intermediate_size": GEMMA4_INTER,
            "num_attention_heads": GEMMA4_Q_HEADS,
            "num_key_value_heads": GEMMA4_KV_HEADS,
            "head_dim": GEMMA4_HEAD_DIM,
            "global_head_dim": GEMMA4_GLOBAL_HEAD_DIM,
            "num_global_key_value_heads": GEMMA4_GLOBAL_KV_HEADS,
            "attention_k_eq_v": true,
            "attention_bias": false,
            "enable_moe_block": true,
            "num_experts": GEMMA4_EXPERTS,
            "top_k_experts": GEMMA4_TOP_K,
            "moe_intermediate_size": GEMMA4_MOE_INTER,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": GEMMA4_VOCAB,
            "use_double_wide_mlp": false,
            "num_kv_shared_layers": 0,
            "use_bidirectional_attention": "vision",
            "vocab_size": GEMMA4_VOCAB,
            "sliding_window": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": GEMMA4_PARTIAL_ROTARY,
                    "rope_theta": GEMMA4_FULL_THETA,
                    "rope_type": "proportional"
                },
                "sliding_attention": { "rope_theta": GEMMA4_SLIDING_THETA, "rope_type": "default" }
            },
        "layer_types": layer_types,
        "tie_word_embeddings": true
    });
    let vision_config = serde_json::json!({
        "model_type": "gemma4_vision",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "intermediate_size": 64,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 100.0, "rope_type": "default" },
            "attention_bias": false,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 64,
            "default_output_length": 4,
            "standardize": true,
            "use_clipped_linears": false,
            "id2label": { "0": "LABEL_0" },
            "label2id": { "LABEL_0": 0 },
            "problem_type": null,
            "return_dict": true,
            "output_attentions": false,
            "output_hidden_states": false,
        "is_encoder_decoder": false,
        "chunk_size_feed_forward": 0
    });
    let mut config = serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "gemma4",
        "audio_config": null,
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "boi_token_id": 255999,
        "eoa_token_id": 258883,
        "eoa_token_index": 258883,
        "eoi_token_id": 258882,
        "image_token_id": 258880,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 4,
        "tie_word_embeddings": true,
        "text_config": text_config,
        "vision_config": vision_config
    });
    mutate_config(&mut config);

    let h = GEMMA4_HIDDEN;
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        (
            "model.language_model.embed_tokens.weight".into(),
            vec![GEMMA4_VOCAB, h],
        ),
        ("model.language_model.norm.weight".into(), vec![h]),
        (
            "model.embed_vision.embedding_projection.weight".into(),
            vec![h, 32],
        ),
        (
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight".into(),
            vec![32, 32],
        ),
    ];
    for layer in 0..GEMMA4_FIXTURE_LAYERS {
        let stack = format!("model.language_model.layers.{layer}");
        let full = layer == GEMMA4_FULL_LAYER;
        let (head_dim, kv_heads) = if full {
            (GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_KV_HEADS)
        } else {
            (GEMMA4_HEAD_DIM, GEMMA4_KV_HEADS)
        };
        let q_rows = GEMMA4_Q_HEADS * head_dim;
        let kv_rows = kv_heads * head_dim;
        tensors.push((format!("{stack}.self_attn.q_proj.weight"), vec![q_rows, h]));
        tensors.push((format!("{stack}.self_attn.k_proj.weight"), vec![kv_rows, h]));
        if !full {
            tensors.push((format!("{stack}.self_attn.v_proj.weight"), vec![kv_rows, h]));
        }
        tensors.push((format!("{stack}.self_attn.o_proj.weight"), vec![h, q_rows]));
        tensors.push((format!("{stack}.self_attn.q_norm.weight"), vec![head_dim]));
        tensors.push((format!("{stack}.self_attn.k_norm.weight"), vec![head_dim]));
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            tensors.push((format!("{stack}.{norm}.weight"), vec![h]));
        }
        tensors.push((format!("{stack}.layer_scalar"), vec![1]));
        tensors.push((
            format!("{stack}.mlp.gate_proj.weight"),
            vec![GEMMA4_INTER, h],
        ));
        tensors.push((format!("{stack}.mlp.up_proj.weight"), vec![GEMMA4_INTER, h]));
        tensors.push((
            format!("{stack}.mlp.down_proj.weight"),
            vec![h, GEMMA4_INTER],
        ));
        tensors.push((
            format!("{stack}.router.proj.weight"),
            vec![GEMMA4_EXPERTS, h],
        ));
        tensors.push((format!("{stack}.router.scale"), vec![h]));
        tensors.push((
            format!("{stack}.router.per_expert_scale"),
            vec![GEMMA4_EXPERTS],
        ));
        tensors.push((
            format!("{stack}.experts.gate_up_proj"),
            vec![GEMMA4_EXPERTS, 2 * GEMMA4_MOE_INTER, h],
        ));
        tensors.push((
            format!("{stack}.experts.down_proj"),
            vec![GEMMA4_EXPERTS, h, GEMMA4_MOE_INTER],
        ));
    }
    mutate_tensors(&mut tensors);
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in &tensors {
        push_tensor(&mut header, &mut offset, name, shape);
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

// ── Gemma 3 ──────────────────────────────────────────────────────────
//
// The miniature mirrors `google/gemma-3-4b-it`'s config shape and tensor
// spelling: a multimodal root with `*_index` token roles and
// `mm_tokens_per_image`; a `text_config` that declares NO `vocab_size`,
// NO head count and NO head width (HF leaves all three at class defaults
// — 8 / 4 / 256 — which the parser supplies and the tensors below are
// shaped to, so the estate binds only if the defaults are right), a
// flat `rope_scaling = {linear, 8.0}` and a sliding window; a SigLIP
// `vision_config` with `vision_use_head: false`.

pub const GEMMA3_FIXTURE_LAYERS: usize = 12;
/// HF's `sliding_window_pattern` class default is 6: every sixth layer
/// is full attention, so at twelve layers two are — an instrument, not a
/// single special case.
pub const GEMMA3_FULL_LAYERS: [usize; 2] = [5, 11];
pub const GEMMA3_HIDDEN: usize = 32;
pub const GEMMA3_INTER: usize = 64;
/// The attention class defaults the real 4B config leaves implicit.
pub const GEMMA3_Q_HEADS: usize = 8;
pub const GEMMA3_KV_HEADS: usize = 4;
pub const GEMMA3_HEAD_DIM: usize = 256;
/// Deliberately not a power of two and not a multiple of anything the
/// stack pads to: a width read off the wrong axis or rounded shows.
pub const GEMMA3_VOCAB: usize = 300;
pub const GEMMA3_GLOBAL_THETA: f64 = 1_000_000.0;
pub const GEMMA3_LOCAL_THETA: f64 = 10_000.0;
pub const GEMMA3_ROPE_FACTOR: f64 = 8.0;
pub const GEMMA3_SLIDING_WINDOW: usize = 16;
pub const GEMMA3_VISION_HIDDEN: usize = 16;
/// The embedding as the checkpoint spells it — the tensor that answers
/// `vocab_size` when the config does not.
pub const GEMMA3_EMBED_TENSOR: &str = "language_model.model.embed_tokens.weight";

pub fn gemma3_shaped_target(dir: &Path) -> ArchitectureInventory {
    gemma3_shaped_target_with(dir, |_| {}, |_| {})
}

/// The same fixture with `mutate_config` applied to its config and
/// `mutate_tensors` to its `(name, shape)` list before writing.
pub fn gemma3_shaped_target_with(
    dir: &Path,
    mutate_config: impl FnOnce(&mut serde_json::Value),
    mutate_tensors: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
) -> ArchitectureInventory {
    let mut config = serde_json::json!({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "boi_token_index": 255999,
        "eoi_token_index": 256000,
        "eos_token_id": [1, 106],
        "image_token_index": 262144,
        "initializer_range": 0.02,
        "mm_tokens_per_image": 256,
        "model_type": "gemma3",
        "text_config": {
            "hidden_size": GEMMA3_HIDDEN,
            "intermediate_size": GEMMA3_INTER,
            "model_type": "gemma3_text",
            "num_hidden_layers": GEMMA3_FIXTURE_LAYERS,
            "rope_scaling": { "factor": GEMMA3_ROPE_FACTOR, "rope_type": "linear" },
            "sliding_window": GEMMA3_SLIDING_WINDOW
        },
        "torch_dtype": "bfloat16",
        "transformers_version": "4.50.0.dev0",
        "vision_config": {
            "hidden_size": GEMMA3_VISION_HIDDEN,
            "image_size": 896,
            "intermediate_size": 32,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "patch_size": 14,
            "vision_use_head": false
        }
    });
    mutate_config(&mut config);

    let h = GEMMA3_HIDDEN;
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        (GEMMA3_EMBED_TENSOR.into(), vec![GEMMA3_VOCAB, h]),
        ("language_model.model.norm.weight".into(), vec![h]),
        (
            "multi_modal_projector.mm_input_projection_weight".into(),
            vec![GEMMA3_VISION_HIDDEN, h],
        ),
        (
            "multi_modal_projector.mm_soft_emb_norm.weight".into(),
            vec![GEMMA3_VISION_HIDDEN],
        ),
        (
            "vision_tower.vision_model.embeddings.patch_embedding.weight".into(),
            vec![GEMMA3_VISION_HIDDEN, 3, 14, 14],
        ),
        (
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight".into(),
            vec![GEMMA3_VISION_HIDDEN, GEMMA3_VISION_HIDDEN],
        ),
    ];
    let q_rows = GEMMA3_Q_HEADS * GEMMA3_HEAD_DIM;
    let kv_rows = GEMMA3_KV_HEADS * GEMMA3_HEAD_DIM;
    for layer in 0..GEMMA3_FIXTURE_LAYERS {
        let stack = format!("language_model.model.layers.{layer}");
        tensors.push((format!("{stack}.self_attn.q_proj.weight"), vec![q_rows, h]));
        tensors.push((format!("{stack}.self_attn.k_proj.weight"), vec![kv_rows, h]));
        tensors.push((format!("{stack}.self_attn.v_proj.weight"), vec![kv_rows, h]));
        tensors.push((format!("{stack}.self_attn.o_proj.weight"), vec![h, q_rows]));
        tensors.push((
            format!("{stack}.self_attn.q_norm.weight"),
            vec![GEMMA3_HEAD_DIM],
        ));
        tensors.push((
            format!("{stack}.self_attn.k_norm.weight"),
            vec![GEMMA3_HEAD_DIM],
        ));
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            tensors.push((format!("{stack}.{norm}.weight"), vec![h]));
        }
        tensors.push((
            format!("{stack}.mlp.gate_proj.weight"),
            vec![GEMMA3_INTER, h],
        ));
        tensors.push((format!("{stack}.mlp.up_proj.weight"), vec![GEMMA3_INTER, h]));
        tensors.push((
            format!("{stack}.mlp.down_proj.weight"),
            vec![h, GEMMA3_INTER],
        ));
    }
    mutate_tensors(&mut tensors);
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in &tensors {
        push_tensor(&mut header, &mut offset, name, shape);
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

/// A drafter-shaped artifact declaring `target_layer_ids` taps into a
/// deeper producer.
pub fn drafter_shaped(dir: &Path) -> ArchitectureInventory {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerAssistantModel"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "sliding_window": 16,
        "block_size": 4,
        "mask_token_id": 99,
        "target_layer_ids": [1, 3, 5]
    });
    // The first four tensors keep their historical offsets — encode tests
    // slice the source pattern at [8320, 32896) for the projector. The
    // appended estate mirrors the real assistant's two-norm + QK-norm
    // layer anatomy (11 tensors per layer).
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    push_tensor(
        &mut header,
        &mut offset,
        "layers.0.self_attn.q_proj.weight",
        &[64, 64],
    );
    push_tensor(&mut header, &mut offset, "norm.weight", &[64]);
    push_tensor(&mut header, &mut offset, "encoder.fc.weight", &[192, 64]);
    push_tensor(
        &mut header,
        &mut offset,
        "encoder.output_norm_enc.weight",
        &[64],
    );
    for layer in 0..2 {
        if layer > 0 {
            push_tensor(
                &mut header,
                &mut offset,
                &format!("layers.{layer}.self_attn.q_proj.weight"),
                &[64, 64],
            );
        }
        // 8 q-heads * head_dim 8 = 64 rows; 4 kv-heads * 8 = 32 rows.
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.k_proj.weight"),
            &[32, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.v_proj.weight"),
            &[32, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.o_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.q_norm.weight"),
            &[8],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.k_norm.weight"),
            &[8],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.input_layernorm.weight"),
            &[64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.post_attention_layernorm.weight"),
            &[64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.gate_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.up_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.down_proj.weight"),
            &[64, 256],
        );
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}
