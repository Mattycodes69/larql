//! Gemma 4 — per-layer geometry, partial RoPE, V-norm, KV sharing
//! Gemma 2 — softcapping, QK norm with +1 offset
//! Gemma 3 — sliding window, dual RoPE, QK norm offset
//! Mistral — sliding window, Llama-compatible keys

use super::*;

#[test]
fn gemma4_per_layer_head_dim() {
    let arch = gemma4_e2b_arch();
    // Sliding: head_dim=256, Global: head_dim=512
    assert_eq!(arch.head_dim_for_layer(0), 256);
    assert_eq!(arch.head_dim_for_layer(3), 256);
    assert_eq!(arch.head_dim_for_layer(4), 512); // first global
    assert_eq!(arch.head_dim_for_layer(9), 512);
    assert_eq!(arch.head_dim_for_layer(34), 512); // last layer (global)

    // Q heads constant across all layers
    assert_eq!(arch.num_q_heads_for_layer(0), 8);
    assert_eq!(arch.num_q_heads_for_layer(4), 8);

    // KV heads: 1 everywhere (E2B has MQA, no num_global_key_value_heads)
    assert_eq!(arch.num_kv_heads_for_layer(0), 1);
    assert_eq!(arch.num_kv_heads_for_layer(4), 1);
}

#[test]
fn gemma4_partial_rotary() {
    let arch = gemma4_e2b_arch();
    // Sliding: full rotation
    assert_eq!(arch.rotary_fraction_for_layer(0), 1.0);
    assert_eq!(arch.rotary_fraction_for_layer(3), 1.0);
    // Global: 25% rotation
    assert_eq!(arch.rotary_fraction_for_layer(4), 0.25);
    assert_eq!(arch.rotary_fraction_for_layer(9), 0.25);
}

#[test]
fn gemma4_rope_bases() {
    let arch = gemma4_e2b_arch();
    // Sliding: 10k, Global: 1M
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(4), 1_000_000.0);
}

#[test]
fn gemma4_attention_scale_is_one() {
    let arch = gemma4_e2b_arch();
    // QK-norm makes explicit scaling unnecessary
    assert_eq!(arch.attention_scale(), 1.0);
    assert_eq!(arch.attention_scale_for_layer(0), 1.0);
    assert_eq!(arch.attention_scale_for_layer(4), 1.0);
}

#[test]
fn gemma4_v_norm() {
    let arch = gemma4_e2b_arch();
    assert!(arch.has_v_norm());
}

#[test]
fn gemma4_norm_offset_zero() {
    let arch = gemma4_e2b_arch();
    // Gemma 4 stores weights as full multiplier (no +1 like Gemma 2/3)
    assert_eq!(arch.norm_weight_offset(), 0.0);
}

#[test]
fn gemma4_kv_sharing() {
    let arch = gemma4_e2b_arch();
    // First 15 layers: no sharing
    for l in 0..15 {
        assert!(
            arch.kv_shared_source_layer(l).is_none(),
            "L{l} should not be shared"
        );
    }
    // Layers 15-34: shared
    // Sliding shared layers → last non-shared sliding (L13)
    assert_eq!(arch.kv_shared_source_layer(15), Some(13));
    assert_eq!(arch.kv_shared_source_layer(16), Some(13));
    // Global shared layers → last non-shared global (L14)
    assert_eq!(arch.kv_shared_source_layer(19), Some(14));
    assert_eq!(arch.kv_shared_source_layer(34), Some(14));
}

#[test]
fn gemma4_ple() {
    let arch = gemma4_e2b_arch();
    assert!(arch.has_per_layer_embeddings());
    assert_eq!(arch.per_layer_embed_dim(), 256);
    assert_eq!(
        arch.per_layer_input_gate_key(5),
        Some("layers.5.per_layer_input_gate.weight".to_string())
    );
    assert_eq!(
        arch.per_layer_projection_key(5),
        Some("layers.5.per_layer_projection.weight".to_string())
    );
}

#[test]
fn gemma4_layer_scalar() {
    let arch = gemma4_e2b_arch();
    assert_eq!(
        arch.layer_scalar_key(10),
        Some("layers.10.layer_scalar".to_string())
    );
}

#[test]
fn gemma4_prefix_strip() {
    let arch = gemma4_e2b_arch();
    let prefixes = arch.key_prefixes_to_strip();
    // Must strip model.language_model. for multimodal Gemma 4
    assert!(prefixes.contains(&"model.language_model."));
    assert!(prefixes.contains(&"model.language_model.model."));
}

#[test]
fn gemma4_gemma_family_traits() {
    let arch = gemma4_e2b_arch();
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert!(arch.has_post_norms());
    assert!(arch.attn_q_norm_key(0).is_some());
    assert!(arch.attn_k_norm_key(0).is_some());
    // embed_scale = sqrt(hidden_size)
    assert_eq!(arch.embed_scale(), Some((1536.0f32).sqrt()));
}

#[test]
fn gemma2_detection() {
    let arch = gemma2_arch();
    assert_eq!(arch.family(), "gemma2");
    assert_eq!(arch.config().num_layers, 26);
}

#[test]
fn gemma2_softcapping() {
    let arch = gemma2_arch();
    assert_eq!(arch.attn_logit_softcapping(), Some(50.0));
    assert_eq!(arch.final_logit_softcapping(), Some(30.0));
}

#[test]
fn gemma2_norm_offsets() {
    let arch = gemma2_arch();
    assert_eq!(arch.norm_weight_offset(), 1.0);
    assert_eq!(arch.qk_norm_weight_offset(), 1.0);
}

#[test]
fn gemma2_qk_norm_keys() {
    let arch = gemma2_arch();
    assert_eq!(
        arch.attn_q_norm_key(5).unwrap(),
        "layers.5.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.attn_k_norm_key(5).unwrap(),
        "layers.5.self_attn.k_norm.weight"
    );
}

#[test]
fn gemma2_attention_scale() {
    let arch = gemma2_arch();
    // query_pre_attn_scalar = 256 → scale = 256^(-0.5) = 1/16 = 0.0625
    let expected = (256.0f64).powf(-0.5);
    assert_eq!(arch.attention_scale(), expected);
}

#[test]
fn gemma2_gemma_family_traits() {
    let arch = gemma2_arch();
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert!(arch.has_post_norms());
    assert_eq!(arch.embed_scale(), Some((2304.0f32).sqrt()));
}

#[test]
fn gemma2_sliding_window_alternates_every_other_layer() {
    // HF `Gemma2DecoderLayer.is_sliding = not bool(layer_idx % 2)`: even
    // layers slide, odd layers see full attention — a fixed period-2
    // pattern, not a declared `layer_types` interleave.
    let arch = gemma2_arch();
    for layer in 0..26 {
        assert_eq!(
            arch.is_sliding_window_layer(layer),
            layer.is_multiple_of(2),
            "layer {layer} sliding-window mismatch"
        );
    }
    assert_eq!(arch.sliding_window_size(), None); // no `sliding_window` in gemma2_arch()'s fixture
}

#[test]
fn gemma3_detection() {
    let arch = gemma3_arch();
    assert_eq!(arch.family(), "gemma3");
    assert_eq!(arch.config().num_layers, 34);
}

#[test]
fn gemma3_sliding_window_pattern() {
    let arch = gemma3_arch();
    // Every 6th layer (0-indexed: 5, 11, 17, ...) is full attention
    assert!(arch.is_sliding_window_layer(0));
    assert!(arch.is_sliding_window_layer(4));
    assert!(!arch.is_sliding_window_layer(5)); // full
    assert!(arch.is_sliding_window_layer(6));
    assert!(!arch.is_sliding_window_layer(11)); // full
}

#[test]
fn gemma3_dual_rope_bases() {
    let arch = gemma3_arch();
    // Sliding layers: 10k, full layers: 1M
    assert_eq!(arch.rope_base_for_layer(0), 10_000.0);
    assert_eq!(arch.rope_base_for_layer(5), 1_000_000.0);
}

#[test]
fn gemma3_norm_offsets() {
    let arch = gemma3_arch();
    assert_eq!(arch.norm_weight_offset(), 1.0);
    assert_eq!(arch.qk_norm_weight_offset(), 1.0);
}

#[test]
fn gemma3_gemma_family_traits() {
    let arch = gemma3_arch();
    assert_eq!(arch.activation(), larql_models::Activation::GeluTanh);
    assert!(arch.has_post_norms());
    assert_eq!(arch.embed_scale(), Some((2560.0f32).sqrt()));
    assert!(arch.attn_q_norm_key(0).is_some());
    // No softcapping on Gemma 3
    assert!(arch.attn_logit_softcapping().is_none());
    assert!(arch.final_logit_softcapping().is_none());
}

#[test]
fn mistral_detection_and_keys() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "mistral",
        "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8, "sliding_window": 4096
    }));
    assert_eq!(arch.family(), "mistral");
    assert_eq!(arch.sliding_window_size(), Some(4096));
    // Mistral uses same keys as Llama
    assert_eq!(arch.attn_q_key(0), "layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.ffn_gate_key(0), "layers.0.mlp.gate_proj.weight");
    // RMSNorm, SiLU, gated FFN
    assert_eq!(arch.norm_type(), larql_models::NormType::RmsNorm);
    assert_eq!(arch.activation(), larql_models::Activation::Silu);
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Gated);
}
