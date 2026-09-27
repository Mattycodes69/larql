//! ModelWeights — drop_attn_weights, drop_lm_head, drop_embed, get_packed_bytes

use super::*;

#[test]
fn drop_attn_weights_removes_qkvo_and_norms() {
    let mut w = minimal_weights();
    assert_eq!(w.tensors.len(), 9);
    let freed = w.drop_attn_weights();
    assert!(freed > 0);
    // q/k/v/o + q_norm removed (5 tensors); FFN + norm remain (4)
    assert_eq!(w.tensors.len(), 4, "expected ffn + layernorm to remain");
    assert!(!w.tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    assert!(!w.tensors.contains_key("layers.0.self_attn.q_norm.weight"));
    assert!(w.tensors.contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(w.tensors.contains_key("layers.0.input_layernorm.weight"));
}

#[test]
fn drop_attn_weights_frees_correct_byte_count() {
    let mut w = minimal_weights();
    // 5 attn tensors × (2×4 elements) × 4 bytes = 160 bytes
    let freed = w.drop_attn_weights();
    assert_eq!(freed, 5 * 2 * 4 * 4);
}

#[test]
fn drop_lm_head_zeroes_matrix_and_reports_freed() {
    let mut w = minimal_weights();
    let freed = w.drop_lm_head();
    assert_eq!(freed, 2 * 4 * 4, "freed should be elem_count × sizeof(f32)");
    assert_eq!(w.lm_head.shape(), &[0, 0]);
}

#[test]
fn drop_embed_zeroes_matrix_and_reports_freed() {
    let mut w = minimal_weights();
    let freed = w.drop_embed();
    assert_eq!(freed, 2 * 4 * 4);
    assert_eq!(w.embed.shape(), &[0, 0]);
}

#[test]
fn get_packed_bytes_from_raw_bytes() {
    let mut w = minimal_weights();
    w.raw_bytes
        .insert("experts.gate_up_proj".into(), vec![1u8, 2, 3, 4]);
    let bytes = w.get_packed_bytes("experts.gate_up_proj").unwrap();
    assert_eq!(bytes, &[1u8, 2, 3, 4]);
}

#[test]
fn get_packed_bytes_from_mmap_range_takes_precedence() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[10u8, 11, 12, 13, 14, 15]).unwrap();
    file.flush().unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut w = minimal_weights();
    w.raw_bytes.insert("tensor.key".into(), vec![1u8, 2, 3]);
    w.packed_mmaps.insert("packed.bin".into(), mmap);
    w.packed_byte_ranges
        .insert("tensor.key".into(), ("packed.bin".into(), 2, 3));

    assert_eq!(w.get_packed_bytes("tensor.key").unwrap(), &[12u8, 13, 14]);
}

#[test]
fn get_packed_bytes_out_of_bounds_mmap_range_returns_none() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[10u8, 11, 12, 13]).unwrap();
    file.flush().unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut w = minimal_weights();
    w.packed_mmaps.insert("packed.bin".into(), mmap);
    w.packed_byte_ranges
        .insert("tensor.key".into(), ("packed.bin".into(), 3, 4));

    assert!(w.get_packed_bytes("tensor.key").is_none());
}

#[test]
fn per_layer_ffn_bytes_detects_and_loads_entries() {
    let mut w = minimal_weights();
    w.raw_bytes.insert(
        larql_models::weights::per_layer_ffn_key(
            2,
            7,
            larql_models::weights::PER_LAYER_FFN_GATE_UP,
        ),
        vec![1u8, 2, 3],
    );
    w.raw_bytes.insert(
        larql_models::weights::per_layer_ffn_key(2, 7, larql_models::weights::PER_LAYER_FFN_DOWN),
        vec![4u8, 5],
    );
    w.packed_byte_ranges.insert(
        larql_models::weights::per_layer_ffn_key(
            9,
            1,
            larql_models::weights::PER_LAYER_FFN_GATE_UP,
        ),
        ("missing.bin".into(), 0, 1),
    );

    assert!(w.has_per_layer_ffn());
    let (gate_up, down) = w.get_layer_entry_bytes(2, 7).unwrap();
    assert_eq!(gate_up, &[1u8, 2, 3]);
    assert_eq!(down, &[4u8, 5]);
    assert!(w.get_layer_entry_bytes(2, 8).is_none());
    assert_eq!(
        larql_models::weights::per_layer_ffn_key(3, 4, larql_models::weights::PER_LAYER_FFN_DOWN,),
        "layers/3/4/down"
    );
}

#[test]
fn drop_ffn_weights_removes_raw_packed_expert_bytes() {
    let mut w = minimal_weights();
    w.raw_bytes
        .insert("layers.0.experts.gate_up_proj".into(), vec![1u8; 8]);
    w.raw_bytes
        .insert("layers.0.experts.down_proj".into(), vec![2u8; 4]);
    w.raw_bytes.insert("attention.cache".into(), vec![3u8; 2]);

    let freed = w.drop_ffn_weights();

    assert!(freed >= 12);
    assert!(!w.raw_bytes.contains_key("layers.0.experts.gate_up_proj"));
    assert!(!w.raw_bytes.contains_key("layers.0.experts.down_proj"));
    assert!(w.raw_bytes.contains_key("attention.cache"));
}

#[test]
fn drop_ffn_weights_releases_unreferenced_mmaps() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed.bin");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&[0u8; 16]).unwrap();
    file.flush().unwrap();
    drop(file);

    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut w = minimal_weights();
    w.packed_mmaps.insert("packed.bin".into(), mmap);
    w.packed_byte_ranges.insert(
        "layers.0.experts.gate_up_proj".into(),
        ("packed.bin".into(), 0, 8),
    );

    let freed = w.drop_ffn_weights();

    assert!(freed >= 8);
    assert!(w.packed_byte_ranges.is_empty());
    assert!(w.packed_mmaps.is_empty());
}

#[test]
fn get_packed_bytes_missing_key_returns_none() {
    let w = minimal_weights();
    assert!(w.get_packed_bytes("nonexistent.key").is_none());
}

#[test]
fn get_packed_bytes_mmap_range_missing_file_falls_through_to_raw() {
    // packed_byte_ranges points to a file not in packed_mmaps → falls through to raw_bytes.
    let mut w = minimal_weights();
    w.raw_bytes.insert("tensor.key".into(), vec![9u8, 8]);
    w.packed_byte_ranges
        .insert("tensor.key".into(), ("missing_file.bin".into(), 0, 2));
    // mmap file absent → fallback to raw_bytes
    let bytes = w.get_packed_bytes("tensor.key").unwrap();
    assert_eq!(bytes, &[9u8, 8]);
}

#[test]
fn olmoe_moe_shape_and_keys() {
    let a = olmoe();
    assert_eq!(a.family(), "olmoe");
    assert!(a.is_moe());
    assert_eq!(a.num_experts(), 64);
    assert_eq!(a.num_experts_per_token(), 8);
    // No `moe_intermediate_size` field: expert width comes from
    // `intermediate_size`, not an `unwrap_or(0)`.
    assert_eq!(a.moe_intermediate_size(), 1024);
    // `norm_topk_prob: false` keeps raw softmax probabilities.
    assert_eq!(
        a.expert_routing_policy(),
        larql_models::ExpertRoutingPolicy::SoftmaxThenSelect
    );
    let p = a.layer_prefix(2);
    assert_eq!(a.moe_router_key(2), Some(format!("{p}mlp.gate.weight")));
    assert_eq!(
        a.expert_ffn_gate_key(2, 5),
        Some(format!("{p}mlp.experts.5.gate_proj.weight"))
    );
    assert_eq!(
        a.expert_ffn_up_key(2, 5),
        Some(format!("{p}mlp.experts.5.up_proj.weight"))
    );
    assert_eq!(
        a.expert_ffn_down_key(2, 5),
        Some(format!("{p}mlp.experts.5.down_proj.weight"))
    );
    assert_eq!(
        a.attn_q_norm_key(2),
        Some(format!("{p}self_attn.q_norm.weight"))
    );
    assert_eq!(
        a.attn_k_norm_key(2),
        Some(format!("{p}self_attn.k_norm.weight"))
    );
}

#[test]
fn olmoe_without_experts_is_dense() {
    let a = detect_from_json(&serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
    }));
    assert!(!a.is_moe());
    assert_eq!(a.num_experts_per_token(), 0);
    assert_eq!(a.moe_router_key(0), None);
    assert_eq!(a.expert_ffn_gate_key(0, 0), None);
    assert_eq!(a.expert_ffn_up_key(0, 0), None);
    assert_eq!(a.expert_ffn_down_key(0, 0), None);
}

/// Walk the `ModelArchitecture` defaults through a family that overrides
/// almost nothing, so this binary's instantiation of the default bodies
/// executes. Assertions mirror `config::tests` — see the header comment.
#[test]
fn trait_defaults_via_public_api() {
    let a = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 32,
    }));
    // Dense-model MoE surface.
    assert!(!a.is_moe());
    assert!(!a.is_hybrid_moe());
    assert_eq!(a.num_experts(), 0);
    assert_eq!(a.num_shared_experts(), 0);
    assert_eq!(a.moe_intermediate_size(), 0);
    assert_eq!(a.moe_router_type(), "top_k_softmax");
    assert_eq!(a.expert_format(), ExpertFormat::PerExpert);
    assert_eq!(
        a.expert_gate_policy(),
        larql_models::ExpertGatePolicy::Gated
    );
    assert_eq!(
        a.expert_routing_policy(),
        larql_models::ExpertRoutingPolicy::SoftmaxThenSelect
    );
    for k in [
        a.moe_router_bias_key(0),
        a.moe_router_scale_key(0),
        a.moe_router_per_expert_scale_key(0),
        a.moe_router_norm_key(0),
        a.shared_expert_gate_key(0),
        a.shared_expert_up_key(0),
        a.shared_expert_down_key(0),
        a.packed_gate_up_blocks_key(0),
        a.packed_gate_up_scales_key(0),
        a.packed_gate_up_bias_key(0),
        a.packed_down_blocks_key(0),
        a.packed_down_scales_key(0),
        a.packed_down_bias_key(0),
        a.packed_experts_gate_up_key(0),
        a.packed_experts_down_key(0),
        a.moe_post_outer_norm_key(0),
        a.moe_post_ffn1_norm_key(0),
        a.moe_pre_experts_norm_key(0),
        a.moe_post_experts_norm_key(0),
        a.attn_sinks_key(0),
        a.fused_qkv_key(0),
        a.fused_qkv_bias_key(0),
        a.ffn_up_bias_key(0),
        a.ffn_down_bias_key(0),
        a.layer_scalar_key(0),
        a.mla_kv_a_key(0),
        a.mla_kv_b_key(0),
        a.mla_q_a_key(0),
        a.mla_q_b_key(0),
    ] {
        assert_eq!(k, None);
    }
    assert!(!a.moe_router_norm_parameter_free());
    assert_eq!(a.moe_router_input_scalar(), None);
    assert!(!a.moe_has_combined_output_norm());
    // MLA off means standard GQA.
    assert!(!a.uses_mla());
    assert_eq!(a.kv_lora_rank(), 0);
    assert_eq!(a.q_lora_rank(), 0);
    assert_eq!(a.mla_qk_nope_head_dim(), None);
    assert_eq!(a.mla_qk_rope_head_dim(), None);
    assert_eq!(a.mla_v_head_dim(), None);
    // Identity multipliers, no softcapping, no PLE, no rope scaling.
    assert_eq!(a.residual_multiplier(), 1.0);
    assert_eq!(a.attention_multiplier(), 1.0);
    assert_eq!(a.logits_scaling(), 1.0);
    assert_eq!(a.attn_logit_softcapping(), None);
    assert_eq!(a.final_logit_softcapping(), None);
    assert!(!a.has_per_layer_embeddings());
    assert_eq!(a.per_layer_embed_dim(), 0);
    assert_eq!(a.per_layer_embed_key(), None);
    assert_eq!(a.per_layer_model_projection_key(), None);
    assert_eq!(a.per_layer_projection_norm_key(), None);
    assert_eq!(a.per_layer_input_gate_key(0), None);
    assert_eq!(a.per_layer_projection_key(0), None);
    assert_eq!(a.post_per_layer_input_norm_key(0), None);
    assert_eq!(a.rope_position_divisor_for_layer(0), 1.0);
    // llama overrides llama3_rope_scaling, so only walk it — the default's
    // None is asserted by `config::tests` on an override-free arch.
    let _ = a.llama3_rope_scaling();
    assert!(a.multimodal().is_none());
    assert_eq!(a.kv_shared_source_layer(0), None);
    // Scale falls back to head_dim^-0.5 without query_pre_attn_scalar.
    assert_eq!(a.attention_scale_for_layer(0), (16.0f64).powf(-0.5));
    // The markov-residual precondition holds for the default surface.
    assert!(a.kv_recomputable_from_residuals());
}
