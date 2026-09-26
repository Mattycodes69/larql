//! GPT-OSS MXFP4 (load_mxfp4_expert_tensors): full-load path

use super::*;

#[test]
fn load_full_gpt_oss_dequantises_packed_mxfp4_experts() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 1,
        "num_experts_per_tok": 1,
        "head_dim": 2,
        "vocab_size": 10,
    });
    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.mlp.router.weight",
                "F32",
                &[1, 4],
                f32_bytes(&[1.0f32; 4]),
            ),
            // Packed MXFP4: gate+up fused, 1 expert, out_features=8 (2×inter=4),
            // groups=1, packed_bytes=16 per (expert, out, group).
            // shape = [num_experts=1, out_features=8, groups=1, 16].
            (
                "layers.0.mlp.experts.gate_up_proj_blocks",
                "U8",
                &[1, 8, 1, 16],
                vec![0x22; 8 * 16],
            ),
            (
                "layers.0.mlp.experts.gate_up_proj_scales",
                "U8",
                &[1, 8, 1],
                vec![127; 8],
            ),
            // Down: 1 expert, out_features=4 (=hidden), groups=1.
            (
                "layers.0.mlp.experts.down_proj_blocks",
                "U8",
                &[1, 4, 1, 16],
                vec![0x22; 4 * 16],
            ),
            (
                "layers.0.mlp.experts.down_proj_scales",
                "U8",
                &[1, 4, 1],
                vec![127; 4],
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).expect("full GPT-OSS load");
    // The dequantiser splits gate_up into gate (w1) + up (w3) per expert,
    // and emits down (w2). Each per-expert key uses the
    // `block_sparse_moe.experts.<E>.<proj>` shape from
    // `mxfp4_expert_key`. Pin that we got non-empty tensors back.
    let any_expert_key = weights
        .tensors
        .keys()
        .any(|k| k.contains("block_sparse_moe.experts.0"));
    assert!(
        any_expert_key,
        "load_mxfp4_expert_tensors must populate per-expert keys; got: {:?}",
        weights.tensors.keys().collect::<Vec<_>>()
    );
}
