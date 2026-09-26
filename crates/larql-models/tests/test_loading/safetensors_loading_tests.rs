//! Safetensors loading tests

use super::*;

#[test]
fn load_f32_tensors_correct_values() {
    let dir = TempDir::new().unwrap();
    let known: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
    write_model_dir(
        dir.path(),
        &[
            ("embed_tokens.weight", "F32", &[10, 4], f32_bytes(&known)),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
    // First element: known[0] = 0.0
    assert!((weights.embed[[0, 0]] - known[0]).abs() < 1e-6);
    // Last element: known[39] = 3.9
    assert!((weights.embed[[9, 3]] - known[39]).abs() < 1e-5);
}

#[test]
fn load_model_dir_validated_rejects_invalid_config() {
    let dir = TempDir::new().unwrap();
    write_model_dir_with_config(
        dir.path(),
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 5,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 0,
            "vocab_size": 10,
        }),
        &minimal_tensors(),
    );

    let permissive = load_model_dir(dir.path()).unwrap();
    assert_eq!(permissive.hidden_size, 5);

    match load_model_dir_validated(dir.path()) {
        Err(ModelError::ConfigValidation(errors)) => {
            assert!(errors.iter().any(|error| error.field == FIELD_HEAD_DIM));
        }
        _ => panic!("expected config validation error"),
    }
}

#[test]
fn load_model_dir_walk_only_validated_rejects_invalid_config() {
    let dir = TempDir::new().unwrap();
    write_model_dir_with_config(
        dir.path(),
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 5,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 0,
            "vocab_size": 10,
        }),
        &minimal_tensors(),
    );

    match load_model_dir_walk_only_validated(dir.path()) {
        Err(ModelError::ConfigValidation(errors)) => {
            assert!(errors.iter().any(|error| error.field == FIELD_HEAD_DIM));
        }
        _ => panic!("expected config validation error"),
    }
}

#[test]
fn load_f16_tensors_converts_to_f32() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            ("embed_tokens.weight", "F16", &[10, 4], f16_ones(40)),
            ("norm.weight", "F16", &[4], f16_ones(4)),
            ("lm_head.weight", "F16", &[10, 4], f16_ones(40)),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
    // f16 1.0 → f32 1.0
    assert!((weights.embed[[0, 0]] - 1.0).abs() < 1e-4);
}

#[test]
fn load_bf16_tensors_converts_to_f32() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            ("embed_tokens.weight", "BF16", &[10, 4], bf16_ones(40)),
            ("norm.weight", "BF16", &[4], bf16_ones(4)),
            ("lm_head.weight", "BF16", &[10, 4], bf16_ones(40)),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
    assert!((weights.embed[[0, 0]] - 1.0).abs() < 1e-4);
}

#[test]
fn load_1d_norm_tensor_goes_into_vectors() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[2.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            (
                "layers.0.input_layernorm.weight",
                "F32",
                &[4],
                f32_bytes(&[3.0f32; 4]),
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    let norm = weights.vectors.get("norm.weight").unwrap();
    assert_eq!(norm.len(), 4);
    assert!((norm[0] - 2.0).abs() < 1e-6);

    let ln = weights
        .vectors
        .get("layers.0.input_layernorm.weight")
        .unwrap();
    assert!((ln[0] - 3.0).abs() < 1e-6);
}

#[test]
fn walk_only_excludes_ffn_tensors() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
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
                "layers.0.self_attn.q_proj.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[1.0f32; 8]),
            ),
            (
                "layers.0.mlp.gate_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[1.0f32; 16]),
            ),
            (
                "layers.0.mlp.up_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[1.0f32; 16]),
            ),
            (
                "layers.0.mlp.down_proj.weight",
                "F32",
                &[4, 4],
                f32_bytes(&[1.0f32; 16]),
            ),
        ],
    );

    let weights = load_model_dir_walk_only(dir.path()).unwrap();
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(!weights.tensors.contains_key("layers.0.mlp.up_proj.weight"));
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.down_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn walk_only_excludes_starcoder2_ffn_tensors() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
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
                "layers.0.self_attn.q_proj.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[1.0f32; 8]),
            ),
            (
                "layers.0.mlp.c_fc.weight",
                "F32",
                &[16, 4],
                f32_bytes(&[1.0f32; 64]),
            ),
            (
                "layers.0.mlp.c_proj.weight",
                "F32",
                &[4, 16],
                f32_bytes(&[1.0f32; 64]),
            ),
            (
                "layers.0.mlp.c_fc.bias",
                "F32",
                &[16],
                f32_bytes(&[1.0f32; 16]),
            ),
            (
                "layers.0.mlp.c_proj.bias",
                "F32",
                &[4],
                f32_bytes(&[1.0f32; 4]),
            ),
        ],
    );

    let weights = load_model_dir_walk_only(dir.path()).unwrap();
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_fc.weight"));
    assert!(!weights.tensors.contains_key("layers.0.mlp.c_proj.weight"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_fc.bias"));
    assert!(!weights.vectors.contains_key("layers.0.mlp.c_proj.bias"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn walk_only_excludes_gpt_oss_packed_mxfp4_experts() {
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
            (
                "layers.0.mlp.experts.gate_up_proj_blocks",
                "U8",
                &[1, 2, 1, 16],
                vec![0x22; 32],
            ),
            (
                "layers.0.mlp.experts.gate_up_proj_scales",
                "U8",
                &[1, 2, 1],
                vec![127; 2],
            ),
            (
                "layers.0.mlp.experts.down_proj_blocks",
                "U8",
                &[1, 1, 1, 16],
                vec![0x22; 16],
            ),
            (
                "layers.0.mlp.experts.down_proj_scales",
                "U8",
                &[1, 1, 1],
                vec![127; 1],
            ),
        ],
    );

    let weights = load_model_dir_walk_only(dir.path()).unwrap();
    assert!(!weights
        .tensors
        .keys()
        .any(|key| key.contains("block_sparse_moe.experts")));
    assert!(weights.tensors.contains_key("layers.0.mlp.router.weight"));
}

#[test]
fn packed_bf16_experts_are_mmap_backed_not_copied() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "vocab_size": 10,
            "enable_moe_block": true,
            "num_experts": 1,
            "top_k_experts": 1,
            "moe_intermediate_size": 1
        }
    });
    let gate_up_bytes: Vec<u8> = (0u8..16).collect();
    let down_bytes: Vec<u8> = (16u8..24).collect();
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
                "layers.0.experts.gate_up_proj",
                "BF16",
                &[1, 2, 4],
                gate_up_bytes.clone(),
            ),
            (
                "layers.0.experts.down_proj",
                "BF16",
                &[1, 4, 1],
                down_bytes.clone(),
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();

    assert!(
        weights.raw_bytes.is_empty(),
        "large packed BF16 tensors should stay in mmap ranges, not heap raw_bytes"
    );
    assert_eq!(weights.packed_mmaps.len(), 1);
    assert_eq!(
        weights
            .get_packed_bytes("layers.0.experts.gate_up_proj")
            .unwrap(),
        gate_up_bytes.as_slice()
    );
    assert_eq!(
        weights
            .get_packed_bytes("layers.0.experts.down_proj")
            .unwrap(),
        down_bytes.as_slice()
    );
}

#[test]
fn filtered_custom_predicate_skips_target() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
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
                "layers.0.self_attn.q_proj.weight",
                "F32",
                &[2, 4],
                f32_bytes(&[1.0f32; 8]),
            ),
        ],
    );

    let weights = load_model_dir_filtered(dir.path(), |k| k.contains("q_proj")).unwrap();
    assert!(!weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));
    // embed and lm_head are not filtered
    assert_eq!(weights.embed.shape(), &[10, 4]);
}

#[test]
fn unsupported_dtype_goes_to_skipped_tensors() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[1.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            // attention_mask is typically I64 — should be skipped, not crash
            ("attention_mask", "I64", &[1, 10], i64_bytes(10)),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert!(
        !weights.skipped_tensors.is_empty(),
        "I64 tensor should be in skipped_tensors"
    );
    let (key, dtype) = &weights.skipped_tensors[0];
    assert_eq!(key, "attention_mask");
    assert!(
        dtype.contains("I64"),
        "dtype string should mention I64, got: {dtype}"
    );
}

#[test]
fn missing_embed_returns_missing_tensor_error() {
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            // no embed_tokens.weight
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
        ],
    );

    match load_model_dir(dir.path()) {
        Err(ModelError::MissingTensor(k)) => assert_eq!(k, "embed_tokens.weight"),
        Err(e) => panic!("expected MissingTensor, got error: {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn tied_lm_head_falls_back_to_embed() {
    // No lm_head.weight → falls back to embed clone.
    let dir = TempDir::new().unwrap();
    write_model_dir(
        dir.path(),
        &[
            (
                "embed_tokens.weight",
                "F32",
                &[10, 4],
                f32_bytes(&[2.0f32; 40]),
            ),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
        ],
    );

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.lm_head.shape(), &[10, 4]);
    assert!((weights.lm_head[[0, 0]] - 2.0).abs() < 1e-6);
}

#[test]
fn mlx_weights_subdir_is_found() {
    // MLX layout: safetensors lives in a weights/ subdirectory.
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "llama", "hidden_size": 4, "num_hidden_layers": 1,
        "intermediate_size": 16, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 2, "vocab_size": 10,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let weights_dir = dir.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();
    let tensors = minimal_tensors();
    std::fs::write(
        weights_dir.join("model.safetensors"),
        make_safetensors(&tensors),
    )
    .unwrap();

    let weights = load_model_dir(dir.path()).unwrap();
    assert_eq!(weights.embed.shape(), &[10, 4]);
}

#[test]
fn no_safetensors_files_returns_error() {
    let dir = TempDir::new().unwrap();
    // Complete config.json: the loader must reach the safetensors-scan
    // step (and report no files) rather than erroring earlier on a
    // partial config.
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    // No .safetensors files → NoSafetensors error
    match load_model_dir(dir.path()) {
        Err(ModelError::NoSafetensors(_)) => {}
        Err(e) => panic!("expected NoSafetensors, got error: {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn non_directory_non_gguf_file_returns_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("not_a_model.txt");
    std::fs::write(&path, b"hello").unwrap();
    match load_model_dir(&path) {
        Err(ModelError::NotADirectory(_)) => {}
        Err(e) => panic!("expected NotADirectory, got error: {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}
