//! DeepSeek-V4 per-expert MXFP4 (dequantize_per_expert_mxfp4)

use super::*;

#[test]
fn load_full_deepseek_v4_dequantises_per_expert_mxfp4() {
    let dir = TempDir::new().unwrap();
    // V4 detection in detect.rs requires `model_type = "deepseek_v4"`,
    // and uses_mla() requires kv/q_lora_rank present.
    let config = serde_json::json!({
        "model_type": "deepseek_v4",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "n_shared_experts": 0,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
    });
    // V4 strips no `model.` prefix and uses `embed.weight` + `norm.weight`
    // (see `architectures/deepseek_v4.rs`).
    //
    // Per-expert MXFP4 layout: weight [out_features, packed_cols=groups*16],
    // scale [out_features, groups]. We use out_features=2, groups=1 →
    // packed_cols=16 → unpacked_cols=32.
    let out_features = 2usize;
    let groups = 1usize;
    let packed_cols = groups * 16;
    let weight_bytes = vec![0u8; out_features * packed_cols]; // all-zero nibbles → all-zero unpacked
    let scale_bytes = f8_e8m0_bytes(out_features * groups);

    write_model_dir_with_config(
        dir.path(),
        config,
        &[
            ("embed.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
            ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
            // V4 doesn't necessarily have lm_head; loader falls back to embed.
            // Per-expert MXFP4 weight + scale pair for w1 (gate_proj).
            (
                "layers.0.ffn.experts.0.w1.weight",
                "I8",
                &[out_features, packed_cols],
                weight_bytes.clone(),
            ),
            (
                "layers.0.ffn.experts.0.w1.scale",
                "F8_E8M0",
                &[out_features, groups],
                scale_bytes.clone(),
            ),
            // Plus w2 (down) and w3 (up) — same shape.
            (
                "layers.0.ffn.experts.0.w2.weight",
                "I8",
                &[out_features, packed_cols],
                weight_bytes.clone(),
            ),
            (
                "layers.0.ffn.experts.0.w2.scale",
                "F8_E8M0",
                &[out_features, groups],
                scale_bytes.clone(),
            ),
            (
                "layers.0.ffn.experts.0.w3.weight",
                "I8",
                &[out_features, packed_cols],
                weight_bytes,
            ),
            (
                "layers.0.ffn.experts.0.w3.scale",
                "F8_E8M0",
                &[out_features, groups],
                scale_bytes,
            ),
        ],
    );

    let weights = load_model_dir(dir.path()).expect("full V4 load");
    // The V4 dequantiser writes the dequantised weight under the
    // (prefix-stripped) tensor name. With V4's empty prefix list, that's
    // exactly `layers.0.ffn.experts.0.w1.weight`.
    assert!(
        weights
            .tensors
            .contains_key("layers.0.ffn.experts.0.w1.weight"),
        "V4 dequantiser must emit the unpacked weight; got: {:?}",
        weights.tensors.keys().collect::<Vec<_>>()
    );
    let arr = weights
        .tensors
        .get("layers.0.ffn.experts.0.w1.weight")
        .unwrap();
    // Output cols = packed_cols * 2 = 32 (the dequantiser unpacks
    // nibbles).
    assert_eq!(arr.shape(), &[out_features, packed_cols * 2]);
}

#[test]
fn load_filtered_validated_runs_with_validation() {
    // `load_model_dir_filtered_validated` is the public validated
    // entrypoint that callers like the streaming extractor use. Pin that
    // it accepts a passing predicate + invokes the validation path.
    let dir = TempDir::new().unwrap();
    write_model_dir(dir.path(), &minimal_tensors());
    let weights = larql_models::load_model_dir_filtered_validated(dir.path(), |_| false)
        .expect("validated filter accepts minimal model");
    assert!(weights.tensors.contains_key("embed_tokens.weight"));
}

#[test]
fn load_model_dir_errors_when_path_points_at_a_non_gguf_file() {
    // `path.is_file()` true, extension != gguf → NotADirectory.
    let dir = TempDir::new().unwrap();
    let bogus = dir.path().join("random.txt");
    std::fs::write(&bogus, "not a model").unwrap();
    match load_model_dir(&bogus) {
        Err(ModelError::NotADirectory(p)) => assert_eq!(p, bogus),
        Err(e) => panic!("expected NotADirectory, got error: {e}"),
        Ok(_) => panic!("expected NotADirectory, got Ok"),
    }
}

#[test]
fn load_model_dir_errors_when_path_does_not_exist() {
    // Neither file nor directory — falls into the second NotADirectory
    // branch (line 121).
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("does-not-exist");
    match load_model_dir(&missing) {
        Err(ModelError::NotADirectory(p)) => assert_eq!(p, missing),
        Err(e) => panic!("expected NotADirectory, got error: {e}"),
        Ok(_) => panic!("expected NotADirectory, got Ok"),
    }
}

#[test]
fn load_model_dir_errors_when_safetensors_missing_in_directory() {
    // Empty directory (no safetensors, no GGUF) — exercises the
    // "no safetensors files in {path}" branch.
    let dir = TempDir::new().unwrap();
    // Must still have config.json so detect_architecture passes.
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    match load_model_dir(dir.path()) {
        Err(ModelError::NoSafetensors(p)) => assert_eq!(p, dir.path()),
        Err(e) => panic!("expected NoSafetensors, got error: {e}"),
        Ok(_) => panic!("expected NoSafetensors, got Ok"),
    }
}

#[test]
fn load_model_dir_filtered_skip_predicate_omits_filtered_tensors() {
    // `load_model_dir_filtered` accepts a closure that returns true for
    // tensors to skip. Covers the filter branch + the resulting
    // skipped-tensor accounting.
    let dir = TempDir::new().unwrap();
    write_model_dir(dir.path(), &minimal_tensors());
    let weights = load_model_dir_filtered(dir.path(), |name| name == "norm.weight")
        .expect("filtered load should succeed");
    assert!(weights.tensors.contains_key("embed_tokens.weight"));
    assert!(!weights.tensors.contains_key("norm.weight"));
}

/// A config declaring `tie_word_embeddings: false` promises a real lm_head.
///
/// Silently falling back to the embedding matrix there is not a smaller
/// error: an untied model served with tied weights produces confident,
/// fluent, wrong text. The loader must refuse rather than substitute.
#[test]
fn untied_config_without_an_lm_head_tensor_is_refused_not_silently_tied() {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
        "tie_word_embeddings": false,
    });
    // Deliberately omit lm_head.weight.
    let embed = f32_bytes(&[1.0f32; 40]);
    let norm = f32_bytes(&[1.0f32; 4]);
    let entries: Vec<(&str, &str, &[usize], Vec<u8>)> = vec![
        ("embed_tokens.weight", "F32", &[10, 4], embed),
        ("norm.weight", "F32", &[4], norm),
    ];
    write_model_dir_with_config(dir.path(), config, &entries);

    match load_model_dir(dir.path()) {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("lm_head") && msg.contains("tie_word_embeddings"),
                "error must name both the tensor and the config field that \
                 makes it mandatory, got: {msg}"
            );
        }
        Ok(_) => panic!("untied config with no lm_head must not load"),
    }
}

/// The same model *with* the tensor loads — proves the refusal above is
/// about the missing lm_head, not about `tie_word_embeddings: false` itself.
#[test]
fn untied_config_with_an_lm_head_tensor_loads() {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
        "tie_word_embeddings": false,
    });
    write_model_dir_with_config(dir.path(), config, &minimal_tensors());
    load_model_dir(dir.path()).expect("untied config with lm_head must load");
}

/// 0-D scalar tensors are stored as one-element vectors rather than dropped.
///
/// The `match shape.len()` arm for `0` existed but nothing exercised it, so
/// a regression to `_ => {}` would have silently discarded the scalar.
#[test]
fn zero_dimensional_scalar_tensors_are_kept_as_one_element_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let mut entries = minimal_tensors();
    entries.push(("layer_scalar", "F32", &[], f32_bytes(&[2.5])));
    write_model_dir(dir.path(), &entries);

    let weights = load_model_dir(dir.path()).expect("scalar tensor must not break the load");
    let v = weights
        .vectors
        .get("layer_scalar")
        .expect("0-D tensor should land in `vectors`");
    assert_eq!(v.len(), 1, "a scalar is one element");
    assert!((v[0] - 2.5).abs() < 1e-6, "value must survive: {v:?}");
}

/// A tensor of a dtype the converter cannot handle is recorded and skipped,
/// not fatal — one exotic auxiliary tensor must not make a model unloadable.
#[test]
fn an_unsupported_dtype_is_skipped_rather_than_failing_the_whole_load() {
    let dir = tempfile::tempdir().unwrap();
    let mut entries = minimal_tensors();
    // I64 is a real safetensors dtype with no f32 conversion here.
    entries.push(("aux.counter", "I64", &[2], i64_bytes(2)));
    write_model_dir(dir.path(), &entries);

    let weights = load_model_dir(dir.path()).expect("unsupported dtype must not be fatal");
    assert!(
        weights.tensors.contains_key("embed_tokens.weight"),
        "supported tensors must still load"
    );
    assert!(
        !weights.tensors.contains_key("aux.counter")
            && !weights.vectors.contains_key("aux.counter"),
        "the unsupported tensor must be skipped, not half-converted"
    );
}

/// The MXFP4 branch has its own copy of the normal-tensor loop, so the
/// filter, dtype and rank handling there need their own coverage — a
/// regression in one is invisible from the dense path's tests.
#[test]
fn mxfp4_branch_honours_skip_key_for_ordinary_tensors() {
    let dir = TempDir::new().unwrap();
    gpt_oss_dir(
        dir.path(),
        &[(
            "layers.0.mlp.router.weight",
            "F32",
            &[1, 4],
            f32_bytes(&[1.0f32; 4]),
        )],
    );
    let weights = load_model_dir_filtered(dir.path(), |k| k.contains("router"))
        .expect("filtered MXFP4 load should succeed");
    assert!(
        !weights.tensors.keys().any(|k| k.contains("router")),
        "skip_key must apply inside the MXFP4 branch too"
    );
    assert!(weights.tensors.contains_key("embed_tokens.weight"));
}

#[test]
fn mxfp4_branch_keeps_one_dimensional_tensors_as_vectors() {
    let dir = TempDir::new().unwrap();
    gpt_oss_dir(
        dir.path(),
        &[(
            "layers.0.input_layernorm.weight",
            "F32",
            &[4],
            f32_bytes(&[3.0f32; 4]),
        )],
    );
    let weights = load_model_dir(dir.path()).expect("MXFP4 load should succeed");
    let v = weights
        .vectors
        .get("layers.0.input_layernorm.weight")
        .expect("1-D tensor belongs in `vectors`");
    assert_eq!(v.len(), 4);
}

#[test]
fn mxfp4_branch_skips_an_unsupported_dtype_without_failing() {
    let dir = TempDir::new().unwrap();
    gpt_oss_dir(dir.path(), &[("aux.counter", "I64", &[2], i64_bytes(2))]);
    let weights = load_model_dir(dir.path()).expect("unsupported dtype must not be fatal");
    assert!(!weights.tensors.contains_key("aux.counter"));
    assert!(!weights.vectors.contains_key("aux.counter"));
}

/// PackedBF16 experts (Gemma 4 26B A4B) are retained as raw mmap byte
/// ranges rather than converted to f32.
///
/// The point of the branch is memory: these are 3-D `[experts, out, in]`
/// BF16 tensors and converting them would double a multi-GB footprint, so
/// the compute path dequantises per expert on demand. A regression that
/// dropped them into the ordinary f32 path would still *load* — it would
/// just quietly cost twice the RAM, which no other assertion notices.
#[test]
fn packed_bf16_experts_are_kept_raw_not_converted_to_f32() {
    let dir = TempDir::new().unwrap();
    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "vocab_size": 10,
            "enable_moe_block": true,
            "num_experts": 2,
            "top_k_experts": 1,
            "moe_intermediate_size": 4,
        }
    });
    // [experts=2, out=8, in=4] BF16 gate_up, and [2, 4, 8] down.
    let gate_up = bf16_ones(2 * 8 * 4);
    let down = bf16_ones(2 * 4 * 8);
    let entries: Vec<(&str, &str, &[usize], Vec<u8>)> = vec![
        (
            "embed_tokens.weight",
            "F32",
            &[10, 4],
            f32_bytes(&[1.0f32; 40]),
        ),
        ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
        (
            "layers.0.mlp.experts.gate_up_proj",
            "BF16",
            &[2, 8, 4],
            gate_up,
        ),
        ("layers.0.mlp.experts.down_proj", "BF16", &[2, 4, 8], down),
    ];
    write_model_dir_with_config(dir.path(), config, &entries);

    let weights = load_model_dir(dir.path()).expect("PackedBF16 model must load");

    for key in [
        "layers.0.mlp.experts.gate_up_proj",
        "layers.0.mlp.experts.down_proj",
    ] {
        assert!(
            !weights.tensors.contains_key(key),
            "{key} must NOT be materialised as an f32 Array2 — that is the \
             memory regression this branch exists to avoid"
        );
        assert!(
            !weights.vectors.contains_key(key),
            "{key} must not land in `vectors` either"
        );
    }
    // The ordinary tensors alongside them still load normally.
    assert!(weights.tensors.contains_key("embed_tokens.weight"));
}
