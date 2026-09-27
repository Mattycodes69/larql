use super::*;

#[test]
fn test_normalize_gguf_key() {
    assert_eq!(
        normalize_gguf_key("blk.0.attn_q.weight"),
        "layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        normalize_gguf_key("blk.15.ffn_gate.weight"),
        "layers.15.mlp.gate_proj.weight"
    );
    assert_eq!(
        normalize_gguf_key("token_embd.weight"),
        "embed_tokens.weight"
    );
    assert_eq!(normalize_gguf_key("output.weight"), "lm_head.weight");
}

#[test]
fn test_normalize_gguf_key_gemma_layout() {
    assert_eq!(
        normalize_gguf_key_for_arch("blk.0.attn_q_norm.weight", "gemma4"),
        "layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        normalize_gguf_key_for_arch("blk.0.attn_k_norm.weight", "gemma4_unified"),
        "layers.0.self_attn.k_norm.weight"
    );
    assert_eq!(
        normalize_gguf_key_for_arch("blk.0.post_attention_norm.weight", "gemma2"),
        "layers.0.post_attention_layernorm.weight"
    );
    // gemma's ffn_norm is the PRE-feedforward norm...
    assert_eq!(
        normalize_gguf_key_for_arch("blk.3.ffn_norm.weight", "gemma3"),
        "layers.3.pre_feedforward_layernorm.weight"
    );
    // ...while llama's ffn_norm keeps the generic post-attention mapping.
    assert_eq!(
        normalize_gguf_key_for_arch("blk.3.ffn_norm.weight", "llama"),
        "layers.3.post_attention_layernorm.weight"
    );
    // Gemma 1 is llama-layout too.
    assert_eq!(
        normalize_gguf_key_for_arch("blk.3.ffn_norm.weight", "gemma"),
        "layers.3.post_attention_layernorm.weight"
    );
    assert_eq!(
        normalize_gguf_key_for_arch("blk.47.post_ffw_norm.weight", "gemma4"),
        "layers.47.post_feedforward_layernorm.weight"
    );
    assert_eq!(
        normalize_gguf_key_for_arch("blk.5.layer_output_scale.weight", "gemma4"),
        "layers.5.layer_scalar"
    );
    // Mapped projections are untouched by the gemma pre-pass.
    assert_eq!(
        normalize_gguf_key_for_arch("blk.0.attn_q.weight", "gemma4"),
        "layers.0.self_attn.q_proj.weight"
    );
}

#[test]
fn test_load_tensors_swaps_gguf_2d_dims_to_rows_cols() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    let mut file = std::fs::File::create(&path).unwrap();

    // Header
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap(); // version
    file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
    file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

    // Tensor info: ggml dims order is [cols, rows].
    let name = b"blk.0.ffn_down.weight";
    file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    file.write_all(name).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
    file.write_all(&4u64.to_le_bytes()).unwrap(); // cols
    file.write_all(&2u64.to_le_bytes()).unwrap(); // rows
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor data offset

    // Pad tensor data start to 32-byte boundary.
    let pos = file.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    file.write_all(&vec![0u8; (aligned - pos) as usize])
        .unwrap();

    // Raw row-major data for a logical [2, 4] matrix.
    for v in 1u32..=8 {
        file.write_all(&(v as f32).to_le_bytes()).unwrap();
    }
    file.flush().unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    let (tensors, _) = gguf.load_tensors().unwrap();
    let down = tensors.get("layers.0.mlp.down_proj.weight").unwrap();

    assert_eq!(down.shape(), &[2, 4]);
    assert_eq!(down[[0, 0]], 1.0);
    assert_eq!(down[[0, 1]], 2.0);
    assert_eq!(down[[0, 2]], 3.0);
    assert_eq!(down[[0, 3]], 4.0);
    assert_eq!(down[[1, 0]], 5.0);
    assert_eq!(down[[1, 1]], 6.0);
    assert_eq!(down[[1, 2]], 7.0);
    assert_eq!(down[[1, 3]], 8.0);
}

#[test]
fn keep_quant_captures_i2s_bytes_and_trailing_scale() {
    let scale = 0.527_f32;
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(scale)));

    let weights =
        load_gguf_keep_quant(&path, &[crate::quant::ggml::TYPE_I2_S]).expect("keep-quant load");

    // The raw I2_S bytes survive into ModelWeights::raw_bytes under
    // the normalized tensor key, byte-for-byte.
    let key = crate::loading::safetensors::normalize_key(
        &normalize_gguf_key("blk.0.bitlinear.weight"),
        weights.arch.key_prefixes_to_strip(),
    );
    let raw = weights.raw_bytes.get(&key).unwrap_or_else(|| {
        panic!(
            "missing raw I2_S bytes for {key}; have {:?}",
            weights.raw_bytes.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(
        raw.as_slice(),
        &KQ_I2S_PACKED,
        "raw trits preserved verbatim"
    );

    // The trailing per-tensor scale is captured under the sentinel key.
    let scale_bytes = weights
        .raw_bytes
        .get(&format!("{key}{}", crate::I2S_SCALE_SUFFIX))
        .expect("missing captured I2_S scale");
    let got = f32::from_le_bytes(scale_bytes.as_slice().try_into().unwrap());
    assert!((got - scale).abs() < 1e-6, "scale {got} != {scale}");
}

#[test]
fn keep_quant_method_captures_scale_and_filters_by_type() {
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(1.5)));
    let gguf = GgufFile::open(&path).unwrap();
    let (_tensors, _vectors, raw) = gguf
        .load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_I2_S])
        .unwrap();

    // Only the I2_S tensor (and its scale) are retained — the F32
    // embed/output/norm tensors are not in `keep_raw_for_types`.
    let i2s_key = normalize_gguf_key("blk.0.bitlinear.weight");
    assert!(raw.contains_key(&i2s_key));
    assert!(raw.contains_key(&format!("{i2s_key}{}", crate::I2S_SCALE_SUFFIX)));
    assert!(
        !raw.contains_key("embed_tokens.weight"),
        "F32 tensors must not be retained when only I2_S is requested"
    );
}

#[test]
fn keep_quant_non_i2s_type_retained_without_scale() {
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(2.0)));
    let gguf = GgufFile::open(&path).unwrap();
    // Request F32 retention: bytes are kept, but no scale sentinel
    // is written (scale capture is I2_S-specific).
    let (_t, _v, raw) = gguf
        .load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_F32])
        .unwrap();
    assert!(
        raw.contains_key("embed_tokens.weight"),
        "F32 bytes retained"
    );
    assert!(
        raw.keys().all(|k| !k.ends_with(crate::I2S_SCALE_SUFFIX)),
        "no scale sentinel for non-I2_S types"
    );
}

#[test]
fn keep_quant_method_honours_skip_key() {
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(1.0)));
    let gguf = GgufFile::open(&path).unwrap();
    let (_t, _v, raw) = gguf
        .load_tensors_filtered_keep_quant(
            &|k| k.contains("bitlinear"),
            &[crate::quant::ggml::TYPE_I2_S],
        )
        .unwrap();
    // The only keep-eligible tensor was skipped, so nothing is retained.
    assert!(
        raw.is_empty(),
        "skipped tensor must not be retained: {raw:?}"
    );
}

#[test]
fn keep_quant_scale_absent_when_no_trailing_room() {
    // No trailing scale f32 → the [end, end+4) window runs past EOF
    // and the scale is simply not captured (the raw trits still are).
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(None));
    let gguf = GgufFile::open(&path).unwrap();
    let (_t, _v, raw) = gguf
        .load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_I2_S])
        .unwrap();
    let i2s_key = normalize_gguf_key("blk.0.bitlinear.weight");
    assert!(raw.contains_key(&i2s_key), "raw trits still captured");
    assert!(
        !raw.contains_key(&format!("{i2s_key}{}", crate::I2S_SCALE_SUFFIX)),
        "scale must be absent when there is no trailing f32"
    );
}

#[test]
fn load_gguf_keep_quant_with_empty_keep_types_matches_plain_load() {
    // keep_types = [] → no raw bytes retained; otherwise identical
    // to the plain load path.
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(1.0)));
    let weights = load_gguf_keep_quant(&path, &[]).unwrap();
    assert!(
        weights.raw_bytes.is_empty(),
        "no raw bytes when keep_types is empty"
    );
    assert_eq!(weights.embed.shape(), &[8, 4]);
}

#[test]
fn load_gguf_resolves_embed_lm_head_and_vocab() {
    // Drives the unified orchestrator over the GGUF path (load_gguf now
    // routes through the keep-quant orchestrator with no retained types):
    // embed orientation, the present-`output.weight` lm_head branch, and
    // vocab-size resolution from the embedding shape.
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(None));
    let weights = load_gguf(&path).unwrap();
    assert_eq!(weights.embed.shape(), &[8, 4]);
    assert_eq!(weights.lm_head.shape(), &[8, 4]);
    assert_eq!(weights.vocab_size, 8);
    assert!(weights.raw_bytes.is_empty());
}

#[test]
fn load_gguf_lm_head_falls_back_to_embed_when_output_absent() {
    // No `output.weight` → lm_head ties to the embedding.
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf_opts(None, true, false));
    let weights = load_gguf(&path).unwrap();
    assert_eq!(weights.lm_head.shape(), weights.embed.shape());
}

#[test]
fn load_gguf_missing_embed_is_missing_tensor_error() {
    // No `token_embd.weight` → the embed-key lookup fails.
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf_opts(None, false, true));
    match load_gguf(&path) {
        Ok(_) => panic!("expected MissingTensor error for embed-less GGUF"),
        Err(e) => assert!(
            matches!(e, ModelError::MissingTensor(_)),
            "expected MissingTensor, got {e:?}"
        ),
    }
}

#[test]
fn load_gguf_validated_routes_through_unified_pipeline() {
    // load_gguf_validated shares the same orchestrator (validate=true);
    // a well-formed llama GGUF passes architecture validation.
    let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(None));
    let weights = load_gguf_validated(&path).unwrap();
    assert_eq!(weights.embed.shape(), &[8, 4]);
}

#[test]
fn keep_quant_oob_tensor_offset_errors() {
    // Corrupt the last tensor-info offset so the declared data runs past
    // EOF — exercises the bounds-check error arm in the loader loop.
    let mut bytes = build_i2s_gguf(Some(1.0));
    // The data section is well under 1 MiB; an offset of 1 MiB is past EOF
    // for every tensor.  Rewrite the final tensor info's offset field.
    // Tensor-info offset is the last u64 of the I2_S tensor info, which is
    // the last tensor-info written; locate it by scanning for the packed
    // marker is fragile, so instead rebuild with a deliberately bad file:
    // truncate the data section to force the bounds check.
    bytes.truncate(bytes.len() - 40);
    let (_dir, path) = write_tmp_gguf(&bytes);
    let err = GgufFile::open(&path)
        .and_then(|g| {
            g.load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_I2_S])
                .map(|_| ())
        })
        .unwrap_err();
    assert!(
        matches!(err, ModelError::Parse(_)),
        "expected a Parse (out-of-bounds) error, got {err:?}"
    );
}

#[test]
fn test_gemma4_gguf_to_config_json_maps_arch_and_overrides_head_dim() {
    // Synthesize GGUF metadata matching gemma-4-e2b's shape.
    // Exercises: (a) gemma4 name pass-through, (b) head_dim=256 override,
    // (c) array metadata (per-layer variable FFN sizes → take max).
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("gemma4".to_string()),
    );
    metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(1536));
    metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(35));
    metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(8));
    metadata.insert(
        "gemma4.attention.head_count_kv".to_string(),
        GgufValue::U32(1),
    );
    // Gemma 4 reports attention.key_length=512 (global head_dim), not the
    // per-head 256 we want. Loader must override to 256 for arch="gemma4".
    metadata.insert(
        "gemma4.attention.key_length".to_string(),
        GgufValue::U32(512),
    );
    metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(262144));
    // Per-layer variable FFN — some layers 6144, some 12288. Must take max.
    metadata.insert(
        "gemma4.feed_forward_length".to_string(),
        GgufValue::Array(vec![
            GgufValue::U32(6144),
            GgufValue::U32(12288),
            GgufValue::U32(6144),
        ]),
    );

    let gguf = GgufFile {
        metadata,
        tensor_infos: Vec::new(),
        data_offset: 0,
        path: std::path::PathBuf::from("<no-file>"),
        shards: vec![ShardInfo {
            path: std::path::PathBuf::from("<no-file>"),
            data_offset: 0,
        }],
    };
    let cfg = gguf.to_config_json();

    assert_eq!(cfg["model_type"], "gemma4");
    assert_eq!(cfg["hidden_size"], 1536);
    assert_eq!(cfg["num_hidden_layers"], 35);
    // head_dim override: 256 despite attention.key_length=512
    assert_eq!(cfg["head_dim"], 256);
    // intermediate_size: max of the per-layer FFN array (12288), not 6144
    assert_eq!(cfg["intermediate_size"], 12288);
    assert_eq!(cfg["num_attention_heads"], 8);
    assert_eq!(cfg["num_key_value_heads"], 1);
    assert_eq!(cfg["vocab_size"], 262144);
}

#[test]
fn test_gguf_to_config_json_omits_absent_rope_base_for_arch_default() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("llama".to_string()),
    );
    metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
    metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
    metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::U32(11008),
    );
    metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
    metadata.insert(
        "llama.attention.head_count_kv".to_string(),
        GgufValue::U32(8),
    );
    metadata.insert(
        "llama.attention.key_length".to_string(),
        GgufValue::U32(128),
    );

    let gguf = GgufFile {
        metadata,
        tensor_infos: Vec::new(),
        data_offset: 0,
        path: std::path::PathBuf::from("<no-file>"),
        shards: vec![ShardInfo {
            path: std::path::PathBuf::from("<no-file>"),
            data_offset: 0,
        }],
    };
    let cfg = gguf.to_config_json();

    assert!(cfg.get(HF_ROPE_THETA).is_none());
    let arch = crate::detect_from_json_validated(&cfg).unwrap();
    assert_eq!(arch.config().rope_base, 10_000.0);
}

#[test]
fn test_kimi_k2_gguf_to_config_json_extracts_mla_fields() {
    // Synthesize GGUF metadata matching Kimi K2.6's unsloth Q8_K_XL shape.
    // Verifies the MLA fields surface into the HF-style config that the
    // parser → ModelConfig path consumes, so that PR #96's MLA absorption
    // fires for GGUF-sourced DeepSeek-V2/V3/Kimi-K2 models. Closes #67.
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("deepseek2".to_string()),
    );
    metadata.insert(
        "deepseek2.embedding_length".to_string(),
        GgufValue::U32(7168),
    );
    metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(61));
    metadata.insert(
        "deepseek2.attention.head_count".to_string(),
        GgufValue::U32(64),
    );
    metadata.insert(
        "deepseek2.attention.head_count_kv".to_string(),
        GgufValue::U32(1),
    );
    metadata.insert(
        "deepseek2.feed_forward_length".to_string(),
        GgufValue::U32(18432),
    );
    metadata.insert("deepseek2.vocab_size".to_string(), GgufValue::U32(163840));
    // MLA-specific keys emitted by llama.cpp for DeepSeek-V2/V3 family.
    // `_mla` carries the pre-absorption per-head split that PR #96 needs.
    metadata.insert(
        "deepseek2.attention.q_lora_rank".to_string(),
        GgufValue::U32(1536),
    );
    metadata.insert(
        "deepseek2.attention.kv_lora_rank".to_string(),
        GgufValue::U32(512),
    );
    metadata.insert(
        "deepseek2.attention.key_length".to_string(),
        GgufValue::U32(576),
    );
    metadata.insert(
        "deepseek2.attention.value_length".to_string(),
        GgufValue::U32(512),
    );
    metadata.insert(
        "deepseek2.attention.key_length_mla".to_string(),
        GgufValue::U32(192),
    );
    metadata.insert(
        "deepseek2.attention.value_length_mla".to_string(),
        GgufValue::U32(128),
    );
    metadata.insert(
        "deepseek2.rope.dimension_count".to_string(),
        GgufValue::U32(64),
    );

    let gguf = GgufFile {
        metadata,
        tensor_infos: Vec::new(),
        data_offset: 0,
        path: std::path::PathBuf::from("<no-file>"),
        shards: vec![ShardInfo {
            path: std::path::PathBuf::from("<no-file>"),
            data_offset: 0,
        }],
    };
    let cfg = gguf.to_config_json();

    // Model type maps deepseek2 → deepseek_v2 (existing logic).
    assert_eq!(cfg["model_type"], "deepseek_v2");
    // MLA fields populated from GGUF metadata.
    assert_eq!(cfg["q_lora_rank"], 1536);
    assert_eq!(cfg["kv_lora_rank"], 512);
    assert_eq!(cfg["qk_rope_head_dim"], 64);
    // qk_nope_head_dim = key_length_mla - rope.dimension_count = 192-64 = 128
    // (prefers _mla variant over the absorbed key_length=576).
    assert_eq!(cfg["qk_nope_head_dim"], 128);
    // v_head_dim prefers the _mla variant (128 pre-absorption, not 512).
    assert_eq!(cfg["v_head_dim"], 128);

    // Architecture-detection path picks the fields up into ModelConfig.
    let arch = crate::detect_from_json(&cfg);
    assert_eq!(arch.mla_qk_nope_head_dim(), Some(128));
    assert_eq!(arch.mla_qk_rope_head_dim(), Some(64));
    assert_eq!(arch.mla_v_head_dim(), Some(128));
    assert_eq!(arch.q_lora_rank(), 1536);
    assert_eq!(arch.kv_lora_rank(), 512);
    assert!(arch.uses_mla());
}

#[test]
fn test_gguf_mla_falls_back_to_non_mla_key_length_when_mla_keys_absent() {
    // Some DeepSeek-V2 GGUFs may not emit the `_mla` variants. The
    // loader must fall back to attention.key_length / value_length so
    // the pre-absorption split is still computed.
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("deepseek2".to_string()),
    );
    metadata.insert(
        "deepseek2.embedding_length".to_string(),
        GgufValue::U32(5120),
    );
    metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(27));
    metadata.insert(
        "deepseek2.attention.head_count".to_string(),
        GgufValue::U32(128),
    );
    metadata.insert(
        "deepseek2.attention.head_count_kv".to_string(),
        GgufValue::U32(128),
    );
    metadata.insert(
        "deepseek2.feed_forward_length".to_string(),
        GgufValue::U32(12288),
    );
    metadata.insert(
        "deepseek2.attention.q_lora_rank".to_string(),
        GgufValue::U32(1536),
    );
    metadata.insert(
        "deepseek2.attention.kv_lora_rank".to_string(),
        GgufValue::U32(512),
    );
    // Only non-`_mla` variants present.
    metadata.insert(
        "deepseek2.attention.key_length".to_string(),
        GgufValue::U32(192),
    );
    metadata.insert(
        "deepseek2.attention.value_length".to_string(),
        GgufValue::U32(128),
    );
    metadata.insert(
        "deepseek2.rope.dimension_count".to_string(),
        GgufValue::U32(64),
    );

    let gguf = GgufFile {
        metadata,
        tensor_infos: Vec::new(),
        data_offset: 0,
        path: std::path::PathBuf::from("<no-file>"),
        shards: vec![ShardInfo {
            path: std::path::PathBuf::from("<no-file>"),
            data_offset: 0,
        }],
    };
    let cfg = gguf.to_config_json();
    assert_eq!(cfg["qk_nope_head_dim"], 128); // 192 - 64
    assert_eq!(cfg["qk_rope_head_dim"], 64);
    assert_eq!(cfg["v_head_dim"], 128);
}
