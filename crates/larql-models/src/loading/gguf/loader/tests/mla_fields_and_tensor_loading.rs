use super::*;

#[test]
fn test_gguf_mla_fields_absent_for_non_mla_architectures() {
    // Llama / Qwen / Mistral GGUFs do not emit MLA keys. The config
    // builder must leave the optional MLA fields out so `uses_mla()`
    // stays false and the streaming path keeps its existing behaviour.
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

    assert!(cfg.get("q_lora_rank").is_none());
    assert!(cfg.get("kv_lora_rank").is_none());
    assert!(cfg.get("qk_nope_head_dim").is_none());
    assert!(cfg.get("v_head_dim").is_none());
    assert!(cfg.get("qk_rope_head_dim").is_none());
}

#[test]
fn load_tensors_filtered_skips_key() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("skip.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
    file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
                                                  // Tensor 0: kept
    let n0 = b"blk.0.attn_q.weight";
    file.write_all(&(n0.len() as u64).to_le_bytes()).unwrap();
    file.write_all(n0).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    // Tensor 1: skipped by key
    let n1 = b"blk.0.ffn_gate.weight";
    file.write_all(&(n1.len() as u64).to_le_bytes()).unwrap();
    file.write_all(n1).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&16u64.to_le_bytes()).unwrap();
    // Data
    let pos = file.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    file.write_all(&vec![0u8; (aligned - pos) as usize])
        .unwrap();
    for i in 0..8 {
        file.write_all(&(i as f32).to_le_bytes()).unwrap();
    }
    file.flush().unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    let skip: &dyn Fn(&str) -> bool = &|k| k.contains("gate_proj");
    let (tensors, _) = gguf.load_tensors_filtered(skip).unwrap();
    assert_eq!(tensors.len(), 1);
    assert!(tensors.contains_key("layers.0.self_attn.q_proj.weight"));
}

#[test]
fn load_tensors_handles_1d_and_higher_dim() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dims.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
    file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
                                                  // Tensor 0: 1D norm vector (4 elements)
    let n0 = b"blk.0.attn_norm.weight";
    file.write_all(&(n0.len() as u64).to_le_bytes()).unwrap();
    file.write_all(n0).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap(); // 1D
    file.write_all(&4u64.to_le_bytes()).unwrap();
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    // Tensor 1: 3D tensor (should be skipped)
    let n1 = b"blk.0.expert.weight";
    file.write_all(&(n1.len() as u64).to_le_bytes()).unwrap();
    file.write_all(n1).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap(); // 3D
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&16u64.to_le_bytes()).unwrap();
    // Data
    let pos = file.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    file.write_all(&vec![0u8; (aligned - pos) as usize])
        .unwrap();
    for i in 0..12 {
        file.write_all(&(i as f32).to_le_bytes()).unwrap();
    }
    file.flush().unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    let (tensors, vectors) = gguf.load_tensors().unwrap();
    // 1D → vectors map
    assert_eq!(vectors.len(), 1);
    assert!(vectors.contains_key("layers.0.input_layernorm.weight"));
    assert_eq!(vectors["layers.0.input_layernorm.weight"].len(), 4);
    // 3D → skipped (not in tensors or vectors)
    assert!(tensors.is_empty());
}

#[test]
fn to_config_json_head_dim_falls_back_to_hidden_div_heads() {
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
    // No attention.key_length → head_dim = hidden / heads = 128

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
    assert_eq!(cfg["head_dim"], 128);
}

#[test]
fn to_config_json_kv_heads_defaults_to_heads_when_absent() {
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
    // No head_count_kv → defaults to head_count

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
    assert_eq!(cfg["num_key_value_heads"], 32);
}

#[test]
fn to_config_json_maps_deepseek_v4_arch() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("deepseek_v4".to_string()),
    );
    metadata.insert(
        "deepseek_v4.embedding_length".to_string(),
        GgufValue::U32(4096),
    );
    metadata.insert("deepseek_v4.block_count".to_string(), GgufValue::U32(32));
    metadata.insert(
        "deepseek_v4.feed_forward_length".to_string(),
        GgufValue::U32(11008),
    );
    metadata.insert(
        "deepseek_v4.attention.head_count".to_string(),
        GgufValue::U32(32),
    );
    metadata.insert(
        "deepseek_v4.attention.head_count_kv".to_string(),
        GgufValue::U32(8),
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
    assert_eq!(cfg["model_type"], "deepseek_v4");
}

#[test]
fn to_config_json_maps_unknown_arch_passthrough() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("futurearch".to_string()),
    );
    metadata.insert(
        "futurearch.embedding_length".to_string(),
        GgufValue::U32(512),
    );
    metadata.insert("futurearch.block_count".to_string(), GgufValue::U32(6));
    metadata.insert(
        "futurearch.feed_forward_length".to_string(),
        GgufValue::U32(2048),
    );
    metadata.insert(
        "futurearch.attention.head_count".to_string(),
        GgufValue::U32(8),
    );
    metadata.insert(
        "futurearch.attention.head_count_kv".to_string(),
        GgufValue::U32(8),
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
    assert_eq!(cfg["model_type"], "futurearch");
}

#[test]
fn test_gguf_to_config_json_falls_back_to_expert_feed_forward_length_on_moe() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("deepseek2".to_string()),
    );
    metadata.insert(
        "deepseek2.embedding_length".to_string(),
        GgufValue::U32(4096),
    );
    metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(43));
    metadata.insert(
        "deepseek2.attention.head_count".to_string(),
        GgufValue::U32(64),
    );
    metadata.insert(
        "deepseek2.attention.head_count_kv".to_string(),
        GgufValue::U32(1),
    );
    metadata.insert(
        "deepseek2.attention.key_length".to_string(),
        GgufValue::U32(128),
    );
    metadata.insert(
        "deepseek2.expert_feed_forward_length".to_string(),
        GgufValue::U32(2048),
    );
    metadata.insert("deepseek2.vocab_size".to_string(), GgufValue::U32(129280));

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
    assert_eq!(cfg["intermediate_size"], 2048);
    crate::detect_from_json_validated(&cfg)
        .expect("MoE-only GGUF config should pass validation after fallback");
}

#[test]
fn test_gguf_to_config_json_prefers_global_feed_forward_length_when_both_present() {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("deepseek2".to_string()),
    );
    metadata.insert(
        "deepseek2.embedding_length".to_string(),
        GgufValue::U32(2048),
    );
    metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(27));
    metadata.insert(
        "deepseek2.attention.head_count".to_string(),
        GgufValue::U32(16),
    );
    metadata.insert(
        "deepseek2.attention.head_count_kv".to_string(),
        GgufValue::U32(16),
    );
    metadata.insert(
        "deepseek2.attention.key_length".to_string(),
        GgufValue::U32(192),
    );
    metadata.insert(
        "deepseek2.feed_forward_length".to_string(),
        GgufValue::U32(10944),
    );
    metadata.insert(
        "deepseek2.expert_feed_forward_length".to_string(),
        GgufValue::U32(1408),
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
    assert_eq!(cfg["intermediate_size"], 10944);
}

/// Build a minimal GGUF file with one 2-D F32 tensor, but truncate the
/// tensor data region so that `offset + size > file len`. Loader must
/// reject this cleanly, not panic on a slice OOB.
#[test]
fn test_load_tensors_rejects_truncated_tensor_data() {
    use std::io::{Seek, Write};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.gguf");
    let mut file = std::fs::File::create(&path).unwrap();

    // Header
    file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap(); // version
    file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
    file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

    // Tensor info: declares 2x4 F32 (32 bytes of data) at tensor offset 0.
    let name = b"blk.0.ffn_down.weight";
    file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    file.write_all(name).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&4u64.to_le_bytes()).unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();

    // Pad to 32-byte boundary, then write only 16 bytes of tensor data
    // (half of the declared 32). Loader must detect the shortfall.
    let pos = file.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    file.write_all(&vec![0u8; (aligned - pos) as usize])
        .unwrap();
    file.write_all(&[0u8; 16]).unwrap();
    file.flush().unwrap();

    let gguf = GgufFile::open(&path).unwrap();
    match gguf.load_tensors() {
        Err(ModelError::Parse(msg)) => {
            assert!(
                msg.contains("out of bounds") || msg.contains("too short"),
                "unexpected error: {msg}"
            );
        }
        Err(other) => panic!("expected Parse error, got {other:?}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

#[test]
fn read_tokenizer_vocab_size_reads_vocab_object_length() {
    let dir = tempfile::TempDir::new().unwrap();
    let gguf = dir.path().join("model.gguf");
    let tokenizer_json = serde_json::json!({
        TOKENIZER_MODEL: {
            TOKENIZER_VOCAB: {
                "<unk>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "a": 3,
                "b": 4,
            }
        }
    });
    std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
    assert_eq!(read_tokenizer_vocab_size(&gguf), Some(5));
}

#[test]
fn read_tokenizer_vocab_size_returns_none_when_tokenizer_json_absent() {
    let dir = tempfile::TempDir::new().unwrap();
    // model.gguf path with no tokenizer.json next to it.
    assert_eq!(
        read_tokenizer_vocab_size(&dir.path().join("model.gguf")),
        None
    );
}

#[test]
fn read_tokenizer_vocab_size_returns_none_when_vocab_empty() {
    let dir = tempfile::TempDir::new().unwrap();
    let gguf = dir.path().join("model.gguf");
    // Empty vocab object — filtered out by `.filter(|&v| v > 0)`.
    let tokenizer_json = serde_json::json!({
        TOKENIZER_MODEL: {
            TOKENIZER_VOCAB: {}
        }
    });
    std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
    assert_eq!(read_tokenizer_vocab_size(&gguf), None);
}

#[test]
fn read_tokenizer_vocab_size_returns_none_on_malformed_json() {
    let dir = tempfile::TempDir::new().unwrap();
    let gguf = dir.path().join("model.gguf");
    std::fs::write(dir.path().join(TOKENIZER_JSON), b"not-json").unwrap();
    assert_eq!(read_tokenizer_vocab_size(&gguf), None);
}

#[test]
fn read_tokenizer_vocab_size_returns_none_when_path_has_no_parent() {
    assert_eq!(read_tokenizer_vocab_size(std::path::Path::new("")), None);
}
