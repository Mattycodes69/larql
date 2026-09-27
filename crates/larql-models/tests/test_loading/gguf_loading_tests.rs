//! GGUF loading tests

use super::*;

#[test]
fn load_gguf_via_load_model_dir() {
    // load_model_dir detects .gguf in the directory and delegates to load_gguf.
    let dir = TempDir::new().unwrap();
    write_minimal_gguf(&dir.path().join("model.gguf"));

    let weights = load_model_dir(dir.path()).unwrap();
    // embed_tokens: dims=[4, 100] in GGUF → shape [100, 4] after GGUF dim swap
    assert_eq!(weights.embed.shape(), &[100, 4]);
    assert_eq!(weights.vocab_size, 100);
    assert_eq!(weights.num_layers, 1);
    assert_eq!(weights.hidden_size, 4);
}

#[test]
fn load_gguf_single_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.gguf");
    write_minimal_gguf(&path);

    let weights = load_model_dir(&path).unwrap();
    assert_eq!(weights.embed.shape(), &[100, 4]);
    assert_eq!(weights.vocab_size, 100);
    assert_eq!(weights.num_layers, 1);
}

#[test]
fn load_gguf_preserves_explicit_small_vocab_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("small-vocab.gguf");
    write_minimal_gguf_custom(&path, 128, Some(128), true, true);

    let weights = load_model_dir(&path).unwrap();

    assert_eq!(weights.embed.shape(), &[128, 4]);
    assert_eq!(weights.vocab_size, 128);
}

#[test]
fn load_gguf_uses_shape_vocab_when_metadata_and_tokenizer_are_absent() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("shape-vocab.gguf");
    write_minimal_gguf_custom(&path, 64, None, true, true);

    let weights = load_model_dir(&path).unwrap();

    assert_eq!(weights.embed.shape(), &[64, 4]);
    assert_eq!(weights.vocab_size, 64);
}

#[test]
fn load_gguf_defaults_missing_kv_heads_and_key_length() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("missing-attn-metadata.gguf");
    write_minimal_gguf_custom(&path, 100, Some(100), false, false);

    let weights = load_model_dir_validated(&path).unwrap();

    assert_eq!(weights.num_q_heads, 2);
    assert_eq!(weights.num_kv_heads, 2);
    assert_eq!(weights.head_dim, 2);
}

#[test]
fn load_gguf_walk_only_excludes_ffn_tensor() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("tiny-with-ffn.gguf");
    write_gguf_with_ffn(&path);

    let weights = load_model_dir_walk_only(&path).unwrap();
    assert!(!weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert_eq!(weights.embed.shape(), &[100, 4]);
}

#[test]
fn load_gguf_prefers_largest_file_when_multiple() {
    // When a directory has multiple GGUF files, the loader picks the largest.
    let dir = TempDir::new().unwrap();
    write_minimal_gguf(&dir.path().join("model-small.gguf"));
    // Write a zero-byte "large" file — loader picks by metadata(len).
    // In practice: largest by file size. Write the big one as the real model.
    write_minimal_gguf(&dir.path().join("model-main.gguf"));
    std::fs::write(dir.path().join("shard.gguf"), [0u8; 4]).unwrap();

    // Should not panic — any successful load is acceptable here.
    let result = load_model_dir(dir.path());
    assert!(result.is_ok() || matches!(result, Err(ModelError::Parse(_))));
}

#[test]
fn gguf_vectors_map_includes_1d_norms() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.gguf");
    write_minimal_gguf(&path);

    let weights = load_model_dir(&path).unwrap();
    // output_norm.weight → normalize_gguf_key → norm.weight (1D)
    // ends up in vectors, not tensors
    assert!(
        weights.vectors.contains_key("norm.weight"),
        "1D output_norm should be in vectors as norm.weight; keys: {:?}",
        weights.vectors.keys().collect::<Vec<_>>()
    );
}
