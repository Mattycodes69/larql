//! Integration tests for model loading — safetensors and GGUF.
//!
//! Each test builds a minimal synthetic binary in a tempdir and exercises the
//! public loading API. No real model files required.

use std::io::{Seek, Write};
use std::path::Path;
use tempfile::TempDir;

use larql_models::{
    load_model_dir, load_model_dir_filtered, load_model_dir_validated, load_model_dir_walk_only,
    load_model_dir_walk_only_validated, validation::FIELD_HEAD_DIM, ModelError,
};

// Safetensors binary builder

/// Build a valid safetensors file in memory.
///
/// `entries`: (tensor_name, dtype_string, shape, raw_data_bytes)
///
/// The dtype string must match the safetensors spec: "F32", "F16", "BF16",
/// "I64", etc. `raw_data_bytes` must be exactly the right number of bytes for
/// the given shape × element size.
fn make_safetensors(entries: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
    let mut data_offset = 0usize;
    let mut meta = serde_json::Map::new();
    let mut tensor_data = Vec::<u8>::new();

    for &(name, dtype, shape, ref bytes) in entries {
        let end = data_offset + bytes.len();
        meta.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [data_offset, end],
            }),
        );
        tensor_data.extend_from_slice(bytes);
        data_offset = end;
    }
    meta.insert("__metadata__".into(), serde_json::json!({}));

    let header = serde_json::to_vec(&serde_json::Value::Object(meta)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&tensor_data);
    out
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Encode `n` elements as f16 1.0 (0x3C00).
fn f16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| [0x00u8, 0x3C]).collect()
}

/// Encode `n` elements as bf16 1.0 (0x3F80).
fn bf16_ones(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| [0x80u8, 0x3F]).collect()
}

/// Encode `n` elements as I64 42.
fn i64_bytes(n: usize) -> Vec<u8> {
    (0..n).flat_map(|_| 42i64.to_le_bytes()).collect()
}

/// Write config.json and a single `model.safetensors` into `dir`.
fn write_model_dir(dir: &Path, entries: &[(&str, &str, &[usize], Vec<u8>)]) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 10,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("model.safetensors"), make_safetensors(entries)).unwrap();
}

fn write_model_dir_with_config(
    dir: &Path,
    config: serde_json::Value,
    entries: &[(&str, &str, &[usize], Vec<u8>)],
) {
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("model.safetensors"), make_safetensors(entries)).unwrap();
}

/// Minimal embed + lm_head + norm for a successful Llama-like load (hidden=4, vocab=10).
fn minimal_tensors() -> Vec<(&'static str, &'static str, &'static [usize], Vec<u8>)> {
    let embed_data = f32_bytes(&[1.0f32; 40]); // [10, 4]
    let norm_data = f32_bytes(&[1.0f32; 4]); // [4]
    let head_data = f32_bytes(&[1.0f32; 40]); // [10, 4]
    vec![
        ("embed_tokens.weight", "F32", &[10, 4], embed_data),
        ("norm.weight", "F32", &[4], norm_data),
        ("lm_head.weight", "F32", &[10, 4], head_data),
    ]
}

// GGUF binary builder

const GGUF_MAGIC: u32 = 0x46554747;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_F32: u32 = 0; // tensor type F32

fn gguf_str(f: &mut impl Write, s: &str) {
    let b = s.as_bytes();
    f.write_all(&(b.len() as u64).to_le_bytes()).unwrap();
    f.write_all(b).unwrap();
}

fn gguf_meta_str(f: &mut impl Write, key: &str, val: &str) {
    gguf_str(f, key);
    f.write_all(&GGUF_TYPE_STRING.to_le_bytes()).unwrap();
    gguf_str(f, val);
}

fn gguf_meta_u32(f: &mut impl Write, key: &str, val: u32) {
    gguf_str(f, key);
    f.write_all(&GGUF_TYPE_UINT32.to_le_bytes()).unwrap();
    f.write_all(&val.to_le_bytes()).unwrap();
}

fn gguf_meta_f32(f: &mut impl Write, key: &str, val: f32) {
    gguf_str(f, key);
    f.write_all(&GGUF_TYPE_FLOAT32.to_le_bytes()).unwrap();
    f.write_all(&val.to_le_bytes()).unwrap();
}

fn gguf_tensor_info(f: &mut impl Write, name: &str, dims: &[u64], ty: u32, offset: u64) {
    gguf_str(f, name);
    f.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
    for &d in dims {
        f.write_all(&d.to_le_bytes()).unwrap();
    }
    f.write_all(&ty.to_le_bytes()).unwrap();
    f.write_all(&offset.to_le_bytes()).unwrap();
}

fn write_minimal_gguf(path: &Path) {
    write_minimal_gguf_custom(path, 100, None, true, true);
}

/// Write a minimal but complete GGUF file that `load_gguf` can successfully parse.
///
/// Architecture: llama, hidden=4, 1 layer.
/// Tensors: token_embd (embed), output (lm_head), output_norm (norm vector).
fn write_minimal_gguf_custom(
    path: &Path,
    vocab: u64,
    metadata_vocab_size: Option<u32>,
    include_kv_heads: bool,
    include_key_length: bool,
) {
    const HIDDEN: u64 = 4;
    let embed_elems = (HIDDEN * vocab) as usize;
    let norm_elems = HIDDEN as usize;

    let embed_bytes = (embed_elems * 4) as u64; // F32
    let norm_bytes = (norm_elems * 4) as u64;
    let metadata_count: u64 = 6
        + if include_kv_heads { 1 } else { 0 }
        + if include_key_length { 1 } else { 0 }
        + if metadata_vocab_size.is_some() { 1 } else { 0 };

    let mut f = std::fs::File::create(path).unwrap();

    // Header
    f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
    f.write_all(&3u64.to_le_bytes()).unwrap(); // n_tensors
    f.write_all(&metadata_count.to_le_bytes()).unwrap(); // n_metadata

    // Metadata
    gguf_meta_str(&mut f, "general.architecture", "llama");
    gguf_meta_u32(&mut f, "llama.embedding_length", HIDDEN as u32);
    gguf_meta_u32(&mut f, "llama.block_count", 1);
    gguf_meta_u32(&mut f, "llama.feed_forward_length", 16);
    gguf_meta_u32(&mut f, "llama.attention.head_count", 2);
    if include_kv_heads {
        gguf_meta_u32(&mut f, "llama.attention.head_count_kv", 2);
    }
    if include_key_length {
        gguf_meta_u32(&mut f, "llama.attention.key_length", 2);
    }
    gguf_meta_f32(&mut f, "llama.rope.freq_base", 10000.0);
    if let Some(vocab_size) = metadata_vocab_size {
        gguf_meta_u32(&mut f, "llama.vocab_size", vocab_size);
    }

    // Tensor infos (offsets are relative to the data section start)
    gguf_tensor_info(&mut f, "token_embd.weight", &[HIDDEN, vocab], GGUF_F32, 0);
    gguf_tensor_info(
        &mut f,
        "output.weight",
        &[HIDDEN, vocab],
        GGUF_F32,
        embed_bytes,
    );
    gguf_tensor_info(
        &mut f,
        "output_norm.weight",
        &[HIDDEN],
        GGUF_F32,
        embed_bytes * 2,
    );

    // Pad to 32-byte boundary (start of data section)
    let pos = f.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    f.write_all(&vec![0u8; (aligned - pos) as usize]).unwrap();

    // Tensor data: all 1.0f32
    // Write tensor data (all zeros — we just check shape loads correctly)
    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; norm_bytes as usize]).unwrap();
    f.flush().unwrap();
}

/// Write a minimal GGUF with one FFN tensor, used to prove walk-only filtering
/// is applied before/at GGUF tensor loading.
fn write_gguf_with_ffn(path: &Path) {
    const VOCAB: u64 = 100;
    const HIDDEN: u64 = 4;
    const INTERMEDIATE: u64 = 16;
    let embed_elems = (HIDDEN * VOCAB) as usize;
    let norm_elems = HIDDEN as usize;
    let ffn_elems = (HIDDEN * INTERMEDIATE) as usize;

    let embed_bytes = (embed_elems * 4) as u64;
    let norm_bytes = (norm_elems * 4) as u64;
    let ffn_bytes = (ffn_elems * 4) as u64;

    let mut f = std::fs::File::create(path).unwrap();

    f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&4u64.to_le_bytes()).unwrap();
    f.write_all(&8u64.to_le_bytes()).unwrap();

    gguf_meta_str(&mut f, "general.architecture", "llama");
    gguf_meta_u32(&mut f, "llama.embedding_length", HIDDEN as u32);
    gguf_meta_u32(&mut f, "llama.block_count", 1);
    gguf_meta_u32(&mut f, "llama.feed_forward_length", INTERMEDIATE as u32);
    gguf_meta_u32(&mut f, "llama.attention.head_count", 2);
    gguf_meta_u32(&mut f, "llama.attention.head_count_kv", 2);
    gguf_meta_u32(&mut f, "llama.attention.key_length", 2);
    gguf_meta_f32(&mut f, "llama.rope.freq_base", 10000.0);

    gguf_tensor_info(&mut f, "token_embd.weight", &[HIDDEN, VOCAB], GGUF_F32, 0);
    gguf_tensor_info(
        &mut f,
        "output.weight",
        &[HIDDEN, VOCAB],
        GGUF_F32,
        embed_bytes,
    );
    gguf_tensor_info(
        &mut f,
        "output_norm.weight",
        &[HIDDEN],
        GGUF_F32,
        embed_bytes * 2,
    );
    gguf_tensor_info(
        &mut f,
        "blk.0.ffn_gate.weight",
        &[HIDDEN, INTERMEDIATE],
        GGUF_F32,
        embed_bytes * 2 + norm_bytes,
    );

    let pos = f.stream_position().unwrap();
    let aligned = pos.div_ceil(32) * 32;
    f.write_all(&vec![0u8; (aligned - pos) as usize]).unwrap();

    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; embed_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; norm_bytes as usize]).unwrap();
    f.write_all(&vec![0u8; ffn_bytes as usize]).unwrap();
    f.flush().unwrap();
}

// Safetensors loading tests

// GGUF loading tests

// GPT-OSS MXFP4 (load_mxfp4_expert_tensors): full-load path
//
// `walk_only_excludes_gpt_oss_packed_mxfp4_experts` exercises only the
// walk-only path which short-circuits before dequantising. The full
// `load_model_dir` path runs `load_mxfp4_expert_tensors`, which:
//   1. iterates safetensors looking for `*.gate_up_proj_blocks` tensors,
//   2. pairs each with its scales companion + the down companion,
//   3. dequantises them via `crate::quant::mxfp4::split_gate_up_experts`.
//
// This test pins that path against a tiny synthetic GPT-OSS fixture.

// DeepSeek-V4 per-expert MXFP4 (dequantize_per_expert_mxfp4)
//
// V4 stores expert weights as separate `(.weight=I8, .scale=F8_E8M0)`
// pairs per (expert, projection). The `dequantize_per_expert_mxfp4`
// detector inside the standard load branch finds those pairs by
// dtype + naming pattern, regardless of architecture metadata. This
// test exercises that path against a synthetic V4-style fixture.

fn f8_e8m0_bytes(n: usize) -> Vec<u8> {
    // Byte = 127 → 2^0 = 1.0 (per-group scale of unity).
    vec![127u8; n]
}

// Coverage for the file-vs-directory + missing-file early-return arms
// in `loading/safetensors.rs::load_model_dir_filtered_with_validation`.

// Branches the split left uncovered (issue #214)
//
// #204 carved `loading/safetensors.rs` into mod/mxfp4/dtype/paths, which
// concentrated the old file's uncovered remainder into two files. These three
// are the contracts in `mod.rs` that no test reached.

/// GPT-OSS config plus whatever extra tensors a test needs.
fn gpt_oss_dir(dir: &Path, extra: &[(&str, &str, &[usize], Vec<u8>)]) {
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
    let mut entries: Vec<(&str, &str, &[usize], Vec<u8>)> = vec![
        (
            "embed_tokens.weight",
            "F32",
            &[10, 4],
            f32_bytes(&[1.0f32; 40]),
        ),
        ("norm.weight", "F32", &[4], f32_bytes(&[1.0f32; 4])),
        ("lm_head.weight", "F32", &[10, 4], f32_bytes(&[1.0f32; 40])),
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
            vec![127u8; 2],
        ),
    ];
    entries.extend(extra.iter().cloned());
    write_model_dir_with_config(dir, config, &entries);
}

mod deepseek_v4_per_expert_mxfp4_dequantize;
mod gguf_loading_tests;
mod gpt_oss_mxfp4_load_mxfp4_expert_tensors;
mod safetensors_loading_tests;
