use super::super::types::{GgufValue, ShardInfo};
use super::*;

//
// Exercise the `--keep-quant` path (`load_gguf_keep_quant` /
// `GgufFile::load_tensors_filtered_keep_quant`) that preserves the
// raw pre-dequant I2_S BitLinear bytes plus the trailing per-tensor
// scale f32 (microsoft/BitNet layout — see `dequantize_i2_s`).

fn kq_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}
fn kq_meta_str(buf: &mut Vec<u8>, key: &str, val: &str) {
    kq_str(buf, key);
    buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF string type
    kq_str(buf, val);
}
fn kq_meta_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
    kq_str(buf, key);
    buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF uint32 type
    buf.extend_from_slice(&val.to_le_bytes());
}
fn kq_meta_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
    kq_str(buf, key);
    buf.extend_from_slice(&6u32.to_le_bytes()); // GGUF float32 type
    buf.extend_from_slice(&val.to_le_bytes());
}
fn kq_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, offset: u64) {
    kq_str(buf, name);
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&ty.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// 32 bytes of I2_S packed trits — one full 128-element block.
const KQ_I2S_PACKED: [u8; 32] = [
    0b00_01_10_00,
    0x55,
    0xAA,
    0x0F,
    0xF0,
    0x33,
    0xCC,
    0x12,
    0x00,
    0x55,
    0xAA,
    0x0F,
    0xF0,
    0x33,
    0xCC,
    0x12,
    0x00,
    0x55,
    0xAA,
    0x0F,
    0xF0,
    0x33,
    0xCC,
    0x12,
    0x00,
    0x55,
    0xAA,
    0x0F,
    0xF0,
    0x33,
    0xCC,
    0x12,
];

/// Build a complete minimal llama GGUF: token_embd / output /
/// output_norm (F32) plus one I2_S "BitLinear" tensor
/// `blk.0.bitlinear.weight` (128 trits → 32 packed bytes). The
/// name is deliberately *not* an attn/ffn projection so the
/// orientation passes leave it untouched. When `scale` is `Some`,
/// the per-tensor scale f32 is appended immediately after the
/// packed trits (where `load_tensors_filtered_keep_quant` looks
/// for it).
fn build_i2s_gguf(scale: Option<f32>) -> Vec<u8> {
    build_i2s_gguf_opts(scale, true, true)
}

/// As [`build_i2s_gguf`] but lets a test omit `token_embd` (to drive the
/// missing-embed error) or `output` (to drive the tied-lm_head fallback).
fn build_i2s_gguf_opts(scale: Option<f32>, include_embed: bool, include_output: bool) -> Vec<u8> {
    const HIDDEN: u64 = 4;
    const VOCAB: u64 = 8;
    let f32t = crate::quant::ggml::TYPE_F32;
    let i2st = crate::quant::ggml::TYPE_I2_S;
    let align32 = |n: u64| n.div_ceil(32) * 32;

    // (name, dims, ggml_type, data) — the I2_S tensor's trailing scale
    // is part of its `data` here (it lives in the alignment slack after
    // the 32 declared bytes; the I2_S tensor is always last).
    let embed = vec![0u8; (HIDDEN * VOCAB * 4) as usize];
    let norm = vec![0u8; (HIDDEN * 4) as usize];
    let mut i2s = KQ_I2S_PACKED.to_vec();
    if let Some(s) = scale {
        i2s.extend_from_slice(&s.to_le_bytes());
    }
    let mut tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    if include_embed {
        tensors.push((
            "token_embd.weight",
            vec![HIDDEN, VOCAB],
            f32t,
            embed.clone(),
        ));
    }
    if include_output {
        tensors.push(("output.weight", vec![HIDDEN, VOCAB], f32t, embed));
    }
    tensors.push(("output_norm.weight", vec![HIDDEN], f32t, norm));
    tensors.push(("blk.0.bitlinear.weight", vec![32, 4], i2st, i2s));

    // Lay out 32-aligned data-section-relative offsets.
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut running = 0u64;
    for (_, _, _, data) in &tensors {
        offsets.push(running);
        running = align32(running + data.len() as u64);
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&8u64.to_le_bytes()); // n_metadata
    kq_meta_str(&mut buf, "general.architecture", "llama");
    kq_meta_u32(&mut buf, "llama.embedding_length", HIDDEN as u32);
    kq_meta_u32(&mut buf, "llama.block_count", 1);
    kq_meta_u32(&mut buf, "llama.feed_forward_length", 16);
    kq_meta_u32(&mut buf, "llama.attention.head_count", 2);
    kq_meta_u32(&mut buf, "llama.attention.head_count_kv", 2);
    kq_meta_u32(&mut buf, "llama.attention.key_length", 2);
    kq_meta_f32(&mut buf, "llama.rope.freq_base", 10000.0);
    for ((name, dims, ty, _), &off) in tensors.iter().zip(&offsets) {
        kq_tensor_info(&mut buf, name, dims, *ty, off);
    }
    // Pad to the 32-byte data-section boundary, then write each tensor
    // at its declared offset.
    let pad = (32 - (buf.len() % 32)) % 32;
    buf.resize(buf.len() + pad, 0);
    let data_start = buf.len();
    for ((_, _, _, data), &off) in tensors.iter().zip(&offsets) {
        buf.resize(data_start + off as usize, 0);
        buf.extend_from_slice(data);
    }
    buf
}

fn write_tmp_gguf(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bitnet.gguf");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
    (dir, path)
}

// Dequant tests are in format::quant::ggml::tests

mod key_normalisation_and_config;
mod mla_fields_and_tensor_loading;
