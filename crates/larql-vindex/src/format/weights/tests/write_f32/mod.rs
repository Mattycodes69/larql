//! Colocated tests for `write_f32` — the split-file f32 weight writer.
//!
//! The writer is verified primarily by ROUND-TRIP through the f32
//! loader (`load::f32`): write a small synthetic model's weights, load
//! them back, and require numerical equality plus manifest and
//! index.json correctness. Writer-only branches (MoE experts, MLA
//! absorption, tier gating, error paths) are pinned against the files
//! and manifest directly.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use larql_models::ModelWeights;

use super::super::load::load_model_weights;
use super::super::write_f32::*;
use crate::config::dtype::{encode_floats, StorageDtype};
use crate::config::types::QuantFormat;
use crate::config::{VindexConfig, VindexLayerInfo};
use crate::extract::callbacks::SilentBuildCallbacks;
use crate::format::filenames::{
    ATTN_WEIGHTS_BIN, DOWN_WEIGHTS_BIN, EMBEDDINGS_BIN, GATE_VECTORS_BIN, INDEX_JSON, LM_HEAD_BIN,
    NORMS_BIN, UP_WEIGHTS_BIN, WEIGHT_MANIFEST_JSON,
};
use crate::index::SilentLoadCallbacks;

const HIDDEN: usize = 4;
const NUM_LAYERS: usize = 2;
const INTERMEDIATE: usize = 6;
const VOCAB: usize = 5;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 2;
const GATE_FEATURES: usize = 2;
/// Base offsets so every tensor's values are distinct.
const VALUE_STEP: f32 = 0.25;

fn qwen2_arch_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    })
}

/// Deterministic distinct fill: `base + i * VALUE_STEP`.
fn fill(rows: usize, cols: usize, base: f32) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        base + (r * cols + c) as f32 * VALUE_STEP
    })
}

fn empty_model_weights(arch_json: &serde_json::Value) -> ModelWeights {
    let arch = larql_models::detect_from_json(arch_json);
    let cfg = arch.config();
    let embed = fill(VOCAB, HIDDEN, 900.0);
    let lm_head = fill(VOCAB, HIDDEN, 950.0);
    ModelWeights {
        tensors: HashMap::new(),
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: embed.into_shared(),
        lm_head: lm_head.into_shared(),
        position_embed: None,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: VOCAB,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    }
}

/// Fully-populated qwen2 model: attention projections + biases, norms,
/// dense FFN, distinct lm_head/embed.
fn qwen2_model_weights() -> ModelWeights {
    dense_model_weights(&qwen2_arch_json())
}

/// The qwen2 fixture's geometry, declared as another family.
fn dense_model_weights_as(model_type: &str) -> ModelWeights {
    let mut json = qwen2_arch_json();
    json["model_type"] = serde_json::json!(model_type);
    dense_model_weights(&json)
}

fn dense_model_weights(arch_json: &serde_json::Value) -> ModelWeights {
    let mut w = empty_model_weights(arch_json);
    let q_dim = NUM_Q_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    for layer in 0..NUM_LAYERS {
        let base = layer as f32 * 100.0;
        let arch = &w.arch;
        let tensors: Vec<(String, Array2<f32>)> = vec![
            (arch.attn_q_key(layer), fill(q_dim, HIDDEN, base + 1.0)),
            (arch.attn_k_key(layer), fill(kv_dim, HIDDEN, base + 2.0)),
            (arch.attn_v_key(layer), fill(kv_dim, HIDDEN, base + 3.0)),
            (arch.attn_o_key(layer), fill(HIDDEN, q_dim, base + 4.0)),
            (
                arch.ffn_up_key(layer),
                fill(INTERMEDIATE, HIDDEN, base + 5.0),
            ),
            (
                arch.ffn_down_key(layer),
                fill(HIDDEN, INTERMEDIATE, base + 6.0),
            ),
        ];
        let vectors: Vec<(Option<String>, usize, f32)> = vec![
            (arch.attn_q_bias_key(layer), q_dim, base + 7.0),
            (arch.attn_k_bias_key(layer), kv_dim, base + 8.0),
            (arch.attn_v_bias_key(layer), kv_dim, base + 9.0),
            (Some(arch.input_layernorm_key(layer)), HIDDEN, base + 10.0),
            (
                Some(arch.post_attention_layernorm_key(layer)),
                HIDDEN,
                base + 11.0,
            ),
        ];
        for (key, t) in tensors {
            w.tensors.insert(key, t.into_shared());
        }
        for (key, len, b) in vectors {
            // A family that declares no key for a vector (no QKV bias,
            // say) simply has none.
            let Some(key) = key else { continue };
            w.vectors
                .insert(key, (0..len).map(|i| b + i as f32 * VALUE_STEP).collect());
        }
    }
    let final_norm = w.arch.final_norm_key().to_string();
    w.vectors
        .insert(final_norm, (0..HIDDEN).map(|i| 800.0 + i as f32).collect());
    w
}

fn gate_layer_matrix(layer: usize) -> Vec<f32> {
    (0..GATE_FEATURES * HIDDEN)
        .map(|i| 500.0 + layer as f32 * 50.0 + i as f32)
        .collect()
}

/// Write the index.json + embeddings.bin + gate_vectors.bin the writer
/// and loader expect around the weight files.
fn write_vindex_scaffolding(dir: &Path, weights: &ModelWeights) {
    let layer_bytes = (GATE_FEATURES * HIDDEN * 4) as u64;
    let config = VindexConfig {
        version: 2,
        model: "test/write-f32".into(),
        family: "test".into(),
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        embed_scale: 1.0,
        layers: (0..NUM_LAYERS)
            .map(|layer| VindexLayerInfo {
                layer,
                num_features: GATE_FEATURES,
                offset: layer as u64 * layer_bytes,
                length: layer_bytes,
                num_experts: None,
                num_features_per_expert: None,
            })
            .collect(),
        down_top_k: 1,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: crate::ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: crate::LayerBands::for_family("test", NUM_LAYERS),
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    std::fs::write(
        dir.join(INDEX_JSON),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    let embed_floats: Vec<f32> = weights.embed.iter().copied().collect();
    std::fs::write(
        dir.join(EMBEDDINGS_BIN),
        encode_floats(&embed_floats, StorageDtype::F32),
    )
    .unwrap();
    let gate: Vec<f32> = (0..NUM_LAYERS).flat_map(gate_layer_matrix).collect();
    std::fs::write(
        dir.join(GATE_VECTORS_BIN),
        encode_floats(&gate, StorageDtype::F32),
    )
    .unwrap();
}

fn manifest_entries(dir: &Path) -> Vec<WeightEntry> {
    let text = std::fs::read_to_string(dir.join(WEIGHT_MANIFEST_JSON)).unwrap();
    serde_json::from_str(&text).unwrap()
}

mod mla_absorption;
mod round_trip;
mod streamingweights;
