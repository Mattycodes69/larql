//! Synthetic index, config and model fixtures for the feature demo.

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{ArcArray2, Array2};
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

pub(super) fn section(name: &str) {
    println!("\n── {} ──\n", name);
}

pub(super) fn meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(),
        top_token_id: id,
        c_score: score,
        top_k: vec![TopKEntry {
            token: token.into(),
            token_id: id,
            logit: score,
        }],
    }
}

pub(super) fn build_demo_index() -> VectorIndex {
    let h = 4;
    let mut g0 = Array2::<f32>::zeros((5, h));
    g0[[0, 0]] = 10.0;
    g0[[1, 1]] = 10.0;
    g0[[2, 2]] = 10.0;
    g0[[3, 0]] = 5.0;
    g0[[3, 1]] = 5.0;
    let g1 = Array2::<f32>::zeros((5, h));
    let m0 = vec![
        Some(meta("Paris", 100, 0.95)),
        Some(meta("Berlin", 101, 0.92)),
        Some(meta("Tokyo", 102, 0.88)),
        Some(meta("European", 103, 0.70)),
        None,
    ];
    VectorIndex::new(
        vec![Some(g0), Some(g1)],
        vec![Some(m0), Some(vec![None; 5])],
        2,
        h,
    )
}

pub(super) fn build_moe_index() -> VectorIndex {
    let h = 4;
    let mut g = Array2::<f32>::zeros((6, h));
    g[[0, 0]] = 10.0;
    g[[1, 1]] = 10.0;
    g[[2, 2]] = 10.0;
    g[[3, 3]] = 10.0;
    g[[4, 0]] = 5.0;
    g[[4, 3]] = 5.0;
    g[[5, 1]] = 3.0;
    let m = vec![
        Some(meta("Paris", 100, 0.95)),
        Some(meta("Berlin", 101, 0.92)),
        Some(meta("Tokyo", 102, 0.88)),
        Some(meta("London", 103, 0.90)),
        Some(meta("Rome", 104, 0.85)),
        Some(meta("Madrid", 105, 0.80)),
    ];
    VectorIndex::new(vec![Some(g)], vec![Some(m)], 1, h)
}

pub(super) fn make_config(
    model: &str,
    layers: usize,
    hidden: usize,
    intermediate: usize,
    layer_infos: Vec<larql_vindex::VindexLayerInfo>,
    dtype: larql_vindex::StorageDtype,
) -> VindexConfig {
    VindexConfig {
        version: 2,
        model: model.into(),
        family: "demo".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some(format!("demo/{model}")),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: "2026-04-01T00:00:00Z".into(),
            larql_version: env!("CARGO_PKG_VERSION").into(),
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        }),
        checksums: larql_vindex::format::checksums::compute_checksums(
            &std::env::temp_dir().join("larql_vindex_showcase"),
        )
        .ok(),
        num_layers: layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size: 200,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    }
}

pub(super) fn make_synthetic_model() -> larql_models::ModelWeights {
    let (num_layers, hidden, intermediate, vocab_size) = (2, 8, 4, 16);
    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..num_layers {
        let mut gate = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + layer as f32;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        let mut up = Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        let mut down = Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate {
            down[[i % hidden, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        for s in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut a = Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden {
                a[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{s}.weight"),
                a.into_shared(),
            );
        }
        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; hidden],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; hidden],
        );
    }
    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    let mut embed = Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size {
        embed[[i, i % hidden]] = 1.0;
    }

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": hidden,
        "num_hidden_layers": num_layers, "intermediate_size": intermediate,
        "head_dim": hidden, "num_attention_heads": 1,
        "num_key_value_heads": 1, "rope_theta": 10000.0, "vocab_size": vocab_size,
    }));

    let embed = embed.into_shared();
    larql_models::ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: embed.clone(),
        lm_head: embed.clone(),
        position_embed: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim: hidden,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}
