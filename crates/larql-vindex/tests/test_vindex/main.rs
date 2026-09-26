//! Tests for the larql-vindex crate.

use larql_vindex::format::filenames::*;
use larql_vindex::{
    FeatureMeta, GateIndex, PatchOverrides, VectorIndex, VindexConfig, VindexLayerInfo,
};
use ndarray::{ArcArray2, Array1, Array2};

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![make_top_k(token, id, score)],
    }
}

/// Build a small in-memory VectorIndex for testing.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    // Layer 0: 3 features × 4 hidden
    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0; // feature 0 responds to dim 0
    gate0[[1, 1]] = 1.0; // feature 1 responds to dim 1
    gate0[[2, 2]] = 1.0; // feature 2 responds to dim 2

    // Layer 1: 3 features × 4 hidden
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let gate_vectors = vec![Some(gate0), Some(gate1)];

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None, // feature 1 has no metadata
        Some(make_meta("Spain", 202, 0.70)),
    ];

    let down_meta = vec![Some(meta0), Some(meta1)];

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

// CONSTRUCTION

// FEATURE LOOKUP

// GATE KNN

// WALK

// MUTATION

// DOWN VECTOR OVERRIDES (used by COMPILE INTO VINDEX baker)

// SAVE / LOAD ROUND-TRIP

// BINARY DOWN_META

// ERROR HANDLING

// LAYER BANDS

// CHECKSUM VERIFICATION

// EXTRACT LEVEL

// DESCRIBE TYPES

// SOURCE PROVENANCE

// PATCHES

// WEIGHTS (split file write/read)

// DTYPE

// LOADER (HF cache resolution)

// PATCH EDGE CASES

// FULL VINDEX LIFECYCLE

// EXTRACT PIPELINE (synthetic model)

fn make_synthetic_model() -> larql_models::ModelWeights {
    use std::collections::HashMap;

    let num_layers = 2;
    let hidden = 8;
    let intermediate = 4;
    let vocab_size = 16;

    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    // Per layer: gate, up, down, attn Q/K/V/O, norms
    for layer in 0..num_layers {
        // FFN gate (intermediate × hidden)
        let mut gate = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + layer as f32;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        // FFN up (intermediate × hidden)
        let mut up = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        // FFN down (hidden × intermediate)
        let mut down = ndarray::Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate {
            down[[i % hidden, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        // Attention Q/K/V/O (hidden × hidden)
        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut attn = ndarray::Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden {
                attn[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{suffix}.weight"),
                attn.into_shared(),
            );
        }

        // Norms
        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; hidden],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; hidden],
        );
    }

    // Final norm
    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    // Embeddings (vocab × hidden)
    let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size {
        embed[[i, i % hidden]] = 1.0;
    }

    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "head_dim": hidden,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));

    larql_models::ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
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

// GGUF tests

// PatchedVindex insert/delete/gate_knn tests

// Vindexfile parse + build test

// HuggingFace path tests

// Streaming extraction test

// GateIndex trait tests

// GATE WALK (BLAS gemv path)

// Q4 GATE KNN

// LM HEAD KNN

// HNSW INTEGRATION

// ADAPTIVE RESIDENCY

//
// Writes a Gemma-4-shaped HuggingFace model dir on disk: config.json
// with `hidden_size_per_layer_input` set (the knob
// `has_per_layer_embeddings()` keys off), tokenizer.json stub, and a
// safetensors with every tensor the extractor expects — including the
// six PLE tensors per layer plus the three globals AND the per-layer
// `layer_scalar` (Gemma-4-only, 0-D scalar surfaced as a 1-element
// vector). Shared between the Q4_K and the `--quant none` regression
// tests so they exercise the exact same fixture.
//
// Returns the populated model dir path; caller decides where to write
// the vindex output and what quant to extract with.
#[allow(clippy::type_complexity)]
fn write_gemma4_ple_fixture(
    model_dir: &std::path::Path,
    num_layers: usize,
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    ple_dim: usize,
) {
    use std::collections::HashMap;

    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden,
            "hidden_size_per_layer_input": ple_dim,
            "vocab_size": vocab,
            // Gemma 4 ships with a final-logit tanh softcap of 30.0. This
            // must survive extract → load; without it predict_kquant peaks
            // on the wrong token on E2B.
            "final_logit_softcapping": 30.0,
        }
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    // Core Gemma 4 tensors (with the multimodal `model.language_model.` prefix
    // the arch strips on load). Attn/FFN dims kept small but 256-aligned.
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.norm.weight",
        vec![hidden],
    );

    for layer in 0..num_layers {
        let lp = format!("model.language_model.layers.{layer}");
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_norm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_norm.weight"),
            vec![hidden],
        );
        // Gemma-4 per-layer scalar multiplier (`layer_scalar`). Real
        // models ship it as a 0-D scalar; we use a 1-element vector
        // because that's how `WeightSource::get_vector` surfaces it.
        // Omitting it on the writer side silently broke Gemma-4
        // inference (same family of bug as #49 for PLE).
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.layer_scalar"),
            vec![1],
        );

        // ── PLE per-layer tensors (the regression surface) ──
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.per_layer_input_gate.weight"),
            vec![ple_dim, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.per_layer_projection.weight"),
            vec![hidden, ple_dim],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_per_layer_input_norm.weight"),
            vec![hidden],
        );
    }

    // ── PLE global tensors ──
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.per_layer_model_projection.weight",
        vec![ple_dim * num_layers, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.embed_tokens_per_layer.weight",
        vec![vocab, ple_dim * num_layers],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.per_layer_projection_norm.weight",
        vec![ple_dim],
    );

    // Serialise as f32 safetensors.
    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
}

mod adaptive_residency;
mod binary_down_meta;
mod construction;
mod down_vector_overrides_used_by_compile_in;
mod extract_pipeline_synthetic_model;
mod full_vindex_lifecycle;
mod gateindex_trait_tests;
mod layer_bands;
mod load_model_weights_rejects_ple_arch_vind;
mod patch_edge_cases;
mod ple_tensors_survive_q4_k_extract_load_ro;
mod ple_tensors_survive_quant_none_extract_l;
mod q4_gate_knn;
mod save_load_round_trip;
mod source_provenance;
mod streaming_extract_with_quantformat_q4k;
mod streaming_extraction_test;
mod variable_per_layer_intermediate_size_gem;
mod vindexfile_parse_build_test;
