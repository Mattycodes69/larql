//! Rich, full and large vindex fixtures and sessions.

use crate::parser;

#[allow(unused_imports)]
use super::*;

/// Spin up a session and `USE` the rich test vindex.
pub(super) fn rich_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_rich_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on rich synthetic vindex should succeed");
    (session, dir)
}

//
// `make_full_test_vindex_dir` produces a vindex with `has_model_weights
// = true`, populated via `larql_inference::test_utils::make_test_weights`
// (TinyModelArch, 2 layers × 16 hidden × 32 intermediate × vocab 32).
// `larql_vindex::write_model_weights` writes the full attention + FFN
// + lm_head + norm weight files into the vindex directory, so
// `load_model_weights` succeeds and downstream INFER / TRACE / EXPLAIN
// INFER / COMPACT MAJOR / REBALANCE-with-installs / INSERT-compose all
// have real (random) weights to forward through.
//
// The same WordLevel tokenizer that `make_test_tokenizer(32)` produces
// is written to the vindex so prompts tokenise to ids 0..31.

pub(super) fn make_full_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_vindex::{
        ExtractLevel, MoeConfig, QuantFormat, SilentBuildCallbacks, StorageDtype, VindexConfig,
        VindexLayerInfo, VindexModelConfig,
    };

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_full_test_vindex_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // `make_test_weights()` produces vocab_size = 32 with embed shape
    // [32, 16]. The companion `make_test_tokenizer(32)` adds `[UNK]`
    // at id 32 — out of the embed's bounds. Extend the embed by one
    // row so any UNK-tagged token still resolves to a valid embedding.
    let mut weights = larql_inference::test_utils::make_test_weights();
    {
        use larql_inference::ndarray::Array2;
        let new_vocab = weights.vocab_size + 1;
        let hidden = weights.hidden_size;
        let mut extended = Array2::<f32>::zeros((new_vocab, hidden));
        for (i, row) in weights.embed.rows().into_iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                extended[[i, j]] = *v;
            }
        }
        // UNK row gets a small constant so it embeds to something
        // distinguishable from id 0.
        for j in 0..hidden {
            extended[[weights.vocab_size, j]] = 0.01_f32 * (j as f32 + 1.0);
        }
        weights.embed = extended.into_shared();
        weights.vocab_size = new_vocab;
        // Mirror into the lm_head if it shared the original embed.
        let mut lm_extended = Array2::<f32>::zeros((new_vocab, hidden));
        for (i, row) in weights.lm_head.rows().into_iter().enumerate() {
            if i >= new_vocab {
                break;
            }
            for (j, v) in row.iter().enumerate() {
                lm_extended[[i, j]] = *v;
            }
        }
        weights.lm_head = lm_extended.into_shared();
        // Update embed_key tensor too so manifest stays consistent.
        let embed_key = weights.arch.embed_key().to_string();
        weights.tensors.insert(embed_key, weights.embed.clone());
    }
    let vindex = larql_inference::test_utils::make_test_vindex(&weights);

    // Gate offsets: each layer's gate matrix is `intermediate × hidden`
    // floats. Lay them out contiguously starting at 0.
    let bpf = 4_usize; // f32
    let row_bytes = weights.hidden_size * bpf;
    let layer_bytes = weights.intermediate_size * row_bytes;
    let mut layers: Vec<VindexLayerInfo> = Vec::new();
    for li in 0..weights.num_layers {
        layers.push(VindexLayerInfo {
            layer: li,
            offset: (li * layer_bytes) as u64,
            length: layer_bytes as u64,
            num_features: weights.intermediate_size,
            num_experts: None,
            num_features_per_expert: None,
        });
    }

    let model_config = VindexModelConfig {
        model_type: weights.arch.family().to_string(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
        ..Default::default()
    };

    let mut config = VindexConfig {
        version: 2,
        model: format!("test/full-fixture-{tag}"),
        family: weights.arch.family().to_string(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: layers.clone(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    // 1. Save the index (gate vectors + down_meta + config).
    vindex.save_vindex(&dir, &mut config).unwrap();

    // 2. Write the model weight files (attn / up / down / norms / lm_head).
    let mut build_cb = SilentBuildCallbacks;
    larql_vindex::write_model_weights(&weights, &dir, &mut build_cb).unwrap();

    // 3. Write embeddings.bin from `weights.embed`.
    let embed_slice = weights.embed.as_slice().unwrap();
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * bpf);
    for v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // 4. Write a tokenizer.json that maps token-id N to "[N]" string.
    //    Pass `vocab_size - 1` so the tokenizer's `[UNK]` lands at id
    //    `vocab_size - 1` (we extended the embed by one above), keeping
    //    every produced id inside the embed table.
    let tok = larql_inference::test_utils::make_test_tokenizer(weights.vocab_size - 1);
    tok.save(dir.join("tokenizer.json").to_str().unwrap(), false)
        .unwrap();

    dir
}

/// Build a synthetic vindex with hidden_size=1024.
///
/// `make_test_weights()` is hardcoded to hidden=16, but `COMPACT MAJOR`
/// guards on `hidden_dim >= 1024`. This fixture mirrors the full-vindex
/// builder with parameterised dimensions large enough to clear that
/// guard while staying small enough for unit-test runtime (~5 MB on
/// disk, sub-second forward pass per fact).
pub(super) fn make_large_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_inference::ndarray::Array2;
    use larql_models::{detect_from_json, ModelWeights, WeightArray};
    use larql_vindex::{
        ExtractLevel, MoeConfig, QuantFormat, SilentBuildCallbacks, StorageDtype, VindexConfig,
        VindexLayerInfo, VindexModelConfig,
    };
    use std::collections::HashMap;

    // Just over the COMPACT MAJOR threshold; intermediate kept tiny so
    // gate/up/down stay under 1 MB each.
    const VOCAB: usize = 32;
    const HIDDEN: usize = 1024;
    const INTER: usize = 64;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 64;
    const NUM_LAYERS: usize = 2;

    let dir = std::env::temp_dir().join(format!(
        "larql_lql_large_test_vindex_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
    });
    let arch = detect_from_json(&arch_json);
    let arch_family = arch.family().to_string();

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0x600d_face_u64;
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    };

    // Reserve one extra vocab row for the [UNK] token (mirrors the
    // small-fixture extension trick).
    let new_vocab = VOCAB + 1;
    let mut embed_arr = Array2::<f32>::zeros((new_vocab, HIDDEN));
    let base_embed = rand_mat(VOCAB, HIDDEN, 0.05);
    for (i, row) in base_embed.rows().into_iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            embed_arr[[i, j]] = *v;
        }
    }
    for j in 0..HIDDEN {
        embed_arr[[VOCAB, j]] = 0.005_f32 * ((j % 13) as f32 + 1.0);
    }
    let embed = embed_arr.into_shared();
    let lm_head = rand_mat(new_vocab, HIDDEN, 0.05);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    for layer in 0..NUM_LAYERS {
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 0.05));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 0.05));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 0.05));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 0.05));
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 0.05));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 0.05));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 0.05));
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    let weights = ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed: embed.clone(),
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: new_vocab,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    };

    // Build vindex with random gate vectors, mirroring make_test_vindex.
    let n_features = INTER;
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..NUM_LAYERS)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * HIDDEN)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, HIDDEN), data).unwrap())
        })
        .collect();
    let down_meta = vec![None; NUM_LAYERS];
    let vindex = larql_vindex::VectorIndex::new(gate_vectors, down_meta, NUM_LAYERS, HIDDEN);

    let bpf = 4_usize;
    let row_bytes = HIDDEN * bpf;
    let layer_bytes = INTER * row_bytes;
    let layers: Vec<VindexLayerInfo> = (0..NUM_LAYERS)
        .map(|li| VindexLayerInfo {
            layer: li,
            offset: (li * layer_bytes) as u64,
            length: layer_bytes as u64,
            num_features: INTER,
            num_experts: None,
            num_features_per_expert: None,
        })
        .collect();

    let model_config = VindexModelConfig {
        model_type: arch_family.clone(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
        ..Default::default()
    };

    let mut config = VindexConfig {
        version: 2,
        model: format!("test/large-fixture-{tag}"),
        family: arch_family,
        source: None,
        checksums: None,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: new_vocab,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        layers: layers.clone(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    vindex.save_vindex(&dir, &mut config).unwrap();

    let mut build_cb = SilentBuildCallbacks;
    larql_vindex::write_model_weights(&weights, &dir, &mut build_cb).unwrap();

    let embed_slice = embed.as_slice().unwrap();
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * bpf);
    for v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    let tok = larql_inference::test_utils::make_test_tokenizer(VOCAB);
    tok.save(dir.join("tokenizer.json").to_str().unwrap(), false)
        .unwrap();

    dir
}

pub(super) fn large_vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_large_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, lql_path(&dir))).unwrap();
    session
        .execute(&stmt)
        .expect("USE on large synthetic vindex should succeed");
    (session, dir)
}
