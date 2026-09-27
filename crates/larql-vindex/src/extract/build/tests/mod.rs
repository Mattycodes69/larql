use ndarray::ArcArray2;
use std::collections::HashMap;
use tempfile::TempDir;

use super::{build_vindex, knowledge_layer_range};
use crate::{ExtractLevel, SilentBuildCallbacks, SilentLoadCallbacks, StorageDtype, VectorIndex};

// ── synthetic model fixture ──────────────────────────────────────────

const NUM_LAYERS: usize = 2;
const HIDDEN: usize = 8;
const INTERMEDIATE: usize = 4;
const VOCAB: usize = 16;

#[test]
fn knowledge_layer_range_uses_model_band_policy() {
    assert_eq!(knowledge_layer_range("llama", 32), Some((13, 26)));
    assert_eq!(knowledge_layer_range("gemma3", 34), Some((14, 28)));
    assert_eq!(knowledge_layer_range("tiny", 4), None);
}

fn make_weights() -> larql_models::ModelWeights {
    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..NUM_LAYERS {
        let mut gate = ndarray::Array2::<f32>::zeros((INTERMEDIATE, HIDDEN));
        for i in 0..INTERMEDIATE {
            gate[[i, i % HIDDEN]] = 1.0;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        let mut up = ndarray::Array2::<f32>::zeros((INTERMEDIATE, HIDDEN));
        for i in 0..INTERMEDIATE {
            up[[i, (i + 1) % HIDDEN]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        let mut down = ndarray::Array2::<f32>::zeros((HIDDEN, INTERMEDIATE));
        for i in 0..INTERMEDIATE {
            down[[i % HIDDEN, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut a = ndarray::Array2::<f32>::zeros((HIDDEN, HIDDEN));
            for i in 0..HIDDEN {
                a[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{suffix}.weight"),
                a.into_shared(),
            );
        }
        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; HIDDEN],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; HIDDEN],
        );
    }
    vectors.insert("norm.weight".into(), vec![1.0; HIDDEN]);

    let mut embed = ndarray::Array2::<f32>::zeros((VOCAB, HIDDEN));
    for i in 0..VOCAB {
        embed[[i, i % HIDDEN]] = 1.0;
    }
    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "head_dim": HIDDEN,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    }));
    larql_models::ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        head_dim: HIDDEN,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

const TOK_JSON: &str =
    r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

fn tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(TOK_JSON).unwrap()
}

fn run_build(dir: &std::path::Path, level: ExtractLevel, dtype: StorageDtype) {
    let weights = make_weights();
    let tok = tokenizer();
    let mut cb = SilentBuildCallbacks;
    build_vindex(&weights, &tok, "test/unit", dir, 3, level, dtype, &mut cb).unwrap();
}

/// Dense-only build: embeddings + norms + lm_head + tokenizer +
/// index.json, and crucially NO gate_vectors.bin and NO
/// down_meta clustering.  `has_model_weights` must still be
/// true (the loader needs the manifested norms/embed/lm_head),
/// and `layers` must be empty (no gate features).
#[test]
fn build_dense_only_skips_gate_and_clustering() {
    let dir = TempDir::new().unwrap();
    let weights = make_weights();
    let tok = tokenizer();
    let mut cb = SilentBuildCallbacks;
    super::build_vindex_dense_only(
        &weights,
        &tok,
        "test/unit",
        dir.path(),
        StorageDtype::F32,
        &mut cb,
    )
    .unwrap();

    // Present: the dense pieces the BitNet loader needs.
    assert!(
        dir.path().join("embeddings.bin").exists(),
        "embeddings.bin missing"
    );
    assert!(
        dir.path().join("norms.bin").exists(),
        "norms.bin missing — dense-only must still write norms (decoupled \
         from the skipped attention projections)"
    );
    assert!(dir.path().join("index.json").exists(), "index.json missing");
    assert!(
        dir.path().join("tokenizer.json").exists(),
        "tokenizer.json missing"
    );

    // Absent: the expensive walk-mode artifacts.
    assert!(
        !dir.path().join("gate_vectors.bin").exists(),
        "dense-only must NOT write gate_vectors.bin"
    );
    // Absent: the dense f32 attention/FFN projections (they live
    // in the bitnet/ I2_S artifacts; writing them dense would be
    // ~6 GB of duplicate weights the BitNet path never reads).
    assert!(
        !dir.path().join("attn_weights.bin").exists(),
        "dense-only must NOT write attn_weights.bin"
    );
    assert!(
        !dir.path().join("up_weights.bin").exists(),
        "dense-only must NOT write up_weights.bin"
    );
    assert!(
        !dir.path().join("down_weights.bin").exists(),
        "dense-only must NOT write down_weights.bin"
    );

    // index.json: has_model_weights true (manifest written),
    // zero gate layers.
    let cfg = crate::load_vindex_config(dir.path()).unwrap();
    assert!(
        cfg.has_model_weights,
        "dense-only must set has_model_weights=true"
    );
    assert!(
        cfg.layers.is_empty(),
        "dense-only must carry zero gate layers, got {}",
        cfg.layers.len()
    );
}

/// A dense-only vindex must load as a degenerate VectorIndex
/// (empty gate) without erroring on the missing gate_vectors.bin
/// — this is what lets larql-server boot a dense-only BitNet
/// vindex.
#[test]
fn dense_only_vindex_loads_without_gate_vectors() {
    let dir = TempDir::new().unwrap();
    let weights = make_weights();
    let tok = tokenizer();
    let mut cb = SilentBuildCallbacks;
    super::build_vindex_dense_only(
        &weights,
        &tok,
        "test/unit",
        dir.path(),
        StorageDtype::F32,
        &mut cb,
    )
    .unwrap();
    let mut lcb = SilentLoadCallbacks;
    let idx = VectorIndex::load_vindex_with_range(dir.path(), &mut lcb, None);
    assert!(
        idx.is_ok(),
        "dense-only vindex failed to load: {:?}",
        idx.err()
    );
}

// ── build output file inventory ──────────────────────────────────────

#[test]
fn build_browse_writes_required_files() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    assert!(
        dir.path().join("gate_vectors.bin").exists(),
        "gate_vectors.bin missing"
    );
    assert!(
        dir.path().join("embeddings.bin").exists(),
        "embeddings.bin missing"
    );
    assert!(
        dir.path().join("down_meta.bin").exists(),
        "down_meta.bin missing"
    );
    assert!(dir.path().join("index.json").exists(), "index.json missing");
    assert!(
        dir.path().join("tokenizer.json").exists(),
        "tokenizer.json missing"
    );
}

#[test]
fn build_browse_does_not_write_weight_files() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    // Browse level: no model weights
    assert!(!dir.path().join("attn_weights.bin").exists());
    assert!(!dir.path().join("up_weights.bin").exists());
    assert!(!dir.path().join("down_weights.bin").exists());
}

#[test]
fn build_all_writes_weight_files() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::All, StorageDtype::F32);
    assert!(
        dir.path().join("attn_weights.bin").exists(),
        "attn_weights.bin missing"
    );
    assert!(
        dir.path().join("up_weights.bin").exists(),
        "up_weights.bin missing"
    );
    assert!(
        dir.path().join("down_weights.bin").exists(),
        "down_weights.bin missing"
    );
}

// ── index.json content ───────────────────────────────────────────────

#[test]
fn build_index_json_has_correct_shape() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    assert_eq!(cfg.num_layers, NUM_LAYERS);
    assert_eq!(cfg.hidden_size, HIDDEN);
    assert_eq!(cfg.intermediate_size, INTERMEDIATE);
    assert_eq!(cfg.vocab_size, VOCAB);
    assert_eq!(cfg.model, "test/unit");
    assert_eq!(cfg.version, 2);
}

#[test]
fn build_browse_has_model_weights_false() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    assert!(!cfg.has_model_weights);
}

#[test]
fn build_all_has_model_weights_true() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::All, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    assert!(cfg.has_model_weights);
}

#[test]
fn build_records_source_provenance() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    let src = cfg.source.unwrap();
    assert_eq!(src.huggingface_repo.as_deref(), Some("test/unit"));
    assert!(!src.larql_version.is_empty());
}

#[test]
fn build_records_checksums() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    let checksums = cfg.checksums.unwrap();
    assert!(
        checksums.contains_key("gate_vectors.bin"),
        "gate_vectors.bin not in checksums"
    );
}

#[test]
fn build_layer_infos_match_num_layers() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    assert_eq!(cfg.layers.len(), NUM_LAYERS);
    for (i, info) in cfg.layers.iter().enumerate() {
        assert_eq!(info.layer, i, "layer index mismatch at position {i}");
        assert_eq!(
            info.num_features, INTERMEDIATE,
            "wrong feature count at layer {i}"
        );
    }
}

// ── gate_vectors.bin content ─────────────────────────────────────────

#[test]
fn build_gate_vectors_bin_size_matches_config() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    let expected: u64 = cfg.layers.iter().map(|l| l.length).sum();
    let actual = std::fs::metadata(dir.path().join("gate_vectors.bin"))
        .unwrap()
        .len();
    assert_eq!(actual, expected, "gate_vectors.bin size mismatch");
}

// ── round-trip: build then load ──────────────────────────────────────

#[test]
fn build_then_load_vindex_succeeds() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
    assert_eq!(index.num_layers, NUM_LAYERS);
    assert_eq!(index.hidden_size, HIDDEN);
    assert_eq!(index.total_gate_vectors(), NUM_LAYERS * INTERMEDIATE);
}

#[test]
fn build_then_load_gate_knn_returns_results() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
    let query = ndarray::Array1::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = index.gate_knn(0, &query, 2);
    assert!(!hits.is_empty(), "gate_knn returned no results after build");
}

#[test]
fn build_f16_dtype_round_trips() {
    let dir = TempDir::new().unwrap();
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F16);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    assert_eq!(cfg.dtype, StorageDtype::F16);
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
    assert_eq!(index.num_layers, NUM_LAYERS);
}

#[test]
fn build_idempotent_on_existing_dir() {
    let dir = TempDir::new().unwrap();
    // First build
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    // Second build into same directory should overwrite cleanly
    run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
    let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
    assert_eq!(cfg.num_layers, NUM_LAYERS);
}

// ── architecture capability gate ─────────────────────────────────────

fn make_mla_weights() -> larql_models::ModelWeights {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": HIDDEN,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    }));
    assert!(arch.uses_mla(), "fixture must produce an MLA architecture");

    let mut embed = ndarray::Array2::<f32>::zeros((VOCAB, HIDDEN));
    for i in 0..VOCAB {
        embed[[i, i % HIDDEN]] = 1.0;
    }
    let embed = embed.into_shared();
    let lm_head = embed.clone();

    larql_models::ModelWeights {
        tensors: HashMap::new(),
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        head_dim: HIDDEN,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

#[test]
fn build_browse_passes_for_mla_arch() {
    // Browse-level extracts don't need attention; MLA must succeed.
    let dir = TempDir::new().unwrap();
    let weights = make_mla_weights();
    let tok = tokenizer();
    let mut cb = SilentBuildCallbacks;
    build_vindex(
        &weights,
        &tok,
        "test/mla",
        dir.path(),
        3,
        ExtractLevel::Browse,
        StorageDtype::F32,
        &mut cb,
    )
    .expect("Browse-level MLA extract should succeed (no attention written)");
    // Sanity: gate_vectors / embeddings still got written.
    assert!(dir.path().join("gate_vectors.bin").exists());
    assert!(dir.path().join("embeddings.bin").exists());
}

#[test]
fn build_inference_rejects_mla_before_writing() {
    // The capability gate must fire before any output file is created
    // so a failed extract leaves no half-populated vindex on disk.
    let dir = TempDir::new().unwrap();
    let weights = make_mla_weights();
    let tok = tokenizer();
    let mut cb = SilentBuildCallbacks;
    let err = build_vindex(
        &weights,
        &tok,
        "test/mla",
        dir.path(),
        3,
        ExtractLevel::Inference,
        StorageDtype::F32,
        &mut cb,
    )
    .expect_err("Inference-level MLA extract must be rejected up front");
    let msg = err.to_string();
    assert!(
        msg.contains("MLA"),
        "error should mention MLA capability gap: {msg}"
    );

    let written: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        written.is_empty(),
        "MLA rejection must happen before any file is written; \
         found leftovers: {written:?}"
    );
}

mod gate_layouts;
