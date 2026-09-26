//! Round trip
//! Tier / skip gating
//! WeightSource impl on ModelWeights
//! Sparse-source writer behavior
//! Error paths
//! MoE expert writing

use super::*;

#[test]
fn round_trip_through_f32_loader_is_numerically_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);

    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).expect("write");
    let loaded = load_model_weights(tmp.path(), &mut SilentLoadCallbacks).expect("load");

    // Every written 2D tensor comes back bit-exact.
    for (key, tensor) in &weights.tensors {
        let got = loaded
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("tensor {key} missing after round trip"));
        assert_eq!(got.shape(), tensor.shape(), "{key} shape");
        assert_eq!(
            got.as_slice().unwrap(),
            tensor.as_slice().unwrap(),
            "{key} data"
        );
    }
    // Every written 1D vector (norms + qwen2 attention biases).
    for (key, vector) in &weights.vectors {
        assert_eq!(loaded.vectors.get(key), Some(vector), "{key}");
    }
    // lm_head + embed.
    assert_eq!(
        loaded.lm_head.as_slice().unwrap(),
        weights.lm_head.as_slice().unwrap()
    );
    assert_eq!(
        loaded.embed.as_slice().unwrap(),
        weights.embed.as_slice().unwrap()
    );
    // Gate vectors surface as FFN gate tensors.
    for layer in 0..NUM_LAYERS {
        let key = weights.arch.ffn_gate_key(layer);
        let got = loaded.tensors.get(&key).expect("gate tensor");
        assert_eq!(got.shape(), &[GATE_FEATURES, HIDDEN]);
        assert_eq!(got.as_slice().unwrap(), gate_layer_matrix(layer));
    }
}

#[test]
fn manifest_and_config_are_correct_after_write() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);
    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();

    let entries = manifest_entries(tmp.path());
    // 6 tensors + 5 vectors per layer, + final norm + lm_head.
    assert_eq!(entries.len(), NUM_LAYERS * 11 + 2);

    // Per-file contiguity: offsets start at 0 and are gapless, and each
    // entry's byte length matches its f32 shape.
    let mut next_offset: HashMap<&str, u64> = HashMap::new();
    for e in &entries {
        assert!(!e.file.is_empty(), "{} has no file", e.key);
        assert!(
            e.kind == kind::TENSOR || e.kind == kind::VECTOR,
            "{} kind {}",
            e.key,
            e.kind
        );
        let expected_len = e.shape.iter().product::<usize>() as u64 * 4;
        assert_eq!(e.length, expected_len, "{} length", e.key);
        let cursor = next_offset.entry(e.file.as_str()).or_insert(0);
        assert_eq!(e.offset, *cursor, "{} offset in {}", e.key, e.file);
        *cursor += e.length;
    }
    // Physical file sizes match the manifest's account of them.
    for (file, total) in &next_offset {
        let on_disk = std::fs::metadata(tmp.path().join(file)).unwrap().len();
        assert_eq!(on_disk, *total, "{file} size");
    }

    // index.json updated in place.
    let config: VindexConfig =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join(INDEX_JSON)).unwrap())
            .unwrap();
    assert!(config.has_model_weights);
    let model_cfg = config.model_config.expect("model_config recorded");
    assert_eq!(model_cfg.model_type, "qwen2");
    assert_eq!(model_cfg.num_q_heads, NUM_Q_HEADS);
    assert_eq!(model_cfg.num_kv_heads, NUM_KV_HEADS);
}

#[test]
fn attention_level_skips_ffn_and_lm_head() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);
    let opts = WriteWeightsOptions {
        level: crate::ExtractLevel::Attention,
        ..Default::default()
    };
    write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts).unwrap();

    assert!(tmp.path().join(ATTN_WEIGHTS_BIN).exists());
    assert!(tmp.path().join(NORMS_BIN).exists());
    assert!(!tmp.path().join(UP_WEIGHTS_BIN).exists());
    assert!(!tmp.path().join(DOWN_WEIGHTS_BIN).exists());
    assert!(!tmp.path().join(LM_HEAD_BIN).exists());
}

#[test]
fn skip_attn_still_writes_norms() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);
    let opts = WriteWeightsOptions {
        skip_attn: true,
        ..Default::default()
    };
    write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts).unwrap();

    assert!(!tmp.path().join(ATTN_WEIGHTS_BIN).exists());
    assert!(tmp.path().join(NORMS_BIN).exists());
    assert!(tmp.path().join(UP_WEIGHTS_BIN).exists());
    // Norm entries survive; attention projections are absent.
    let entries = manifest_entries(tmp.path());
    assert!(entries.iter().all(|e| e.file != ATTN_WEIGHTS_BIN));
    assert!(entries.iter().any(|e| e.file == NORMS_BIN));
}

#[test]
fn ffn_compact_and_skip_ffn_suppress_up_down_files() {
    for opts in [
        WriteWeightsOptions {
            ffn_compact: true,
            ..Default::default()
        },
        WriteWeightsOptions {
            skip_ffn: true,
            ..Default::default()
        },
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let weights = qwen2_model_weights();
        write_vindex_scaffolding(tmp.path(), &weights);
        write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts)
            .unwrap();
        assert!(!tmp.path().join(UP_WEIGHTS_BIN).exists());
        assert!(!tmp.path().join(DOWN_WEIGHTS_BIN).exists());
        assert!(tmp.path().join(ATTN_WEIGHTS_BIN).exists());
    }
}

#[test]
fn model_weights_source_trait_surface() {
    let mut weights = qwen2_model_weights();
    weights
        .raw_bytes
        .insert("packed.experts".into(), vec![1, 2, 3, 4]);
    let source: &dyn WeightSource = &weights;

    let q_key = weights.arch.attn_q_key(0);
    let (data, rows, cols) = source.get_tensor(&q_key).expect("q tensor");
    assert_eq!((rows, cols), (NUM_Q_HEADS * HEAD_DIM, HIDDEN));
    assert_eq!(data.len(), rows * cols);
    assert!(source.get_tensor("no.such.tensor").is_none());

    let norm_key = weights.arch.final_norm_key().to_string();
    assert!(source.get_vector(&norm_key).is_some());
    assert!(source.get_vector("no.such.vector").is_none());

    assert_eq!(source.num_layers(), NUM_LAYERS);
    let (_, head_rows, head_cols) = source.lm_head().expect("lm_head");
    assert_eq!((head_rows, head_cols), (VOCAB, HIDDEN));

    let names = source.vector_names();
    assert!(names.contains(&norm_key));
    assert_eq!(names.len(), weights.vectors.len());

    assert_eq!(
        source.get_packed_bf16("packed.experts"),
        Some(vec![1, 2, 3, 4])
    );
    assert!(source.get_packed_bf16("absent").is_none());
}

/// Missing tensors are skipped (manifest simply has no entry), and the
/// BitNet `*_sub_norm.weight` vectors are picked up from the source map
/// even though no arch norm-key lists them.
#[test]
fn missing_tensors_skip_and_sub_norms_are_captured() {
    let tmp = tempfile::tempdir().unwrap();
    let mut weights = dense_model_weights_as("bitnet");
    // Drop layer 1's FFN entirely — dense writer must skip, not fail.
    let up1 = weights.arch.ffn_up_key(1);
    let down1 = weights.arch.ffn_down_key(1);
    weights.tensors.remove(&up1);
    weights.tensors.remove(&down1);
    // The family's declared sub-layer norms on layer 0.
    let sub_norms = weights.arch.sub_norm_keys(0);
    assert!(!sub_norms.is_empty(), "bitnet declares sub-layer norms");
    for key in &sub_norms {
        weights.vectors.insert(key.clone(), vec![1.0; HIDDEN]);
    }

    write_vindex_scaffolding(tmp.path(), &weights);
    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();

    let entries = manifest_entries(tmp.path());
    assert!(entries.iter().all(|e| e.key != up1 && e.key != down1));
    assert!(entries.iter().any(|e| e.key == weights.arch.ffn_up_key(0)));
    for key in &sub_norms {
        let e = entries
            .iter()
            .find(|e| &e.key == key)
            .expect("sub-norm entry");
        assert_eq!(e.file, NORMS_BIN);
        assert_eq!(e.kind, kind::VECTOR);
    }
}

/// A family that declares no sub-layer norms writes none, whatever
/// vectors the source happens to carry.
#[test]
fn undeclared_sub_norms_are_not_written() {
    let tmp = tempfile::tempdir().unwrap();
    let mut weights = qwen2_model_weights();
    assert!(weights.arch.sub_norm_keys(0).is_empty());
    let stray = format!("{}attn_sub_norm.weight", weights.arch.layer_prefix(0));
    weights.vectors.insert(stray.clone(), vec![1.0; HIDDEN]);

    write_vindex_scaffolding(tmp.path(), &weights);
    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();

    assert!(manifest_entries(tmp.path()).iter().all(|e| e.key != stray));
}

#[test]
fn mla_without_full_geometry_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    // deepseek_v2 with lora ranks but NO qk_nope/qk_rope/v_head_dim —
    // absorption cannot run, the standard-attention guard must fire.
    let weights = empty_model_weights(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
        "vocab_size": VOCAB,
    }));
    assert!(weights.arch.uses_mla());
    let err = write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks)
        .expect_err("MLA without geometry must be rejected");
    assert!(err.to_string().contains("latent attention"), "{err}");
}

#[test]
fn missing_index_json_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    // No scaffolding: the final config update has nothing to read.
    let err = write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks)
        .expect_err("must fail without index.json");
    assert!(!err.to_string().is_empty());
}

#[test]
fn ffn_compact_on_moe_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = empty_model_weights(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_local_experts": 2,
        "num_experts_per_tok": 2,
        "vocab_size": VOCAB,
    }));
    let opts = WriteWeightsOptions {
        ffn_compact: true,
        ..Default::default()
    };
    let err = write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts)
        .expect_err("MoE + ffn_compact must be refused");
    assert!(err.to_string().contains("ffn_compact"), "{err}");
}

#[test]
fn moe_writer_emits_expert_and_router_entries() {
    const NUM_EXPERTS: usize = 2;
    // Two layers, but tensors only on layer 0 — layer 1 exercises the
    // "expert key resolves but tensor is absent" skip branches.
    const MOE_LAYERS: usize = 2;
    let tmp = tempfile::tempdir().unwrap();
    let mut weights = empty_model_weights(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": HIDDEN,
        "num_hidden_layers": MOE_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "vocab_size": VOCAB,
    }));
    for expert in 0..NUM_EXPERTS {
        let base = 10.0 * expert as f32;
        let up_key = weights.arch.expert_ffn_up_key(0, expert).unwrap();
        let down_key = weights.arch.expert_ffn_down_key(0, expert).unwrap();
        weights
            .tensors
            .insert(up_key, fill(INTERMEDIATE, HIDDEN, base + 1.0).into_shared());
        weights.tensors.insert(
            down_key,
            fill(HIDDEN, INTERMEDIATE, base + 2.0).into_shared(),
        );
    }
    let router_key = weights.arch.moe_router_key(0).unwrap();
    weights.tensors.insert(
        router_key.clone(),
        fill(NUM_EXPERTS, HIDDEN, 70.0).into_shared(),
    );

    // Minimal index.json (the writer's final config update needs it).
    write_vindex_scaffolding(tmp.path(), &qwen2_model_weights());

    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();
    let entries = manifest_entries(tmp.path());

    // Experts' up tensors + router live in up_weights.bin; down tensors
    // in down_weights.bin. Verify bytes at the recorded offsets.
    let up_bytes = std::fs::read(tmp.path().join(UP_WEIGHTS_BIN)).unwrap();
    let down_bytes = std::fs::read(tmp.path().join(DOWN_WEIGHTS_BIN)).unwrap();
    for expert in 0..NUM_EXPERTS {
        for (key, file_bytes, file) in [
            (
                weights.arch.expert_ffn_up_key(0, expert).unwrap(),
                &up_bytes,
                UP_WEIGHTS_BIN,
            ),
            (
                weights.arch.expert_ffn_down_key(0, expert).unwrap(),
                &down_bytes,
                DOWN_WEIGHTS_BIN,
            ),
        ] {
            let entry = entries
                .iter()
                .find(|e| e.key == key)
                .unwrap_or_else(|| panic!("no manifest entry for {key}"));
            assert_eq!(entry.file, file);
            let raw = &file_bytes[entry.offset as usize..(entry.offset + entry.length) as usize];
            let decoded = crate::config::dtype::decode_floats(raw, StorageDtype::F32);
            let expected = weights.tensors.get(&key).unwrap();
            assert_eq!(decoded, expected.as_slice().unwrap());
        }
    }
    let router = entries.iter().find(|e| e.key == router_key).unwrap();
    assert_eq!(router.file, UP_WEIGHTS_BIN);
    assert_eq!(router.shape, vec![NUM_EXPERTS, HIDDEN]);
    // Layer 1 contributed nothing.
    let layer1_router = weights.arch.moe_router_key(1).unwrap();
    assert!(entries.iter().all(|e| e.key != layer1_router));
}
