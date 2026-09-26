//! SAVE / LOAD ROUND-TRIP

use super::*;

#[test]
fn save_and_load_down_meta_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_down_meta_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Save gate vectors + down_meta + config (needed for load_vindex)
    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    let count = idx.save_down_meta(&dir).unwrap();
    assert_eq!(count, 5); // 3 + 2 (one None skipped)

    let config = VindexConfig {
        version: 2,
        model: "test".into(),
        family: "test".into(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Write a minimal tokenizer (needed for binary down_meta loading)
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    // Load it back via the proper load path
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let idx2 = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

    // Verify content — binary down_meta stores token IDs, not strings.
    // With an empty tokenizer vocab, strings decode to empty or token IDs.
    // Check that the data round-trips (token_id and c_score preserved).
    let meta = idx2.feature_meta(0, 0).unwrap();
    assert_eq!(meta.top_token_id, 100);
    assert!((meta.c_score - 0.95).abs() < 0.01);

    let meta1 = idx2.feature_meta(1, 0).unwrap();
    assert_eq!(meta1.top_token_id, 200);

    // Feature 1 at layer 1 should still be None
    assert!(idx2.feature_meta(1, 1).is_none());

    // Gate vectors should also round-trip
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx2.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 0); // feature 0

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_and_load_gate_vectors_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_gate_rt");
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    assert_eq!(layer_infos.len(), 2);
    assert_eq!(layer_infos[0].layer, 0);
    assert_eq!(layer_infos[0].num_features, 3);
    assert_eq!(layer_infos[1].layer, 1);

    // Verify file exists with expected size
    let gate_path = dir.join("gate_vectors.bin");
    assert!(gate_path.exists());
    let file_size = std::fs::metadata(&gate_path).unwrap().len();
    // 2 layers × 3 features × 4 hidden × 4 bytes = 96 bytes
    assert_eq!(file_size, 96);

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_config_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_config_rt");
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-model".into(),
        family: "test".into(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: vec![
            VindexLayerInfo {
                layer: 0,
                num_features: 3,
                offset: 0,
                length: 48,
                num_experts: None,
                num_features_per_expert: None,
            },
            VindexLayerInfo {
                layer: 1,
                num_features: 3,
                offset: 48,
                length: 48,
                num_experts: None,
                num_features_per_expert: None,
            },
        ],
        down_top_k: 10,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();

    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(loaded.model, "test-model");
    assert_eq!(loaded.num_layers, 2);
    assert_eq!(loaded.hidden_size, 4);
    assert_eq!(loaded.layers.len(), 2);
    assert_eq!(loaded.layers[0].num_features, 3);

    let _ = std::fs::remove_dir_all(&dir);
}
