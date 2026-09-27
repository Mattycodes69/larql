//! FULL VINDEX LIFECYCLE

use super::*;

#[test]
fn full_lifecycle_build_query_mutate_save_reload() {
    // Build → query → mutate → save → reload → verify
    let hidden = 4;
    let mut g0 = Array2::<f32>::zeros((4, hidden));
    g0[[0, 0]] = 10.0; // Paris
    g0[[1, 1]] = 10.0; // Berlin
    g0[[2, 2]] = 10.0; // Tokyo
                       // F3 is empty (free slot)
    let gate_vectors = vec![Some(g0)];

    let meta = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("Berlin", 101, 0.92)),
        Some(make_meta("Tokyo", 102, 0.88)),
        None,
    ];
    let down_meta = vec![Some(meta)];

    let mut idx = VectorIndex::new(gate_vectors, down_meta, 1, hidden);

    // Query
    let q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(idx.gate_knn(0, &q, 1)[0].0, 0); // Paris

    // Mutate
    let slot = idx.find_free_feature(0).unwrap();
    assert_eq!(slot, 3);
    idx.set_gate_vector(0, slot, &Array1::from_vec(vec![0.0, 0.0, 0.0, 10.0]));
    idx.set_feature_meta(0, slot, make_meta("Canberra", 103, 0.85));

    // Save
    let dir = std::env::temp_dir().join("larql_test_lifecycle");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    idx.save_down_meta(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "lifecycle-test".into(),
        family: "test".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: hidden,
        intermediate_size: 4,
        vocab_size: 200,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Write tokenizer for binary down_meta loading
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    // Reload
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

    // Verify — token IDs and scores round-trip through binary
    assert_eq!(loaded.total_gate_vectors(), 4);
    assert_eq!(loaded.total_down_meta(), 4);
    assert_eq!(loaded.feature_meta(0, 0).unwrap().top_token_id, 100);
    assert_eq!(loaded.feature_meta(0, 3).unwrap().top_token_id, 103);

    // KNN should find Canberra for dim 3
    let q2 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(loaded.gate_knn(0, &q2, 1)[0].0, 3);

    let _ = std::fs::remove_dir_all(&dir);
}
