//! Streaming extraction test

#[test]
fn streaming_extract_from_safetensors() {
    // Create a minimal safetensors model directory with synthetic weights
    let model_dir = std::env::temp_dir().join("larql_test_streaming_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    std::fs::create_dir_all(&model_dir).unwrap();

    // Write config.json
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_theta": 10000.0,
        "vocab_size": 16,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    // Write a minimal safetensors file with gate + down + embed tensors
    let mut tensors: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    // Embeddings: 16 × 8
    let embed: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    tensors.insert("model.embed_tokens.weight".into(), embed);
    metadata.push(("model.embed_tokens.weight".into(), vec![16, 8]));

    // Per-layer: gate (4×8), down (8×4)
    for layer in 0..2 {
        let gate: Vec<f32> = (0..32).map(|i| (i as f32 + layer as f32) * 0.1).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.gate_proj.weight"), gate);
        metadata.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            vec![4, 8],
        ));

        let down: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.down_proj.weight"), down);
        metadata.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            vec![8, 4],
        ));
    }

    // Build safetensors file
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

    // Write tokenizer
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();

    // Run streaming extraction
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;

    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None,
        &mut cb,
    )
    .unwrap();

    // Verify output
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());
    assert!(output_dir.join("index.json").exists());
    assert!(output_dir.join("tokenizer.json").exists());

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.model, "test/streaming");

    // Load and verify KNN works
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut lcb).unwrap();
    assert_eq!(index.total_gate_vectors(), 8); // 2 layers × 4 features
    assert!(index.is_mmap()); // should use mmap mode

    let query = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = index.gate_knn(0, &query, 2);
    assert!(!hits.is_empty());

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}
