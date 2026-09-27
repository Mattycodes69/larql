//! LAYER BANDS
//! CHECKSUM VERIFICATION
//! EXTRACT LEVEL
//! DESCRIBE TYPES

use super::*;

#[test]
fn layer_bands_gemma3_4b() {
    let bands = larql_vindex::LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));
}

#[test]
fn layer_bands_gemma2_9b() {
    let bands = larql_vindex::LayerBands::for_family("gemma2", 42).unwrap();
    assert_eq!(bands.syntax, (0, 16));
    assert_eq!(bands.knowledge, (17, 34));
    assert_eq!(bands.output, (35, 41));
}

#[test]
fn layer_bands_llama3_70b() {
    let bands = larql_vindex::LayerBands::for_family("llama", 80).unwrap();
    assert_eq!(bands.syntax, (0, 31));
    assert_eq!(bands.knowledge, (32, 63));
    assert_eq!(bands.output, (64, 79));
}

#[test]
fn layer_bands_llama3_8b() {
    let bands = larql_vindex::LayerBands::for_family("llama", 32).unwrap();
    assert_eq!(bands.syntax, (0, 12));
    assert_eq!(bands.knowledge, (13, 25));
    assert_eq!(bands.output, (26, 31));
}

#[test]
fn layer_bands_mixtral() {
    let bands = larql_vindex::LayerBands::for_family("mixtral", 32).unwrap();
    assert_eq!(bands.syntax, (0, 12));
    assert_eq!(bands.knowledge, (13, 25));
    assert_eq!(bands.output, (26, 31));
}

#[test]
fn layer_bands_gpt2_small() {
    let bands = larql_vindex::LayerBands::for_family("gpt2", 12).unwrap();
    assert_eq!(bands.syntax, (0, 4));
    assert_eq!(bands.knowledge, (5, 9));
    assert_eq!(bands.output, (10, 11));
}

#[test]
fn layer_bands_unknown_family_fallback() {
    // Unknown family with enough layers → falls back to heuristic
    let bands = larql_vindex::LayerBands::for_family("unknown_model", 40).unwrap();
    assert_eq!(bands.syntax.0, 0);
    assert!(bands.knowledge.0 > bands.syntax.1);
    assert!(bands.output.0 > bands.knowledge.1);
    assert_eq!(bands.output.1, 39);
}

#[test]
fn layer_bands_tiny_model_returns_none() {
    // Too few layers to band meaningfully
    assert!(larql_vindex::LayerBands::for_family("test", 2).is_none());
    assert!(larql_vindex::LayerBands::for_family("test", 4).is_none());
}

#[test]
fn layer_bands_band_for_layer() {
    let bands = larql_vindex::LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.band_for_layer(0), "syntax");
    assert_eq!(bands.band_for_layer(13), "syntax");
    assert_eq!(bands.band_for_layer(14), "knowledge");
    assert_eq!(bands.band_for_layer(27), "knowledge");
    assert_eq!(bands.band_for_layer(28), "output");
    assert_eq!(bands.band_for_layer(33), "output");
}

#[test]
fn v1_config_loads_with_defaults() {
    // Simulate a v1 index.json that lacks new fields
    let dir = std::env::temp_dir().join("larql_test_v1_compat");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let v1_json = r#"{
        "version": 1,
        "model": "old-model",
        "family": "test",
        "num_layers": 32,
        "hidden_size": 4,
        "intermediate_size": 3,
        "vocab_size": 100,
        "embed_scale": 1.0,
        "layers": [],
        "down_top_k": 10
    }"#;
    std::fs::write(dir.join("index.json"), v1_json).unwrap();

    let config = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(config.version, 1);
    assert_eq!(config.model, "old-model");
    // New fields should have sensible defaults
    assert_eq!(config.extract_level, larql_vindex::ExtractLevel::Browse);
    assert!(config.layer_bands.is_none());
    assert!(config.source.is_none());
    assert!(config.checksums.is_none());
    assert!(!config.has_model_weights);
    assert!(config.model_config.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_config_full_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_v2_full_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write a dummy gate_vectors.bin so checksums have something to hash
    std::fs::write(dir.join("gate_vectors.bin"), b"test data").unwrap();

    let checksums = larql_vindex::checksums::compute_checksums(&dir).ok();

    let config = VindexConfig {
        version: 2,
        model: "google/gemma-3-4b-it".into(),
        family: "gemma3".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("abc123".into()),
            safetensors_sha256: None,
            extracted_at: "2026-04-01T12:00:00Z".into(),
            larql_version: "0.1.0".into(),
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        }),
        checksums,
        num_layers: 34,
        hidden_size: 2560,
        intermediate_size: 10240,
        vocab_size: 262144,
        embed_scale: 50.596,
        extract_level: larql_vindex::ExtractLevel::Inference,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }),
        layers: vec![],
        down_top_k: 10,
        has_model_weights: true,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "gemma3".into(),
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: 10000.0,
            sliding_window: Some(1024),
            moe: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            layer_rope_theta: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
            ..Default::default()
        }),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    // Verify all v2 fields round-trip
    assert_eq!(loaded.version, 2);
    assert_eq!(loaded.model, "google/gemma-3-4b-it");
    assert_eq!(loaded.extract_level, larql_vindex::ExtractLevel::Inference);
    assert!(loaded.has_model_weights);

    let source = loaded.source.unwrap();
    assert_eq!(
        source.huggingface_repo.as_deref(),
        Some("google/gemma-3-4b-it")
    );
    assert_eq!(source.huggingface_revision.as_deref(), Some("abc123"));
    assert_eq!(source.larql_version, "0.1.0");

    let bands = loaded.layer_bands.unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));

    let mc = loaded.model_config.unwrap();
    assert_eq!(mc.model_type, "gemma3");
    assert_eq!(mc.head_dim, 256);
    assert_eq!(mc.sliding_window, Some(1024));
    assert!(mc.moe.is_none());

    assert!(loaded.checksums.is_some());
    let cs = loaded.checksums.unwrap();
    assert!(cs.contains_key("gate_vectors.bin"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_config_with_moe() {
    let dir = std::env::temp_dir().join("larql_test_v2_moe");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "mistralai/Mixtral-8x7B".into(),
        family: "mixtral".into(),
        source: None,
        checksums: None,
        num_layers: 32,
        hidden_size: 4096,
        intermediate_size: 14336,
        vocab_size: 32000,
        embed_scale: 64.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands::for_family("mixtral", 32).unwrap()),
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "mixtral".into(),
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            rope_base: 1000000.0,
            sliding_window: None,
            moe: Some(larql_vindex::MoeConfig {
                num_experts: 8,
                top_k: 2,
                shared_expert: false,
                shared_expert_intermediate_size: None,
                router_type: "top_k_softmax".into(),
                moe_intermediate_size: None,
                hybrid: false,
            }),
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            layer_rope_theta: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
            ..Default::default()
        }),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let mc = loaded.model_config.unwrap();
    let moe = mc.moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);
    assert!(!moe.shared_expert);
    assert_eq!(moe.router_type, "top_k_softmax");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn moe_index_gate_knn_across_experts() {
    // Simulate a MoE layer: 2 experts × 3 features = 6 total features
    // Expert 0 features respond to dims 0,1,2
    // Expert 1 features respond to dim 3
    let hidden = 4;
    let features_per_expert = 3;
    let num_experts = 2;

    // Concatenate expert gate matrices (as build_vindex would)
    let mut gate0 = Array2::<f32>::zeros((num_experts * features_per_expert, hidden));
    // Expert 0
    gate0[[0, 0]] = 10.0; // E0F0 responds to dim 0
    gate0[[1, 1]] = 10.0; // E0F1 responds to dim 1
    gate0[[2, 2]] = 10.0; // E0F2 responds to dim 2
                          // Expert 1
    gate0[[3, 3]] = 10.0; // E1F0 responds to dim 3
    gate0[[4, 0]] = 5.0;
    gate0[[4, 3]] = 5.0; // E1F1 mixed
    gate0[[5, 1]] = 3.0; // E1F2 weak dim 1

    let gate_vectors = vec![Some(gate0)];

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),  // E0F0
        Some(make_meta("Berlin", 101, 0.92)), // E0F1
        Some(make_meta("Tokyo", 102, 0.88)),  // E0F2
        Some(make_meta("London", 103, 0.90)), // E1F0
        Some(make_meta("Rome", 104, 0.85)),   // E1F1
        Some(make_meta("Madrid", 105, 0.80)), // E1F2
    ];
    let down_meta = vec![Some(meta0)];

    let idx = VectorIndex::new(gate_vectors, down_meta, 1, hidden);
    assert_eq!(idx.num_features(0), 6); // 2 experts × 3 features

    // Query dim 0 → should match E0F0 (Paris) strongest
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 2);
    assert_eq!(hits[0].0, 0); // E0F0 = Paris
    assert_eq!(hits[1].0, 4); // E1F1 = Rome (has dim 0 component)

    // Query dim 3 → should match E1F0 (London) strongest
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = idx.gate_knn(0, &query, 2);
    assert_eq!(hits[0].0, 3); // E1F0 = London
    assert_eq!(hits[1].0, 4); // E1F1 = Rome (has dim 3 component)

    // Walk should find features across experts
    let query = Array1::from_vec(vec![0.5, 0.0, 0.0, 0.5]);
    let trace = idx.walk(&query, &[0], 3);
    let (_, hits) = &trace.layers[0];
    // Both E0F0 (Paris, dim0) and E1F0 (London, dim3) should appear
    let tokens: Vec<&str> = hits.iter().map(|h| h.meta.top_token.as_str()).collect();
    assert!(tokens.contains(&"Paris") || tokens.contains(&"London"));
}

#[test]
fn moe_layer_info_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_moe_layer_info");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-moe".into(),
        family: "mixtral".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: larql_vindex::LayerBands::for_family("mixtral", 32),
        layers: vec![VindexLayerInfo {
            layer: 0,
            num_features: 24, // 8 experts × 3 features
            offset: 0,
            length: 384,
            num_experts: Some(8),
            num_features_per_expert: Some(3),
        }],
        down_top_k: 10,
        has_model_weights: false,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "mixtral".into(),
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            rope_base: 1000000.0,
            sliding_window: None,
            moe: Some(larql_vindex::MoeConfig {
                num_experts: 8,
                top_k: 2,
                shared_expert: false,
                shared_expert_intermediate_size: None,
                router_type: "top_k_softmax".into(),
                moe_intermediate_size: None,
                hybrid: false,
            }),
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            layer_rope_theta: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
            ..Default::default()
        }),
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    // Verify MoE layer info round-trips
    assert_eq!(loaded.layers[0].num_experts, Some(8));
    assert_eq!(loaded.layers[0].num_features_per_expert, Some(3));
    assert_eq!(loaded.layers[0].num_features, 24);

    // Verify MoE config round-trips
    let moe = loaded.model_config.unwrap().moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn layer_bands_config_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_bands_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-bands".into(),
        family: "test".into(),
        num_layers: 34,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }),
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let bands = loaded.layer_bands.unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checksum_compute_and_verify() {
    let dir = std::env::temp_dir().join("larql_test_checksums");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write some test data
    std::fs::write(dir.join("gate_vectors.bin"), b"test gate data").unwrap();
    std::fs::write(dir.join("embeddings.bin"), b"test embed data").unwrap();
    std::fs::write(dir.join("down_meta.bin"), b"test down data").unwrap();

    // Compute checksums
    let checksums = larql_vindex::checksums::compute_checksums(&dir).unwrap();
    assert_eq!(checksums.len(), 3); // 3 files present
    assert!(checksums.contains_key("gate_vectors.bin"));
    assert!(checksums.contains_key("embeddings.bin"));
    assert!(checksums.contains_key("down_meta.bin"));

    // Verify — should all pass
    let results = larql_vindex::checksums::verify_checksums(&dir, &checksums).unwrap();
    assert!(results.iter().all(|(_, ok)| *ok));

    // Corrupt a file
    std::fs::write(dir.join("gate_vectors.bin"), b"corrupted!").unwrap();
    let results = larql_vindex::checksums::verify_checksums(&dir, &checksums).unwrap();
    let gate_result = results
        .iter()
        .find(|(f, _)| f == "gate_vectors.bin")
        .unwrap();
    assert!(!gate_result.1); // should fail

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checksum_individual_file() {
    let dir = std::env::temp_dir().join("larql_test_sha256");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("test.bin"), b"hello world").unwrap();
    let hash = larql_vindex::checksums::sha256_file(&dir.join("test.bin")).unwrap();
    // SHA256 of "hello world" is known
    assert_eq!(
        hash,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_level_serialization() {
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::Browse), "browse");
    assert_eq!(
        format!("{}", larql_vindex::ExtractLevel::Inference),
        "inference"
    );
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::All), "all");

    // serde round-trip
    let json = serde_json::to_string(&larql_vindex::ExtractLevel::Inference).unwrap();
    assert_eq!(json, "\"inference\"");
    let parsed: larql_vindex::ExtractLevel = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, larql_vindex::ExtractLevel::Inference);
}

#[test]
fn extract_level_default_is_browse() {
    let level: larql_vindex::ExtractLevel = Default::default();
    assert_eq!(level, larql_vindex::ExtractLevel::Browse);
}

#[test]
fn label_source_display() {
    assert_eq!(format!("{}", larql_vindex::LabelSource::Probe), "probe");
    assert_eq!(format!("{}", larql_vindex::LabelSource::Cluster), "cluster");
    assert_eq!(format!("{}", larql_vindex::LabelSource::Pattern), "pattern");
    assert_eq!(format!("{}", larql_vindex::LabelSource::None), "");
}

#[test]
fn describe_edge_construction() {
    let edge = larql_vindex::DescribeEdge {
        relation: Some("capital".into()),
        source: larql_vindex::LabelSource::Probe,
        target: "Paris".into(),
        gate_score: 1436.9,
        layer_min: 27,
        layer_max: 27,
        count: 1,
        also_tokens: vec![],
    };
    assert_eq!(edge.relation.as_deref(), Some("capital"));
    assert_eq!(edge.source, larql_vindex::LabelSource::Probe);
    assert_eq!(edge.target, "Paris");
}
