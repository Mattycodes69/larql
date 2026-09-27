//! Vindexfile parse + build test
//! HuggingFace path tests

use super::*;

#[test]
fn vindexfile_parse_and_build() {
    let base_dir = std::env::temp_dir().join("larql_test_vindexfile_base");
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir).unwrap();

    // Save a base vindex (with tokenizer for binary down_meta loading)
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(base_dir.join("tokenizer.json"), tok_json).unwrap();

    let index = test_index();
    let mut config = VindexConfig {
        version: 2,
        model: "test/vindexfile".into(),
        family: "llama".into(),
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 10,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        has_model_weights: false,
        layer_bands: None,
        layers: vec![],
        down_top_k: 5,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    index.save_vindex(&base_dir, &mut config).unwrap();

    // Create a patch
    let patch_dir = std::env::temp_dir().join("larql_test_vindexfile_patch");
    let _ = std::fs::remove_dir_all(&patch_dir);
    std::fs::create_dir_all(&patch_dir).unwrap();

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test/vindexfile".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some("test".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "PATCHED".into(),
                top_token_id: 999,
                c_score: 9.0,
            }),
        }],
    };
    let patch_path = patch_dir.join("test.vlp");
    patch.save(&patch_path).unwrap();

    // Build from Vindexfile
    let vf_content = format!(
        "FROM {}\nPATCH {}\n",
        base_dir.display(),
        patch_path.display()
    );
    let vf = larql_vindex::vindexfile::parse_vindexfile_str(&vf_content).unwrap();
    let result = larql_vindex::build_from_vindexfile(&vf, None, &std::env::temp_dir()).unwrap();

    // Patched feature should have the update applied
    assert_eq!(result.index.feature_meta(0, 0).unwrap().top_token_id, 999);
    assert!((result.index.feature_meta(0, 0).unwrap().c_score - 9.0).abs() < 0.01);
    // Unpatched feature should remain (token_id preserved through binary round-trip)
    assert_eq!(result.index.feature_meta(0, 1).unwrap().top_token_id, 101);
    assert_eq!(result.layers.len(), 2);

    let _ = std::fs::remove_dir_all(&base_dir);
    let _ = std::fs::remove_dir_all(&patch_dir);
}

#[test]
fn hf_path_detection() {
    assert!(larql_vindex::is_hf_path(
        "hf://chrishayuk/gemma-3-4b-it-vindex"
    ));
    assert!(larql_vindex::is_hf_path("hf://user/repo@v2.0"));
    assert!(!larql_vindex::is_hf_path("./local.vindex"));
    assert!(!larql_vindex::is_hf_path("/absolute/path"));
    assert!(!larql_vindex::is_hf_path("google/gemma-3-4b-it"));
}

#[test]
fn hf_path_with_revision() {
    let path = "hf://chrishayuk/gemma-3-4b-it-vindex@v2.0";
    assert!(larql_vindex::is_hf_path(path));
    let stripped = path.strip_prefix("hf://").unwrap();
    let (repo, rev) = stripped.split_once('@').unwrap();
    assert_eq!(repo, "chrishayuk/gemma-3-4b-it-vindex");
    assert_eq!(rev, "v2.0");
}

#[test]
fn hf_resolve_invalid_path_fails() {
    // Non-hf:// path should fail
    let result = larql_vindex::resolve_hf_vindex("./not-an-hf-path");
    assert!(result.is_err());
}
