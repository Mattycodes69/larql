//! PLE tensors survive --quant none extract → load round-trip

use super::*;

//
// The companion regression test to `streaming_extract_q4k_carries_ple_tensors`
// for the `--quant none` / `--quant f16` extract path. Q4_K already wrote
// the PLE sidecar correctly; the float writer silently dropped every PLE
// tensor (the bug from chrishayuk/larql#49). After the fix, both writers
// route through `weights::ple_sidecar::write_ple_weights` and the float
// loader validates the invariant on load, refusing to surface a vindex that
// would silently produce garbage INFER output. ExtractLevel must be
// Inference (not Browse) because non-Q4 extracts only write model weights
// when the level demands attn — see `maybe_write_model_weights`.
#[test]
fn streaming_extract_noquant_carries_ple_tensors() {
    use larql_vindex::QuantFormat;

    let model_dir = std::env::temp_dir().join("larql_test_streaming_noquant_ple_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_noquant_ple_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);

    let hidden = 256usize;
    let intermediate = 256usize;
    let num_layers = 2usize;
    let vocab = 256usize;
    let ple_dim = 256usize;

    write_gemma4_ple_fixture(&model_dir, num_layers, hidden, intermediate, vocab, ple_dim);
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming-noquant-ple",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
        // Inference (not Browse): non-Q4 only writes model weights when
        // the level includes attn.
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None,
        &mut cb,
    )
    .unwrap();

    // ── ple_weights.bin must exist and the manifest must list all 3
    //     global + (2 per-layer) PLE tensor_f16 entries. Identical
    //     layout to the Q4_K case because both writers call into the
    //     shared `weights::ple_sidecar`. ──
    assert!(
        output_dir.join("ple_weights.bin").exists(),
        "non-Q4 extract should emit ple_weights.bin when the arch has PLE (#49)"
    );

    let manifest_json = std::fs::read_to_string(output_dir.join("weight_manifest.json")).unwrap();
    let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_json).unwrap();
    let ple_tensor_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "tensor_f16")
        .filter_map(|e| e["key"].as_str())
        .collect();

    assert_eq!(
        ple_tensor_keys.len(),
        2 + 2 * num_layers,
        "expected {} PLE tensor_f16 entries, got: {:?}",
        2 + 2 * num_layers,
        ple_tensor_keys
    );
    assert!(
        ple_tensor_keys.contains(&"per_layer_model_projection.weight"),
        "global model projection missing from manifest"
    );
    assert!(
        ple_tensor_keys.contains(&"embed_tokens_per_layer.weight"),
        "global per-layer embed missing from manifest"
    );

    // ── PLE norms (per-layer + global) must land in norms.bin as
    //     vector entries. ──
    let ple_vector_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "vector")
        .filter_map(|e| e["key"].as_str())
        .filter(|k| k.contains("per_layer"))
        .collect();
    assert!(
        ple_vector_keys.contains(&"per_layer_projection_norm.weight"),
        "global PLE norm missing from norms.bin manifest: {ple_vector_keys:?}"
    );
    for layer in 0..num_layers {
        let k = format!("layers.{layer}.post_per_layer_input_norm.weight");
        assert!(
            ple_vector_keys.iter().any(|v| *v == k),
            "layer {layer} post-PLE norm missing: {ple_vector_keys:?}"
        );
    }

    // ── layer_scalar (Gemma-4 only) must also land in norms.bin. ──
    let scalar_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "vector")
        .filter_map(|e| e["key"].as_str())
        .filter(|k| k.contains("layer_scalar"))
        .collect();
    assert_eq!(
        scalar_keys.len(),
        num_layers,
        "expected one layer_scalar entry per layer, got: {scalar_keys:?}"
    );

    // ── Load back via the float loader and verify every PLE tensor +
    //     vector is hydrated. The loader now validates this invariant
    //     and would refuse to return Ok if anything is missing. ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights(&output_dir, &mut lcb).unwrap();

    let proj = weights
        .tensors
        .get("per_layer_model_projection.weight")
        .expect("per_layer_model_projection missing after load");
    assert_eq!(proj.shape(), &[ple_dim * num_layers, hidden]);

    let embed_ple = weights
        .tensors
        .get("embed_tokens_per_layer.weight")
        .expect("embed_tokens_per_layer missing after load");
    assert_eq!(embed_ple.shape(), &[vocab, ple_dim * num_layers]);

    for layer in 0..num_layers {
        let gate_key = format!("layers.{layer}.per_layer_input_gate.weight");
        let proj_key = format!("layers.{layer}.per_layer_projection.weight");
        let gate = weights
            .tensors
            .get(&gate_key)
            .unwrap_or_else(|| panic!("{gate_key} missing"));
        assert_eq!(gate.shape(), &[ple_dim, hidden]);
        let proj = weights
            .tensors
            .get(&proj_key)
            .unwrap_or_else(|| panic!("{proj_key} missing"));
        assert_eq!(proj.shape(), &[hidden, ple_dim]);

        let norm_key = format!("layers.{layer}.post_per_layer_input_norm.weight");
        assert!(
            weights.vectors.contains_key(&norm_key),
            "{norm_key} missing from weights.vectors"
        );

        let scalar_key = format!("layers.{layer}.layer_scalar");
        assert!(
            weights.vectors.contains_key(&scalar_key),
            "{scalar_key} missing from weights.vectors"
        );
    }

    assert!(
        weights
            .vectors
            .contains_key("per_layer_projection_norm.weight"),
        "global PLE norm missing from loaded weights.vectors"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}
