//! PLE tensors survive Q4_K extract → load round-trip

use super::*;

//
// Regression test for the Gemma 4 E2B "predict returns garbage on
// Q4K vindex" bug: the extractor used to drop the six Per-Layer
// Embedding tensors, so `precompute_per_layer_inputs` silently
// returned an empty Vec and PLE was never applied. Extraction now
// writes `ple_weights.bin` (Q4_K-packed tensors) plus the two small
// PLE norms into norms.bin. This test builds a Gemma 4-shaped
// synthetic safetensors, runs the real extract pipeline, loads via
// `load_model_weights_kquant`, and asserts every PLE tensor is back in
// `weights.tensors` / `weights.vectors` with the right shape.
#[test]
fn streaming_extract_q4k_carries_ple_tensors() {
    use larql_vindex::QuantFormat;

    let model_dir = std::env::temp_dir().join("larql_test_streaming_q4k_ple_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_q4k_ple_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);

    // E2B-shaped config at a test-friendly scale. `hidden_size_per_layer_input`
    // is the knob `has_per_layer_embeddings()` keys off, so it must be present
    // AND non-zero for the extractor to hit the PLE path. Gemma 4 uses the
    // text_config wrapper; detect_from_json handles that.
    let hidden = 256usize; // multiple of 256 so Q/K/V/O skip the padder
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
        "test/streaming-q4k-ple",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None,
        &mut cb,
    )
    .unwrap();

    // ── ple_weights.bin must exist and the manifest must list all 3
    //     global + (2 per-layer) PLE tensor entries as `tensor_q4k`. ──
    assert!(
        output_dir.join("ple_weights.bin").exists(),
        "Q4 extract should emit ple_weights.bin when the arch has PLE"
    );

    let manifest_json = std::fs::read_to_string(output_dir.join("weight_manifest.json")).unwrap();
    let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_json).unwrap();
    // PLE tensors are stored as f16 (not Q4_K) — Q4_K's per-super-block
    // calibration zeros out the non-outlier cells of embedding-style
    // tensors, compounding to garbage across Gemma 4 E2B's 35 layers.
    let ple_tensor_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "tensor_f16")
        .filter_map(|e| e["key"].as_str())
        .collect();

    // 2 global tensors (per_layer_model_projection, embed_tokens_per_layer)
    // + 2 per-layer tensors × num_layers. per_layer_projection_norm is a
    // vector and belongs in norms.bin, not here.
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

    // ── post_per_layer_input_norm + per_layer_projection_norm must land
    //     in norms.bin as vector entries. ──
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

    // ── Load back and verify the dequantised PLE tensors surface in
    //     weights.tensors with the expected shapes. ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&output_dir, &mut lcb).unwrap();

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
    }

    // Norms land in weights.vectors (f32 raw).
    assert!(
        weights
            .vectors
            .contains_key("per_layer_projection_norm.weight"),
        "global PLE norm missing from loaded weights.vectors"
    );

    // Gemma-4 layer_scalar: a per-layer 0-D scalar surfaced as a 1-element
    // vector. The forward path multiplies h by this value after FFN; omitting
    // it silently produced garbage on the 31B model. Previously uncovered
    // even on the Q4_K path — same failure mode shape as the PLE drop in #49,
    // so we pin it here too.
    for layer in 0..num_layers {
        let key = format!("layers.{layer}.layer_scalar");
        assert!(
            weights.vectors.contains_key(&key),
            "layer {layer} layer_scalar missing from loaded weights.vectors"
        );
    }

    // final_logit_softcapping must survive the round-trip. Missing it
    // lets predict_kquant peak the softmax on the wrong token.
    let cfg = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(
        cfg.model_config
            .as_ref()
            .and_then(|m| m.final_logit_softcapping),
        Some(30.0),
        "final_logit_softcapping dropped from vindex model_config"
    );
    assert_eq!(
        weights.arch.final_logit_softcapping(),
        Some(30.0),
        "loaded arch must surface the softcap via final_logit_softcapping()"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}
