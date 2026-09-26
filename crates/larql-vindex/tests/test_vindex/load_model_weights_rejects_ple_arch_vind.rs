//! load_model_weights rejects PLE-arch vindexes with missing sidecars

use super::*;

//
// Pre-#49, dropping PLE tensors on the writer side was paired with a
// silent "missing tensor → return empty" in the forward path, so an
// affected vindex loaded fine but produced garbage INFER. The fix moves
// the failure into the load path: any PLE-active vindex whose
// `weights.tensors` / `weights.vectors` don't carry the required PLE
// keys is rejected with an actionable rebuild hint. This test stages
// that failure by writing the vindex normally, then nuking the manifest
// entries for the PLE tensors before calling `load_model_weights` —
// exercises the error branch in `load/f32.rs` end-to-end.
#[test]
fn load_model_weights_rejects_ple_arch_with_missing_sidecars() {
    use larql_vindex::QuantFormat;

    let model_dir = std::env::temp_dir().join("larql_test_ple_missing_sidecar_model");
    let output_dir = std::env::temp_dir().join("larql_test_ple_missing_sidecar_output");
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
        "test/ple-missing-sidecar",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
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

    // Simulate a stale (pre-fix) vindex: drop the manifest entries that
    // would otherwise hydrate the PLE tensors at load time, then re-write
    // the manifest. The bytes in `ple_weights.bin` / `norms.bin` are left
    // alone — only the manifest table changes, which is exactly what the
    // pre-fix writer did (it never wrote the entries in the first place).
    let manifest_path = output_dir.join("weight_manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
    let mut entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text).unwrap();
    entries.retain(|e| {
        let key = e["key"].as_str().unwrap_or("");
        !key.contains("per_layer")
            && !key.contains("embed_tokens_per_layer")
            && !key.contains("layer_scalar")
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&entries).unwrap(),
    )
    .unwrap();

    let mut lcb = larql_vindex::SilentLoadCallbacks;
    // `ModelWeights` doesn't implement Debug, so use a match here
    // instead of `Result::expect_err` (which requires T: Debug).
    let err = match larql_vindex::load_model_weights(&output_dir, &mut lcb) {
        Ok(_) => panic!("load must reject PLE-arch vindex with missing sidecars"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("Gemma-4 PLE sidecar"),
        "error must call out the PLE sidecars — got: {msg}"
    );
    assert!(
        msg.contains("chrishayuk/larql#49"),
        "error must point at the issue for the rebuild hint — got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}
