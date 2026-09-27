//! GGUF-backed streaming extract

use super::*;

#[test]
fn streaming_extract_gguf_llama_browse_runs_end_to_end() {
    let num_layers = 2usize;
    let vocab = 24usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("gguf_model");
    std::fs::create_dir_all(&model_dir).unwrap();
    // A directory containing exactly one `.gguf` — exercises the
    // dir-scan branch of `detect_gguf_entry` as well as the GGUF arms.
    write_synthetic_llama_gguf(&model_dir.join("model.gguf"), num_layers, vocab);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let output_dir = tmp.path().join("vindex");
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/llama-gguf",
        &output_dir,
        5, // down_top_k
        0, // summary_features_per_expert (off)
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None, // expert_banks_out
        &mut cb,
    )
    .expect("streaming extract on GGUF llama fixture");

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(
        config.layers.len(),
        num_layers,
        "GGUF extract should record one layer_info per block"
    );
    assert!(
        output_dir.join("gate_vectors.bin").exists(),
        "GGUF extract should write gate_vectors.bin"
    );
    assert!(
        output_dir.join("embeddings.bin").exists(),
        "GGUF extract should write embeddings.bin"
    );
}

#[test]
fn streaming_extract_gguf_single_file_path_is_accepted() {
    // Point `build_vindex_streaming` directly at the `.gguf` file rather
    // than its parent dir — exercises the single-file arm of
    // `detect_gguf_entry` through the full pipeline.
    let tmp = tempfile::tempdir().unwrap();
    let gguf_path = tmp.path().join("solo.gguf");
    write_synthetic_llama_gguf(&gguf_path, 1, 16);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let output_dir = tmp.path().join("vindex");
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &gguf_path,
        &tokenizer,
        "test/llama-gguf-solo",
        &output_dir,
        3,
        0,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None,
        &mut cb,
    )
    .expect("streaming extract on single-file GGUF");

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.layers.len(), 1);
}

#[test]
fn streaming_extract_drop_gate_without_q4k_is_rejected() {
    // `--drop-gate-vectors` is only recoverable when interleaved Q4K is
    // also written; with `QuantFormat::None` the orchestrator must refuse
    // before touching the output dir.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");
    let tokenizer = write_synthetic_mixtral_model(&model_dir, 8, 4, 1, 2, 1, 16);

    let mut cb = SilentBuildCallbacks;
    let err = build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/drop-gate-bad",
        &output_dir,
        5,
        0,
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None, // not Q4K — drop_gate is invalid here
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        true,
        larql_vindex::ExtractionRequest::Legacy,
        None, // expert_banks_out
        &mut cb,
    )
    .expect_err("drop_gate_vectors without Q4K must be rejected");

    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("drop-gate-vectors") || msg.contains("q4k"),
        "unexpected error message: {msg}"
    );
}
