//! down_meta edge arms (missing-tensor + resume skips)

use super::*;

#[test]
fn streaming_extract_dense_with_missing_ffn_down_skips_down_projection() {
    // Dense llama GGUF with no `blk.L.ffn_down.weight` — `down_meta`'s
    // dense arm must hit its missing-tensor `continue` for every layer
    // (no down projection), yet the extract still succeeds and writes a
    // config with the right layer count.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("gguf_model");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_synthetic_llama_gguf_opts(&model_dir.join("model.gguf"), 2, 16, false);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let output_dir = tmp.path().join("vindex");
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/llama-gguf-nodown",
        &output_dir,
        5,
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
    .expect("extract should succeed even when ffn_down is absent");

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.layers.len(), 2);
}

#[test]
#[serial_test::serial]
fn streaming_extract_moe_with_missing_expert_down_skips_layer() {
    // Mixtral fixture with no expert `w2` (down) tensors — `down_meta`'s
    // MoE arm gathers an empty `down_matrices` and takes the
    // `is_empty()` skip branch for every layer. The other stages
    // (gate / router / embeddings) still have what they need.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");
    let tokenizer = write_synthetic_mixtral_model_opts(
        &model_dir, 8, 4, 2, 2, 1, 16, /* include_expert_down = */ false,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-nodown",
        &output_dir,
        5,
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
    .expect("extract should succeed when expert down tensors are absent");

    // Gate still written; the run completes despite the empty down arm.
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(larql_vindex::load_vindex_config(&output_dir).is_ok());
}

#[test]
#[serial_test::serial]
fn streaming_extract_resumes_and_skips_down_meta_when_checkpoint_marks_it() {
    use larql_vindex::extract::{Checkpoint, ExtractPhase};

    // Full extract once, then plant a checkpoint that marks the down_meta
    // phase complete and re-run. The second run must take `write_down_meta`'s
    // resume-skip branch — `down_meta.bin` is reused byte-for-byte, never
    // recomputed.
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");
    let num_layers = 2usize;
    let tokenizer = write_synthetic_mixtral_model(&model_dir, 8, 4, num_layers, 2, 1, 16);
    let model_name = "test/resume-down-meta";

    let run = |out: &Path, tok: &larql_vindex::tokenizers::Tokenizer| {
        let mut cb = SilentBuildCallbacks;
        build_vindex_streaming(
            &model_dir,
            tok,
            model_name,
            out,
            5,
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
        .expect("streaming extract");
    };

    run(&output_dir, &tokenizer);
    let down_meta_before = std::fs::read(output_dir.join("down_meta.bin")).unwrap();

    // Plant a checkpoint marking *only* down_meta complete (the gate stage
    // re-runs fresh, so `layer_infos` is rebuilt for index.json — we just
    // want down_meta to be skipped). `mark` persists to disk.
    let mut cp = Checkpoint::fresh(&model_dir, model_name, num_layers);
    cp.mark(ExtractPhase::DownMeta, &output_dir).unwrap();
    assert!(cp.is_complete(ExtractPhase::DownMeta));

    run(&output_dir, &tokenizer);
    let down_meta_after = std::fs::read(output_dir.join("down_meta.bin")).unwrap();

    assert_eq!(
        down_meta_before, down_meta_after,
        "resumed down_meta.bin must be reused unchanged, not recomputed"
    );
}
