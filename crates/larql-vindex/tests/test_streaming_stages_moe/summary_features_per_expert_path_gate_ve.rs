//! summary-features-per-expert path (gate_vectors SVD + down_meta cap)

use super::*;

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_resumes_when_run_twice_on_same_output_dir() {
    // First-pass extract writes outputs + clears the checkpoint on
    // success. To exercise the resumed_* branches at the head of each
    // streaming stage, we pre-seed the output dir with a checkpoint
    // marking every phase complete before the second pass — then re-run.
    // Each stage's `resumed_*` early-return branch fires, and the
    // pre-existing output files are left untouched.
    use larql_vindex::extract::{Checkpoint, ExtractPhase};

    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    // ── First pass: build the full vindex (and have all output files
    //    on disk so the resume path doesn't choke).
    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-resume",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
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
    .expect("first-pass extract");
    let first_down_meta_size = std::fs::metadata(output_dir.join("down_meta.bin"))
        .unwrap()
        .len();

    // ── Seed a checkpoint that marks every phase complete. This is the
    //    on-disk state an extractor would observe after crashing right
    //    before the final cleanup step. Forces the second pass to hit
    //    the resumed_* branches in every stage.
    let mut cp = Checkpoint::default();
    for phase in [ExtractPhase::Gate, ExtractPhase::DownMeta] {
        cp.mark(phase, &output_dir).expect("seed checkpoint");
    }
    assert!(output_dir.join(".extract_checkpoint.json").exists());

    // ── Second pass on the SAME output_dir — hits resumed_* paths.
    let mut cb2 = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-resume",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::None,
        WriteWeightsOptions::default(),
        KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None,
        &mut cb2,
    )
    .expect("second-pass extract reuses checkpoint");
    let second_down_meta_size = std::fs::metadata(output_dir.join("down_meta.bin"))
        .unwrap()
        .len();
    // Resumed phases don't re-write the file, so size is identical.
    assert_eq!(first_down_meta_size, second_down_meta_size);
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_with_real_tokenizer_records_top_k_entries() {
    // Same fixture as the baseline, but with a richer tokenizer so
    // down_meta can actually decode token IDs to strings. Exercises the
    // `TopKEntry` construction inside `down_meta::write_down_meta`'s
    // inner loop (lines 192-213) that was previously skipped because
    // every decode returned an empty string.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model_with_real_tokenizer(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-real-tok",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
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
    .expect("streaming extract with real tokenizer");

    assert!(output_dir.join("down_meta.bin").exists());
    let bytes = std::fs::metadata(output_dir.join("down_meta.bin"))
        .unwrap()
        .len();
    assert!(
        bytes > 0,
        "down_meta.bin must be non-empty when at least one feature decodes a token"
    );
}

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_with_drop_gate_vectors_removes_zero_byte_file() {
    // `drop_gate_vectors: true` requires `quant: Q4K` (the gate is
    // rebuilt from Q4K weights at load time). The streaming extractor
    // still walks the gate loop to populate `layer_infos` for
    // index.json, but pipes bytes to /dev/null and removes the
    // zero-byte gate_vectors.bin afterward.
    //
    // This exercises the `GateSink::Discard` path + the cleanup at
    // the end of `write_gate_vectors`.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-drop-gate",
        &output_dir,
        5,
        0, // summary_features_per_expert (off)
        ExtractLevel::Browse,
        StorageDtype::F32,
        QuantFormat::Q4K, // required by drop_gate_vectors
        WriteWeightsOptions {
            level: ExtractLevel::Browse,
            ffn_compact: false,
            skip_attn: false,
            skip_ffn: false,
        },
        KquantWriteOptions::default(),
        true,
        larql_vindex::ExtractionRequest::Legacy,
        None, // expert_banks_out
        &mut cb,
    )
    .expect("streaming extract with drop_gate_vectors=true");

    // gate_vectors.bin should have been removed (was zero bytes).
    assert!(
        !output_dir.join("gate_vectors.bin").exists(),
        "drop_gate_vectors=true must clean up the empty file"
    );

    // index.json still records per-layer geometry (the gate loop walked
    // every layer to populate layer_infos).
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.layers.len(), num_layers);
}

#[test]
fn streaming_extract_mixtral_with_summary_k_runs_svd_and_caps_down_meta() {
    // Same Mixtral fixture as the baseline test, but with
    // `summary_features_per_expert = 2`. Triggers:
    //   - the SVD-summary path in `gate_vectors.rs` Standard-MoE branch
    //     (writes K rows per expert instead of full intermediate=4)
    //   - the down_meta `summary_k`-cap branch (truncates `num_features`
    //     to K=2 per expert)
    //
    // Output assertions confirm both paths fired by checking that
    // num_features_per_expert collapses from intermediate=4 to K=2 in
    // the recorded layer_infos.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;
    let summary_k = 2usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_mixtral_model(
        &model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/mixtral-summary-k",
        &output_dir,
        5,
        summary_k, // summary_features_per_expert (SVD-summary tier)
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
    .expect("streaming extract on mixtral fixture with summary_k");

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        // SVD path: num_features_per_expert collapses to K, regardless
        // of original intermediate.
        assert_eq!(layer_info.num_features_per_expert, Some(summary_k));
        assert_eq!(layer_info.num_features, num_experts * summary_k);
    }

    // gate_vectors.bin now stores K floats per expert per layer instead
    // of `intermediate`. Same hidden width (=8 floats/row).
    let gate_bytes = std::fs::metadata(output_dir.join("gate_vectors.bin"))
        .unwrap()
        .len();
    let expected = (num_layers * num_experts * summary_k * hidden * 4) as u64;
    assert_eq!(
        gate_bytes, expected,
        "gate_vectors.bin sized for K rows × hidden, not intermediate"
    );
}
