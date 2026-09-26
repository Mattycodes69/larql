//! Gemma 4 hybrid MoE (PackedBF16 expert format)
//! gpt-oss (PackedMxfp4 expert format)

use super::*;

#[test]
#[serial_test::serial]
fn streaming_extract_gemma4_hybrid_moe_exercises_packed_bf16_arms() {
    // Tiny dims; enable_moe_block flips Gemma4Arch into the hybrid
    // MoE configuration where expert_format == PackedBF16 and is_moe
    // is true, hitting the `PackedBF16 && is_moe` arms in
    // stages/gate_vectors.rs and stages/down_meta.rs.
    let hidden = 8usize;
    let intermediate = 4usize;
    let moe_intermediate = 4usize;
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_gemma4_hybrid_moe(
        &model_dir,
        hidden,
        intermediate,
        moe_intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/gemma4-hybrid-moe-synthetic",
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
    .expect("streaming extract on gemma4 hybrid MoE fixture");

    // Outputs the hybrid MoE arms must produce.
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(
        output_dir.join("router_weights.bin").exists(),
        "MoE arm must write router_weights.bin"
    );
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());

    // index.json carries hybrid-MoE config — exercises
    // `arch.is_hybrid_moe()` branch in write_index_json.
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.family, "gemma4");
    let model_cfg = config.model_config.expect("model_config present");
    let moe = model_cfg.moe.expect("MoE config recorded");
    assert!(moe.hybrid, "Gemma 4 26B A4B is hybrid MoE");
    assert_eq!(moe.num_experts, num_experts);
    assert_eq!(moe.top_k, num_experts_per_tok);
    assert_eq!(moe.moe_intermediate_size, Some(moe_intermediate));

    // The hybrid arm uses the dense FFN gate for routing (NOT per-expert
    // gate), so layer_infos.num_features should match the dense width
    // (`intermediate`), with no per-expert breakdown.
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(
            layer_info.num_features, intermediate,
            "hybrid MoE routes through dense gate (intermediate width)"
        );
        assert_eq!(layer_info.num_experts, None);
        assert_eq!(layer_info.num_features_per_expert, None);
    }
}

#[test]
#[serial_test::serial]
fn streaming_extract_gpt_oss_exercises_packed_mxfp4_arms() {
    let num_layers = 2usize;
    let num_experts = 2usize;
    let num_experts_per_tok = 1usize;
    let vocab = 16usize;

    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    let output_dir = tmp.path().join("vindex");

    let tokenizer = write_synthetic_gpt_oss_model(
        &model_dir,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    let mut cb = SilentBuildCallbacks;
    build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/gpt-oss-synthetic",
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
    .expect("streaming extract on gpt-oss MXFP4 fixture");

    // Outputs the PackedMxfp4 arm must produce.
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(output_dir.join("router_weights.bin").exists());
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.family, "gpt_oss");

    // num_features per layer = num_experts × (out_features_gate_up / 2)
    // = num_experts × intermediate (since out_features_gate_up = 2*intermediate).
    let intermediate = 32usize;
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(layer_info.num_experts, Some(num_experts));
        assert_eq!(layer_info.num_features_per_expert, Some(intermediate));
        assert_eq!(layer_info.num_features, num_experts * intermediate);
    }
}
