use super::*;

#[test]
#[serial_test::serial]
fn streaming_extract_mixtral_exercises_moe_arms() {
    // Tiny dims chosen so each FFN row pads to a clean Q4_K boundary if
    // the test ever extends to quant=Q4K. For now we extract f32 at
    // Browse level — covers gate / down_meta / router / index_json.
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
        "test/mixtral-synthetic",
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
    .expect("streaming extract on mixtral fixture");

    // ── Outputs the MoE arms must produce ───────────────────────
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(
        output_dir.join("router_weights.bin").exists(),
        "MoE arm must write router_weights.bin (router_weights.rs whole body)"
    );
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());
    assert!(output_dir.join("index.json").exists());

    // ── index.json carries MoE config (index_json.rs MoE branch) ──
    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    let model_cfg = config.model_config.expect("model_config present");
    let moe = model_cfg.moe.expect("MoE config recorded");
    assert_eq!(moe.num_experts, num_experts);
    assert_eq!(moe.top_k, num_experts_per_tok);

    // ── layer_infos record per-expert geometry (gate_vectors arm) ──
    assert_eq!(config.layers.len(), num_layers);
    for layer_info in &config.layers {
        assert_eq!(layer_info.num_experts, Some(num_experts));
        assert_eq!(layer_info.num_features_per_expert, Some(intermediate));
        // Total = num_experts × intermediate.
        assert_eq!(layer_info.num_features, num_experts * intermediate);
    }

    // ── router_weights.bin shape: per-layer router (+ optional bias) ──
    // Each router is `num_experts × hidden` f32 = 4 floats × 4 bytes = 16 B.
    // Two layers → ≥ 32 B (more if biases happened to be present).
    let router_bytes = std::fs::metadata(output_dir.join("router_weights.bin"))
        .unwrap()
        .len();
    let min_expected = (num_layers * num_experts * hidden * 4) as u64;
    assert!(
        router_bytes >= min_expected,
        "router_weights.bin {router_bytes} B < expected {min_expected} B"
    );
}
