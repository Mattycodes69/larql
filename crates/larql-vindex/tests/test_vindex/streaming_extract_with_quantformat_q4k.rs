//! streaming_extract with QuantFormat::Q4K

//
// End-to-end coverage for `write_model_weights_kquant`:
//   - Manifest shape: attn has 4 entries per layer, FFN has 3;
//     V and down carry Q6_K, everything else Q4_K.
//   - Offsets tile start-to-end with no gaps.
//   - `config.quant = Q4K` and `has_model_weights = true` land in
//     `index.json` so loaders can dispatch without sniffing files.
//   - The non-Q4 `attn_weights.bin` / `interleaved.bin` are absent.
#[test]
fn streaming_extract_q4k_from_safetensors() {
    use larql_vindex::QuantFormat;
    use std::collections::HashMap;

    let model_dir = std::env::temp_dir().join("larql_test_streaming_q4k_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_q4k_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    std::fs::create_dir_all(&model_dir).unwrap();

    // Small llama config — dims chosen so each tensor pads to exactly
    // one 256-element Q4_K/Q6_K super-block (256 elems = 2×128 or 8×32
    // or 16×16). Hidden=8 keeps padding overhead visible; the padder
    // zero-fills to the next 256-multiple.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;

    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    push(
        &mut tensors,
        &mut metadata,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.norm.weight",
        vec![hidden],
    );

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Attention: Q/K/V/O all [hidden, hidden]
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        // FFN: gate [inter, hidden], up [inter, hidden], down [hidden, inter]
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        // Norms
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    // Run with QuantFormat::Q4K — also verifies the Browse-level auto-
    // promotion to "all" that the streaming extractor applies when
    // quant != None.
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming-q4k",
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

    // ── File layout ──
    //
    // Writers emit the new kquant-canonical filenames. Legacy q4k-named
    // files must NOT be written (they're read-only back-compat fallbacks).
    assert!(output_dir.join("attn_weights_kquant.bin").exists());
    assert!(output_dir
        .join("attn_weights_kquant_manifest.json")
        .exists());
    assert!(output_dir.join("interleaved_kquant.bin").exists());
    assert!(output_dir.join("interleaved_kquant_manifest.json").exists());
    assert!(output_dir.join("norms.bin").exists());
    assert!(output_dir.join("weight_manifest.json").exists());
    assert!(output_dir.join("index.json").exists());
    assert!(
        !output_dir.join("attn_weights_q4k.bin").exists(),
        "writer must NOT emit the legacy q4k filename"
    );
    assert!(
        !output_dir.join("interleaved_q4k.bin").exists(),
        "writer must NOT emit the legacy q4k filename"
    );

    // k-quant path writes its own filenames; the non-quantised names should be absent.
    assert!(
        !output_dir.join("attn_weights.bin").exists(),
        "k-quant path should not emit attn_weights.bin"
    );

    // ── Config schema ──
    let cfg = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(cfg.num_layers, num_layers);
    assert_eq!(cfg.quant, QuantFormat::Q4K, "config.quant must be Q4K");
    assert!(
        cfg.has_model_weights,
        "config.has_model_weights must flip true"
    );

    // ── attn manifest ──
    let attn_manifest_json =
        std::fs::read_to_string(output_dir.join("attn_weights_kquant_manifest.json")).unwrap();
    let attn_entries: Vec<serde_json::Value> = serde_json::from_str(&attn_manifest_json).unwrap();

    // 4 tensors (Q, K, V, O) × num_layers
    assert_eq!(
        attn_entries.len(),
        num_layers * 4,
        "attn manifest should have 4N entries (Q/K/V/O per layer)"
    );

    // Per-layer slot order: Q=Q4_K, K=Q4_K, V=Q6_K, O=Q4_K.
    // Offsets must chain start-to-end with no gaps.
    let mut expected_offset: u64 = 0;
    for (i, entry) in attn_entries.iter().enumerate() {
        let slot = i % 4;
        let format = entry["format"].as_str().unwrap();
        let expected_format = if slot == 2 { "Q6_K" } else { "Q4_K" };
        assert_eq!(
            format, expected_format,
            "entry {i} slot {slot}: expected {expected_format}, got {format}"
        );
        let offset = entry["offset"].as_u64().unwrap();
        assert_eq!(offset, expected_offset, "offsets must tile with no gaps");
        let length = entry["length"].as_u64().unwrap();
        assert!(length > 0, "each entry must carry bytes");
        expected_offset += length;
    }

    // ── interleaved (FFN) manifest ──
    let ff_manifest_json =
        std::fs::read_to_string(output_dir.join("interleaved_kquant_manifest.json")).unwrap();
    let ff_entries: Vec<serde_json::Value> = serde_json::from_str(&ff_manifest_json).unwrap();

    // 3 tensors (gate, up, down) × num_layers
    assert_eq!(
        ff_entries.len(),
        num_layers * 3,
        "FFN manifest should have 3N entries (gate/up/down per layer)"
    );

    // Per-layer slot order: gate=Q4_K, up=Q4_K, down=Q6_K.
    let mut expected_offset: u64 = 0;
    for (i, entry) in ff_entries.iter().enumerate() {
        let slot = i % 3;
        let format = entry["format"].as_str().unwrap();
        let expected_format = if slot == 2 { "Q6_K" } else { "Q4_K" };
        assert_eq!(
            format, expected_format,
            "FFN entry {i} slot {slot}: expected {expected_format}, got {format}"
        );
        let offset = entry["offset"].as_u64().unwrap();
        assert_eq!(
            offset, expected_offset,
            "FFN offsets must tile with no gaps"
        );
        expected_offset += entry["length"].as_u64().unwrap();
    }

    // ── manifest byte counts match file sizes ──
    let attn_bytes = std::fs::metadata(output_dir.join("attn_weights_kquant.bin"))
        .unwrap()
        .len();
    let attn_manifest_total: u64 = attn_entries
        .iter()
        .map(|e| e["length"].as_u64().unwrap())
        .sum();
    assert_eq!(
        attn_bytes, attn_manifest_total,
        "attn_weights_kquant.bin size must equal sum of manifest lengths"
    );

    let ff_bytes = std::fs::metadata(output_dir.join("interleaved_kquant.bin"))
        .unwrap()
        .len();
    let ff_manifest_total: u64 = ff_entries
        .iter()
        .map(|e| e["length"].as_u64().unwrap())
        .sum();
    assert_eq!(
        ff_bytes, ff_manifest_total,
        "interleaved_kquant.bin size must equal sum of manifest lengths"
    );

    // ── load_model_weights on a Q4K vindex must surface a clear error ──
    // The float-weight loader can't reconstruct a ModelWeights struct
    // from Q4_K/Q6_K blocks; callers must go through
    // `VectorIndex::load_attn_kquant` / `load_interleaved_kquant` instead.
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    match larql_vindex::load_model_weights(&output_dir, &mut lcb) {
        Ok(_) => panic!("load_model_weights on a Q4K vindex must error"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("quantised") && msg.contains("load_attn_kquant"),
                "expected quant-dispatch error, got: {msg}"
            );
        }
    }

    // ── VectorIndex::load_attn_kquant + load_interleaved_kquant must read
    //     back what the writer emitted ──
    let mut index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut lcb).unwrap();
    index.load_attn_kquant(&output_dir).unwrap();
    index.load_interleaved_kquant(&output_dir).unwrap();
    assert!(
        index.has_interleaved_kquant(),
        "interleaved Q4K should be loaded"
    );
    // Layer 0 attn slices: [Q/Q4_K, K/Q4_K, V/Q6_K, O/Q4_K]
    let slices = index.attn_kquant_layer_data(0).expect("layer 0 attn data");
    assert_eq!(slices[0].1, "Q4_K", "Q slot format");
    assert_eq!(slices[1].1, "Q4_K", "K slot format");
    assert_eq!(slices[2].1, "Q6_K", "V slot format");
    assert_eq!(slices[3].1, "Q4_K", "O slot format");

    // ── Write-side correctness: dequantize the bytes the writer emitted
    //     and confirm they round-trip back to the source within block
    //     error tolerance. Proves the writer's manifest → data
    //     correspondence is correct (not just a shape assertion).
    //
    // Source data for every tensor: (0..n).map(|i| i as f32 * 0.01).
    // Q/K/V/O are hidden×hidden = 64 elems each, zero-padded to 256.
    //
    // Block-level error on a 64-value-then-192-zero-padded 256-value
    // super-block: ~0.02 for Q4_K and ~0.006 for Q6_K on this linear
    // ramp. Use 0.03 / 0.01 as ceilings — loose enough for the
    // quantiser's block allocation on this padding-heavy synthetic
    // case, tight enough to catch a manifest that points at the wrong
    // bytes (which would produce garbage orders of magnitude worse).
    let expected: Vec<f32> = (0..(hidden * hidden)).map(|i| (i as f32) * 0.01).collect();

    // The writer's `pad_rows_to_256` zero-extends each row from `hidden`
    // to 256 cols before quantising, so the dequantised output is a
    // [hidden × 256] padded matrix, not a flat copy of `expected`.
    // Map (row, col) of the original to the padded layout for comparison.
    let padded_cols = 256;
    let padded_at = |row: usize, col: usize| -> usize { row * padded_cols + col };

    let q_dequant =
        larql_models::quant::ggml::dequantize_q4_k(slices[0].0, hidden * padded_cols).unwrap();
    for row in 0..hidden {
        for col in 0..hidden {
            let i = row * hidden + col;
            let v = expected[i];
            let got = q_dequant[padded_at(row, col)];
            assert!(
                (got - v).abs() < 0.03,
                "Q[r{row} c{col}] round-trip diverged: got {got}, expected {v}",
            );
        }
        // Per-row zero pad: cols [hidden..256] should dequantise near zero
        // (within block error — the row's value range sets the scale).
        for col in hidden..padded_cols {
            let got = q_dequant[padded_at(row, col)];
            assert!(
                got.abs() < 0.05,
                "Q padding[r{row} c{col}] expected ~0, got {got}",
            );
        }
    }

    let v_dequant =
        larql_models::quant::ggml::dequantize_q6_k(slices[2].0, hidden * padded_cols).unwrap();
    for row in 0..hidden {
        for col in 0..hidden {
            let i = row * hidden + col;
            let v = expected[i];
            let got = v_dequant[padded_at(row, col)];
            assert!(
                (got - v).abs() < 0.01,
                "V[r{row} c{col}] round-trip diverged (Q6_K): got {got}, expected {v}",
            );
        }
    }

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

#[test]
fn quant_block_format_serde_roundtrip() {
    // The manifest format strings are load-bearing — llama.cpp / Ollama
    // expect the literal "Q4_K" and "Q6_K" on the wire. The enum uses
    // #[serde(rename)] to keep those strings; a future refactor must
    // not drift to e.g. "Q4K" without also updating every reader.
    use larql_vindex::format::weights::write_kquant::QuantBlockFormat;
    let q4 = serde_json::to_string(&QuantBlockFormat::Q4K).unwrap();
    let q6 = serde_json::to_string(&QuantBlockFormat::Q6K).unwrap();
    assert_eq!(q4, "\"Q4_K\"");
    assert_eq!(q6, "\"Q6_K\"");

    let parsed: QuantBlockFormat = serde_json::from_str("\"Q4_K\"").unwrap();
    assert_eq!(parsed, QuantBlockFormat::Q4K);
    let parsed: QuantBlockFormat = serde_json::from_str("\"Q6_K\"").unwrap();
    assert_eq!(parsed, QuantBlockFormat::Q6K);
}
