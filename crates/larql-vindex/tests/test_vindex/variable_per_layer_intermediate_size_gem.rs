//! Variable per-layer intermediate size (Gemma 4 E2B double-wide MLP)

//
// E2B's `use_double_wide_mlp=True` gives half the layers a 2× intermediate
// dimension (6144 → 12288 on the real model). `predict_kquant` previously
// hardcoded `weights.intermediate_size` for every layer's FFN dequant,
// so the wide layers' weights were read at half-size and the forward
// pass computed garbage. Fix: read per-layer feature count from the
// vindex via `VectorIndex::num_features(layer)`. This test locks the
// invariant that num_features matches the real per-layer shape so the
// fix stays honest.
#[test]
fn streaming_extract_preserves_per_layer_intermediate_for_variable_ffn() {
    use larql_vindex::QuantFormat;
    use std::collections::HashMap;

    let model_dir = std::env::temp_dir().join("larql_test_variable_ffn_model");
    let output_dir = std::env::temp_dir().join("larql_test_variable_ffn_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    std::fs::create_dir_all(&model_dir).unwrap();

    let hidden = 256usize;
    let num_layers = 4usize;
    let vocab = 256usize;
    // Layers 0,1 narrow (256), layers 2,3 double-wide (512). Matches the
    // E2B pattern: the last half of the stack doubles the FFN width.
    let intermediates = [256usize, 256, 512, 512];
    let max_intermediate = *intermediates.iter().max().unwrap();

    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": max_intermediate,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
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
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
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

    for (layer, &inter) in intermediates.iter().enumerate() {
        let lp = format!("model.layers.{layer}");
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
        // Per-layer FFN width.
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![inter, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![inter, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, inter],
        );
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

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/variable-ffn",
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

    // ── Per-layer num_features in index.json ──
    let cfg = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(cfg.layers.len(), num_layers);
    for (layer, li) in cfg.layers.iter().enumerate() {
        assert_eq!(
            li.num_features, intermediates[layer],
            "layer {layer} num_features must equal source FFN intermediate"
        );
    }

    // ── VectorIndex::num_features(layer) — the accessor predict_kquant calls ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut lcb).unwrap();
    for (layer, &inter) in intermediates.iter().enumerate().take(num_layers) {
        assert_eq!(
            index.num_features(layer),
            inter,
            "VectorIndex::num_features(layer={layer}) wrong"
        );
    }

    // ── FFN manifest shape — the raw Q4K bytes must match the per-layer
    //     intermediate, NOT the model-wide max. Earlier predict_kquant bug:
    //     dequantising with the wrong width silently produced half-width
    //     weights on wide layers, so this assertion is the invariant. ──
    let ff_manifest_json =
        std::fs::read_to_string(output_dir.join("interleaved_kquant_manifest.json")).unwrap();
    let ff_entries: Vec<serde_json::Value> = serde_json::from_str(&ff_manifest_json).unwrap();
    for (layer, &inter) in intermediates.iter().enumerate() {
        let base = layer * 3; // gate, up, down per layer
        let gate_shape: Vec<usize> = ff_entries[base]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let up_shape: Vec<usize> = ff_entries[base + 1]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let down_shape: Vec<usize> = ff_entries[base + 2]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(gate_shape, vec![inter, hidden], "layer {layer} gate shape");
        assert_eq!(up_shape, vec![inter, hidden], "layer {layer} up shape");
        assert_eq!(down_shape, vec![hidden, inter], "layer {layer} down shape");
    }

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}
