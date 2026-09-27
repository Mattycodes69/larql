//! Synthetic fixtures for the streaming-extract pipeline.
//!
//! Hand-built, deterministic, in-process — no HuggingFace, no large
//! model downloads. Fixtures write a tempdir tree (safetensors:
//! `config.json` + `tokenizer.json` + `model.safetensors`; GGUF: a
//! single `model.gguf`) shaped like a real architecture and drive
//! [`larql_vindex::build_vindex_streaming`] against it.
//!
//! Coverage targets across `extract::streaming::{mod, context, stages}`
//! that the dense Llama fixture in `test_vindex.rs` doesn't reach:
//!
//! - `gate_vectors::write_gate_vectors` / `down_meta::write_down_meta` —
//!   standard MoE arms (Mixtral happy path)
//! - `router_weights::write_router_weights` — whole body (early-returns
//!   on dense; only fires when `is_moe`)
//! - `index_json::write_index_json` — MoE config branch + has-experts
//!   per-layer tracking
//! - the **GGUF arms**: arch detection in `streaming::mod`, the
//!   `GgufTensorSource` branch of `context::new`, and `detect_gguf_entry`
//!   (single-file, multi-shard, largest-fallback) — driven by a hand-
//!   built `GgufWriter` llama model
//! - `down_meta` **edge arms**: missing-tensor `continue`s (dense and
//!   per-expert MoE down absent), the resume-skip path (checkpoint marks
//!   down_meta complete), and the `TopKEntry` keep arm (full-vocab
//!   tokenizer so the down argmax decodes to a non-empty token)

use std::collections::HashMap;
use std::path::Path;

use larql_vindex::{
    build_vindex_streaming, ExtractLevel, KquantWriteOptions, QuantFormat, SilentBuildCallbacks,
    StorageDtype, WriteWeightsOptions,
};

/// Build a tiny Mixtral-shaped model (block-sparse MoE FFN with
/// `num_experts` experts per layer). Deterministic per-tensor ramps
/// so two runs against the same dims produce identical vindexes.
///
/// Returns the in-memory tokenizer so callers can drive
/// `build_vindex_streaming` without re-reading the JSON file.
fn write_synthetic_mixtral_model(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    write_synthetic_mixtral_model_opts(
        model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
        true,
    )
}

/// As [`write_synthetic_mixtral_model`], but `include_expert_down = false`
/// omits every expert's `w2` (down) tensor. With no per-expert down
/// matrices to gather, `down_meta`'s MoE arm produces an empty
/// `down_matrices` and takes its `is_empty()` skip branch for each layer.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_mixtral_model_opts(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab: usize,
    include_expert_down: bool,
) -> larql_vindex::tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        "num_local_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    // Embedding + final norm.
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard Llama-style attention.
        push(
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Block-sparse MoE: router + per-expert gate (w1) / down (w2) / up (w3).
        push(
            &format!("{lp}.block_sparse_moe.gate.weight"),
            vec![num_experts, hidden],
        );
        for e in 0..num_experts {
            let ep = format!("{lp}.block_sparse_moe.experts.{e}");
            push(&format!("{ep}.w1.weight"), vec![intermediate, hidden]);
            if include_expert_down {
                push(&format!("{ep}.w2.weight"), vec![hidden, intermediate]);
            }
            push(&format!("{ep}.w3.weight"), vec![intermediate, hidden]);
        }
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
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();

    // Minimal BPE tokenizer — enough for safetensors-backed extracts
    // that don't need to encode strings.
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Build a tiny Gemma 4 26B-A4B-shaped hybrid MoE model: dense MLP +
/// per-layer expert block, with experts stored packed as the
/// `experts.gate_up_proj` / `experts.down_proj` BF16 tensor pair.
///
/// `extract::streaming::stages::gate_vectors` and `down_meta` both have
/// dedicated `PackedBF16 + is_moe` arms that route through the **dense**
/// MLP gate/down for KNN routing while leaving the packed tensors
/// untouched (the q4k writer consumes those later). This fixture
/// exercises that route end-to-end.
///
/// Note: the dense FFN keys overlap with Llama's, but the runtime
/// dispatch picks the PackedBF16 arm because Gemma4Arch advertises
/// `expert_format() == ExpertFormat::PackedBF16` whenever
/// `enable_moe_block=true`.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_gemma4_hybrid_moe(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    moe_intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    std::fs::create_dir_all(model_dir).unwrap();

    // Gemma 4 detection: model_type that starts with "gemma4". We use
    // a flat config (no `text_config` nesting) — `detect_from_json`
    // falls back to the top level when `text_config` is absent.
    let config = serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        // Hybrid MoE flag — flips Gemma4Arch.is_moe / is_hybrid_moe / expert_format.
        "enable_moe_block": true,
        "num_experts": num_experts,
        "top_k_experts": num_experts_per_token,
        "moe_intermediate_size": moe_intermediate,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    // Globals
    push("model.embed_tokens.weight", vec![vocab, hidden]);
    push("model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard Llama-style attention (Gemma 4 inherits this).
        push(
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(&format!("{lp}.input_layernorm.weight"), vec![hidden]);
        // Hybrid MoE renames post_feedforward_layernorm → _1 (dense
        // branch); the streaming pipeline doesn't need it for Browse
        // level, but the loader at the end will look for it.
        push(
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Dense MLP — both branches coexist in hybrid MoE; gate_vectors'
        // PackedBF16 arm reads from here.
        push(
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        // Packed expert tensors — only consumed by the q4k writer at
        // QuantFormat::Q4K. With QuantFormat::None they're present but
        // unused by Browse-level extraction.
        push(
            &format!("{lp}.experts.gate_up_proj"),
            vec![num_experts, 2 * moe_intermediate, hidden],
        );
        push(
            &format!("{lp}.experts.down_proj"),
            vec![num_experts, hidden, moe_intermediate],
        );
        // Router (hybrid MoE: `router.proj` not `gate.weight`).
        push(
            &format!("{lp}.router.proj.weight"),
            vec![num_experts, hidden],
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
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Build a tiny gpt-oss-shaped MoE model: experts packed as MXFP4
/// (e8m0 scales + 4-bit nibbles), gate and up projections fused into
/// one tensor pair (`gate_up_proj_blocks` + `gate_up_proj_scales`),
/// down projections in a separate pair.
///
/// `extract::streaming::stages::gate_vectors` and `down_meta` both have
/// dedicated `PackedMxfp4` arms that:
/// 1. Find the packed tensor pair via `arch.packed_gate_up_blocks_key` /
///    `arch.packed_down_blocks_key`.
/// 2. Deserialise safetensors directly to read the U8 byte payload.
/// 3. Call `format::quant::mxfp4::dequantize_all_experts` to recover
///    f32 expert matrices.
/// 4. (gate) Slice the first half of each expert's rows as the gate
///    portion and write to `gate_vectors.bin`.
/// 5. (down_meta) Use the recovered down matrices for embed-projection
///    top-K extraction.
///
/// Block byte payload is all-zero (MXFP4 nibble 0 dequantises to 0.0)
/// and scales are all `127` (e8m0 → scale=1.0). The decoded tensors
/// are therefore zero-filled but the *dispatch path* runs end-to-end,
/// which is what we're after for coverage.
///
/// MXFP4 constraint: `in_features = groups × 32`, so dimensions must
/// be multiples of 32. We use `hidden = intermediate = 32` (groups=1)
/// for the smallest possible payload.
fn write_synthetic_gpt_oss_model(
    model_dir: &Path,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_token: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    // Fixed dims chosen to satisfy MXFP4's `in_features = groups × 32`
    // constraint with the smallest-possible groups=1.
    let hidden = 32usize;
    let intermediate = 32usize;
    let groups = 1usize; // hidden / 32
    let groups_down = 1usize; // intermediate / 32
    let out_features_gate_up = 2 * intermediate; // fused gate+up: 64

    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_token,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    // Track tensors as either F32 or U8. Each entry is
    // (name, dtype, shape, raw_bytes).
    let mut entries: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    let push_f32 = |entries: &mut Vec<_>, name: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        entries.push((name, safetensors::Dtype::F32, shape, bytes));
    };

    let push_u8 = |entries: &mut Vec<_>, name: String, shape: Vec<usize>, fill: u8| {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = vec![fill; n];
        entries.push((name, safetensors::Dtype::U8, shape, bytes));
    };

    // Globals
    push_f32(
        &mut entries,
        "model.embed_tokens.weight".into(),
        vec![vocab, hidden],
    );
    push_f32(&mut entries, "model.norm.weight".into(), vec![hidden]);

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Standard attention.
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push_f32(
            &mut entries,
            format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        // Router (gpt-oss: `mlp.router.weight`, NOT `block_sparse_moe.gate`).
        push_f32(
            &mut entries,
            format!("{lp}.mlp.router.weight"),
            vec![num_experts, hidden],
        );

        // Packed MXFP4 expert blocks (U8) + e8m0 scales (U8). All-zero
        // blocks → MXFP4_TABLE[0]=0.0 → zero-filled dequant. Scale 127
        // → e8m0 → 1.0. The dequantize path runs end-to-end either way.
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.gate_up_proj_blocks"),
            vec![num_experts, out_features_gate_up, groups, 16],
            0,
        );
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.gate_up_proj_scales"),
            vec![num_experts, out_features_gate_up, groups],
            127, // e8m0 = 1.0
        );
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.down_proj_blocks"),
            vec![num_experts, hidden, groups_down, 16],
            0,
        );
        push_u8(
            &mut entries,
            format!("{lp}.mlp.experts.down_proj_scales"),
            vec![num_experts, hidden, groups_down],
            127,
        );
    }

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = entries
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

// The summary tier is now a `build_vindex_streaming` parameter
// (`summary_features_per_expert`), passed directly per call — no
// process-global env, so these tests need no serialisation.

/// Build a Mixtral fixture with a tokenizer that has a non-empty vocab,
/// so `down_meta` actually decodes some `token_id → string` and exercises
/// the `TopKEntry` construction branch (lines 200-204) instead of having
/// every decode return an empty string.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_mixtral_model_with_real_tokenizer(
    model_dir: &Path,
    hidden: usize,
    intermediate: usize,
    num_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab: usize,
) -> larql_vindex::tokenizers::Tokenizer {
    // Reuse the model-side fixture (config + safetensors); only the
    // tokenizer side gets the richer vocab.
    let _ = write_synthetic_mixtral_model(
        model_dir,
        hidden,
        intermediate,
        num_layers,
        num_experts,
        num_experts_per_tok,
        vocab,
    );

    // Populate the BPE `vocab` map for *every* ID so `decode(&[id], true)`
    // returns a printable, non-empty token for whichever feature the
    // down-projection argmax selects. This exercises the `TopKEntry`
    // keep arm in `down_meta` (the `.map(|token| TopKEntry { .. })` that
    // earlier capped at ID 8 never reached, because the argmax always
    // landed on a higher ID). The complementary empty-string skip path
    // stays covered by the empty-tokenizer fixtures above.
    let vocab_entries: Vec<String> = (0..vocab).map(|i| format!("\"tok{i}\":{i}")).collect();
    let tok_json = format!(
        r#"{{"version":"1.0","model":{{"type":"BPE","vocab":{{{}}},"merges":[]}},"added_tokens":[]}}"#,
        vocab_entries.join(",")
    );
    std::fs::write(model_dir.join("tokenizer.json"), &tok_json).unwrap();
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

// Every fixture above is safetensors-backed. This section drives
// `build_vindex_streaming` against a hand-built GGUF model so the GGUF
// arms exercise end-to-end: arch detection in `streaming::mod`
// (`GgufFile::open` → `to_config_json` → `detect_from_json`) and the
// `GgufTensorSource` setup branch in `streaming::context::new`.

use larql_models::loading::gguf::{GgufTensor, GgufValue, GgufWriter};

/// Deterministic f32 ramp → little-endian bytes (non-degenerate so the
/// down-projection argmax isn't a tie across the whole vocab).
fn gguf_f32_ramp(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect()
}

/// Write a tiny but complete llama-architecture GGUF model.
///
/// FFN is square (`hidden == intermediate == 4`) on purpose: the
/// canonical FFN orientation in `GgufTensorSource::get_tensor_f32`
/// becomes a no-op (`orient` short-circuits when rows == cols), so the
/// synthetic data can't trip a transpose mismatch. Tensors use GGUF
/// naming (`blk.L.ffn_gate.weight`); the source adapter maps them back
/// to HF keys via `normalize_gguf_key`.
fn write_synthetic_llama_gguf(path: &Path, num_layers: usize, vocab: usize) {
    write_synthetic_llama_gguf_opts(path, num_layers, vocab, true);
}

/// As [`write_synthetic_llama_gguf`], but `include_ffn_down = false`
/// omits the `blk.L.ffn_down.weight` tensors — so `down_meta`'s dense
/// arm hits its missing-tensor `continue` (the down projection is
/// skipped for every layer).
fn write_synthetic_llama_gguf_opts(
    path: &Path,
    num_layers: usize,
    vocab: usize,
    include_ffn_down: bool,
) {
    const DIM: u64 = 4; // hidden == intermediate
    let v = vocab as u64;

    let mut w = GgufWriter::new();
    w.meta("general.architecture", GgufValue::String("llama".into()))
        .meta("llama.embedding_length", GgufValue::U32(DIM as u32))
        .meta("llama.block_count", GgufValue::U32(num_layers as u32))
        .meta("llama.feed_forward_length", GgufValue::U32(DIM as u32))
        .meta("llama.attention.head_count", GgufValue::U32(2))
        .meta("llama.attention.head_count_kv", GgufValue::U32(2))
        .meta("llama.attention.key_length", GgufValue::U32(2))
        .meta("llama.rope.freq_base", GgufValue::F32(10000.0));

    // GGUF dims are innermost-first: [hidden, vocab] reshapes to the
    // Array2 (vocab, hidden) the embeddings stage expects.
    w.tensor(GgufTensor {
        name: "token_embd.weight".into(),
        dims: vec![DIM, v],
        ggml_type: 0, // GGML_TYPE_F32
        data: gguf_f32_ramp((DIM * v) as usize),
    });
    w.tensor(GgufTensor {
        name: "output.weight".into(),
        dims: vec![DIM, v],
        ggml_type: 0,
        data: gguf_f32_ramp((DIM * v) as usize),
    });
    w.tensor(GgufTensor {
        name: "output_norm.weight".into(),
        dims: vec![DIM],
        ggml_type: 0,
        data: gguf_f32_ramp(DIM as usize),
    });
    for layer in 0..num_layers {
        w.tensor(GgufTensor {
            name: format!("blk.{layer}.ffn_gate.weight"),
            dims: vec![DIM, DIM],
            ggml_type: 0,
            data: gguf_f32_ramp((DIM * DIM) as usize),
        });
        if include_ffn_down {
            w.tensor(GgufTensor {
                name: format!("blk.{layer}.ffn_down.weight"),
                dims: vec![DIM, DIM],
                ggml_type: 0,
                data: gguf_f32_ramp((DIM * DIM) as usize),
            });
        }
    }
    w.write_to_file(path).unwrap();
}

mod down_meta_edge_arms_missing_tensor_resum;
mod gemma_4_hybrid_moe_packedbf16_expert_for;
mod gguf_backed_streaming_extract;
mod streaming_stages_moe_basics;
mod summary_features_per_expert_path_gate_ve;
