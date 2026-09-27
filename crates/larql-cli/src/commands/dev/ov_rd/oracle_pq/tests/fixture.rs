//! A synthetic, self-contained Q4K vindex small enough for an end-to-end
//! `oracle-pq` run in a unit test: a two-layer dense llama-shaped model
//! with deterministic pseudo-random weights, a word-level tokenizer, and a
//! stratified JSONL prompt file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::Dtype;

/// Q4_K packs 256-element super-blocks, so every row width is a multiple
/// of it and the writer never pads.
const HIDDEN: usize = 256;
const INTERMEDIATE: usize = 256;
const NUM_LAYERS: usize = 2;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;
/// Weight scale: small enough that the residual stream stays well
/// conditioned through two layers, large enough that attention and FFN
/// activations are not uniform.
const WEIGHT_SCALE: f32 = 0.08;

const WORDS: [&str; 40] = [
    "the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "park", "one", "two", "three",
    "plus", "equals", "four", "five", "six", "fn", "main", "let", "x", "return", "if", "else",
    "red", "blue", "green", "sky", "sea", "is", "was", "and", "or", "big", "small", "sun", "moon",
    "day", "night",
];

/// `(id, stratum, prompt)` rows. Strata cover the literals the probes
/// branch on (`natural_prose`, `arithmetic`) plus a third.
pub(super) const PROMPTS: [(&str, &str, &str); 12] = [
    (
        "p0",
        "natural_prose",
        "the cat sat on a mat and the dog ran",
    ),
    ("p1", "arithmetic", "one plus two equals three"),
    ("p2", "code", "fn main let x return x"),
    (
        "p3",
        "natural_prose",
        "the sky is blue and the sea is green",
    ),
    ("p4", "arithmetic", "two plus two equals four"),
    ("p5", "code", "if x return one else return two"),
    ("p6", "natural_prose", "a big dog ran to the park"),
    ("p7", "arithmetic", "three plus three equals six"),
    ("p8", "code", "let x equals five if x"),
    (
        "p9",
        "natural_prose",
        "the sun was red and the moon was small",
    ),
    ("p10", "arithmetic", "one plus four equals five"),
    ("p11", "code", "fn main return x or x"),
];

/// Deterministic values in `[-scale, scale)`: a 64-bit LCG seeded by the
/// tensor name, so every tensor differs and every run is identical.
fn pseudo_random(name: &str, n: usize, scale: f32) -> Vec<f32> {
    const LCG_MUL: u64 = 6364136223846793005;
    const LCG_ADD: u64 = 1442695040888963407;
    const MANTISSA_BITS: u32 = 24;
    let mut state = name.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    });
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
            let unit = (state >> (64 - MANTISSA_BITS)) as f32 / (1u64 << MANTISSA_BITS) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

fn vocab_size() -> usize {
    WORDS.len() + 1
}

fn write_config(model_dir: &Path) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size(),
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();
}

fn write_safetensors(model_dir: &Path) {
    let vocab = vocab_size();
    let mut shapes: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![vocab, HIDDEN]),
        ("lm_head.weight".into(), vec![vocab, HIDDEN]),
        ("model.norm.weight".into(), vec![HIDDEN]),
    ];
    for layer in 0..NUM_LAYERS {
        let lp = format!("model.layers.{layer}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            shapes.push((
                format!("{lp}.self_attn.{proj}.weight"),
                vec![HIDDEN, HIDDEN],
            ));
        }
        shapes.push((
            format!("{lp}.mlp.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        ));
        shapes.push((
            format!("{lp}.mlp.up_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        ));
        shapes.push((
            format!("{lp}.mlp.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
        ));
        shapes.push((format!("{lp}.input_layernorm.weight"), vec![HIDDEN]));
        shapes.push((
            format!("{lp}.post_attention_layernorm.weight"),
            vec![HIDDEN],
        ));
    }
    let bytes: HashMap<String, Vec<u8>> = shapes
        .iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values = if name.ends_with("norm.weight") {
                vec![1.0f32; n]
            } else {
                pseudo_random(name, n, WEIGHT_SCALE)
            };
            let raw = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            (name.clone(), raw)
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = shapes
        .iter()
        .map(|(name, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), &bytes[name]).unwrap();
            (name.clone(), view)
        })
        .collect();
    let blob = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), blob).unwrap();
}

fn word_tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    let vocab: HashMap<String, u32> = std::iter::once("[UNK]")
        .chain(WORDS.iter().copied())
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
}

/// Build the Q4K vindex under `root/vindex` and the prompt file at
/// `root/prompts.jsonl`; returns `(vindex_dir, prompts_path)`.
pub(super) fn build(root: &Path) -> (PathBuf, PathBuf) {
    let model_dir = root.join("model");
    let vindex_dir = root.join("vindex");
    std::fs::create_dir_all(&model_dir).unwrap();
    write_config(&model_dir);
    write_safetensors(&model_dir);
    let tokenizer = word_tokenizer();
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .unwrap();

    let mut callbacks = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/oracle-pq-fixture",
        &vindex_dir,
        5,
        0,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        larql_vindex::ExtractionRequest::Legacy,
        None,
        &mut callbacks,
    )
    .unwrap();
    let vindex_tokenizer = vindex_dir.join("tokenizer.json");
    if !vindex_tokenizer.exists() {
        tokenizer.save(&vindex_tokenizer, false).unwrap();
    }

    let prompts_path = root.join("prompts.jsonl");
    let lines = PROMPTS
        .iter()
        .map(|(id, stratum, prompt)| {
            serde_json::json!({ "id": id, "stratum": stratum, "prompt": prompt }).to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&prompts_path, lines).unwrap();
    (vindex_dir, prompts_path)
}
