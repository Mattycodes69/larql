//! Safetensors writer + config/tokenizer copy logic for compiled checkpoints.
//!
//! The skip patterns drop vision/multimodal tensors so the output is
//! a text-only language model. Tied lm_head is dropped when `embed_tokens` is
//! present, matching HuggingFace's tied-embedding convention.

use larql_vindex::format::filenames::*;
use std::collections::HashMap;
use std::path::Path;

use ndarray::ArcArray2;

use larql_models::ModelWeights;

pub const SKIP_PATTERNS: &[&str] = &[
    "vision_tower",
    "multi_modal_projector",
    "vision_model",
    "image_projection",
];

pub struct MergedWeights {
    pub tensors: HashMap<String, ArcArray2<f32>>,
    pub vectors: HashMap<String, Vec<f32>>,
}

/// Merge `modified` 2D tensors over the original weight set, drop multimodal
/// tensors, and dedup tied lm_head/embed_tokens. 1D vectors pass through unchanged.
pub fn merge_for_save(
    weights: &ModelWeights,
    modified: HashMap<String, ArcArray2<f32>>,
) -> MergedWeights {
    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    for (k, v) in &weights.tensors {
        if SKIP_PATTERNS.iter().any(|p| k.contains(p)) {
            continue;
        }
        tensors.insert(k.clone(), v.clone());
    }
    for (k, v) in modified {
        tensors.insert(k, v);
    }

    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    for (k, v) in &weights.vectors {
        if SKIP_PATTERNS.iter().any(|p| k.contains(p)) {
            continue;
        }
        vectors.insert(k.clone(), v.clone());
    }

    if tensors.contains_key("model.embed_tokens.weight") && tensors.contains_key("lm_head.weight") {
        tensors.remove("lm_head.weight");
    }

    MergedWeights { tensors, vectors }
}

/// Write tensors as bf16 — Gemma / Llama / most modern transformers' native
/// dtype. Halves file size vs f32 (~15 GB → ~7.8 GB on Gemma 3 4B).
///
/// Uses `larql_models::quant::half::encode_bf16` which does the standard
/// `f32 → bf16` truncation (keep top 16 bits, round-to-nearest-even on the
/// dropped mantissa via hardware semantics). Round-trip through our own
/// `decode_bf16` is bit-exact for the subset of f32 values bf16 can represent,
/// which is the regime the trained weights + our compile-installed edges
/// both live in.
pub fn write_safetensors(
    tensors: &HashMap<String, ArcArray2<f32>>,
    vectors: &HashMap<String, Vec<f32>>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_models::quant::half::encode_bf16;
    use safetensors::tensor::{serialize, TensorView};

    let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for (name, arr) in tensors {
        let shape = arr.shape().to_vec();
        // Tensors from safetensors loading are row-major contiguous; use
        // as_slice when possible, fall back to iterator collect otherwise.
        let owned: Vec<f32>;
        let slice: &[f32] = match arr.as_slice() {
            Some(s) => s,
            None => {
                owned = arr.iter().copied().collect();
                &owned
            }
        };
        byte_bufs.insert(name.clone(), encode_bf16(slice));
        shapes.insert(name.clone(), shape);
    }

    for (name, vec) in vectors {
        if tensors.contains_key(name) {
            continue;
        }
        let bytes = encode_bf16(vec);
        byte_bufs.insert(name.clone(), bytes);
        shapes.insert(name.clone(), vec![vec.len()]);
    }

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, bytes) in &byte_bufs {
        let shape = &shapes[name];
        views.insert(
            name.clone(),
            TensorView::new(safetensors::Dtype::BF16, shape.clone(), bytes)?,
        );
    }

    let serialized = serialize(&views, None)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

/// Suffix of a Hugging Face multimodal wrapper class and of the text-only
/// causal-LM class it wraps (`XForConditionalGeneration` → `XForCausalLM`).
const WRAPPER_CLASS_SUFFIX: &str = "ForConditionalGeneration";
const CAUSAL_LM_CLASS_SUFFIX: &str = "ForCausalLM";

/// Copy tokenizer files and write config.json so the output stands alone as
/// a text-only checkpoint (multimodal tensors were skipped above).
///
/// A plain config is copied as-is: the compiled model is the same
/// architecture as its base. A multimodal wrapper (one with `text_config`)
/// is unwrapped to its text config, keeping that config's own
/// `model_type`; `architectures` and `tie_word_embeddings` are carried over
/// from the source, never invented.
pub fn copy_model_config(base: &Path, output: &Path) -> std::io::Result<()> {
    for name in &[
        TOKENIZER_JSON,
        TOKENIZER_CONFIG_JSON,
        "special_tokens_map.json",
        GENERATION_CONFIG_JSON,
        "tokenizer.model", // SentencePiece model — required by llama.cpp's GGUF converter
    ] {
        let src = base.join(name);
        if src.exists() {
            std::fs::copy(&src, output.join(name))?;
        }
    }

    let config_src = base.join("config.json");
    if !config_src.exists() {
        return Ok(());
    }
    let text = std::fs::read_to_string(&config_src)?;
    let cfg: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let out = text_only_config(&cfg);
    let body = serde_json::to_string_pretty(&out)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(output.join("config.json"), body)
}

/// The text-only config for `cfg`: `cfg` itself unless it wraps a
/// `text_config`.
pub fn text_only_config(cfg: &serde_json::Value) -> serde_json::Value {
    let Some(text_cfg) = cfg.get("text_config").and_then(|t| t.as_object()) else {
        return cfg.clone();
    };
    let mut out = text_cfg.clone();
    if !out.contains_key("architectures") {
        let text_classes: Vec<serde_json::Value> = cfg
            .get("architectures")
            .and_then(|a| a.as_array())
            .into_iter()
            .flatten()
            .filter_map(|c| c.as_str()?.strip_suffix(WRAPPER_CLASS_SUFFIX))
            .map(|stem| serde_json::json!(format!("{stem}{CAUSAL_LM_CLASS_SUFFIX}")))
            .collect();
        if !text_classes.is_empty() {
            out.insert("architectures".into(), text_classes.into());
        }
    }
    if let (false, Some(tie)) = (
        out.contains_key("tie_word_embeddings"),
        cfg.get("tie_word_embeddings"),
    ) {
        out.insert("tie_word_embeddings".into(), tie.clone());
    }
    serde_json::Value::Object(out)
}

#[cfg(test)]
mod tests;
