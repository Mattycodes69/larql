//! GGUF tensor loading, config building, and entry points.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use crate::detect::{
    detect_from_json_validated, find_gguf_architecture, gguf_model_type, ModelError,
};
use crate::loading::lm_head::{resolve_lm_head, LM_HEAD_KEY};
use crate::weights::ModelWeights;

use super::constants::*;
use super::orient::{
    orient_attention_tensors, orient_embedding, orient_ffn_tensors, split_fused_qkv,
};
use super::types::GgufFile;

/// Sentinel suffix appended to a BitNet I2_S tensor's key under which
/// its per-tensor scale f32 is stashed in `ModelWeights::raw_bytes`
/// during a keep-quant load.  The packed-trit bytes live under the
/// bare key; the 4-byte little-endian scale lives under
/// `"{key}{I2S_SCALE_SUFFIX}"`.  The NUL byte guarantees the sentinel
/// can never collide with a real GGUF tensor name.
pub const I2S_SCALE_SUFFIX: &str = "\0i2s_scale";

impl GgufFile {
    /// Load all tensors, dequantizing to f32.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors(
        &self,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        self.load_tensors_filtered(&|_| false)
    }

    /// Load tensors, skipping normalized keys before reading/dequantizing tensor data.
    ///
    /// `skip_key` sees keys after GGUF-to-HF normalization but before architecture-specific
    /// prefix stripping. GGUF keys do not carry the HF wrapper prefixes, so this is enough for
    /// the current GGUF path and lets walk-only loading avoid FFN dequantization.
    ///
    /// Multi-shard models: tensors are read from `self.shards[info.shard_idx]`,
    /// which is mmap'd lazily on first use within this call. Shards that
    /// contain no surviving tensors after `skip_key` are not mmap'd at all.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        // Single source of truth: the keep-quant variant with no retained
        // types does exactly this (lazy per-shard mmap, normalize, bounds-check,
        // dequantize, dim-swap) and simply returns an empty raw-bytes map.
        let (tensors, vectors, _raw) = self.load_tensors_filtered_keep_quant(skip_key, &[])?;
        Ok((tensors, vectors))
    }

    /// As [`Self::load_tensors_filtered`] but also returns the raw
    /// pre-dequant bytes for tensors whose ggml type matches
    /// `keep_raw_for_types`.  Used by the `--keep-quant` convert
    /// path so I2_S BitLinear tensors can be written verbatim to
    /// the vindex (see `larql_vindex::extract::bitnet_writer`).
    ///
    /// The returned tensors map still contains the dequantised
    /// f32 view (for shape inspection downstream); the raw bytes
    /// are an additional sidecar.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered_keep_quant(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
        keep_raw_for_types: &[u32],
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
            HashMap<String, Vec<u8>>,
        ),
        ModelError,
    > {
        let mut shard_mmaps: Vec<Option<memmap2::Mmap>> =
            (0..self.shards.len()).map(|_| None).collect();

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

        let arch = self
            .metadata
            .get(GGUF_GENERAL_ARCHITECTURE)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        for info in &self.tensor_infos {
            let key = normalize_gguf_key_for_arch(&info.name, &arch);
            if skip_key(&key) {
                continue;
            }

            let shard = &self.shards[info.shard_idx];
            if shard_mmaps[info.shard_idx].is_none() {
                let f = std::fs::File::open(&shard.path)?;
                let m = unsafe { memmap2::Mmap::map(&f)? };
                shard_mmaps[info.shard_idx] = Some(m);
            }
            let mmap = shard_mmaps[info.shard_idx]
                .as_ref()
                .expect("mmap initialised above");

            let abs_offset = shard.data_offset.checked_add(info.offset).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: data_offset {} + tensor offset {} overflows u64",
                    info.name, shard.data_offset, info.offset,
                ))
            })?;
            let n_elements = info
                .dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    ModelError::Parse(format!(
                        "tensor {}: element count of dims {:?} overflows usize",
                        info.name, info.dims,
                    ))
                })?;

            let data_size = tensor_data_size(info.tensor_type, n_elements)?;
            let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
                ModelError::Parse(format!(
                    "tensor {}: absolute offset {} exceeds usize on this platform",
                    info.name, abs_offset,
                ))
            })?;
            let end = abs_offset_usize.checked_add(data_size).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: offset {} + size {} overflows usize",
                    info.name, abs_offset_usize, data_size,
                ))
            })?;
            if end > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > shard {} file {})",
                    info.name,
                    abs_offset,
                    data_size,
                    info.shard_idx,
                    mmap.len()
                )));
            }

            let raw = &mmap[abs_offset_usize..end];
            if keep_raw_for_types.contains(&info.tensor_type) {
                raw_bytes.insert(key.clone(), raw.to_vec());
                // BitNet I2_S stores a single per-tensor scale f32
                // (= max|W| at quant time) immediately AFTER the n/4
                // packed-trit bytes (microsoft/BitNet
                // ggml-bitnet-mad.cpp::quantize_i2_s writes
                // `scale_ptr[0]` at byte offset n/4).  tensor_data_size
                // returns only n/4, so the scale lives in the
                // 32-byte-aligned padding at [end, end+4).  Capture it
                // under a sentinel key so the keep-quant writer can set
                // BitLinearWeight.channel_scales without re-reading the
                // GGUF.  Reconstruction convention: W = trit * scale.
                if info.tensor_type == crate::quant::ggml::TYPE_I2_S {
                    if let Some(se) = end.checked_add(4) {
                        if se <= mmap.len() {
                            let sb = &mmap[end..se];
                            let scale = f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
                            raw_bytes.insert(
                                format!("{key}{I2S_SCALE_SUFFIX}"),
                                scale.to_le_bytes().to_vec(),
                            );
                        }
                    }
                }
            }
            let floats = dequantize(raw, info.tensor_type, n_elements)?;

            match info.n_dims {
                2 => {
                    let ne0 = info.dims[0] as usize;
                    let ne1 = info.dims[1] as usize;
                    let arr = Array2::from_shape_vec((ne1, ne0), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {}
            }
        }

        Ok((tensors, vectors, raw_bytes))
    }

    /// Build a config.json-equivalent from GGUF metadata for architecture detection.
    ///
    /// The flat mapping here is shared by every family; anything a family's
    /// export needs beyond it comes from that family's registry row
    /// ([`crate::detect::GgufTranslation`]).
    pub fn to_config_json(&self) -> serde_json::Value {
        let arch = self.architecture();
        let family = find_gguf_architecture(arch);

        let hidden_size = self.arch_u32(GGUF_EMBEDDING_LENGTH);
        let num_heads = self.arch_u32(GGUF_ATTENTION_HEAD_COUNT);
        let num_kv_heads = self.arch_u32(GGUF_ATTENTION_HEAD_COUNT_KV);
        let head_dim = match self.arch_u32(GGUF_ATTENTION_KEY_LENGTH) {
            0 => hidden_size.checked_div(num_heads).unwrap_or(0),
            key_length => key_length,
        };
        let num_kv_heads = if num_kv_heads > 0 {
            num_kv_heads
        } else {
            num_heads
        };

        // intermediate_size: prefer the global `feed_forward_length`. For
        // MoE-only models (DeepSeek-V4 family) the global key is omitted,
        // so we fall back to the per-expert size. The HF config exposes
        // `intermediate_size` as a single number even on MoE archs because
        // per-expert and per-layer FFNs share that dim in every
        // llama.cpp-supported architecture.
        let intermediate_size = match self.arch_u32(GGUF_FEED_FORWARD_LENGTH) {
            0 => self.arch_u32(GGUF_EXPERT_FEED_FORWARD_LENGTH),
            global => global,
        };
        let mut config = serde_json::json!({
            HF_MODEL_TYPE: gguf_model_type(arch),
            HF_HIDDEN_SIZE: hidden_size,
            HF_NUM_HIDDEN_LAYERS: self.arch_u32(GGUF_BLOCK_COUNT),
            HF_INTERMEDIATE_SIZE: intermediate_size,
            HF_NUM_ATTENTION_HEADS: num_heads,
            HF_NUM_KEY_VALUE_HEADS: num_kv_heads,
            HF_HEAD_DIM: head_dim,
        });

        if let Some(rope_base) = self.arch_f64(GGUF_ROPE_FREQ_BASE) {
            config[HF_ROPE_THETA] = serde_json::json!(rope_base);
        }
        if let Some(vocab_size) = self.arch_u32_opt(GGUF_VOCAB_SIZE).filter(|&v| v > 0) {
            config[HF_VOCAB_SIZE] = serde_json::json!(vocab_size);
        }

        if let Some(hook) = family.and_then(|entry| entry.gguf.config) {
            hook(self, &mut config);
        }

        // RMSNorm epsilon — llama.cpp emits it for every RMSNorm family
        // under the arch prefix. Absent → detect_from_json falls back to
        // its default.
        if let Some(eps) = self.arch_f64("attention.layer_norm_rms_epsilon") {
            config["rms_norm_eps"] = serde_json::json!(eps);
        }

        // ── MLA fields (DeepSeek-V2/V3 family, e.g. Kimi K2) ─────────────────
        // The HF config exposes `q_lora_rank` / `kv_lora_rank` /
        // `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim`. llama.cpp
        // emits the equivalent fields under the `{arch}.attention.*` and
        // `{arch}.rope.dimension_count` namespace; we surface them here so
        // the existing parser → `ModelConfig` path picks them up and MLA
        // absorption (PR #96) fires for GGUF-sourced inputs.
        //
        // For per-head dims we prefer the `_mla` variants when present —
        // those carry the pre-absorption (DeepSeek-V3 standard) split that
        // `mla_absorb::absorb()` operates on. The non-`_mla` keys can hold
        // post-absorption / "effective" widths (576/512 on Kimi K2.6) which
        // are too large to feed back into the absorption math.
        if let Some(q_lora) = self
            .arch_u32_opt(GGUF_ATTENTION_Q_LORA_RANK)
            .filter(|&v| v > 0)
        {
            config["q_lora_rank"] = serde_json::json!(q_lora);
        }
        if let Some(kv_lora) = self
            .arch_u32_opt(GGUF_ATTENTION_KV_LORA_RANK)
            .filter(|&v| v > 0)
        {
            config["kv_lora_rank"] = serde_json::json!(kv_lora);
        }
        let qk_rope = self
            .arch_u32_opt(GGUF_ROPE_DIMENSION_COUNT)
            .filter(|&v| v > 0);
        if let Some(rope) = qk_rope {
            config["qk_rope_head_dim"] = serde_json::json!(rope);
        }
        // qk_head_dim total: prefer key_length_mla, fall back to key_length.
        let key_length_mla = self
            .arch_u32_opt(GGUF_ATTENTION_KEY_LENGTH_MLA)
            .filter(|&v| v > 0);
        let key_length = self
            .arch_u32_opt(GGUF_ATTENTION_KEY_LENGTH)
            .filter(|&v| v > 0);
        let qk_head_dim = key_length_mla.or(key_length);
        if let (Some(qk_total), Some(rope)) = (qk_head_dim, qk_rope) {
            if qk_total > rope {
                config["qk_nope_head_dim"] = serde_json::json!(qk_total - rope);
            }
        }
        // v_head_dim: prefer value_length_mla, fall back to value_length.
        let v_head = self
            .arch_u32_opt(GGUF_ATTENTION_VALUE_LENGTH_MLA)
            .filter(|&v| v > 0)
            .or_else(|| {
                self.arch_u32_opt(GGUF_ATTENTION_VALUE_LENGTH)
                    .filter(|&v| v > 0)
            });
        if let Some(v) = v_head {
            config["v_head_dim"] = serde_json::json!(v);
        }

        config
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered(path, &|_| false)
}

/// Load a GGUF file into ModelWeights, retaining the original
/// pre-dequant bytes for tensors of the listed types.
///
/// Used by `larql convert gguf-to-vindex --keep-quant` so the
/// BitNet 1.58 I2_S BitLinear bytes survive into
/// `ModelWeights::raw_bytes` (rather than being dropped after
/// dequantization).  See BUG-infer-deadlock §5.4.
///
/// `keep_types` should be the list of GGML type IDs whose bytes
/// you want preserved.  For BitNet pass `&[36]` (TYPE_I2_S); for
/// future TQ1_0/TQ2_0 native paths pass `&[34, 35]`.
pub fn load_gguf_keep_quant(path: &Path, keep_types: &[u32]) -> Result<ModelWeights, ModelError> {
    load_gguf_keep_quant_filtered(path, &|_| false, keep_types)
}

pub(crate) fn load_gguf_keep_quant_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation_and_keep(path, skip_key, false, keep_types)
}

/// Load and validate a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf_validated(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation(path, &|_| false, true)
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation_and_keep(path, skip_key, false, &[])
}

/// Same as [`load_gguf_filtered_with_validation`] but also retains
/// raw pre-dequant bytes for tensors whose GGML type appears in
/// `keep_types`.
pub(crate) fn load_gguf_filtered_with_validation_and_keep(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    let config_json = gguf.to_config_json();
    let arch = if validate_config {
        detect_from_json_validated(&config_json)?
    } else {
        crate::detect_from_json(&config_json)
    };
    let prefixes = arch.key_prefixes_to_strip();

    let (mut tensors, mut vectors, mut raw_keep) =
        gguf.load_tensors_filtered_keep_quant(skip_key, keep_types)?;

    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_tensors.insert(key, v);
    }
    // Re-key the raw_bytes map through the same normalisation so
    // downstream consumers can look up by the canonical name.
    let mut normalized_raw: HashMap<String, Vec<u8>> = HashMap::new();
    for (k, v) in raw_keep.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_raw.insert(key, v);
    }

    orient_ffn_tensors(&mut normalized_tensors, &*arch);
    orient_attention_tensors(&mut normalized_tensors, &*arch);
    split_fused_qkv(&mut normalized_tensors, &mut vectors, &*arch);

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    let cfg = arch.config();
    let tokenizer_vocab_size = read_tokenizer_vocab_size(path);
    let configured_vocab_size = cfg.vocab_size.filter(|&v| v > 0);
    let expected_vocab_size = configured_vocab_size.or(tokenizer_vocab_size);
    let embed = orient_embedding(embed_raw, cfg.hidden_size, expected_vocab_size);

    let lm_head = resolve_lm_head(
        normalized_tensors
            .get(LM_HEAD_KEY)
            .or_else(|| normalized_tensors.get(GGUF_OUTPUT_WEIGHT)),
        &embed,
        &*arch,
    )?;
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| normalized_tensors.get(key).cloned());

    let vocab_size = expected_vocab_size
        .or_else(|| (embed.shape()[0] > 0).then_some(embed.shape()[0]))
        .ok_or_else(|| {
            ModelError::Parse(format!(
                "{}: no vocab_size in metadata or tokenizer, and {embed_key} has no rows",
                path.display()
            ))
        })?;

    let cfg_clone = cfg.clone();
    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        raw_bytes: normalized_raw,
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed,
        num_layers: cfg_clone.num_layers,
        hidden_size: cfg_clone.hidden_size,
        intermediate_size: cfg_clone.intermediate_size,
        vocab_size,
        head_dim: cfg_clone.head_dim,
        num_q_heads: cfg_clone.num_q_heads,
        num_kv_heads: cfg_clone.num_kv_heads,
        rope_base: cfg_clone.rope_base,
        arch,
    })
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered_with_validation(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
) -> Result<ModelWeights, ModelError> {
    // Identical to the keep-quant path with no retained types — that variant
    // runs the same detect/load/orient/embed pipeline and leaves `raw_bytes`
    // empty.  Single source of truth for the GGUF load orchestration.
    load_gguf_filtered_with_validation_and_keep(path, skip_key, validate_config, &[])
}

pub(super) fn read_tokenizer_vocab_size(path: &Path) -> Option<usize> {
    let parent = path.parent()?;
    let tok_path = parent.join(TOKENIZER_JSON);
    let data = std::fs::read_to_string(tok_path).ok()?;
    let json = serde_json::from_str::<serde_json::Value>(&data).ok()?;
    json[TOKENIZER_MODEL][TOKENIZER_VOCAB]
        .as_object()
        .map(|v| v.len())
        .filter(|&v| v > 0)
}

pub(super) fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    crate::quant::ggml::tensor_data_size(tensor_type, n_elements)
}

pub(super) fn dequantize(
    data: &[u8],
    tensor_type: u32,
    n_elements: usize,
) -> Result<Vec<f32>, ModelError> {
    crate::quant::ggml::dequantize(data, tensor_type, n_elements)
}

/// Normalize GGUF tensor key names to match HuggingFace conventions.
pub fn normalize_gguf_key(name: &str) -> String {
    // GGUF uses "blk.N.attn_q.weight" format
    // HF uses "model.layers.N.self_attn.q_proj.weight" format
    // We normalize to the HF style since that's what ModelArchitecture expects

    GGUF_TO_HF_KEY_REPLACEMENTS
        .iter()
        .fold(name.to_string(), |acc, (from, to)| acc.replace(from, to))
}

/// As [`normalize_gguf_key`], but arch-aware: gemma 2/3/4 GGUFs use a
/// four-norm layer layout (attn_norm / post_attention_norm / ffn_norm /
/// post_ffw_norm) plus QK-norms and, on Gemma 4, a per-layer output
/// scalar. The generic table maps `ffn_norm.` to the llama-style
/// post-attention slot — actively wrong for gemma — and drops the rest
/// on the floor, so gemma-specific replacements run first. Gemma 1 has
/// the llama two-norm layout and stays on the generic path.
pub fn normalize_gguf_key_for_arch(name: &str, arch: &str) -> String {
    let family_rules = find_gguf_architecture(arch).map_or(&[][..], |e| e.gguf.key_replacements);
    family_rules
        .iter()
        .chain(GGUF_TO_HF_KEY_REPLACEMENTS)
        .fold(name.to_string(), |acc, (from, to)| acc.replace(from, to))
}

#[cfg(test)]
mod tests;
