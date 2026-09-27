//! SigLIP vision encoder — config + weights + safetensors loader.
//!
//! Used by Gemma 3 / PaliGemma multimodal checkpoints. Forward pass
//! lives in `larql-compute::encoders::siglip`.
//!
//! Tensor key convention (verified against `google/gemma-3-4b-it`):
//!
//! ```text
//! vision_tower.vision_model.embeddings.patch_embedding.{weight,bias}
//! vision_tower.vision_model.embeddings.position_embedding.weight
//! vision_tower.vision_model.encoder.layers.<L>.layer_norm1.{weight,bias}
//! vision_tower.vision_model.encoder.layers.<L>.self_attn.{q,k,v,out}_proj.{weight,bias}
//! vision_tower.vision_model.encoder.layers.<L>.layer_norm2.{weight,bias}
//! vision_tower.vision_model.encoder.layers.<L>.mlp.{fc1,fc2}.{weight,bias}
//! vision_tower.vision_model.post_layernorm.{weight,bias}
//! ```
//!
//! Phase 1b scope: config + struct definitions + loader. The forward
//! pass and the multi_modal_projector connector are Phase 1b.2 and 1c.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array2, Array4};
use serde::Deserialize;

use crate::detect::ModelError;
use crate::loading::safetensors::tensor_to_f32;

const SIGLIP_VISION_TOWER_PREFIX: &str = "vision_tower.vision_model.";

// ─── Config ──────────────────────────────────────────────────────────────

/// SigLIP vision tower configuration. Parsed from the `vision_config`
/// sub-object of a multimodal model's `config.json`.
///
/// Matches HuggingFace `SiglipVisionConfig`. Field names lower-snake
/// because that's what the JSON uses.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub image_size: usize,
    /// Defaults to 3 (RGB) if absent.
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    /// LayerNorm epsilon. HF default for SigLIP is 1e-6.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Activation function in the encoder MLP. Defaults to `"gelu_pytorch_tanh"` (SigLIP).
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Normalization type. `"layer_norm"` for SigLIP, `"rms_norm"` for SigLIP2 variants.
    #[serde(default = "default_norm_type")]
    pub norm_type: String,
}

fn default_num_channels() -> usize {
    3
}
fn default_layer_norm_eps() -> f64 {
    1e-6
}
fn default_hidden_act() -> String {
    "gelu_pytorch_tanh".to_string()
}
fn default_norm_type() -> String {
    "layer_norm".to_string()
}

impl VisionConfig {
    /// Parse from the `vision_config` sub-object of a model's `config.json`.
    /// Most multimodal HF configs nest the vision encoder config under this
    /// key (Gemma 3, PaliGemma, LLaVA-Next-Vision-style).
    pub fn from_json(value: &serde_json::Value) -> Result<Self, ModelError> {
        serde_json::from_value(value.clone()).map_err(|e| ModelError::Parse(e.to_string()))
    }

    /// Number of patches along one image dimension. SigLIP at 896 × 14 = 64.
    pub fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches per image. SigLIP at 64×64 = 4096.
    pub fn num_patches(&self) -> usize {
        let s = self.patches_per_side();
        s * s
    }

    /// Per-head dimension (hidden_size / num_attention_heads). SigLIP at
    /// 1152 / 16 = 72.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether this config describes a SigLIP2 encoder.
    pub fn is_siglip2(&self) -> bool {
        self.norm_type != "layer_norm" || self.hidden_act == "gelu" || self.hidden_act == "silu"
    }
}

// ─── Tensor bundles ──────────────────────────────────────────────────────

/// Weight + bias pair for an affine projection. Encoded `out × in` for
/// the weight (matches HF safetensors row-major convention — `y = x @ W.T + b`).
/// `bias` is `Vec<f32>` (not `Array1`) so it interoperates with
/// `larql_compute::residual::layer_norm_eps`-style APIs that take
/// `Option<&Vec<f32>>` — matches the LM's `weights.vectors` HashMap convention.
#[derive(Debug, Clone)]
pub struct ProjWithBias {
    pub weight: Array2<f32>,
    pub bias: Vec<f32>,
}

/// LayerNorm scale + bias. Both `Vec<f32>` of length hidden_size, same
/// rationale as `ProjWithBias::bias` above.
#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Per-layer SigLIP transformer block weights.
#[derive(Debug, Clone)]
pub struct VisionLayerWeights {
    pub layer_norm1: LayerNormWeights,
    pub q_proj: ProjWithBias,
    pub k_proj: ProjWithBias,
    pub v_proj: ProjWithBias,
    pub out_proj: ProjWithBias,
    pub layer_norm2: LayerNormWeights,
    pub fc1: ProjWithBias,
    pub fc2: ProjWithBias,
}

/// Full SigLIP vision encoder weights.
///
/// Memory footprint for the Gemma 3 4B-it variant (SigLIP at 1152/27/16):
///
/// - patch_embed: 1152·3·14·14 + 1152 bias ≈ 0.7 MB f32
/// - position_embed: 4096·1152 ≈ 18.9 MB f32
/// - 27 × (4 projections + 2 norms + 2 fc) × (1152² or 1152·4304) ≈ 620 MB f32
/// - post_layernorm: 2 × 1152 ≈ 9 KB f32
/// - **Total ≈ ~640 MB f32**
///
/// Phase 1b: f32 only. f16 quantisation deferred.
#[derive(Debug)]
pub struct VisionWeights {
    pub config: VisionConfig,
    /// Patch projection (Conv2D as 4-D weight).
    /// Shape: `(hidden_size, num_channels, patch_size, patch_size)`.
    pub patch_embed: Array4<f32>,
    pub patch_embed_bias: Vec<f32>,
    /// Learned absolute position embedding. Shape `(num_patches, hidden_size)`.
    pub position_embed: Array2<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub post_layernorm: LayerNormWeights,
}

// ─── Loader ──────────────────────────────────────────────────────────────

/// Load SigLIP weights from a directory of safetensors files.
///
/// Scans every `*.safetensors` in `dir`, picks tensors whose key starts
/// with `vision_tower.vision_model.`, and populates a `VisionWeights`.
///
/// `config` is parsed separately (typically from `dir/config.json`'s
/// `vision_config` field) and passed in — the loader does not crack the
/// model config itself, keeping it usable in tests that synth a config
/// directly.
///
/// Errors:
///   - `ModelError::Parse` on safetensors / dtype / shape mismatch.
///   - `ModelError::Parse` if a required tensor is missing.
pub fn load_vision_tower_from_safetensors(
    dir: impl AsRef<Path>,
    config: VisionConfig,
) -> Result<VisionWeights, ModelError> {
    let dir = dir.as_ref();
    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut patch_embed_raw: Option<(Vec<f32>, Vec<usize>)> = None;

    let entries = std::fs::read_dir(dir).map_err(|e| ModelError::Parse(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| ModelError::Parse(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        load_one_file(&path, &mut tensors, &mut vectors, &mut patch_embed_raw)?;
    }

    if tensors.is_empty() && vectors.is_empty() && patch_embed_raw.is_none() {
        return Err(ModelError::Parse(format!(
            "no vision_tower tensors found under {dir:?}; is this a multimodal checkpoint?"
        )));
    }

    assemble(config, tensors, vectors, patch_embed_raw)
}

fn load_one_file(
    path: &Path,
    tensors: &mut HashMap<String, Array2<f32>>,
    vectors: &mut HashMap<String, Vec<f32>>,
    patch_embed_raw: &mut Option<(Vec<f32>, Vec<usize>)>,
) -> Result<(), ModelError> {
    let file = std::fs::File::open(path).map_err(|e| ModelError::Parse(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| ModelError::Parse(e.to_string()))?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| ModelError::Parse(e.to_string()))?;

    for (name, view) in st.tensors() {
        let key = match name.strip_prefix(SIGLIP_VISION_TOWER_PREFIX) {
            Some(rest) => rest.to_string(),
            None => continue,
        };
        let shape = view.shape().to_vec();
        let data = tensor_to_f32(&view)?;

        match shape.len() {
            4 if key == "embeddings.patch_embedding.weight" => {
                *patch_embed_raw = Some((data, shape));
            }
            2 => {
                let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?;
                tensors.insert(key, arr);
            }
            1 => {
                vectors.insert(key, data);
            }
            _ => {
                // Unknown rank — ignore (a future SigLIP variant might add
                // extra tensors; we shouldn't fail to load on unknowns).
            }
        }
    }
    Ok(())
}

fn assemble(
    config: VisionConfig,
    mut tensors: HashMap<String, Array2<f32>>,
    mut vectors: HashMap<String, Vec<f32>>,
    patch_embed_raw: Option<(Vec<f32>, Vec<usize>)>,
) -> Result<VisionWeights, ModelError> {
    // Patch embedding: Conv2D weight has shape (hidden, channels, patch, patch).
    let (pe_data, pe_shape) = patch_embed_raw.ok_or_else(|| {
        ModelError::Parse("missing embeddings.patch_embedding.weight (rank-4)".to_string())
    })?;
    if pe_shape.len() != 4 {
        return Err(ModelError::Parse(format!(
            "patch_embedding.weight expected rank 4, got shape {pe_shape:?}"
        )));
    }
    let patch_embed = Array4::from_shape_vec(
        (pe_shape[0], pe_shape[1], pe_shape[2], pe_shape[3]),
        pe_data,
    )
    .map_err(|e| ModelError::Parse(e.to_string()))?;

    let patch_embed_bias = take_vec(&mut vectors, "embeddings.patch_embedding.bias")?;
    let position_embed = take_tensor(&mut tensors, "embeddings.position_embedding.weight")?;

    let post_layernorm = LayerNormWeights {
        weight: take_vec(&mut vectors, "post_layernorm.weight")?,
        bias: take_vec(&mut vectors, "post_layernorm.bias")?,
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for l in 0..config.num_hidden_layers {
        layers.push(take_layer(l, &mut tensors, &mut vectors)?);
    }

    Ok(VisionWeights {
        config,
        patch_embed,
        patch_embed_bias,
        position_embed,
        layers,
        post_layernorm,
    })
}

fn take_layer(
    l: usize,
    tensors: &mut HashMap<String, Array2<f32>>,
    vectors: &mut HashMap<String, Vec<f32>>,
) -> Result<VisionLayerWeights, ModelError> {
    let prefix = format!("encoder.layers.{l}.");
    let proj = |t: &mut HashMap<String, Array2<f32>>,
                v: &mut HashMap<String, Vec<f32>>,
                name: &str|
     -> Result<ProjWithBias, ModelError> {
        let weight = take_tensor(t, &format!("{prefix}{name}.weight"))?;
        let bias = take_vec(v, &format!("{prefix}{name}.bias"))?;
        Ok(ProjWithBias { weight, bias })
    };
    let norm =
        |v: &mut HashMap<String, Vec<f32>>, name: &str| -> Result<LayerNormWeights, ModelError> {
            Ok(LayerNormWeights {
                weight: take_vec(v, &format!("{prefix}{name}.weight"))?,
                bias: take_vec(v, &format!("{prefix}{name}.bias"))?,
            })
        };
    Ok(VisionLayerWeights {
        layer_norm1: norm(vectors, "layer_norm1")?,
        q_proj: proj(tensors, vectors, "self_attn.q_proj")?,
        k_proj: proj(tensors, vectors, "self_attn.k_proj")?,
        v_proj: proj(tensors, vectors, "self_attn.v_proj")?,
        out_proj: proj(tensors, vectors, "self_attn.out_proj")?,
        layer_norm2: norm(vectors, "layer_norm2")?,
        fc1: proj(tensors, vectors, "mlp.fc1")?,
        fc2: proj(tensors, vectors, "mlp.fc2")?,
    })
}

fn take_tensor(
    tensors: &mut HashMap<String, Array2<f32>>,
    key: &str,
) -> Result<Array2<f32>, ModelError> {
    tensors
        .remove(key)
        .ok_or_else(|| ModelError::Parse(format!("missing vision_tower tensor: {key}")))
}

fn take_vec(vectors: &mut HashMap<String, Vec<f32>>, key: &str) -> Result<Vec<f32>, ModelError> {
    vectors
        .remove(key)
        .ok_or_else(|| ModelError::Parse(format!("missing vision_tower vector: {key}")))
}

#[cfg(test)]
mod tests;
