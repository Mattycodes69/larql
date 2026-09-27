//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! Both the build path (full ModelWeights in RAM) and the streaming path
//! (mmap'd safetensors) write through the same `write_model_weights` function
//! via the `WeightSource` trait.

use crate::extract::stage_labels::*;
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::format::filenames::*;

use larql_models::ModelWeights;

/// Manifest `kind` discriminators — wire-format strings written into
/// `weights.json`. Constants exist so writers and the loader's match
/// arm dispatch on the same source-of-truth. A typo on a constant
/// fails to compile; a typo in a string literal would silently route
/// the wrong format and reproduce the Q4_K-vs-Q4_0 lm_head bug.
pub mod kind {
    /// 1D float vector (norms, biases, scalars), stored as f32 or f16
    /// raw bytes. Decoded via `crate::config::dtype::decode_floats`.
    pub const VECTOR: &str = "vector";
    /// 2D f32/f16 dense tensor (raw row-major bytes). Used by the legacy
    /// `write_f32` writer for attn/FFN weights.
    pub const TENSOR: &str = "tensor";
    /// 2D Q4_K-quantised tensor (256-element super-blocks, 144 B/block).
    pub const TENSOR_Q4K: &str = "tensor_q4k";
    /// 2D f16 tensor (e.g. Gemma 4 PLE weights).
    pub const TENSOR_F16: &str = "tensor_f16";
    /// 3D BF16-packed expert tensor (Gemma 4 26B-A4B `experts.gate_up_proj`,
    /// `experts.down_proj`). Range-tracked, not cloned (can be 43 GB).
    pub const PACKED_BF16: &str = "packed_bf16";
}

#[derive(Serialize, Deserialize)]
pub struct WeightEntry {
    pub key: String,
    pub kind: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub length: u64,
    #[serde(default)]
    pub file: String,
}

// ── WeightSource trait ──

/// Abstraction over where model weights come from.
///
/// Implemented by `ModelWeights` (build path — everything in RAM)
/// and `StreamingWeights` (streaming path — mmap'd safetensors on demand).
pub trait WeightSource {
    /// Get a 2D weight tensor by normalized key. Returns (data, rows, cols).
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)>;

    /// Get a 1D vector (norm weights, biases) by normalized key.
    fn get_vector(&self, key: &str) -> Option<Vec<f32>>;

    /// Architecture handle for key generation.
    fn arch(&self) -> &dyn larql_models::ModelArchitecture;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// LM head matrix. Returns (data, rows, cols).
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)>;

    /// All 1D vector names (for norms).
    fn vector_names(&self) -> Vec<String>;

    /// Raw BF16 bytes for a packed expert tensor (e.g. Gemma 4 experts.gate_up_proj).
    /// Returns None if the key is absent or the tensor is not BF16.
    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>>;

    /// Raw U8 bytes plus shape for a quantised packed tensor (MXFP4
    /// `*_blocks` / `*_scales`). Returns None if the key is absent or the
    /// tensor is not U8.
    ///
    /// Default `None`: the in-RAM `ModelWeights` source never needs it —
    /// its loader already dequantised packed experts into per-expert
    /// tensors that `get_tensor` resolves. The streaming source overrides
    /// it, because the synthesised per-expert keys name tensors that do
    /// not exist in any shard, and without this raw access the per-expert
    /// writer silently wrote no expert store at all (the fourth appearance
    /// of that failure — see `write_kquant/moe_layers_per_expert.rs`).
    fn get_raw_u8(&self, _key: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        None
    }
}

// ── ModelWeights implementation ──

impl WeightSource for ModelWeights {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let t = self.tensors.get(key)?;
        Some((t.as_slice()?.to_vec(), t.shape()[0], t.shape()[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        self.vectors.get(key).cloned()
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        let h = &self.lm_head;
        Some((h.as_slice()?.to_vec(), h.shape()[0], h.shape()[1]))
    }

    fn vector_names(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }

    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>> {
        self.raw_bytes.get(key).cloned()
    }
}

// ── Streaming implementation ──

/// Weight source backed by mmap'd safetensors files.
/// Tensors are deserialized on demand — peak memory is one tensor at a time.
pub struct StreamingWeights<'a> {
    pub shard_mmaps: &'a [&'a [u8]],
    pub tensor_index: &'a HashMap<String, (usize, String)>,
    pub arch: &'a dyn larql_models::ModelArchitecture,
    pub num_layers: usize,
}

impl<'a> StreamingWeights<'a> {
    fn read_tensor_raw(&self, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        let shape = view.shape().to_vec();

        let data = match view.dtype() {
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
            safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
            _ => return None,
        };
        Some((data, shape))
    }
}

impl<'a> WeightSource for StreamingWeights<'a> {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 2 {
            return None;
        }
        Some((data, shape[0], shape[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 1 {
            return None;
        }
        Some(data)
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        // Try common lm_head key names
        for key in &["lm_head.weight", "output.weight"] {
            if let Some(t) = self.get_tensor(key) {
                return Some(t);
            }
        }
        None
    }

    fn vector_names(&self) -> Vec<String> {
        // Return all 1D tensor keys (norms, biases)
        let mut names = Vec::new();
        for key in self.tensor_index.keys() {
            if key.contains("layernorm") || key.contains("norm") || key.contains("bias") {
                names.push(key.clone());
            }
        }
        names.sort();
        names
    }

    fn get_packed_bf16(&self, key: &str) -> Option<Vec<u8>> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        if view.dtype() != safetensors::Dtype::BF16 {
            return None;
        }
        Some(view.data().to_vec())
    }

    fn get_raw_u8(&self, key: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        if view.dtype() != safetensors::Dtype::U8 {
            return None;
        }
        Some((view.data().to_vec(), view.shape().to_vec()))
    }
}

// ── Write model weights (generic over source) ──

/// Options for [`write_model_weights_with_opts`]. Use
/// `WriteWeightsOptions::default()` to get the legacy behavior (writes
/// every component file — equivalent to `ExtractLevel::All`).
#[derive(Clone, Copy, Debug)]
pub struct WriteWeightsOptions {
    /// Extract tier — controls which component files are written.
    /// Attention tier writes attn + norms only; Inference adds FFN;
    /// All adds lm_head. See [`crate::ExtractLevel`] for full semantics.
    ///
    /// **Default is `All`, not `Browse`.** Callers of `write_model_weights`
    /// have already decided weights should be written; the CLI-facing
    /// `ExtractLevel::default() == Browse` is the "I want a KNN-only
    /// vindex" intent and is gated out earlier in the extract pipeline.
    pub level: crate::ExtractLevel,

    /// Skip writing `up_weights.bin` + `down_weights.bin`. The up/down
    /// weights are expected to be available to sparse walk paths via
    /// feature-major `up_features.bin` + `down_features.bin`.
    ///
    /// On a 4B f16 vindex this saves ~3.4 GB (1.7 GB per tensor). On a
    /// 31B vindex, proportionally ~14 GB. The cost is non-zero load
    /// time (one mmap + transpose per layer for down, direct view for
    /// up).
    ///
    /// Only take this option if `up_features.bin` and `down_features.bin`
    /// are already in the output directory or will be produced afterwards.
    /// Dense `load_model_weights` paths (`WeightFfn::forward`, MEMIT) do
    /// not reconstruct hidden-major tensors from compact feature-major
    /// files and will reject compact vindexes with a clear error.
    pub ffn_compact: bool,

    /// Skip the attention projection tensors entirely (no
    /// `attn_weights.bin`).  Used by the dense-only BitNet build:
    /// the I2_S attention projections live in the `bitnet/`
    /// artifacts, so expanding them to dense f32 here would write
    /// ~2 GB of duplicate weights the BitNet forward pass never
    /// reads.  Norms are still written (they are not attention
    /// *projections* and the BitNet forward pass needs them).
    pub skip_attn: bool,

    /// Skip the FFN projection tensors entirely (no
    /// `up_weights.bin` / `down_weights.bin`).  Companion to
    /// `skip_attn` for the dense-only BitNet build.
    pub skip_ffn: bool,
}

impl Default for WriteWeightsOptions {
    fn default() -> Self {
        Self {
            level: crate::ExtractLevel::All,
            ffn_compact: false,
            skip_attn: false,
            skip_ffn: false,
        }
    }
}

/// Write model weights to split component files.
///
/// Works with any `WeightSource`: ModelWeights (build path) or
/// StreamingWeights (streaming path from mmap'd safetensors).
pub fn write_model_weights(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    write_model_weights_with_opts(source, dir, callbacks, WriteWeightsOptions::default())
}

mod write;
pub use write::*;
