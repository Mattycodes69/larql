//! Llama-family architecture.
//!
//! Covers Llama, Mistral, Qwen, and other Llama-compatible models.
//! Uses all trait defaults (which are Llama-style).

use crate::config::{ModelArchitecture, ModelConfig};
use crate::detect::LayerBandSplit;

pub struct LlamaArch {
    config: ModelConfig,
}

impl LlamaArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for LlamaArch {
    fn family(&self) -> &str {
        "llama"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TensorKeys for LlamaArch {}
impl Norms for LlamaArch {}
impl Position for LlamaArch {}
impl Attention for LlamaArch {}
impl FeedForward for LlamaArch {}
impl LatentAttention for LlamaArch {}
impl Embeddings for LlamaArch {}
impl ModelArchitecture for LlamaArch {}

/// DESCRIBE layer bands for this family at the depths they were set for.
/// Exact `model_type` only: a lookalike falls back to the proportional split.
pub(crate) const LLAMA_LAYER_BANDS: &[LayerBandSplit] = &[
    LayerBandSplit {
        model_type: "llama",
        num_layers: 32,
        syntax_last: 12,
        knowledge_last: 25,
    },
    LayerBandSplit {
        model_type: "llama",
        num_layers: 40,
        syntax_last: 15,
        knowledge_last: 32,
    },
    LayerBandSplit {
        model_type: "llama",
        num_layers: 80,
        syntax_last: 31,
        knowledge_last: 63,
    },
];
