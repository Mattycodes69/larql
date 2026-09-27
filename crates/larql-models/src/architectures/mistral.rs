//! Mistral architecture.
//!
//! Llama-compatible: same tensor keys, norms, activation, RoPE.

use crate::config::{ModelArchitecture, ModelConfig};
use crate::detect::LayerBandSplit;

pub struct MistralArch {
    config: ModelConfig,
}

impl MistralArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for MistralArch {
    fn family(&self) -> &str {
        "mistral"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TensorKeys for MistralArch {}
impl Norms for MistralArch {}
impl Position for MistralArch {}
impl Attention for MistralArch {}
impl FeedForward for MistralArch {}
impl LatentAttention for MistralArch {}
impl Embeddings for MistralArch {}
impl ModelArchitecture for MistralArch {}

/// DESCRIBE layer bands for this family at the depths they were set for.
/// Exact `model_type` only: a lookalike falls back to the proportional split.
pub(crate) const MISTRAL_LAYER_BANDS: &[LayerBandSplit] = &[LayerBandSplit {
    model_type: "mistral",
    num_layers: 32,
    syntax_last: 12,
    knowledge_last: 25,
}];
