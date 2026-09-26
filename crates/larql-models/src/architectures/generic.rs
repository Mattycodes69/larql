//! Generic/fallback architecture for unknown model types.
//!
//! Uses Llama-style defaults: no embedding scaling, no norm offset,
//! no QK norm, no post-norms, standard RoPE base.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct GenericArch {
    config: ModelConfig,
}

impl GenericArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for GenericArch {
    fn family(&self) -> &str {
        "generic"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TensorKeys for GenericArch {}
impl Norms for GenericArch {}
impl Position for GenericArch {}
impl Attention for GenericArch {}
impl FeedForward for GenericArch {}
impl LatentAttention for GenericArch {}
impl Embeddings for GenericArch {}
impl ModelArchitecture for GenericArch {}
