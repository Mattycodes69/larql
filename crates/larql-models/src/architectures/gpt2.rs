//! GPT-2 architecture.
//!
//! Key differences from standard Llama:
//! - Standard LayerNorm (with bias) instead of RMSNorm
//! - Non-gated FFN: activation(x @ c_fc.T + bias) @ c_proj.T + bias, GELU activation
//! - Conv1D-style fused QKV (`attn_qkv.weight`) instead of separate Q/K/V
//! - Learned position embeddings (`position_embd.weight`) instead of RoPE
//! - LM head tied to input embedding
//!
//! Tensor key naming after the GGUF→HF normalization in `loading/gguf.rs`
//! matches the trait defaults (`embed_tokens.weight`, `mlp.up_proj.weight`,
//! `mlp.down_proj.weight`, …), so this arch only overrides behavior flags.

use crate::config::{Activation, FfnType, ModelArchitecture, ModelConfig, NormType};
use crate::detect::LayerBandSplit;
use crate::tensor_keys::attn_bias;

pub struct Gpt2Arch {
    config: ModelConfig,
}

impl Gpt2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for Gpt2Arch {
    fn family(&self) -> &str {
        "gpt2"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TensorKeys for Gpt2Arch {
    /// GPT-2 packs Q, K, V into a single Conv1D `c_attn` projection. The
    /// GGUF→HF normaliser maps `attn_qkv.` → `self_attn.qkv_proj.`; the
    /// loader's `split_fused_qkv` pass then materialises the per-projection
    /// q/k/v tensors at the trait's standard `attn_q_key`/`attn_k_key`/
    /// `attn_v_key` so downstream code stays family-agnostic.
    fn fused_qkv_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.qkv_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn fused_qkv_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.qkv_proj.bias",
            self.layer_prefix(layer)
        ))
    }

    /// GPT-2 has bias on every projection (Conv1D layers carry bias).
    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::q(&self.layer_prefix(layer))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::k(&self.layer_prefix(layer))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::v(&self.layer_prefix(layer))
    }

    fn attn_o_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::o(&self.layer_prefix(layer))
    }

    fn ffn_up_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.up_proj.bias", self.layer_prefix(layer)))
    }

    fn ffn_down_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.down_proj.bias", self.layer_prefix(layer)))
    }

    /// GPT-2 uses learned absolute position embeddings (`wpe`) added to the
    /// token embedding at the input — no rotary positional signal.
    fn position_embed_key(&self) -> Option<&str> {
        Some("wpe.weight")
    }
}

impl Norms for Gpt2Arch {
    fn norm_type(&self) -> NormType {
        NormType::LayerNorm
    }
}

impl FeedForward for Gpt2Arch {
    fn activation(&self) -> Activation {
        Activation::GeluTanh
    }

    fn ffn_type(&self) -> FfnType {
        FfnType::Standard
    }
}

impl Position for Gpt2Arch {}
impl Attention for Gpt2Arch {}
impl LatentAttention for Gpt2Arch {}
impl Embeddings for Gpt2Arch {}
impl ModelArchitecture for Gpt2Arch {}

/// DESCRIBE layer bands for this family at the depths they were set for.
/// Exact `model_type` only: a lookalike falls back to the proportional split.
pub(crate) const GPT2_LAYER_BANDS: &[LayerBandSplit] = &[
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 12,
        syntax_last: 4,
        knowledge_last: 9,
    },
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 24,
        syntax_last: 9,
        knowledge_last: 19,
    },
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 36,
        syntax_last: 14,
        knowledge_last: 28,
    },
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 48,
        syntax_last: 19,
        knowledge_last: 38,
    },
];
