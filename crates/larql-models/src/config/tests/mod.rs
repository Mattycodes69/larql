//! Trait-default behaviour, pinned.
//!
//! An architecture that overrides nothing exercises every default here. The
//! per-architecture files test their own overrides; nothing else pins what a
//! family inherits by *not* overriding, and a changed default silently
//! rewrites the behaviour of every architecture that relied on it.

use super::*;

/// The architecture that overrides nothing.
struct DefaultsArch(ModelConfig);

use crate::config::architecture_prelude::*;

impl ArchitectureCore for DefaultsArch {
    fn family(&self) -> &str {
        "defaults-test"
    }

    fn config(&self) -> &ModelConfig {
        &self.0
    }
}

impl TensorKeys for DefaultsArch {}
impl Norms for DefaultsArch {}
impl Position for DefaultsArch {}
impl Attention for DefaultsArch {}
impl FeedForward for DefaultsArch {}
impl LatentAttention for DefaultsArch {}
impl Embeddings for DefaultsArch {}
impl ModelArchitecture for DefaultsArch {}

/// A minimal parsed config; tests mutate the returned fields directly
/// rather than guessing config.json spellings (parsing is
/// `detect::parser`'s test surface, not this one's).
fn base_config() -> ModelConfig {
    crate::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 32,
    }))
    .config()
    .clone()
}

/// `openai/gpt-oss-20b`'s block verbatim. The scaling type is the only
/// selector, so this doubles as the pin that an architecture needs **no**
/// override to be served under YaRN — the §4.7.8 mechanism fix.
fn yarn_scaling() -> RopeScaling {
    RopeScaling {
        scaling_type: "yarn".to_string(),
        factor: 32.0,
        llama3_low_freq_factor: None,
        llama3_high_freq_factor: None,
        llama3_original_max_position_embeddings: Some(4096.0),
        yarn_beta_fast: Some(32.0),
        yarn_beta_slow: Some(1.0),
        yarn_truncate: Some(false),
        yarn_mscale: None,
        yarn_mscale_all_dim: None,
        gemma3_global_only: false,
    }
}

//    mapping (2026-07-30 review §4 dedupe) ─────────────────────────

//    (K3-ACT-1) ──────────────────────────────────────────────────────
//
// The defect these close: `activation()` used to answer SiLU to BOTH "no
// declaration" and "a declaration this build cannot read", so a
// checkpoint declaring `situ` or `relu2` was executed as SwiGLU. Each
// test below names one of the four states, so a future collapse of two
// of them fails here.

/// An architecture over the default config, declaring one activation.
fn arch_declaring(hidden_act: Option<&str>) -> DefaultsArch {
    let mut config = base_config();
    config.hidden_act = hidden_act.map(str::to_string);
    DefaultsArch(config)
}

/// An architecture over the default config, declaring one `q_lora_rank`.
fn arch_with_q_lora(rank: Option<usize>) -> DefaultsArch {
    let mut config = base_config();
    config.q_lora_rank = rank;
    DefaultsArch(config)
}

/// An architecture over the default config, declaring one latent branch.
fn arch_with_latent(width: Option<usize>, use_norm: Option<bool>) -> DefaultsArch {
    let mut config = base_config();
    config.routed_expert_hidden_size = width;
    config.latent_moe_use_norm = use_norm;
    DefaultsArch(config)
}

//
// Parsed as a *check* on the loader's tie-on-absence behaviour, not as a
// shortcut. See `loading/safetensors.rs`.

mod defaults_and_declarations;
mod gate_up_mla_and_post_norm;
