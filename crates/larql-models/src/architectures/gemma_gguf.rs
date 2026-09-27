//! How Gemma 2/3/4 GGUF exports map back to their HF shape.
//!
//! Referenced from the Gemma rows of the architecture registry; the GGUF
//! loader itself holds no Gemma knowledge.

use serde_json::{json, Value};

use crate::loading::gguf::constants::{
    GGUF_ATTENTION_HEAD_COUNT, GGUF_ATTENTION_HEAD_COUNT_KV, GGUF_ATTENTION_KEY_LENGTH,
    GGUF_BLOCK_COUNT, HF_HEAD_DIM, HF_NUM_KEY_VALUE_HEADS,
};
use crate::loading::gguf::GgufFile;

/// Gemma 2/3/4 layers carry four norms plus QK-norms. The shared GGUF table
/// maps `ffn_norm.` to `post_attention_layernorm.` (right for the llama
/// two-norm layout, wrong here: Gemma's `ffn_norm` is the pre-FFN norm and
/// `post_attention_norm` is the real post-attention one) and has no entries
/// for the rest. Applied before the shared table so `ffn_norm.` is consumed
/// here first. Gemma 1 keeps the llama layout and does not use this.
pub(crate) const GEMMA_GGUF_KEY_REPLACEMENTS: &[(&str, &str)] = &[
    ("attn_q_norm.", "self_attn.q_norm."),
    ("attn_k_norm.", "self_attn.k_norm."),
    ("post_attention_norm.", "post_attention_layernorm."),
    ("ffn_norm.", "pre_feedforward_layernorm."),
    ("post_ffw_norm.", "post_feedforward_layernorm."),
    // Gemma 4 per-layer output scalar; larql's key has no `.weight` suffix.
    ("layer_output_scale.weight", "layer_scalar"),
];

/// The sliding-layer head width known Gemma 4 exports use, applied when the
/// file does not state `attention.key_length_swa`. The file's
/// `attention.key_length` is the *global*-layer width, not this one.
const GEMMA4_SLIDING_HEAD_DIM: u32 = 256;

/// Share of head dims a Gemma 4 global layer rotates. Constant across every
/// Gemma 4 HF config (12B/26B/31B); the GGUF carries no equivalent key.
const GEMMA4_GLOBAL_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

// GGUF metadata suffixes (under `<arch>.`) only Gemma 4 exports carry.
const GGUF_KEY_LENGTH_SWA: &str = "attention.key_length_swa";
const GGUF_ROPE_FREQ_BASE_SWA: &str = "rope.freq_base_swa";
const GGUF_SLIDING_WINDOW: &str = "attention.sliding_window";
const GGUF_FINAL_LOGIT_SOFTCAPPING: &str = "final_logit_softcapping";
const GGUF_ATTN_V_SUFFIX: &str = ".attn_v.weight";

// HF config keys Gemma 4's heterogeneous attention is described with.
const HF_NUM_GLOBAL_KEY_VALUE_HEADS: &str = "num_global_key_value_heads";
const HF_GLOBAL_HEAD_DIM: &str = "global_head_dim";
const HF_PARTIAL_ROTARY_FACTOR: &str = "partial_rotary_factor";
const HF_ROPE_LOCAL_BASE_FREQ: &str = "rope_local_base_freq";
const HF_SLIDING_WINDOW: &str = "sliding_window";
const HF_ATTENTION_K_EQ_V: &str = "attention_k_eq_v";
const HF_FINAL_LOGIT_SOFTCAPPING: &str = "final_logit_softcapping";

/// Gemma 4 per-layer attention geometry.
///
/// Gemma 4 GGUFs describe heterogeneous attention (sliding vs global layers)
/// with per-layer arrays and `*_swa` twin keys; the flat mapping collapses
/// them to one number and drops the rest. Re-emit the HF keys the
/// safetensors path reads so the reconstructed architecture can route per
/// layer. Without them, models whose global layers have no `attn_v` tensor
/// (12B, 31B: `attention_k_eq_v`) index past the V collection at inference.
pub(crate) fn gemma4_gguf_config(gguf: &GgufFile, config: &mut Value) {
    // head_count_kv is per layer (e.g. 8 sliding, 1 global on the 12B).
    // Layer 0 is always sliding; the first differing value is global.
    if let Some(kv_heads) = gguf.arch_u32_array(GGUF_ATTENTION_HEAD_COUNT_KV) {
        if let Some(&sliding) = kv_heads.first() {
            config[HF_NUM_KEY_VALUE_HEADS] = json!(sliding);
            if let Some(&global) = kv_heads.iter().find(|&&v| v != sliding) {
                config[HF_NUM_GLOBAL_KEY_VALUE_HEADS] = json!(global);
            }
        }
    }

    // key_length is the global-layer width; key_length_swa the sliding one,
    // which is the base head_dim the architecture expects.
    let key_len = gguf.arch_u32(GGUF_ATTENTION_KEY_LENGTH);
    let key_len_swa = gguf.arch_u32(GGUF_KEY_LENGTH_SWA);
    if key_len_swa > 0 {
        config[HF_HEAD_DIM] = json!(key_len_swa);
    } else if gguf.arch_u32(GGUF_ATTENTION_HEAD_COUNT) > 0 {
        config[HF_HEAD_DIM] = json!(GEMMA4_SLIDING_HEAD_DIM);
    }
    if key_len > 0 && key_len != key_len_swa {
        config[HF_GLOBAL_HEAD_DIM] = json!(key_len);
        config[HF_PARTIAL_ROTARY_FACTOR] = json!(GEMMA4_GLOBAL_PARTIAL_ROTARY_FACTOR);
    }

    // Dual RoPE bases: `rope.freq_base` (already emitted as rope_theta) is
    // the global-layer clock, `rope.freq_base_swa` the sliding one.
    if let Some(swa) = gguf.arch_f64(GGUF_ROPE_FREQ_BASE_SWA) {
        config[HF_ROPE_LOCAL_BASE_FREQ] = json!(swa);
    }
    if let Some(window) = gguf.arch_u32_opt(GGUF_SLIDING_WINDOW).filter(|&v| v > 0) {
        config[HF_SLIDING_WINDOW] = json!(window);
    }

    // Layers with no attn_v tensor reuse K as V. The metadata has no flag
    // for this; detect it from the tensor inventory, as llama.cpp does.
    let n_blocks = gguf.arch_u32(GGUF_BLOCK_COUNT) as usize;
    let n_v = gguf.count_tensors_ending_with(GGUF_ATTN_V_SUFFIX);
    if n_blocks > 0 && n_v > 0 && n_v < n_blocks {
        config[HF_ATTENTION_K_EQ_V] = json!(true);
    }

    // Final-logit softcap shapes the softmax (never the argmax).
    if let Some(cap) = gguf.arch_f64(GGUF_FINAL_LOGIT_SOFTCAPPING) {
        config[HF_FINAL_LOGIT_SOFTCAPPING] = json!(cap);
    }
}
