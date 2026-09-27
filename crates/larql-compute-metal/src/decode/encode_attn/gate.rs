//! The single authority for whether a layer takes the fully-fused
//! `attn_fused` path.

use crate::ops;
use crate::ops::kv_cache::MAX_HEAD_DIM_SINGLE_SG;
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

impl MetalBackend {
    /// Whether this layer's decode step will take the fully-fused
    /// `attn_fused` path (QK-norm/passthrough + RoPE + KV-append +
    /// attend + optional QKV biases, ONE dispatch).
    ///
    /// The SINGLE authority for that decision: `encode_input_norm_and_qkv`
    /// consults it to skip the separate Q/K/V `bias_add` dispatches (the
    /// fused kernel applies them on load), and `encode_attention_block`
    /// consults it to pick the dispatch. Two sites re-deriving this
    /// independently is the dispatch-geometry-disagreement defect class —
    /// a bias applied twice or not at all, silently.
    pub(in crate::decode) fn attn_fused_will_fire(
        &self,
        layer: &FullPipelineLayer,
        kv_cache: &ops::kv_cache::KVCache,
        layer_idx: usize,
    ) -> bool {
        let attn_spec = layer.attention_spec();
        if layer.kv_shared_source.is_some()
            || !self.decode_flags.fused_attn
            || attn_spec.head_dim > MAX_HEAD_DIM_SINGLE_SG
            || attn_spec.has_v_norm
        {
            return false;
        }
        // QK-norm must be both-or-neither: the kernel norms Q and K under
        // one flag. A half-normed arch takes the unfused chain.
        if attn_spec.q_norm_enabled != attn_spec.k_norm_enabled {
            return false;
        }
        // Same all-or-nothing rule for the projection biases: the kernel
        // reads all three under one flag, and a partially-biased layer
        // bound with stub buffers would read out of bounds. The unfused
        // chain's per-site bias_add dispatches handle mixed presence.
        let n_bias = [layer.attn_q_bias, layer.attn_k_bias, layer.attn_v_bias]
            .iter()
            .filter(|b| b.is_some())
            .count();
        if n_bias != 0 && n_bias != 3 {
            return false;
        }
        let window_size = self.effective_window_for(attn_spec.sliding_window as u32);
        let t_val = (kv_cache.layers[layer_idx].current_len + 1) as u32;
        ops::kv_cache::attention_span(t_val, window_size) <= ops::kv_cache::SHORT_ATTENTION_SPAN
    }
}
