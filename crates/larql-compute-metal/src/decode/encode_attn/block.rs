//! `encode_attention_block` — the stage sequence.
//!
//! Every stage encodes into the one compute encoder it is handed, in the
//! order below; nothing here commits, waits or opens a new encoder. The
//! branch conditions are the ones the single-function block used, so the
//! encoded command stream is unchanged.

use metal::ComputeCommandEncoderRef;

use super::context::AttnCtx;
use super::{AttnBufs, AttnDims};
use crate::ops;
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

impl MetalBackend {
    /// Encode the per-layer attention block (Steps 1.5–5). See the module
    /// doc-comment for the full input/output contract.
    pub(in crate::decode) fn encode_attention_block(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        kv_cache: &mut ops::kv_cache::KVCache,
        layer_idx: usize,
        bufs: AttnBufs<'_>,
        dims: AttnDims,
    ) {
        let c = AttnCtx::new(self, layer, kv_cache, layer_idx, bufs, dims);

        // Path 1: full attention fusion. Skips the qk_norm_rope dispatch,
        // the kv_append_attend_fused dispatch, AND the three Q/K/V
        // bias_add dispatches (when the layer has biases) — all handled
        // inside `attn_fused`. The decision comes from the shared
        // `attn_fused_will_fire` authority, which `encode_input_norm_and_qkv`
        // also consulted to skip its bias dispatches — the two sites must
        // agree or a bias is applied twice / dropped.
        let did_fused_attn = self.attn_fused_will_fire(layer, kv_cache, layer_idx);

        // ── Step 1.5 + 2: QK-norm + RoPE ──
        if did_fused_attn {
            self.encode_attn_fused(enc, &c, kv_cache);
        } else if c.use_fused_qkn_rope
            && layer.q_norm_weight.is_some()
            && layer.k_norm_weight.is_some()
        {
            self.encode_qk_norm_rope_fused(enc, &c);
        } else {
            self.encode_qk_norm_then_rope(enc, &c);
        }

        // ── Step 3: V-norm batched (optional) ──
        if layer.has_v_norm {
            self.encode_v_norm(enc, &c);
        }

        // ── Step 4: KV-append + KV-attend ──
        // Skipped entirely when `did_fused_attn` is true (the unified
        // `attn_fused` kernel above already wrote both cache rows + the
        // attention output and bumped current_len). Shared layers
        // (`kv_shared_source = Some(_)`) skip the append entirely and
        // attend against the source's cache.
        if !did_fused_attn {
            self.encode_kv_append_attend(enc, &c, kv_cache);
        }
        // Only own-cache layers advance current_len; shared layers leave
        // their (unused) cache pointer at 0 forever.
        if !did_fused_attn && c.kv_shared_source.is_none() {
            kv_cache.layers[layer_idx].advance_one();
        }

        // ── Step 5a: O projection (+ optional O bias) ──
        self.encode_o_projection(enc, &c);

        // ── Step 5b: Residual + post-attn norm + ffn-input norm ──
        self.encode_post_attn_residual(enc, &c);
    }
}
