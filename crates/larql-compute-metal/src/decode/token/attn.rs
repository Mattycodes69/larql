//! Steps 1–5 of one layer: input norm + Q/K/V projection, then the
//! attention block (QK-norm/RoPE, V-norm, KV append + attend, O
//! projection, post-attn residual + ffn-input norm).

use metal::{Buffer, ComputeCommandEncoderRef};

use super::super::{encode_attn, encode_qkv};
use super::ctx::TokenCtx;
use crate::ops;
use crate::MetalBackend;

impl MetalBackend {
    pub(super) fn encode_layer_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        t: &TokenCtx<'_, '_>,
        kv_cache: &mut ops::kv_cache::KVCache,
        l: usize,
        h_buf: &Buffer,
    ) {
        let layers = t.layers;
        let layer = &layers[l];
        let s = t.scratch;
        let hidden = t.hidden;
        let norm_offset = layer.norm_offset;
        let eps = layer.eps;
        let layer_head_dim = layer.head_dim;
        let layer_num_q_heads = layer.num_q_heads;
        let layer_num_kv_heads = layer.num_kv_heads;
        let layer_q_dim = layer_num_q_heads * layer_head_dim;
        let layer_kv_dim = layer_num_kv_heads * layer_head_dim;

        // D-RMS-FUSE Phase 1: skip the input rms_norm dispatch when
        // `LARQL_FUSED_PRELAYER_NORM=1` AND we're not the first layer
        // AND the previous layer's `encode_post_ffn_residual` wrote
        // the pre-normalized data into `norm_f32_buf` via
        // `residual_norm_store` (only on the non-post-norms path).
        let prelayer_norm_active =
            l > 0 && !layers[l - 1].has_post_norms && self.decode_flags.fused_prelayer_norm;

        // ── Step 1: Input norm + Q/K/V projection ──
        // Format-aware: Q4_K family routes through fused QKV
        // shaders (uniform / mixed Q4K+Q6K-V / per-projection
        // fallback); Q4_0 routes through fused norm+Q8 then
        // Q8 QKV. Implementation lives in `encode_qkv.rs`.
        //
        // When the fully-fused attention kernel will fire, it applies
        // the Q/K/V projection biases itself — the QKV stage must
        // skip its bias dispatches (shared `attn_fused_will_fire`
        // authority; disagreement = biases applied twice).
        let qkv_bias_deferred = self.attn_fused_will_fire(layer, kv_cache, l);
        self.encode_input_norm_and_qkv(
            enc,
            layer,
            encode_qkv::QkvBufs {
                h_in: h_buf,
                input_norm: &s.input_norm_bufs[l],
                input_norm_bias: layer.input_norm_bias,
                wq: &s.wq_bufs[l],
                wk: &s.wk_bufs[l],
                wv: &s.wv_bufs[l],
                wq_scales: s.wq_scale_bufs[l].as_ref(),
                wk_scales: s.wk_scale_bufs[l].as_ref(),
                wv_scales: s.wv_scale_bufs[l].as_ref(),
                norm_out: &s.norm_f32_buf,
                q_out: &s.q_out,
                k_out: &s.k_out,
                v_out: &s.v_out,
                ffn_q8: &s.ffn_q8,
                ffn_q8s: &s.ffn_q8s,
            },
            encode_qkv::QkvDims {
                hidden,
                layer_q_dim,
                layer_kv_dim,
                eps,
                norm_offset,
            },
            prelayer_norm_active,
            qkv_bias_deferred,
        );

        // ── Steps 1.5–5: attention block ──
        //
        // QK-norm + RoPE (with optional `attn_fused` and `qk_norm_rope_fused`
        // variants), V-norm (Gemma 4), KV append + attend, O projection,
        // post-attn residual + ffn-input norm. See `encode_attn/` for the
        // full path map.
        self.encode_attention_block(
            enc,
            layer,
            kv_cache,
            l,
            encode_attn::AttnBufs {
                h_buf,
                q_out: &s.q_out,
                k_out: &s.k_out,
                v_out: &s.v_out,
                attn_out_buf: &s.attn_out_buf,
                o_out_buf: &s.o_out_buf,
                ffn_norm_out: &s.ffn_norm_out,
                h_post_attn: &s.h_post_attn,
                o_q8_scratch: &s.o_q8_scratch,
                o_q8s_scratch: &s.o_q8s_scratch,
                ffn_q8: &s.ffn_q8,
                ffn_q8s: &s.ffn_q8s,
                normed_scratch: &s.normed_scratch,
                wo: &s.wo_bufs[l],
                wo_scales: s.wo_scale_bufs[l].as_ref(),
                post_attn_norm: &s.post_attn_norm_bufs[l],
            },
            encode_attn::AttnDims {
                hidden,
                layer_q_dim,
                ffn_uses_kquant: layer.gate.format().is_kquant_family(),
            },
        );
    }
}
