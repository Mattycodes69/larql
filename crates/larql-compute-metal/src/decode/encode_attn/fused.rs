//! Path 1: the single-dispatch `attn_fused` kernel.
//!
//! QK-norm (or passthrough) + RoPE + KV append + attend + optional Q/K/V
//! projection biases in ONE dispatch. Fires only where
//! `attn_fused_will_fire` says so; it writes `attn_out_buf` and bumps the
//! layer's `current_len` itself, so steps 1.5–4 are skipped downstream.

use metal::{ComputeCommandEncoderRef, MTLSize};

use super::context::AttnCtx;
use super::{
    ATTN_FUSED_ABS_POS_INDEX, ATTN_FUSED_AMPLITUDE_INDEX, ATTN_FUSED_HAS_QK_NORM_INDEX,
    ATTN_FUSED_INV_FREQ_INDEX, ATTN_FUSED_QKV_BIAS_INDEX, ATTN_FUSED_SINKS_INDEX,
    ATTN_FUSED_SOFTCAP_INDEX,
};
use crate::ops;
use crate::ops::kv_cache::MAX_HEAD_DIM_SINGLE_SG;
use crate::MetalBackend;

impl MetalBackend {
    pub(super) fn encode_attn_fused(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        kv_cache: &mut ops::kv_cache::KVCache,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let layer_idx = c.layer_idx;
        let layer_head_dim = c.head_dim;
        let layer_num_q_heads = c.num_q_heads;
        let cache = &kv_cache.layers[layer_idx];
        // No-QK-norm archs bind the shared empty-slice stub; the
        // has_qk_norm flag gates the read (sinks convention).
        let q_w_buf = self.bufs.get_f32(layer.q_norm_weight.unwrap_or(&[]));
        let k_w_buf = self.bufs.get_f32(layer.k_norm_weight.unwrap_or(&[]));
        let t_val = (cache.current_len + 1) as u32;
        let hd_val = layer_head_dim as u32;
        let nq_val = layer_num_q_heads as u32;
        let nkv_val = cache.num_kv_heads as u32;
        let qk_off = c.qk_norm_offset;
        let rdim = c.rotary_dim as u32;
        let mut tg_w: u64 = 1;
        while tg_w < layer_head_dim as u64 && tg_w < MAX_HEAD_DIM_SINGLE_SG as u64 {
            tg_w <<= 1;
        }
        enc.set_compute_pipeline_state(&self.attention.attn_fused_pipeline);
        enc.set_buffer(0, Some(bufs.q_out), 0);
        enc.set_buffer(1, Some(bufs.k_out), 0);
        enc.set_buffer(2, Some(bufs.v_out), 0);
        enc.set_buffer(3, Some(&cache.k_cache), 0);
        enc.set_buffer(4, Some(&cache.v_cache), 0);
        enc.set_buffer(5, Some(bufs.attn_out_buf), 0);
        enc.set_buffer(6, Some(&q_w_buf), 0);
        enc.set_buffer(7, Some(&k_w_buf), 0);
        enc.set_bytes(8, 4, &t_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(9, 4, &hd_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(10, 4, &nq_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(11, 4, &nkv_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(12, 4, &c.scale as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            13,
            4,
            &c.window_size as *const u32 as *const std::ffi::c_void,
        );
        enc.set_bytes(14, 4, &c.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(15, 4, &qk_off as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(17, 4, &rdim as *const u32 as *const std::ffi::c_void);
        // Attention sinks (GPT-OSS) — see `stages::sinks`.
        crate::stages::sinks::bind(
            enc,
            ATTN_FUSED_SINKS_INDEX,
            layer.attn_sinks,
            layer_num_q_heads,
        );
        crate::stages::rope_freq::bind(
            enc,
            ATTN_FUSED_INV_FREQ_INDEX,
            ATTN_FUSED_AMPLITUDE_INDEX,
            &layer.rope_freq,
            layer_head_dim,
            layer.rotary_dim,
        );
        enc.set_bytes(
            ATTN_FUSED_ABS_POS_INDEX,
            4,
            &c.pos as *const u32 as *const std::ffi::c_void,
        );
        enc.set_bytes(
            ATTN_FUSED_SOFTCAP_INDEX,
            4,
            &layer.attn_softcap as *const f32 as *const std::ffi::c_void,
        );
        // QK-norm flag + Q/K/V projection biases (GPT-OSS). Presence
        // is all-or-nothing here by `attn_fused_will_fire`'s gate;
        // absent slots bind the empty-slice stub, unread under the
        // flags.
        let has_qk_norm: u32 = u32::from(c.q_norm_enabled);
        enc.set_bytes(
            ATTN_FUSED_HAS_QK_NORM_INDEX,
            4,
            &has_qk_norm as *const u32 as *const std::ffi::c_void,
        );
        let has_qkv_bias: u32 = u32::from(layer.attn_q_bias.is_some());
        let qb_buf = self.bufs.get_f32(layer.attn_q_bias.unwrap_or(&[]));
        let kb_buf = self.bufs.get_f32(layer.attn_k_bias.unwrap_or(&[]));
        let vb_buf = self.bufs.get_f32(layer.attn_v_bias.unwrap_or(&[]));
        enc.set_buffer(ATTN_FUSED_QKV_BIAS_INDEX, Some(&qb_buf), 0);
        enc.set_buffer(ATTN_FUSED_QKV_BIAS_INDEX + 1, Some(&kb_buf), 0);
        enc.set_buffer(ATTN_FUSED_QKV_BIAS_INDEX + 2, Some(&vb_buf), 0);
        enc.set_bytes(
            ATTN_FUSED_QKV_BIAS_INDEX + 3,
            4,
            &has_qkv_bias as *const u32 as *const std::ffi::c_void,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(layer_num_q_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
        kv_cache.layers[layer_idx].advance_one();
    }
}
