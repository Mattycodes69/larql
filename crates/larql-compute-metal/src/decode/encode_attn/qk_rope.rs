//! Steps 1.5 + 2: QK-norm + RoPE on the non-`attn_fused` paths.
//!
//! Two shapes: the fused `qk_norm_rope_fused` single dispatch (default when
//! both norm weights exist), and the unfused chain — an optional
//! `qk_norm_qk` dispatch followed by the batched RoPE dispatch.

use metal::{ComputeCommandEncoderRef, MTLSize};

use super::context::AttnCtx;
use crate::ops::kv_cache::MAX_HEAD_DIM_DOUBLE_SG;
use crate::MetalBackend;

impl MetalBackend {
    /// Fused QK-norm + RoPE in one dispatch. The caller has checked both
    /// `q_norm_weight` and `k_norm_weight` are present.
    pub(super) fn encode_qk_norm_rope_fused(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let layer_head_dim = c.head_dim;
        let q_w = layer
            .q_norm_weight
            .expect("caller gated on q_norm_weight.is_some()");
        let k_w = layer
            .k_norm_weight
            .expect("caller gated on k_norm_weight.is_some()");
        let hd_val = layer_head_dim as u32;
        let nq_val = c.num_q_heads as u32;
        let qk_off = c.qk_norm_offset;
        let rdim = c.rotary_dim as u32;
        let mut tg_w: usize = 1;
        while tg_w < layer_head_dim && tg_w < MAX_HEAD_DIM_DOUBLE_SG {
            tg_w <<= 1;
        }
        let q_w_buf = self.bufs.get_f32(q_w);
        let k_w_buf = self.bufs.get_f32(k_w);
        let total_heads = (c.num_q_heads + c.num_kv_heads) as u64;
        enc.set_compute_pipeline_state(&self.norms.qk_norm_rope_fused_pipeline);
        enc.set_buffer(0, Some(bufs.q_out), 0);
        enc.set_buffer(1, Some(bufs.k_out), 0);
        enc.set_buffer(2, Some(&q_w_buf), 0);
        enc.set_buffer(3, Some(&k_w_buf), 0);
        enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &c.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(9, 4, &c.pos as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(10, 4, &rdim as *const u32 as *const std::ffi::c_void);
        crate::stages::rope_freq::bind(
            enc,
            8,
            11,
            &layer.rope_freq,
            layer_head_dim,
            layer.rotary_dim,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(total_heads, 1, 1),
            MTLSize::new(tg_w as u64, 1, 1),
        );
    }

    /// Unfused chain: QK-norm (when both weights exist), then batched RoPE
    /// on the Q and K heads in one dispatch.
    pub(super) fn encode_qk_norm_then_rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let layer_head_dim = c.head_dim;
        let layer_num_q_heads = c.num_q_heads;
        let layer_num_kv_heads = c.num_kv_heads;
        if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
            let hd_val = layer_head_dim as u32;
            let nq_val = layer_num_q_heads as u32;
            let qk_off = c.qk_norm_offset;
            let mut tg_w: usize = 1;
            while tg_w < layer_head_dim && tg_w < MAX_HEAD_DIM_DOUBLE_SG {
                tg_w <<= 1;
            }
            let q_w_buf = self.bufs.get_f32(q_w);
            let k_w_buf = self.bufs.get_f32(k_w);
            let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
            enc.set_compute_pipeline_state(&self.norms.qk_norm_qk_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_buffer(2, Some(&q_w_buf), 0);
            enc.set_buffer(3, Some(&k_w_buf), 0);
            enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &c.eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(total_heads, 1, 1),
                MTLSize::new(tg_w as u64, 1, 1),
            );
        }

        // ── Step 2: RoPE on Q and K heads (batched — one dispatch each) ──
        let hd = layer_head_dim as u32;
        let rdim = c.rotary_dim as u32;
        let rope_pairs = (c.rotary_dim / 2) as u64;
        let num_q = layer_num_q_heads as u32;
        let total_qk_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
        enc.set_compute_pipeline_state(&self.attention.rope_at_pos_batched_qk_pipeline);
        enc.set_buffer(0, Some(bufs.q_out), 0);
        enc.set_buffer(1, Some(bufs.k_out), 0);
        enc.set_bytes(2, 4, &hd as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &c.pos as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &rdim as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
        crate::stages::rope_freq::bind(
            enc,
            3,
            7,
            &layer.rope_freq,
            layer_head_dim,
            layer.rotary_dim,
        );
        enc.dispatch_threads(
            MTLSize::new(rope_pairs, total_qk_heads, 1),
            MTLSize::new(rope_pairs.min(256), 1, 1),
        );
    }
}
