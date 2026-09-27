//! Step 5b: residual + post-attn norm + ffn-input norm.
//!
//! Writes `h_post_attn = h_buf + o_out` (optionally post-attn-normed and
//! residual-scaled) and the FFN input — f32 `ffn_norm_out` when the FFN
//! runs a k-quant family, Q8 `ffn_q8`/`ffn_q8s` otherwise.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::context::AttnCtx;
use crate::MetalBackend;

impl MetalBackend {
    pub(super) fn encode_post_attn_residual(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        if layer.has_post_norms {
            let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                self.bufs.get_f32(pfn)
            } else {
                bufs.post_attn_norm.clone()
            };
            // The triple-fused kernel has no `b_scale` slot, so a
            // Granite-style `residual_multiplier != 1.0` must take the
            // unfused chain below, which binds it. The equivalent guards
            // in `encode_post_ffn.rs` existed for exactly this reason;
            // this site lacked one (capability audit F18).
            let fusable_residual = layer.residual_multiplier == 1.0;
            if c.use_fused_post_attn && c.ffn_uses_kquant && fusable_residual {
                self.encode_post_attn_triple_fused(enc, c, &pre_ffn_buf);
            } else {
                use crate::ops::full_pipeline::encode_rms_norm;
                encode_rms_norm(
                    enc,
                    &self.norms.rms_norm_pipeline,
                    bufs.o_out_buf,
                    bufs.post_attn_norm,
                    bufs.normed_scratch,
                    c.hidden,
                    c.eps,
                    c.norm_offset,
                );
                // Granite residual_multiplier; 1.0 for every other model.
                // `residual_norm_store` reads it at buffer(8),
                // `residual_norm_q8` at buffer(9).
                if c.ffn_uses_kquant {
                    self.encode_residual_norm_store(enc, c, bufs.normed_scratch, &pre_ffn_buf);
                } else {
                    self.encode_residual_norm_q8(enc, c, bufs.normed_scratch, &pre_ffn_buf);
                }
            }
        } else if c.ffn_uses_kquant {
            self.encode_residual_norm_store(enc, c, bufs.o_out_buf, bufs.post_attn_norm);
        } else {
            self.encode_residual_norm_q8(enc, c, bufs.o_out_buf, bufs.post_attn_norm);
        }
    }

    /// Triple-fused: post_attn_norm + residual_norm + h_post_attn store in
    /// ONE dispatch.
    fn encode_post_attn_triple_fused(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        pre_ffn_buf: &Buffer,
    ) {
        let bufs = &c.bufs;
        enc.set_compute_pipeline_state(&self.norms.post_attn_residual_norm_store_pipeline);
        enc.set_buffer(0, Some(bufs.h_buf), 0);
        enc.set_buffer(1, Some(bufs.o_out_buf), 0);
        enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
        enc.set_buffer(3, Some(pre_ffn_buf), 0);
        enc.set_buffer(4, Some(bufs.ffn_norm_out), 0);
        enc.set_buffer(5, Some(bufs.h_post_attn), 0);
        enc.set_bytes(6, 4, &c.hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &c.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            8,
            4,
            &c.norm_offset as *const f32 as *const std::ffi::c_void,
        );
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), hidden_tg(c.hidden));
    }

    /// `residual_norm_store`: `h_post_attn = h_buf + b_scale * addend`, and
    /// the f32 FFN input `ffn_norm_out = rms_norm(h_post_attn) * norm_w`.
    fn encode_residual_norm_store(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        addend: &Buffer,
        norm_w: &Buffer,
    ) {
        let bufs = &c.bufs;
        let b_scale: f32 = c.layer.residual_multiplier;
        enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
        enc.set_buffer(0, Some(bufs.h_buf), 0);
        enc.set_buffer(1, Some(addend), 0);
        enc.set_buffer(2, Some(norm_w), 0);
        enc.set_buffer(3, Some(bufs.ffn_norm_out), 0);
        enc.set_buffer(4, Some(bufs.h_post_attn), 0);
        enc.set_bytes(5, 4, &c.hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &c.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            7,
            4,
            &c.norm_offset as *const f32 as *const std::ffi::c_void,
        );
        enc.set_bytes(8, 4, &b_scale as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), hidden_tg(c.hidden));
    }

    /// `residual_norm_q8`: same residual as `residual_norm_store`, with the
    /// FFN input written as Q8 (`ffn_q8` + per-block `ffn_q8s`).
    fn encode_residual_norm_q8(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        addend: &Buffer,
        norm_w: &Buffer,
    ) {
        let bufs = &c.bufs;
        let b_scale: f32 = c.layer.residual_multiplier;
        enc.set_compute_pipeline_state(&self.norms.residual_norm_q8_pipeline);
        enc.set_buffer(0, Some(bufs.h_buf), 0);
        enc.set_buffer(1, Some(addend), 0);
        enc.set_buffer(2, Some(norm_w), 0);
        enc.set_buffer(3, Some(bufs.ffn_q8), 0);
        enc.set_buffer(4, Some(bufs.ffn_q8s), 0);
        enc.set_buffer(5, Some(bufs.h_post_attn), 0);
        enc.set_bytes(6, 4, &c.hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &c.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            8,
            4,
            &c.norm_offset as *const f32 as *const std::ffi::c_void,
        );
        enc.set_bytes(9, 4, &b_scale as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), hidden_tg(c.hidden));
    }
}

/// One threadgroup spanning `hidden`, capped at the dispatch maximum —
/// the shape every single-threadgroup residual kernel here uses.
fn hidden_tg(hidden: usize) -> MTLSize {
    MTLSize::new(
        crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
        1,
        1,
    )
}
