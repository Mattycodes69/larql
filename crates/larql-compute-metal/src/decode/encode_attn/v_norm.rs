//! Step 3: batched V-norm (Gemma 4), in place on `v_out`.

use metal::{ComputeCommandEncoderRef, MTLSize};

use super::context::AttnCtx;
use crate::ops::kv_cache::MAX_HEAD_DIM_DOUBLE_SG;
use crate::MetalBackend;

impl MetalBackend {
    pub(super) fn encode_v_norm(&self, enc: &ComputeCommandEncoderRef, c: &AttnCtx<'_, '_, '_>) {
        let layer_head_dim = c.head_dim;
        let layer_num_kv_heads = c.num_kv_heads;
        let hd_val = layer_head_dim as u32;
        let num_kv = layer_num_kv_heads as u32;
        let mut tg_w: u64 = 1;
        while tg_w < layer_head_dim as u64 && tg_w < MAX_HEAD_DIM_DOUBLE_SG as u64 {
            tg_w <<= 1;
        }
        enc.set_compute_pipeline_state(&self.norms.v_norm_batched_pipeline);
        enc.set_buffer(0, Some(c.bufs.v_out), 0);
        enc.set_buffer(1, Some(c.bufs.v_out), 0);
        enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(3, 4, &c.eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(layer_num_kv_heads as u64, 1, 1),
            MTLSize::new(tg_w, 1, 1),
        );
    }
}
