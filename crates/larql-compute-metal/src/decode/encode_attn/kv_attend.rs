//! Step 4: KV append + attend on the non-`attn_fused` paths.
//!
//! Three arms, picked in this order:
//! 1. fused `kv_append_attend_fused` (optionally sequence-parallel) —
//!    own-cache layers whose span fits the kernel's score array;
//! 2. shared layer — attend-only against the source layer's cache;
//! 3. unfused `encode_kv_append` + `encode_kv_attend` (or its seqpar arm).
//!
//! `current_len` is advanced by the caller, not here.

use metal::{ComputeCommandEncoderRef, MTLSize};

use super::context::AttnCtx;
use super::{KV_APPEND_ATTEND_SINKS_INDEX, KV_APPEND_ATTEND_SOFTCAP_INDEX};
use crate::ops;
use crate::MetalBackend;

impl MetalBackend {
    pub(super) fn encode_kv_append_attend(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        kv_cache: &ops::kv_cache::KVCache,
    ) {
        if c.use_fused_kv_aa {
            self.encode_kv_append_attend_fused(enc, c, kv_cache);
        } else if let Some(src) = c.kv_shared_source {
            self.encode_kv_attend_shared(enc, c, &kv_cache.layers[src]);
        } else {
            self.encode_kv_append_then_attend(enc, c, kv_cache);
        }
    }

    fn encode_kv_append_attend_fused(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        kv_cache: &ops::kv_cache::KVCache,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let layer_head_dim = c.head_dim;
        let layer_num_q_heads = c.num_q_heads;
        let cache = &kv_cache.layers[c.layer_idx];
        let t_val = (cache.current_len + 1) as u32;
        let hd = cache.head_dim as u32;
        let num_q_val = layer_num_q_heads as u32;
        let num_kv = cache.num_kv_heads as u32;
        // KV-B1: phase 3 walks the span serially on `head_dim` threads
        // in the baseline kernel. When slices are requested, dispatch
        // the sequence-parallel variant at `slices * head_dim` threads
        // instead. The planner resolves the request against this
        // layer's full geometry and returns 0 when the baseline should
        // run.
        let fused_slices = ops::attention_geometry::choose_attention_geometry(
            self.decode_flags.kv_seqpar,
            &ops::attention_geometry::AttentionGeometryQuery {
                head_dim: layer_head_dim,
                num_q_heads: layer_num_q_heads,
                num_kv_heads: cache.num_kv_heads,
                span: c.attn_span,
            },
        )
        .slices();
        enc.set_compute_pipeline_state(if fused_slices > 1 {
            &self.attention.kv_append_attend_fused_seqpar_pipeline
        } else {
            &self.attention.kv_append_attend_fused_pipeline
        });
        enc.set_buffer(0, Some(bufs.q_out), 0);
        enc.set_buffer(1, Some(&cache.k_cache), 0);
        enc.set_buffer(2, Some(&cache.v_cache), 0);
        enc.set_buffer(3, Some(bufs.attn_out_buf), 0);
        enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &num_q_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &c.scale as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(
            9,
            4,
            &c.window_size as *const u32 as *const std::ffi::c_void,
        );
        enc.set_buffer(10, Some(bufs.k_out), 0);
        enc.set_buffer(11, Some(bufs.v_out), 0);
        // Attention sinks (GPT-OSS) — see `stages::sinks`.
        crate::stages::sinks::bind(
            enc,
            KV_APPEND_ATTEND_SINKS_INDEX,
            layer.attn_sinks,
            layer_num_q_heads,
        );
        enc.set_bytes(
            KV_APPEND_ATTEND_SOFTCAP_INDEX,
            4,
            &layer.attn_softcap as *const f32 as *const std::ffi::c_void,
        );
        let fused_width = crate::kernels::DISPATCH_TG_MAX_THREADS.min(layer_head_dim as u64)
            * fused_slices.max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(layer_num_q_heads as u64, 1, 1),
            MTLSize::new(fused_width, 1, 1),
        );
    }

    /// Shared layer: attend against source's cache. Skip append (the
    /// source already appended its K/V row this token). `t_val =
    /// source.current_len` because source has already incremented it to
    /// count the new token.
    fn encode_kv_attend_shared(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        cache: &ops::kv_cache::LayerKVCache,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let layer_num_q_heads = c.num_q_heads;
        let window_size = c.window_size;
        let t_val_u = cache.current_len as u32;
        let hd_u = cache.head_dim as u32;
        let num_q_u = layer_num_q_heads as u32;
        let num_kv_u = cache.num_kv_heads as u32;
        // Pick the same pipeline encode_kv_attend would (long-form for
        // spans past SHORT_ATTENTION_SPAN; standard otherwise).
        let span_u = ops::kv_cache::attention_span(t_val_u, window_size);
        let pipeline = if span_u > ops::kv_cache::SHORT_ATTENTION_SPAN {
            &self.attention.kv_attend_long_pipeline
        } else {
            &self.attention.kv_attend_pipeline
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(bufs.q_out), 0);
        enc.set_buffer(1, Some(&cache.k_cache), 0);
        enc.set_buffer(2, Some(&cache.v_cache), 0);
        enc.set_buffer(3, Some(bufs.attn_out_buf), 0);
        enc.set_bytes(4, 4, &t_val_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &hd_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &num_q_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &num_kv_u as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(8, 4, &c.scale as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
        crate::stages::sinks::bind(enc, 10, layer.attn_sinks, layer_num_q_heads);
        enc.set_bytes(
            12,
            4,
            &layer.attn_softcap as *const f32 as *const std::ffi::c_void,
        );
        enc.dispatch_thread_groups(
            MTLSize::new(layer_num_q_heads as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(cache.head_dim as u64),
                1,
                1,
            ),
        );
    }

    fn encode_kv_append_then_attend(
        &self,
        enc: &ComputeCommandEncoderRef,
        c: &AttnCtx<'_, '_, '_>,
        kv_cache: &ops::kv_cache::KVCache,
    ) {
        let layer = c.layer;
        let bufs = &c.bufs;
        let layer_idx = c.layer_idx;
        let layer_num_q_heads = c.num_q_heads;
        ops::kv_cache::encode_kv_append(
            enc,
            &kv_cache.layers[layer_idx],
            &self.attention.kv_append_pipeline,
            bufs.k_out,
            bufs.v_out,
        );
        // KV-B1 (`LARQL_KV_SEQPAR`): phase 3 — the weighted-V
        // accumulation — is ~85% of long-span attention and walks the
        // span serially with `head_dim` threads. The seqpar arm splits
        // it across N slices. The planner resolves the request against
        // this layer's full geometry — a measured row (gpt-oss's is the
        // A/B/C-licensed policy) or serial where none exists — and
        // returns 0 when the shipped kernel should run instead.
        let slices = ops::attention_geometry::choose_attention_geometry(
            self.decode_flags.kv_seqpar,
            &ops::attention_geometry::AttentionGeometryQuery {
                head_dim: kv_cache.layers[c.attend_cache_idx].head_dim,
                num_q_heads: layer_num_q_heads,
                num_kv_heads: kv_cache.layers[c.attend_cache_idx].num_kv_heads,
                span: c.attn_span,
            },
        )
        .slices();
        if slices > 1 {
            ops::kv_cache::encode_kv_attend_seqpar(
                enc,
                &kv_cache.layers[layer_idx],
                &self.attention.kv_attend_seqpar_pipeline,
                &self.attention.kv_attend_seqpar_long_pipeline,
                bufs.q_out,
                bufs.attn_out_buf,
                layer_num_q_heads,
                c.scale,
                c.window_size,
                layer.attn_sinks,
                layer.attn_softcap,
                slices,
            );
        } else {
            ops::kv_cache::encode_kv_attend(
                enc,
                &kv_cache.layers[layer_idx],
                &self.attention.kv_attend_pipeline,
                Some(&self.attention.kv_attend_long_pipeline),
                bufs.q_out,
                bufs.attn_out_buf,
                layer_num_q_heads,
                c.scale,
                c.window_size,
                layer.attn_sinks,
                layer.attn_softcap,
            );
        }
    }
}
