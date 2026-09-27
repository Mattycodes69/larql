//! One attention op over a block of consecutive positions (VERIFY-N).

use super::super::profile::{Stage, StageEncoders};
use super::super::MatmulRowsTarget;
use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef};

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    /// VERIFY-N: one attention op over `rows` consecutive positions
    /// starting at `shape.position_index`, hidden states `[rows, hidden]`
    /// in and out.
    ///
    /// The same ordered program as [`Self::encode_attention`], with the
    /// projections batched — each weight stream read once for all rows —
    /// and everything position-dependent (norms, rotation, attention)
    /// dispatched per row at its own offset. Row `i` attends the cache up
    /// to and including position `base + i`, so the block is causal by
    /// construction: rows `i+1..` are written to the cache but never read
    /// by row `i`.
    ///
    /// `s`'s `normed`/`q`/`concat`/`attn_out` hold `rows` positions;
    /// `post_scratch` is `[rows, hidden]` and stands in for the post-norm's
    /// single-position scratch. Refuses the attention output gate, which
    /// no verified model needs yet.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_rows(
        &self,
        encs: &mut dyn StageEncoders,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &AttnWeights<'_>,
        s: &AttnScratch<'_>,
        post_scratch: &Buffer,
        shape: &AttnShape,
        rows: usize,
    ) -> Result<(), String> {
        if w.gate.is_some() {
            return Err("multi-position attention has no output-gate lowering yet".into());
        }
        let f = std::mem::size_of::<f32>() as u64;
        let (q_rows, kv_rows, hidden) = (shape.q_rows(), shape.kv_rows(), shape.hidden);
        let base = shape.position_index;
        let at = |i: usize| AttnShape {
            position_index: base + i,
            kv_len: base + i + 1,
            ..shape.clone()
        };
        let h_off = |i: usize| (i * hidden) as u64 * f;
        let q_off = |i: usize| (i * q_rows) as u64 * f;

        let enc = encs.stage(Stage::AttnNorm);
        self.encode_rms_norm_rows(
            enc,
            h_in,
            0,
            w.norm_weight,
            s.normed,
            0,
            hidden,
            rows,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        let enc = encs.stage(Stage::AttnProj);
        let slot = shape.kv_slot_offset();
        for (m, out, off, n) in [
            (&w.q, s.q, 0u64, q_rows),
            (&w.k, s.k_cache, slot, kv_rows),
            (&w.v, s.v_cache, slot, kv_rows),
        ] {
            self.encode_matmul_rows(
                enc,
                m,
                &MatmulRowsTarget {
                    x: s.normed,
                    x_offset: 0,
                    out,
                    out_offset: off,
                    n,
                    k: hidden,
                    rows,
                },
            )?;
        }
        for i in 0..rows {
            let slot_i = at(i).kv_slot_offset();
            if let Some(b) = w.q_bias {
                self.encode_bias_add(enc, s.q, q_off(i), b, q_rows);
            }
            if let Some(b) = w.k_bias {
                self.encode_bias_add(enc, s.k_cache, slot_i, b, kv_rows);
            }
            if let Some(b) = w.v_bias {
                self.encode_bias_add(enc, s.v_cache, slot_i, b, kv_rows);
            }
        }
        // Per-head norms and rotation over the whole block: Q rows and the
        // block's K/V cache slots are each one contiguous `[rows, heads,
        // head_dim]` run, so a per-head kernel covers every position in
        // one dispatch with the same per-head arithmetic.
        let enc = encs.stage(Stage::AttnQkOps);
        let (q_heads, kv_heads) = (rows * shape.num_q_heads, rows * shape.num_kv_heads);
        if shape.parameter_free_v {
            self.encode_parameter_free_qk_norm(
                enc,
                s.v_cache,
                slot,
                kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if let Some(qk) = &w.qk_norm {
            self.encode_weighted_qk_norm(
                enc,
                s.q,
                0,
                qk.q,
                q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
            self.encode_weighted_qk_norm(
                enc,
                s.k_cache,
                slot,
                qk.k,
                kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
        }
        if shape.parameter_free_q {
            self.encode_parameter_free_qk_norm(
                enc,
                s.q,
                0,
                q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if shape.parameter_free_k {
            self.encode_parameter_free_qk_norm(
                enc,
                s.k_cache,
                slot,
                kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        // Query scale is elementwise and position-free.
        if let Some(scale) = shape.query_scale {
            self.encode_scale_vector(enc, s.q, rows * q_rows, scale);
        }
        if let Some(amplitude) = shape.position.amplitude() {
            self.encode_rope_rows(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                s.inv_freq,
                base,
                amplitude,
                rows,
            );
            self.encode_rope_rows(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                s.inv_freq,
                base,
                amplitude,
                rows,
            );
        }
        let enc = encs.stage(Stage::AttnCore);
        if !self.encode_kv_attention_rows(enc, s, shape, w.sinks, rows) {
            for i in 0..rows {
                self.encode_kv_attention(enc, s, &at(i), w.sinks, q_off(i), q_off(i));
            }
        }
        let enc = encs.stage(Stage::AttnOut);
        self.encode_matmul_rows(
            enc,
            &w.o,
            &MatmulRowsTarget {
                x: s.concat,
                x_offset: 0,
                out: s.attn_out,
                out_offset: 0,
                n: hidden,
                k: q_rows,
                rows,
            },
        )?;
        if let Some(b) = w.o_bias {
            for i in 0..rows {
                self.encode_bias_add(enc, s.attn_out, h_off(i), b, hidden);
            }
        }
        self.encode_branch_norm_then_residual_rows(
            enc,
            h_in,
            s.attn_out,
            h_out,
            w.post_norm.as_ref(),
            post_scratch,
            hidden,
            rows,
            shape.residual_scale.unwrap_or(1.0),
        );
        Ok(())
    }
}

impl MetalBackend {
    /// VERIFY-N: the whole block's attention in ONE dispatch (grid
    /// `(num_q, rows)`), when every row would take the same short kernel —
    /// serial (`kv_attention_rows`) or sequence-parallel at one slice count
    /// (`kv_attention_seqpar_rows`). Each rows kernel shares its
    /// per-position kernel's body and threadgroup width, so every row is
    /// bit-identical to its per-position dispatch. Returns `false` (and
    /// encodes nothing) for a block per-position dispatch would route
    /// otherwise — a span past the short kernel, or rows that resolve to
    /// different geometries — so the caller keeps its loop for exactly
    /// those.
    pub(super) fn encode_kv_attention_rows(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
        sinks: Option<&Buffer>,
        rows: usize,
    ) -> bool {
        use crate::ops::kv_cache::{attention_span, SHORT_ATTENTION_SPAN};
        // Split-K first: no short-span bound (each chunk holds its own
        // scores), same per-row arithmetic as the per-position dispatch.
        if self.encode_kv_attention_splitk(enc, s, shape, sinks, 0, 0, rows) {
            return true;
        }
        let window = shape.window.unwrap_or(0) as u32;
        // The last row has the longest span; every row must fit.
        let widest = attention_span((shape.kv_len + rows - 1) as u32, window);
        if widest > SHORT_ATTENTION_SPAN {
            return false;
        }
        // Every row must resolve to the SAME geometry — the per-position
        // path would dispatch each row with its own, and one dispatch can
        // only carry one.
        let slices_at = |i: usize| {
            let span = attention_span((shape.kv_len + i) as u32, window);
            crate::ops::attention_geometry::choose_attention_geometry(
                self.decode_flags.kv_seqpar,
                &crate::ops::attention_geometry::AttentionGeometryQuery {
                    head_dim: shape.head_dim,
                    num_q_heads: shape.num_q_heads,
                    num_kv_heads: shape.num_kv_heads,
                    span,
                },
            )
            .slices()
        };
        let slices = slices_at(0);
        if (1..rows).any(|i| slices_at(i) != slices) {
            return false;
        }
        // Each arm's threadgroup width is the per-position path's own, so
        // the reductions are identical row for row.
        let (pipeline, threads) = if slices > 1 {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SEQPAR);
            (
                &self.attention.kv_attend_seqpar_rows_pipeline,
                (slices * shape.head_dim) as u64,
            )
        } else {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SERIAL);
            (
                &self.attention.kv_attend_rows_pipeline,
                self.attention
                    .kv_attend_pipeline
                    .max_total_threads_per_threadgroup()
                    .min(256),
            )
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.q), 0);
        enc.set_buffer(1, Some(s.k_cache), 0);
        enc.set_buffer(2, Some(s.v_cache), 0);
        enc.set_buffer(3, Some(s.concat), 0);
        super::super::set_u32(enc, 4, shape.kv_len as u32);
        super::super::set_u32(enc, 5, shape.head_dim as u32);
        super::super::set_u32(enc, 6, shape.num_q_heads as u32);
        super::super::set_u32(enc, 7, shape.num_kv_heads as u32);
        super::super::set_f32(enc, 8, shape.score_scale);
        super::super::set_u32(enc, 9, window);
        match sinks {
            Some(sinks) => {
                enc.set_buffer(10, Some(sinks), 0);
                super::super::set_u32(enc, 11, 1);
            }
            None => {
                enc.set_buffer(10, Some(s.inv_freq), 0);
                super::super::set_u32(enc, 11, 0);
            }
        }
        super::super::set_f32(enc, 12, shape.softcap.unwrap_or(0.0));
        enc.dispatch_thread_groups(
            metal::MTLSize::new(shape.num_q_heads as u64, rows as u64, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        true
    }
}
