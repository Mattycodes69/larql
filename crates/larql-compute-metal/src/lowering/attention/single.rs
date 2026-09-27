//! One attention op at one position, including the split-K KV path.

use super::super::profile::{Stage, StageEncoders};
use super::super::{
    nvfp4_residual_fusion_enabled, nvfp4_segment, MatvecOperands, MatvecTarget, Nvfp4Segment,
};
use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef};

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    /// Encode one attention op, hidden state in to hidden state out.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention(
        &self,
        encs: &mut dyn StageEncoders,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &AttnWeights<'_>,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
    ) {
        let (q_rows, kv_rows) = (shape.q_rows(), shape.kv_rows());
        let slot = shape.kv_slot_offset();

        // 1. pre-attention norm.
        let enc = encs.stage(Stage::AttnNorm);
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            h_in,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        // 2. projections. K and V land in their cache slots directly;
        //    a projection's bias joins its output there, before anything
        //    downstream reads it.
        let enc = encs.stage(Stage::AttnProj);
        let projections = [
            (&w.q, w.q_bias, s.q, 0u64, q_rows),
            (&w.k, w.k_bias, s.k_cache, slot, kv_rows),
            (&w.v, w.v_bias, s.v_cache, slot, kv_rows),
        ];
        // A-5b: three NVFP4 projections of one input are one dispatch —
        // the per-dispatch α paid once. Any other mix encodes per matrix.
        let fused: Option<Vec<Nvfp4Segment<'_>>> = projections
            .iter()
            .map(|(p, _, out, off, n)| nvfp4_segment(p, out, *off, *n))
            .collect();
        match fused {
            Some(segments) => {
                self.encode_nvfp4_matvec_segments(enc, s.normed, shape.hidden, &segments)
            }
            None => {
                for (p, _, out, off, n) in &projections {
                    self.encode_matvec(
                        enc,
                        p,
                        &MatvecTarget {
                            x: s.normed,
                            out,
                            out_offset: *off,
                            n: *n,
                            k: shape.hidden,
                        },
                    );
                }
            }
        }
        for (_, bias, out, off, n) in &projections {
            if let Some(bias) = bias {
                self.encode_bias_add(enc, out, *off, bias, *n);
            }
        }
        if let Some(g) = &w.gate {
            // The gate reads the *attention input* — the same normalised
            // vector the projections read, per `GateSource::AttentionInput`.
            self.encode_matvec(
                enc,
                g,
                &MatvecTarget {
                    x: s.normed,
                    out: s.gate,
                    out_offset: 0,
                    n: q_rows,
                    k: shape.hidden,
                },
            );
        }
        // 3. norms on the projections. V first: its norm reads the raw
        //    projection (on a K≡V layer V was projected from the K matrix
        //    into its own slot, so the key's norm below does not touch it).
        let enc = encs.stage(Stage::AttnQkOps);
        if shape.parameter_free_v {
            self.encode_parameter_free_qk_norm(
                enc,
                s.v_cache,
                slot,
                shape.num_kv_heads,
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
                shape.num_q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
            self.encode_weighted_qk_norm(
                enc,
                s.k_cache,
                slot,
                qk.k,
                shape.num_kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
                qk.weight_offset,
            );
        }
        //    Parameter-free QK norm — Q and K independently, per head.
        if shape.parameter_free_q {
            self.encode_parameter_free_qk_norm(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if shape.parameter_free_k {
            self.encode_parameter_free_qk_norm(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        // 4. query scale — Q only, after the norm, before RoPE.
        if let Some(scale) = shape.query_scale {
            self.encode_scale_vector(enc, s.q, q_rows, scale);
        }
        // 5. position encoding, from this layer's policy. `None` means
        //    the op is absent and nothing is encoded — not rotation by
        //    a zero angle, which would also be a no-op here but would
        //    be the wrong reason.
        if let Some(amplitude) = shape.position.amplitude() {
            self.encode_rope(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                s.inv_freq,
                shape.position_index,
                amplitude,
            );
            self.encode_rope(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                s.inv_freq,
                shape.position_index,
                amplitude,
            );
        }
        // 6. attention over the cache.
        let enc = encs.stage(Stage::AttnCore);
        self.encode_kv_attention(enc, s, shape, w.sinks, 0, 0);
        // 7. the judged gate, then the output projection.
        let aggregated = match &w.gate {
            Some(_) => {
                let enc = encs.stage(Stage::AttnOut);
                self.encode_sigmoid_gate(enc, s.concat, s.gate, s.gated, q_rows);
                s.gated
            }
            None => s.concat,
        };
        // A-5b rung 2a: under two-norm placement with no output bias the
        // residual add folds into the o-proj write (bit-identical: the
        // same fp32 add), saving one dispatch per layer. Otherwise the
        // projection, bias and branch-norm/residual encode as before.
        let fused_out = match (w.post_norm.as_ref(), w.o_bias) {
            (None, None) if shape.residual_scale.is_none() && nvfp4_residual_fusion_enabled() => {
                nvfp4_segment(&w.o, h_out, 0, shape.hidden)
            }
            _ => None,
        };
        // The projection is its own stage so the profiler can price the
        // GEMV apart from the gate, bias, branch norm and residual around
        // it; production's `SingleEncoder` ignores the mark.
        let enc = encs.stage(Stage::AttnOProj);
        match fused_out {
            // `_sliced` carries the segment's byte offsets: under the
            // packed attention layout `o` is a row slice of the shared
            // allocation, and binding it at offset 0 would silently
            // compute the Q projection's rows instead.
            Some(seg) => self.encode_nvfp4_matvec_residual_sliced(
                enc,
                &MatvecOperands {
                    packed: seg.packed,
                    scales: seg.scales,
                    x: aggregated,
                    out: h_out,
                    out_offset: 0,
                    n: shape.hidden,
                    k: q_rows,
                },
                seg.tensor_scale,
                h_in,
                seg.packed_offset,
                seg.scales_offset,
            ),
            None => {
                self.encode_matvec(
                    enc,
                    &w.o,
                    &MatvecTarget {
                        x: aggregated,
                        out: s.attn_out,
                        out_offset: 0,
                        n: shape.hidden,
                        k: q_rows,
                    },
                );
                let enc = encs.stage(Stage::AttnOut);
                if let Some(bias) = w.o_bias {
                    self.encode_bias_add(enc, s.attn_out, 0, bias, shape.hidden);
                }
                // 8. post-attention norm (four-norm placement only), then
                //    the residual — branch output normalised *before* the add.
                self.encode_branch_norm_then_residual(
                    enc,
                    h_in,
                    s.attn_out,
                    h_out,
                    w.post_norm.as_ref(),
                    shape.hidden,
                    shape.residual_scale.unwrap_or(1.0),
                );
            }
        }
    }

    /// Attention over the cache: the same tiered dispatch the production
    /// decode path uses (`decode/encode_attn.rs`), transcribed rather than
    /// re-derived.
    ///
    /// - **Span tier.** `kv_attention` holds `SHORT_ATTENTION_SPAN` scores
    ///   in threadgroup memory; a span past it must run the long kernel,
    ///   whose bound is `LONG_ATTENTION_SPAN`. Before this the lowering
    ///   pinned the short kernel for every span, i.e. overflowed its
    ///   threadgroup scratch past 1024 — the sliding layers' 2048 window
    ///   and every full layer past 1K.
    /// - **Sequence-parallel phase 3 (KV-B1).** The weighted-V walk is
    ///   ~85% of long-span cost and the serial kernel walks it with
    ///   `head_dim` threads per head — 32 threadgroups regardless of
    ///   context on Glimmer. `ops::attention_geometry` resolves the
    ///   operator's `LARQL_KV_SEQPAR` against this op's full geometry
    ///   (head_dim, q heads, kv heads, span); serial for any geometry
    ///   without a measured row. The seqpar kernels bind the same
    ///   thirteen slots and differ only in a `slices x head_dim`
    ///   threadgroup and a reassociated sum, gated at 1e-4 against serial
    ///   in `tests/test_kernel_kv_attention_seqpar.rs`.
    pub(super) fn encode_kv_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
        sinks: Option<&Buffer>,
        q_offset: u64,
        out_offset: u64,
    ) {
        use crate::ops::kv_cache::{attention_span, LONG_ATTENTION_SPAN, SHORT_ATTENTION_SPAN};

        let window = shape.window.unwrap_or(0) as u32;
        let span = attention_span(shape.kv_len as u32, window);
        assert!(
            span as usize <= LONG_ATTENTION_SPAN,
            "attention span {span} exceeds the long kernel's threadgroup scratch \
             bound ({LONG_ATTENTION_SPAN})"
        );
        let long = span > SHORT_ATTENTION_SPAN;
        if self.encode_kv_attention_splitk(enc, s, shape, sinks, q_offset, out_offset, 1) {
            return;
        }
        // The planner, not the head_dim-only policy: this op's full
        // semantic geometry decides its execution geometry.
        let slices = crate::ops::attention_geometry::choose_attention_geometry(
            self.decode_flags.kv_seqpar,
            &crate::ops::attention_geometry::AttentionGeometryQuery {
                head_dim: shape.head_dim,
                num_q_heads: shape.num_q_heads,
                num_kv_heads: shape.num_kv_heads,
                span,
            },
        )
        .slices();
        let (pipeline, threads) = if slices > 1 {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SEQPAR);
            let p = if long {
                &self.attention.kv_attend_seqpar_long_pipeline
            } else {
                &self.attention.kv_attend_seqpar_pipeline
            };
            (p, (slices * shape.head_dim) as u64)
        } else {
            crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SERIAL);
            let p = if long {
                &self.attention.kv_attend_long_pipeline
            } else {
                &self.attention.kv_attend_pipeline
            };
            (p, p.max_total_threads_per_threadgroup().min(256))
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.q), q_offset);
        enc.set_buffer(1, Some(s.k_cache), 0);
        enc.set_buffer(2, Some(s.v_cache), 0);
        enc.set_buffer(3, Some(s.concat), out_offset);
        super::super::set_u32(enc, 4, shape.kv_len as u32);
        super::super::set_u32(enc, 5, shape.head_dim as u32);
        super::super::set_u32(enc, 6, shape.num_q_heads as u32);
        super::super::set_u32(enc, 7, shape.num_kv_heads as u32);
        super::super::set_f32(enc, 8, shape.score_scale);
        super::super::set_u32(enc, 9, window);
        // Sinks: the plan's per-head logits when the layer carries the
        // judged semantics. The kernels read slot 10 only when `has_sinks`
        // is non-zero, but Metal still requires the binding to exist, so
        // a sink-free layer binds a one-float placeholder — `inv_freq` is
        // a live buffer of the right kind and is not read by this kernel.
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
            metal::MTLSize::new(shape.num_q_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }

    /// SPLITK-1 (`ops::kv_splitk`): when the planner has a split-K row
    /// for this geometry at every one of the `rows` positions' spans —
    /// and the SAME chunk/slice choice for all of them, so a verify row
    /// is the per-position dispatch's arithmetic exactly — encode both
    /// passes and return `true`. Otherwise encode nothing and return
    /// `false`, leaving the caller's seqpar/serial path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_kv_attention_splitk(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
        sinks: Option<&Buffer>,
        q_offset: u64,
        out_offset: u64,
        rows: usize,
    ) -> bool {
        use crate::ops::attention_geometry::{choose_splitk, AttentionGeometryQuery};
        use crate::ops::kv_cache::attention_span;
        let Some(scratch) = s.splitk else {
            return false;
        };
        if !scratch.fits(rows, shape.num_q_heads, shape.q_rows()) {
            return false;
        }
        let window = shape.window.unwrap_or(0) as u32;
        let at = |i: usize| {
            choose_splitk(
                self.decode_flags.kv_seqpar,
                &AttentionGeometryQuery {
                    head_dim: shape.head_dim,
                    num_q_heads: shape.num_q_heads,
                    num_kv_heads: shape.num_kv_heads,
                    span: attention_span((shape.kv_len + i) as u32, window),
                },
            )
        };
        let Some(geometry) = at(0) else {
            return false;
        };
        if (1..rows).any(|i| at(i) != Some(geometry)) {
            return false;
        }
        crate::route_witness::bump(&crate::route_witness::LOWERED_ATTEND_SPLITK);
        crate::ops::kv_splitk::SplitKDispatch {
            q: s.q,
            q_offset,
            k_cache: s.k_cache,
            v_cache: s.v_cache,
            out: s.concat,
            out_offset,
            o_part: scratch.o_part,
            ml_part: scratch.ml_part,
            sinks,
            // A live buffer bound only to satisfy slot 6; never read
            // without sinks (as `encode_kv_attention`'s slot 10).
            placeholder: s.inv_freq,
            kv_len: shape.kv_len as u32,
            head_dim: shape.head_dim,
            num_q_heads: shape.num_q_heads,
            num_kv_heads: shape.num_kv_heads,
            score_scale: shape.score_scale,
            window,
            softcap: shape.softcap.unwrap_or(0.0),
            n_chunks: geometry.chunks,
            slices: geometry.slices,
            rows,
        }
        .encode(&self.attention, enc);
        true
    }
}
