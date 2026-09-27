//! Steps 6–8 of one layer: dense FFN + post-FFN residual (optionally split
//! across profiling command buffers), Per-Layer Embeddings, and the MoE /
//! layer-scalar tail.

use metal::Buffer;

use super::super::{encode_ffn, encode_ple, encode_post_ffn, gpu_timing, moe_interleave, profile};
use super::cmd::TokenCmd;
use super::ctx::TokenCtx;
use crate::{MetalBackend, PleInputBuffer};

/// How this layer's FFN is scheduled, decided before it is encoded and
/// needed again by the MoE tail.
#[derive(Clone, Copy)]
pub(super) struct FfnSchedule {
    /// Split mode AND this layer has MoE: the dense FFN is re-encoded on a
    /// fresh command buffer inside the MoE block instead of inline.
    pub defer_ffn_for_split: bool,
    /// `LARQL_PROFILE_SPLIT=1` stage-timing boundaries are active.
    pub stage_timing_split: bool,
}

/// Remote-MoE fire / collect callbacks, as the token entry point takes them.
pub(super) type MoeFireFn<'f> = Option<&'f mut dyn FnMut(usize, &[f32]) -> Vec<f32>>;
pub(super) type MoeCollectFn<'f> = Option<&'f mut dyn FnMut(usize) -> Vec<f32>>;

impl MetalBackend {
    /// Steps 6–8 on the dense path: FFN, post-FFN residual, PLE. Encodes
    /// nothing for MoE-deferred, remote-FFN or pure-MoE layers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_layer_dense_ffn(
        &self,
        tc: &mut TokenCmd,
        gpu_time: &mut gpu_timing::TokenGpuTime,
        t: &TokenCtx<'_, '_>,
        l: usize,
        new_h: &Buffer,
        ple_inputs: Option<&PleInputBuffer>,
        split_mode: bool,
    ) -> FfnSchedule {
        let layers = t.layers;
        let layer = &layers[l];
        let s = t.scratch;
        let hidden = t.hidden;
        let inter = t.inter;
        // ── Steps 6-7: FFN + post-FFN residual ──
        //
        // Skip when in split mode AND this layer has MoE — they will be
        // re-encoded on a fresh command buffer inside the MoE block so
        // they can run in parallel with the remote MoE round trip.  For
        // non-MoE layers (or non-split mode) we encode them inline as
        // before.
        //
        // Also skip when ffn_is_remote: the entire FFN for this layer
        // will be provided by the remote server via moe_fn, so there
        // is no local FFN work to encode on the GPU.
        let defer_ffn_for_split = split_mode && layer.moe.is_some();

        // Pure-MoE layers extract no dense FFN weights — encoding the
        // dense branch would run the kernels over empty slices and
        // poison `new_h` with garbage that the expert add can't
        // recover. Their FFN is the expert block alone; the MoE
        // interleave writes `new_h = h_post_attn + moe_out`
        // directly (same combine as the remote-FFN arm).
        let layer_runs_dense_ffn = layer.has_dense_ffn() || layer.moe.is_none();

        // Stage-timing boundary: when LARQL_PROFILE_SPLIT=1 (or the legacy
        // alias LARQL_DECODE_STAGE_TIMING=1), close the encoder here so
        // attention CB time can be recorded separately from FFN CB time.
        // Adds ~1 commit/wait per layer (~30-50µs each on M3 Max) —
        // measurement-only mode, off by default. Skipped on MoE-deferred
        // layers because their interleave block handles its own commits.
        let stage_timing_split = !defer_ffn_for_split && profile::split_profile_requested();
        if stage_timing_split {
            tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:359");
            gpu_time.record_stage(&tc.cmd, gpu_timing::DecodeStage::Attention);
            tc.reopen(&self.queue);
        }

        let schedule = FfnSchedule {
            defer_ffn_for_split,
            stage_timing_split,
        };
        if defer_ffn_for_split || layer.ffn_is_remote || !layer_runs_dense_ffn {
            return schedule;
        }

        let ffn_bufs = encode_ffn::FfnBufs {
            gate_w: &s.gate_bufs[l],
            up_w: &s.up_bufs[l],
            down_w: &s.down_bufs[l],
            ffn_norm_out: &s.ffn_norm_out,
            ffn_q8: &s.ffn_q8,
            ffn_q8s: &s.ffn_q8s,
            gate_out_scratch: &s.gate_out_scratch,
            up_out: &s.up_out,
            act_buf: &s.act_buf,
            down_out: &s.down_out,
        };
        let ffn_dims = encode_ffn::FfnDims {
            hidden,
            inter,
            inter_padded: s.inter_padded,
        };
        let use_fused_post_ffn = self.decode_flags.fused_post_ffn_norm;
        let post_ffn_bufs = encode_post_ffn::PostFfnBufs {
            down_out: &s.down_out,
            h_post_attn: &s.h_post_attn,
            new_h,
            normed_scratch: &s.normed_scratch,
        };

        // D-RMS-FUSE Phase 1: when env var on AND non-Gemma path
        // (no post_norms) AND there's a next layer, hand the next
        // layer's input_norm weight + the shared norm_f32_buf to
        // `encode_post_ffn_residual` so it can fuse the residual
        // add with the next layer's input rms_norm in one
        // `residual_norm_store` dispatch. Saves 1 dispatch/layer.
        let prelayer_fusion = if !layer.has_post_norms && self.decode_flags.fused_prelayer_norm {
            layers
                .get(l + 1)
                .map(|next| encode_post_ffn::PreLayerNormFusion {
                    next_input_norm: next.input_norm,
                    next_norm_out: &s.norm_f32_buf,
                })
        } else {
            None
        };

        if stage_timing_split && !s.has_moe {
            // Fine split: gate+up in one CB, act+down+residual in another.
            // Step 6a: gate+up
            self.encode_ffn_gate_up_phase(&tc.enc, layer, &ffn_bufs, ffn_dims);
            tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:416");
            gpu_time.record_stage(&tc.cmd, gpu_timing::DecodeStage::GateUp);
            tc.cmd = self.queue.new_command_buffer().to_owned();
            tc.enc = tc.cmd.new_compute_command_encoder().to_owned();
            // Step 6b + 7: activation+down + post-FFN residual
            self.encode_ffn_down_phase(&tc.enc, layer, &ffn_bufs, ffn_dims);
            self.encode_post_ffn_residual(
                &tc.enc,
                layer,
                post_ffn_bufs,
                hidden,
                use_fused_post_ffn,
                prelayer_fusion.as_ref(),
            );
            tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:432");
            gpu_time.record_stage(&tc.cmd, gpu_timing::DecodeStage::Down);
            tc.reopen(&self.queue);
        } else {
            // Production path: whole FFN in one encoder block.
            self.encode_ffn_step(&tc.enc, layer, ffn_bufs, ffn_dims);
            self.encode_post_ffn_residual(
                &tc.enc,
                layer,
                post_ffn_bufs,
                hidden,
                use_fused_post_ffn,
                prelayer_fusion.as_ref(),
            );
        }

        // ── Step 8: Per-Layer Embeddings (Gemma 4 E2B) ──
        // Mirrors `crates/larql-inference/src/forward/ple.rs::apply_per_layer_embedding`.
        // Activates only when (a) the layer has the three PLE
        // weights wired and (b) the inference layer uploaded a
        // precomputed per-layer-input table via
        // `MetalBackend::prepare_ple_inputs` for this generation.
        if let Some(pli) = ple_inputs {
            if layer.ple_spec().is_some() {
                // Reuse two scratches that are dead after the
                // post-FFN residual completes:
                //   - `gate_out_scratch` (`inter` f32) holds the
                //     `[ple_dim]` gate (ple_dim ≪ inter);
                //   - `down_out` (`hidden` f32) holds the projection
                //     output (`[hidden]`).
                // Both buffers' previous data is consumed by
                // `encode_post_ffn_residual` above.
                self.encode_per_layer_embed(
                    &tc.enc,
                    layer,
                    encode_ple::PleBufs {
                        h: new_h,
                        per_layer_input: &pli.buffer,
                        per_layer_input_offset: pli.row_offset_bytes(0, l),
                        gate_scratch: &s.gate_out_scratch,
                        contrib_scratch: &s.down_out,
                    },
                    hidden,
                    pli.ple_dim,
                );
            }
        }
        schedule
    }

    /// CPU MoE interleave for hybrid MoE / remote-FFN layers, or the GPU
    /// layer scalar for every other layer. Returns whether the layer took
    /// the MoE arm (which records its own residual-dump row).
    ///
    /// After the GPU dense-FFN pass, the MoE arm flushes the encoder, runs
    /// the expert block on CPU (direct shared-memory access), then restarts
    /// for the next layer. layer_scalar is applied AFTER MoE so it scales
    /// the combined output (dense + MoE). Applying it before would leave
    /// the MoE contribution unscaled.
    ///
    /// Branch on THIS LAYER being a MoE/remote layer, not on the
    /// model-level `has_moe`: with the model-level test, a dense
    /// layer of a hybrid-MoE model entered `handle_moe_interleave`
    /// (which returns immediately for dense layers) and the
    /// `else` arm's layer_scalar was never applied — the same
    /// mis-scaling class as the 14x incident recorded in
    /// `moe_combine.rs`. Capability audit F9. MoE layers get
    /// their scalar inside `moe_combine::apply_outer_combine`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_layer_moe_or_scalar(
        &self,
        tc: &mut TokenCmd,
        gpu_time: &mut gpu_timing::TokenGpuTime,
        residual_dump: &mut super::super::diag::ResidualDump,
        t: &TokenCtx<'_, '_>,
        l: usize,
        new_h: &Buffer,
        schedule: FfnSchedule,
        layer_in_snapshot: Option<&[f32]>,
        dump_l0_dir: Option<&str>,
        moe_fn: &mut MoeFireFn<'_>,
        moe_collect_fn: &mut MoeCollectFn<'_>,
        inline_moe: Option<&moe_interleave::InlineMoeCtx<'_>>,
    ) -> bool {
        let layer = &t.layers[l];
        let s = t.scratch;
        let layer_is_moe = layer.moe.is_some() || layer.ffn_is_remote;
        if layer_is_moe {
            self.handle_moe_interleave(
                layer,
                moe_interleave::MoeInterleaveCtx {
                    layer_idx: l,
                    num_layers: s.num_layers,
                    hidden: t.hidden,
                    inter: t.inter,
                    inter_padded: s.inter_padded,
                    defer_ffn_for_split: schedule.defer_ffn_for_split,
                    stage_timing_split: schedule.stage_timing_split,
                    layer_in_snapshot,
                    dump_l0_dir,
                },
                moe_interleave::MoeInterleaveBufs {
                    gate_w: &s.gate_bufs[l],
                    up_w: &s.up_bufs[l],
                    down_w: &s.down_bufs[l],
                    h_post_attn: &s.h_post_attn,
                    ffn_norm_out: &s.ffn_norm_out,
                    ffn_q8: &s.ffn_q8,
                    ffn_q8s: &s.ffn_q8s,
                    gate_out_scratch: &s.gate_out_scratch,
                    up_out: &s.up_out,
                    act_buf: &s.act_buf,
                    down_out: &s.down_out,
                    normed_scratch: &s.normed_scratch,
                    new_h,
                },
                moe_interleave::MoeCommandState {
                    cmd: &mut tc.cmd,
                    enc: &mut tc.enc,
                    encoder_ended: &mut tc.encoder_ended,
                    gpu_time,
                    residual_dump,
                },
                moe_fn,
                moe_collect_fn,
                inline_moe,
            );
        } else {
            // ── Step 8: Optional layer scalar (non-MoE layers) ──
            // GPU in-place scale on new_h before it becomes the next layer's input.
            if layer.layer_scalar != 0.0 {
                crate::stages::layer_scalar::encode(
                    &tc.enc,
                    &self.norms.scale_vector_pipeline,
                    new_h,
                    1,
                    t.hidden,
                    layer.layer_scalar,
                );
            }
        }
        layer_is_moe
    }
}
