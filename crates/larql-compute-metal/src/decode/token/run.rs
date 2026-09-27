//! `decode_token_with_moe_split_fn` — setup, then the per-layer sequence.
//!
//! Each layer encodes, in order: the state-dump h-input blit, Steps 1–5
//! (`attn.rs`), Steps 6–8 (`ffn.rs`), the K/V staging blit, the
//! diagnostic hooks and the MoE/scalar tail (`hooks.rs`, `ffn.rs`). The
//! token then finishes in `finish.rs`.

use super::super::{diag, encode_ffn, gpu_timing, head::HeadRequest, moe_interleave, setup};
use super::cmd::TokenCmd;
use super::ctx::TokenCtx;
use super::ffn::{MoeCollectFn, MoeFireFn};
use super::finish::FinishInputs;
use super::hooks;
use super::staging::StagingBufs;
use crate::ops;
use crate::MetalBackend;

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_with_moe_split_fn(
        &self,
        kv_cache: &mut ops::kv_cache::KVCache,
        layers: &[larql_compute::FullPipelineLayer],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _rope_base: f32,
        mut moe_fn: MoeFireFn<'_>,
        mut moe_collect_fn: MoeCollectFn<'_>,
        mut state_dump: Option<&mut larql_compute::DecodeStateDump>,
        state_dump_mask: larql_compute::StateDumpMask,
        inline_moe: Option<&moe_interleave::InlineMoeCtx<'_>>,
        head: Option<HeadRequest<'_, '_>>,
    ) -> Vec<f32> {
        // Refuse unroutable FFN formats BEFORE any command buffer or
        // encoder exists: a panic that unwinds past a live Metal
        // encoder trips the ObjC "released without endEncoding"
        // assertion and turns a clean refusal into a process-killing
        // SIGTRAP (the failure mode that hid #229 behind an earlier
        // test's abort). `encode_ffn_step` re-checks as defence in
        // depth for callers that bypass this entry point.
        for layer in layers {
            // A fully-remote FFN never runs locally; its dense weight
            // slots may be placeholders and are not validated.
            if !layer.ffn_is_remote {
                encode_ffn::validate_ffn_formats(layer);
            }
        }
        // W10 Phase B/C: capture flags. `dump_kv` controls the K/V
        // staging + readback (skipped under HOnly + None — Metal's own
        // kv cache still receives the K/V as a side effect for
        // attention). `dump_h` controls the h_in staging + readback
        // (skipped under None only — engines using `None` have no
        // CPU-side use for the residual stream, e.g.
        // MarkovResidualEngine with no window).
        let dump_kv = matches!(state_dump_mask, larql_compute::StateDumpMask::Full);
        let dump_h = !matches!(state_dump_mask, larql_compute::StateDumpMask::None);
        let token_start = std::time::Instant::now();
        let mut gpu_time = gpu_timing::TokenGpuTime::default();

        // Residual dump (env-gated) for HF-reference diffs. Active only when
        // `LARQL_DUMP_RESIDUALS=<path>` is set.
        let mut residual_dump = diag::ResidualDump::from_env();

        // Input RMS debug (first 3 calls, env-gated).
        static CALL_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call_n = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        diag::log_decode_entry(call_n, x, hidden, inter, layers);

        // Per-layer weight-buffer caches + per-stage scratch + ping-pong
        // h-buffers. See `setup.rs` for the full inventory.
        let mut scratch =
            setup::DecodeScratch::new(&self.bufs, layers, x, hidden, inter, q_dim, kv_dim);
        let scratch_clones = std::mem::take(&mut scratch.scratch_clones);
        // Return scratch buffers to the pool when this decode step exits.
        let _scratch_guard = {
            let mut g = crate::buffers::ScratchGuard::new(&self.bufs);
            for buf in scratch_clones {
                g.track(&buf);
            }
            g
        };
        let num_layers = scratch.num_layers;

        // W1-GPU step 7 (blit-encoder fusion): per-layer staging buffers
        // so the layer loop blits k_out / v_out / h_buf instead of
        // committing per layer. See `staging.rs`.
        let staging = StagingBufs::allocate(
            &self.bufs,
            layers,
            hidden,
            num_layers,
            state_dump.is_some(),
            dump_kv,
            dump_h,
        );
        let _staging_guard = StagingBufs::guard(staging.as_ref(), &self.bufs);

        let t = TokenCtx {
            layers,
            hidden,
            inter,
            scratch: &scratch,
        };
        let mut h_buf = &scratch.h_init;
        // Per-Layer Embeddings precomputed table (Gemma 4 E2B): snapshot
        // once per token so the per-layer loop can read it without
        // re-locking the mutex on every iteration. `None` for non-PLE archs.
        let ple_inputs = self.ple_inputs_snapshot();
        // Split mode: when a fire+collect callback pair is present, defer
        // FFN encoding for MoE layers until *after* the remote MoE call has
        // been fired, so dense FFN runs on the GPU in parallel with the
        // network round trip.  Falls back to single-encoder per layer when
        // `moe_collect_fn` is `None` (existing local-MoE / unary HTTP path).
        let split_mode = moe_fn.is_some() && moe_collect_fn.is_some();
        let mut tc = TokenCmd::open(&self.queue);

        // Diagnostic: run only up to (and including) the specified layer,
        // then dump intermediates and exit. Pinpoints which sub-stage in
        // which layer first produces NaN on real-vindex decode.
        let diag_stop_layer: Option<usize> =
            larql_compute::options::env_usize(larql_compute::options::ENV_DECODE_DIAG_LAYER);

        // `num_layers == layers.len()` (setup derives it from `layers`).
        for (l, layer) in layers.iter().enumerate().take(num_layers) {
            // The only place that knows which layer is executing. The served
            // MoE route boundary reads this; without it a routing trace is
            // refused rather than attributed to a guessed layer.
            let _route_scope = larql_compute::moe_route_observe::LayerScope::new(l);

            // Snapshot the layer input for HF-reference diff. Must be taken
            // before any compute since `h_buf` = layer-N input at this point
            // (it's the previous layer's `new_h`, or the embedding for L0).
            // GPU buffers are committed + waited at the end of each MoE
            // iteration so the read returns consistent data.
            let layer_in_snapshot: Option<Vec<f32>> = if residual_dump.is_enabled() {
                Some(crate::buffers::read_buffer_f32(h_buf, hidden))
            } else {
                None
            };

            // W1-GPU step 7 (blit fusion): capture h_in for state dump.
            // - L=0: `x` is on the CPU, push it directly into state_dump.
            // - L>=1: blit `h_buf` (previous layer's output) into the
            //   per-layer h-staging buffer. Drained into state_dump after
            //   the single final commit.
            if dump_h {
                if let Some(s) = state_dump.as_deref_mut() {
                    if l == 0 {
                        s.h_in_per_layer.push(x.to_vec());
                    }
                }
                if let Some(ref st) = staging {
                    st.blit_h_in(&mut tc, l, h_buf, hidden);
                }
            }
            let dump_l0_dir = if l == 0 {
                larql_compute::options::env_value(larql_compute::options::ENV_DUMP_L0)
            } else {
                None
            };

            // ── Steps 1–5: input norm + QKV, attention block ──
            self.encode_layer_attention(&tc.enc, &t, kv_cache, l, h_buf);
            let new_h = t.new_h(l);

            // ── Steps 6–8: dense FFN + post-FFN residual + PLE ──
            let schedule = self.encode_layer_dense_ffn(
                &mut tc,
                &mut gpu_time,
                &t,
                l,
                new_h,
                ple_inputs.as_ref(),
                split_mode,
            );

            h_buf = new_h;

            // W1-GPU step 7 (blit fusion): capture k_new / v_new for state
            // dump inside the same command buffer. Drained into state_dump
            // after the single final commit.
            if dump_kv {
                if let Some(ref st) = staging {
                    st.blit_kv(&mut tc, l, layer, &scratch.k_out, &scratch.v_out);
                }
            }

            self.nan_debug_layer(&mut tc, &t, l, h_buf);

            let layer_is_moe = self.encode_layer_moe_or_scalar(
                &mut tc,
                &mut gpu_time,
                &mut residual_dump,
                &t,
                l,
                new_h,
                schedule,
                layer_in_snapshot.as_deref(),
                dump_l0_dir.as_deref(),
                &mut moe_fn,
                &mut moe_collect_fn,
                inline_moe,
            );

            // Dense layers record their residual here; the MoE arm records
            // at its own boundary (see `hooks::record_dense_residual`).
            if !layer_is_moe {
                if let Some(layer_in) = layer_in_snapshot.as_deref() {
                    self.record_dense_residual(&mut tc, &mut residual_dump, &t, l, layer_in, new_h);
                }
            }

            self.dump_decode_layer(&mut tc, &t, l, new_h);

            // Diagnostic early-exit after layer `l`.
            if diag_stop_layer == Some(l) {
                return hooks::diag_stop_after_layer(&tc, &t, l, new_h);
            }
        }

        self.finish_token(
            &tc,
            gpu_time,
            kv_cache,
            state_dump,
            head,
            FinishInputs {
                layers,
                x,
                hidden,
                h_buf,
                staging: staging.as_ref(),
                dump_kv,
                dump_h,
                call_n,
                token_start,
            },
        )
    }
}
