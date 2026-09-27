//! After the last layer: the fused LM head, the token's single commit +
//! wait, the state-dump drain, env-gated byte dumps, and timing records.

use metal::Buffer;

use super::super::{diag, gpu_timing, head::HeadRequest, profile};
use super::cmd::TokenCmd;
use super::staging::StagingBufs;
use crate::ops;
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

/// What the finish step reads besides the command state.
pub(super) struct FinishInputs<'a, 'w> {
    pub layers: &'a [FullPipelineLayer<'w>],
    pub x: &'a [f32],
    pub hidden: usize,
    /// The last layer's output residual.
    pub h_buf: &'a Buffer,
    pub staging: Option<&'a StagingBufs>,
    pub dump_kv: bool,
    pub dump_h: bool,
    pub call_n: usize,
    pub token_start: std::time::Instant,
}

impl MetalBackend {
    pub(super) fn finish_token(
        &self,
        tc: &TokenCmd,
        mut gpu_time: gpu_timing::TokenGpuTime,
        kv_cache: &ops::kv_cache::KVCache,
        state_dump: Option<&mut larql_compute::DecodeStateDump>,
        head: Option<HeadRequest<'_, '_>>,
        f: FinishInputs<'_, '_>,
    ) -> Vec<f32> {
        let hidden = f.hidden;
        // TOKEN-B1 rung 2: the LM head rides this command buffer rather
        // than a second one. Encoded while `enc` is still open, so the
        // token pays one commit + wait and the hidden state never crosses
        // the host boundary. A refused plan leaves the encoder untouched
        // and `head_bufs` `None`, and the caller runs the unfused head off
        // the hidden state returned below — the path this is pinned to.
        //
        // Skipped when a diagnostic dump already ended the encoder: that
        // buffer is committed, so there is nothing left to ride.
        let mut head_bufs = None;
        if let Some(ref req) = head {
            if !tc.encoder_ended {
                head_bufs = self.encode_decode_head(&tc.enc, f.h_buf, hidden, req.plan);
            }
        }

        if !tc.encoder_ended {
            tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:761");
            // A failed or ignored buffer returns from the wait just like a
            // finished one; only the status tells them apart. The MoE entry
            // seam turns any failure inside this step into `None`.
            gpu_time.record(&tc.cmd);
        }

        // Reduce after the wait — the partials are only settled now — and
        // return the head's scratch to the pool in the same step.
        if let (Some(req), Some(head_out)) = (head, head_bufs) {
            *req.out = Some(head_out.reduce_and_recycle(&self.bufs));
        }

        // W1-GPU step 7 (blit fusion): drain per-layer staging buffers
        // into state_dump now that the single final commit has settled
        // all blits.
        if let Some(s) = state_dump {
            if let Some(staging) = f.staging {
                staging.drain_into(s, f.layers, hidden, f.dump_kv, f.dump_h);
            }
        }

        // Env-gated byte dumps for CPU/Metal bisection. Both are no-ops
        // unless their directory var is set; the bodies live in `diag` so
        // the token loop stays the token's control flow.
        diag::dump_percall_layers(kv_cache, f.h_buf, f.x, hidden, f.call_n);
        diag::dump_kv_caches(kv_cache);

        let result = crate::buffers::read_buffer_f32(f.h_buf, hidden);

        // Print GPU vs CPU split when LARQL_GPU_TIMING=1. Wall covers the
        // entire decode_token_with_moe_fn call including buffer reads;
        // gpu is the sum of MTLCommandBuffer.gpuStartTime/gpuEndTime
        // windows. Delta is CPU encoding + readback overhead.
        let wall_ms = f.token_start.elapsed().as_secs_f64() * 1000.0;
        gpu_time.print_if_enabled(wall_ms);

        // When LARQL_PROFILE_SPLIT=1, store the per-stage breakdown for
        // `decode_token_split_profile` to read back. attn vs full-FFN
        // granularity (gate_up_ms carries the whole FFN block; down_ms
        // reserved for the next-finer split — see profile.rs doc-comment).
        if profile::split_profile_requested() {
            profile::store_last_split_timings(profile::ProfileTimings {
                attn_ms: gpu_time.attn_ms,
                gate_up_ms: gpu_time.gate_up_ms,
                down_ms: gpu_time.down_ms,
                // The GPU/wall pair travels with the stage split so a caller
                // can report how much of the token was on the GPU at all.
                // The numbers were already measured here; they only ever
                // reached stderr via `print_if_enabled`, so every structured
                // consumer — the bench table, `--json` — attributed the whole
                // wall to "GPU fwd".
                gpu_ms: gpu_time.total_gpu_ms,
                wall_ms,
                cmd_buffers: gpu_time.n_cmd_buffers as u32,
            });
        }

        result
    }
}
