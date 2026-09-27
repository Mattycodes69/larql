//! Env-gated per-layer diagnostics run at the end of each layer.
//!
//! Every hook here is a no-op on the production configuration. Each one
//! that reads GPU memory first flushes the token's command buffer (so the
//! read is consistent) and, where more layers follow, reopens it — the
//! exact flush/reopen shape each hook had inline in the token loop.

use metal::Buffer;

use super::super::diag;
use super::cmd::TokenCmd;
use super::ctx::TokenCtx;
use crate::MetalBackend;

impl MetalBackend {
    /// Per-layer NaN diagnostic (LARQL_DEBUG_NAN_LAYERS=1).
    /// Forces a commit+wait per layer — expensive, debug-only.
    pub(super) fn nan_debug_layer(
        &self,
        tc: &mut TokenCmd,
        t: &TokenCtx<'_, '_>,
        l: usize,
        h_buf: &Buffer,
    ) {
        if !larql_compute::options::env_flag(larql_compute::options::ENV_DEBUG_NAN_LAYERS) {
            return;
        }
        let hidden = t.hidden;
        tc.end_encoder_if_open();
        tc.cmd.commit();
        crate::cb_status::wait_or_abort(
            &tc.cmd,
            "crates/larql-compute-metal/src/decode/token.rs:518",
        );
        let h = crate::buffers::read_buffer_f32(h_buf, hidden);
        let nans = h.iter().filter(|v| v.is_nan()).count();
        eprintln!(
            "[nan-debug] layer {l}: {nans}/{hidden} NaN (head_dim={} kv_heads={})",
            t.layers[l].head_dim, t.layers[l].num_kv_heads
        );
        tc.reopen(&self.queue);
    }

    /// Issue #228: record the residual for DENSE layers too.
    ///
    /// `record_layer` used to be reachable only from
    /// `handle_moe_interleave`, so `LARQL_DUMP_RESIDUALS` on a dense
    /// model created the file, wrote its header, printed
    /// "[residual-dump] writing to <path>" and then recorded
    /// nothing — a well-formed 16-byte file that reads as "no
    /// divergence found" rather than "nothing was measured". A
    /// diagnostic that reports success while measuring nothing is
    /// worse than one that is absent, because it gets trusted.
    ///
    /// The caller guards on `!layer_is_moe`: the MoE arm already records
    /// at its own boundary, and recording here as well would double
    /// every MoE layer.
    ///
    /// The MoE arm commits at the end of each iteration, which is
    /// what makes its read of `new_h` consistent. The dense arm
    /// leaves the encoder open, so it must flush first AND restart
    /// the encoder for the next layer — the same shape
    /// `ENV_DECODE_DUMP_LAYERS` uses below. Omitting the restart
    /// encodes the next layer into a finished encoder, which
    /// segfaults rather than failing cleanly.
    pub(super) fn record_dense_residual(
        &self,
        tc: &mut TokenCmd,
        residual_dump: &mut diag::ResidualDump,
        t: &TokenCtx<'_, '_>,
        l: usize,
        layer_in: &[f32],
        new_h: &Buffer,
    ) {
        if !tc.encoder_ended {
            tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:627");
            tc.encoder_ended = true;
        }
        let ha = crate::buffers::read_buffer_f32(&t.scratch.h_post_attn, t.hidden);
        let lo = crate::buffers::read_buffer_f32(new_h, t.hidden);
        residual_dump.record_layer(l, layer_in, &ha, &lo);
        if l + 1 < t.scratch.num_layers {
            tc.reopen(&self.queue);
        }
    }

    /// Optional per-layer end-of-layer dump for decode-path
    /// diagnostics. Flushes the encoder so `new_h` is readable,
    /// writes `decode_layer_{LL}.f32`, then restarts the encoder
    /// for the next layer. Paired with Metal prefill's
    /// `metal_layer_{LL}_h_out.f32` hook so the two paths can be
    /// diffed at the same layer boundaries. Gated on an env var to
    /// keep normal decode free of flush overhead.
    ///
    /// When `LARQL_STAGE_DUMP_LAYER` names the current layer, also
    /// dump every per-sub-stage scratch buffer
    /// (`decode_layer_{LL}_{stage}.f32`). Names match the Metal
    /// prefill side (`metal_layer_NN_{stage}.f32`) so the two
    /// dump dirs can be diffed file-by-file. The end-of-layer
    /// commit above is what makes these reads consistent — the
    /// scratch buffers persist across layers, so without the
    /// per-layer flush we'd be reading the *last* layer's value.
    pub(super) fn dump_decode_layer(
        &self,
        tc: &mut TokenCmd,
        t: &TokenCtx<'_, '_>,
        l: usize,
        new_h: &Buffer,
    ) {
        let Some(dir) =
            larql_compute::options::env_value(larql_compute::options::ENV_DECODE_DUMP_LAYERS)
        else {
            return;
        };
        if !tc.encoder_ended {
            tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:663");
            tc.encoder_ended = true;
        }
        let hidden_bytes = crate::buffers::read_buffer_f32(new_h, t.hidden);
        let as_bytes: Vec<u8> = hidden_bytes.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = format!("{dir}/decode_layer_{l:02}.f32");
        if let Err(e) = std::fs::write(&path, &as_bytes) {
            eprintln!("[decode-dump] failed to write {path}: {e}");
        }

        // Per-stage dump for the layer named by
        // `LARQL_STAGE_DUMP_LAYER` (default 0). Helper lives in
        // `diag.rs`; the bundle of references is the same one
        // the early-exit diag mode uses.
        let stage_layer =
            larql_compute::options::env_usize(larql_compute::options::ENV_STAGE_DUMP_LAYER)
                .unwrap_or(0);
        if l == stage_layer {
            diag::dump_decode_stage_files(&dir, l, &layer_diag_bufs(t, l, new_h));
        }

        if l + 1 < t.scratch.num_layers {
            tc.reopen(&self.queue);
        }
    }
}

/// Diagnostic early-exit after layer `l`. Commits what we have, reads the
/// per-sub-stage buffers, reports NaN counts, and returns the layer's
/// output for the caller to return from the token.
pub(super) fn diag_stop_after_layer(
    tc: &TokenCmd,
    t: &TokenCtx<'_, '_>,
    l: usize,
    new_h: &Buffer,
) -> Vec<f32> {
    if !tc.encoder_ended {
        tc.commit_and_wait("crates/larql-compute-metal/src/decode/token.rs:716");
    }
    diag::dump_layer_buffers(l, &layer_diag_bufs(t, l, new_h));
    crate::buffers::read_buffer_f32(new_h, t.hidden)
}

/// The per-sub-stage buffer bundle both stage dumps read.
fn layer_diag_bufs<'a>(
    t: &'a TokenCtx<'_, '_>,
    l: usize,
    new_h: &'a Buffer,
) -> diag::LayerDiagBufs<'a> {
    let s = t.scratch;
    let layer = &t.layers[l];
    diag::LayerDiagBufs {
        norm_f32_buf: &s.norm_f32_buf,
        q_out: &s.q_out,
        k_out: &s.k_out,
        v_out: &s.v_out,
        attn_out_buf: &s.attn_out_buf,
        o_out_buf: &s.o_out_buf,
        h_post_attn: &s.h_post_attn,
        ffn_norm_out: &s.ffn_norm_out,
        gate_out_scratch: &s.gate_out_scratch,
        up_out: &s.up_out,
        act_buf: &s.act_buf,
        down_out: &s.down_out,
        new_h,
        hidden: t.hidden,
        inter: t.inter,
        layer_q_dim: layer.num_q_heads * layer.head_dim,
        layer_kv_dim: layer.num_kv_heads * layer.head_dim,
    }
}
