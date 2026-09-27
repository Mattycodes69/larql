//! W1-GPU step 7 (blit-encoder fusion): state-dump staging buffers.
//!
//! When `state_dump` is active, per-layer staging buffers are allocated up
//! front so the layer loop can **blit** `k_out` / `v_out` / `h_buf` into
//! them instead of forcing a per-layer commit+wait+CPU-read. Reads run once
//! after the final commit. Saves ~1.7 ms / token (50 µs × num_layers) on
//! M3 Max.

use metal::Buffer;

use super::cmd::TokenCmd;
use crate::buffers::{BufferCache, ScratchGuard};
use larql_compute::FullPipelineLayer;

pub(super) struct StagingBufs {
    /// Per-layer `k_out` copies (empty unless `dump_kv`).
    pub k: Vec<Buffer>,
    /// Per-layer `v_out` copies (empty unless `dump_kv`).
    pub v: Vec<Buffer>,
    /// Per-layer layer-input copies (empty unless `dump_h`).
    pub h: Vec<Buffer>,
}

impl StagingBufs {
    /// Allocate the staging set, or `None` when nothing is being dumped.
    pub(super) fn allocate(
        bufs: &BufferCache,
        layers: &[FullPipelineLayer<'_>],
        hidden: usize,
        num_layers: usize,
        dumping: bool,
        dump_kv: bool,
        dump_h: bool,
    ) -> Option<Self> {
        if dumping && (dump_kv || dump_h) {
            let mut sk = Vec::with_capacity(if dump_kv { num_layers } else { 0 });
            let mut sv = Vec::with_capacity(if dump_kv { num_layers } else { 0 });
            let mut sh = Vec::with_capacity(if dump_h { num_layers } else { 0 });
            let hidden_bytes = (hidden * 4) as u64;
            for layer in layers.iter() {
                if dump_kv {
                    let kv_dim_bytes = (layer.num_kv_heads * layer.head_dim * 4) as u64;
                    sk.push(bufs.output(kv_dim_bytes));
                    sv.push(bufs.output(kv_dim_bytes));
                }
                if dump_h {
                    sh.push(bufs.output(hidden_bytes));
                }
            }
            Some(Self {
                k: sk,
                v: sv,
                h: sh,
            })
        } else {
            None
        }
    }

    /// Track every staging buffer for recycling after the final commit.
    /// Separate from the main scratch guard since these buffers are
    /// allocated post-setup.
    pub(super) fn guard<'c>(staging: Option<&Self>, cache: &'c BufferCache) -> ScratchGuard<'c> {
        let mut g = ScratchGuard::new(cache);
        if let Some(s) = staging {
            for b in s.k.iter().chain(s.v.iter()).chain(s.h.iter()) {
                g.track(b);
            }
        }
        g
    }

    /// Blit layer `l`'s input residual into its h-staging buffer. The blit
    /// is encoded into the same command buffer as the layer compute, so
    /// Metal's command-buffer ordering guarantees it sees the settled
    /// value once committed. Layer 0's input is on the CPU and is pushed
    /// directly by the caller.
    pub(super) fn blit_h_in(&self, tc: &mut TokenCmd, l: usize, h_buf: &Buffer, hidden: usize) {
        if l > 0 && !self.h.is_empty() {
            tc.end_encoder_if_open();
            let blit = tc.cmd.new_blit_command_encoder();
            blit.copy_from_buffer(h_buf, 0, &self.h[l], 0, (hidden * 4) as u64);
            blit.end_encoding();
            tc.reopen_encoder();
        }
    }

    /// Blit layer `l`'s new K/V rows into the K/V staging buffers. The
    /// compute writes to `k_out` / `v_out` happen-before the blit reads
    /// (Metal command-buffer encode order), so the blit captures the
    /// values before the next layer overwrites them.
    pub(super) fn blit_kv(
        &self,
        tc: &mut TokenCmd,
        l: usize,
        layer: &FullPipelineLayer<'_>,
        k_out: &Buffer,
        v_out: &Buffer,
    ) {
        tc.end_encoder_if_open();
        let blit = tc.cmd.new_blit_command_encoder();
        let layer_kv_dim_local = layer.num_kv_heads * layer.head_dim;
        let bytes = (layer_kv_dim_local * 4) as u64;
        blit.copy_from_buffer(k_out, 0, &self.k[l], 0, bytes);
        blit.copy_from_buffer(v_out, 0, &self.v[l], 0, bytes);
        blit.end_encoding();
        tc.reopen_encoder();
    }

    /// Drain the staging buffers into `state_dump` now that the single
    /// final commit has settled all blits. `h_in_per_layer[0]` was already
    /// pushed inline (CPU copy of `x`); indices 1..num_layers come from the
    /// h-staging buffers populated by the blits at the top of each layer.
    pub(super) fn drain_into(
        &self,
        s: &mut larql_compute::DecodeStateDump,
        layers: &[FullPipelineLayer<'_>],
        hidden: usize,
        dump_kv: bool,
        dump_h: bool,
    ) {
        if dump_h {
            for (l, _) in layers.iter().enumerate().skip(1) {
                s.h_in_per_layer
                    .push(crate::buffers::read_buffer_f32(&self.h[l], hidden));
            }
        }
        if dump_kv {
            for (l, layer) in layers.iter().enumerate() {
                let kv_dim_local = layer.num_kv_heads * layer.head_dim;
                s.k_new_per_layer
                    .push(crate::buffers::read_buffer_f32(&self.k[l], kv_dim_local));
                s.v_new_per_layer
                    .push(crate::buffers::read_buffer_f32(&self.v[l], kv_dim_local));
            }
        }
    }
}
