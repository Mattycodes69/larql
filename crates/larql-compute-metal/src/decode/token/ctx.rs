//! Per-token read-only inputs every layer stage reads.

use super::super::setup::DecodeScratch;
use larql_compute::FullPipelineLayer;

pub(super) struct TokenCtx<'a, 'w> {
    pub layers: &'a [FullPipelineLayer<'w>],
    pub hidden: usize,
    pub inter: usize,
    /// Weight caches, per-stage scratch and the ping-pong h-buffers. See
    /// `setup.rs` for the full inventory.
    pub scratch: &'a DecodeScratch,
}

impl TokenCtx<'_, '_> {
    /// The buffer layer `l` writes its output residual into (ping-pong).
    pub(super) fn new_h(&self, l: usize) -> &metal::Buffer {
        if l.is_multiple_of(2) {
            &self.scratch.h_a
        } else {
            &self.scratch.h_b
        }
    }
}
