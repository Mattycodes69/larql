//! Single-row matvec encoders and lowering buffer helpers.

use crate::MetalBackend;
use metal::{Buffer, ComputeCommandEncoderRef};

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    /// Encode `out = W · x` for a matrix in whichever representation it
    /// is resident in.
    pub fn encode_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &LoweredMatrix<'_>,
        at: &MatvecTarget<'_>,
    ) {
        match w {
            LoweredMatrix::Nvfp4 {
                packed,
                packed_offset: 0,
                scales,
                scales_offset: 0,
                tensor_scale,
            } => self.encode_nvfp4_matvec(
                enc,
                &MatvecOperands {
                    packed,
                    scales,
                    x: at.x,
                    out: at.out,
                    out_offset: at.out_offset,
                    n: at.n,
                    k: at.k,
                },
                *tensor_scale,
            ),
            // A sliced matrix (a row slice of a shared allocation): the
            // same kernel, bound at the slice's byte offsets. Not the
            // segmented kernel — that is a different code shape, and
            // under fast-math a different code shape is a different
            // arithmetic, which a layout change must not introduce.
            LoweredMatrix::Nvfp4 {
                packed,
                packed_offset,
                scales,
                scales_offset,
                tensor_scale,
            } => self.encode_nvfp4_matvec_sliced(
                enc,
                &MatvecOperands {
                    packed,
                    scales,
                    x: at.x,
                    out: at.out,
                    out_offset: at.out_offset,
                    n: at.n,
                    k: at.k,
                },
                *tensor_scale,
                *packed_offset,
                *scales_offset,
            ),
            LoweredMatrix::Mxfp4 { packed, scales } => self.encode_mxfp4_matvec(
                enc,
                &MatvecOperands {
                    packed,
                    scales,
                    x: at.x,
                    out: at.out,
                    out_offset: at.out_offset,
                    n: at.n,
                    k: at.k,
                },
            ),
            LoweredMatrix::F16 { bytes } => self.encode_f16_matvec(enc, bytes, at),
        }
    }

    /// Encode `out = W · x` for an f16 matrix into `enc`.
    pub fn encode_f16_matvec(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &Buffer,
        at: &MatvecTarget<'_>,
    ) {
        let kernel = &self.f16_gemv_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(at.x), 0);
        enc.set_buffer(2, Some(at.out), at.out_offset);
        set_u32(enc, 3, at.n as u32);
        set_u32(enc, 4, at.k as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((at.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// Encode the MXFP4 sibling, same contract.
    pub fn encode_mxfp4_matvec(&self, enc: &ComputeCommandEncoderRef, op: &MatvecOperands<'_>) {
        let kernel = &self.quant.mxfp4_matvec_pipeline;
        enc.set_compute_pipeline_state(&kernel.state);
        enc.set_buffer(0, Some(op.packed), 0);
        enc.set_buffer(1, Some(op.scales), 0);
        enc.set_buffer(2, Some(op.x), 0);
        enc.set_buffer(3, Some(op.out), op.out_offset);
        set_u32(enc, 4, op.n as u32);
        set_u32(enc, 5, op.k as u32);
        enc.dispatch_thread_groups(
            metal::MTLSize::new((op.n as u64).div_ceil(kernel.rows_per_tg), 1, 1),
            metal::MTLSize::new(kernel.threads_per_tg, 1, 1),
        );
    }

    /// A pooled device buffer of `floats` f32s, for lowering intermediates
    /// that must never reach the host.
    pub fn lowering_scratch(&self, floats: usize) -> Buffer {
        self.bufs.output((floats * 4) as u64)
    }

    /// Return a lowering scratch buffer to the pool. Only valid after the
    /// command buffer that used it has completed.
    pub fn recycle_lowering_scratch(&self, buf: Buffer) {
        self.bufs.recycle(buf);
    }

    /// Upload `x` into a fresh pooled device buffer — the one host→device
    /// crossing a lowered token needs, at its start.
    pub fn lowering_upload(&self, x: &[f32]) -> Option<Buffer> {
        let buf = self.bufs.output((x.len() * 4) as u64);
        let ptr = buf.contents() as *mut f32;
        if ptr.is_null() {
            return None;
        }
        // SAFETY: pooled buffer is at least x.len()*4 bytes and is not
        // bound to any encoder yet.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), ptr, x.len()) };
        Some(buf)
    }

    /// Read a device buffer back — the one device→host crossing, at the end.
    pub fn lowering_readback(&self, buf: &Buffer, len: usize) -> Option<Vec<f32>> {
        crate::buffers::try_read_buffer_f32(buf, len)
    }

    /// The cached device buffer for a weight stream, keyed on address
    /// identity (see `BufferCache::get_bytes`).
    pub fn lowering_weight(&self, bytes: &[u8]) -> Buffer {
        self.bufs.get_bytes(bytes)
    }

    /// Register a page-aligned, session-lived byte region so a routed
    /// FFN's expert operands can be bound zero-copy (the same
    /// `register_region` the served `--routed-from` path uses). Returns
    /// `false` if `bytes` is not page-aligned — a lowering that copied
    /// 10 GB of experts into owned buffers would defeat the point.
    pub fn lowering_register_region(&self, bytes: &[u8]) -> bool {
        self.bufs.register_region(bytes)
    }

    /// Build (or fetch) a routed layer's expert descriptor table from a
    /// `MoeLayerWeights` whose expert slices lie in registered regions.
    /// `None` = an operand missed its region or the geometry disagrees —
    /// the caller must refuse, never fall back.
    pub fn lowering_moe_descriptor(
        &self,
        layer_idx: usize,
        moe: &larql_compute::MoeLayerWeights<'_>,
        inter: usize,
        hidden: usize,
    ) -> Option<std::sync::Arc<crate::moe_descriptor::MoeExpertDescriptorTable>> {
        self.descriptor_table_for_layer(layer_idx, moe, inter, hidden)
    }

    /// Whether the descriptor MoE path can serve this layer — checked
    /// before encode so a refusal is typed, not a mid-command-buffer
    /// failure.
    pub fn lowering_moe_supported(
        &self,
        moe: &larql_compute::MoeLayerWeights<'_>,
        scratch: &crate::MoeScratch,
    ) -> bool {
        self.gpu_route_supported(moe, scratch)
    }

    /// A command buffer for a lowered unit of work. Owned by the caller,
    /// which decides how much to encode into it before committing —
    /// the decision this whole rung exists to hand over.
    pub fn new_lowering_command_buffer(&self) -> metal::CommandBuffer {
        // `new_command_buffer` hands back an autoreleased reference; a
        // decode loop with no pool of its own would keep every token's
        // command buffer (and what it retains) alive until the thread
        // ends. Retain explicitly, drain the rest here.
        objc::rc::autoreleasepool(|| self.queue.new_command_buffer().to_owned())
    }
}
