//! Operand and target descriptions shared by the lowering encoders.

use metal::{Buffer, ComputeCommandEncoderRef};

#[allow(unused_imports)]
use super::*;

/// The norm applied to a *branch output* before it joins the residual
/// stream, under four-norm (`NormPlacement::PrePost`) placement.
///
/// Its own weight and epsilon because they are not the pre-norm's:
/// Muse-Glimmer uses eps 1e-5 before its blocks and **1e-8** after them,
/// three orders of magnitude apart, and reusing the pre-norm epsilon
/// produces superficially plausible output while lowering a different
/// program.
///
/// `None` means the op is absent (two-norm placement) — a different
/// claim from a norm with a neutral weight.
pub struct PostNorm<'a> {
    pub weight: &'a Buffer,
    pub eps: f32,
    pub weight_offset: f32,
    /// `hidden` floats of scratch. Separate from the branch output
    /// because the norm reduces over the whole vector before writing,
    /// so writing in place would race its own reduction.
    pub scratch: &'a Buffer,
}

/// One matrix operand resident on the device, tagged with the
/// representation it is stored in.
///
/// The lowering dispatches on this rather than taking a single format,
/// so a plan may keep some matrix classes wide and quantise others — the
/// per-class policy VINDEX3 already expresses, executed under one
/// schedule instead of one command buffer per format family.
#[derive(Clone, Copy)]
pub enum LoweredMatrix<'a> {
    /// Little-endian IEEE f16, `[n, k]` row-major.
    F16 { bytes: &'a Buffer },
    /// e2m1 codes + E4M3 group scales + one f32 tensor scale.
    ///
    /// `packed_offset`/`scales_offset` are byte offsets into their
    /// buffers: non-zero when the matrix is a row slice of a SHARED
    /// allocation (the QKV loader-packing rung), so projections fused
    /// into one dispatch stream one contiguous address range. A packed
    /// offset must lie on a row boundary — a multiple of 16 bytes, the
    /// bind alignment the x2 body's `uint2` loads require.
    Nvfp4 {
        packed: &'a Buffer,
        packed_offset: u64,
        scales: &'a Buffer,
        scales_offset: u64,
        tensor_scale: f32,
    },
    /// The same e2m1 codes under E8M0 group scales, 32 to a group. Kept
    /// as a first-class representation, not a deprecated one: gpt-oss
    /// ships its expert matrices in MXFP4 natively, so there it is the
    /// checkpoint's own storage rather than a choice.
    Mxfp4 {
        packed: &'a Buffer,
        scales: &'a Buffer,
    },
}

/// Where a matvec reads from and writes to, and the geometry of the
/// matrix between them. Grouped because these five always travel
/// together and a transposed `n`/`k` at a call site is invisible.
#[derive(Clone, Copy)]
pub struct MatvecTarget<'a> {
    pub x: &'a Buffer,
    pub out: &'a Buffer,
    /// Byte offset into `out` — lets a K/V projection write straight
    /// into its KV-cache slot.
    pub out_offset: u64,
    /// Output rows.
    pub n: usize,
    /// Input width.
    pub k: usize,
}

/// One quantised matrix and the vectors it maps between, as device
/// buffers. Grouped because a lowered matvec genuinely needs weights,
/// scales, input, output and geometry, and an eight-argument call at
/// every encode site is where transposed buffers hide.
pub struct MatvecOperands<'a> {
    pub packed: &'a Buffer,
    pub scales: &'a Buffer,
    pub x: &'a Buffer,
    pub out: &'a Buffer,
    /// Byte offset into `out`. Lets a K/V projection write directly into
    /// its KV-cache slot instead of writing scratch and copying.
    pub out_offset: u64,
    /// Output rows.
    pub n: usize,
    /// Input width; must be a whole number of the format's groups.
    pub k: usize,
}

/// Bind a `u32` at `index` as inline constant bytes.
pub(crate) fn set_u32(enc: &ComputeCommandEncoderRef, index: u64, value: u32) {
    enc.set_bytes(index, 4, &value as *const u32 as *const std::ffi::c_void);
}

/// Bind an `f32` at `index` as inline constant bytes.
pub(crate) fn set_f32(enc: &ComputeCommandEncoderRef, index: u64, value: f32) {
    enc.set_bytes(index, 4, &value as *const f32 as *const std::ffi::c_void);
}

/// Where an N-position matmul reads and writes: `rows` activation rows
/// of `k` floats from `x_offset`, `rows` output rows of `n` floats from
/// `out_offset`, both row-major. `out_offset` may land in a KV cache: the
/// cache is position-major, so N consecutive positions' K (or V) rows are
/// one contiguous `[rows, kv_rows]` block.
#[derive(Clone, Copy)]
pub struct MatmulRowsTarget<'a> {
    pub x: &'a Buffer,
    pub x_offset: u64,
    pub out: &'a Buffer,
    pub out_offset: u64,
    /// Output rows per position.
    pub n: usize,
    /// Input width.
    pub k: usize,
    /// Positions.
    pub rows: usize,
}
