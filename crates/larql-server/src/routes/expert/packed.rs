//! Byte slicing for the legacy packed-BF16 expert table.
//!
//! The packed layout stores every expert of a layer back to back, so an
//! expert's bytes start at `expert_id * stride`. `expert_id` arrives from the
//! request (URL path, JSON body or binary wire), so the offset arithmetic is
//! checked: an unchecked product panics in debug builds and, in release,
//! wraps to a small in-bounds offset that silently serves another expert.

use std::ops::Range;

/// Bytes per element of the packed BF16 expert table.
pub const PACKED_BF16_BYTES: usize = 2;

/// Byte range of `expert_id` in a table of `len` bytes holding experts of
/// `stride` bytes each, or `None` when the expert lies outside the table
/// (including when the offset arithmetic would overflow).
pub fn packed_expert_range(expert_id: usize, stride: usize, len: usize) -> Option<Range<usize>> {
    let start = expert_id.checked_mul(stride)?;
    let end = start.checked_add(stride)?;
    (end <= len).then_some(start..end)
}

/// Gate/up and down byte slices of one expert in the packed BF16 layout,
/// or `None` when `expert_id` is outside either table.
pub fn packed_bf16_expert<'a>(
    gate_up_all: &'a [u8],
    down_all: &'a [u8],
    expert_id: usize,
    hidden: usize,
    inter: usize,
) -> Option<(&'a [u8], &'a [u8])> {
    // Gate and up are fused, so the gate/up block is twice the down block.
    let down_stride = hidden.checked_mul(inter)?.checked_mul(PACKED_BF16_BYTES)?;
    let gate_up_stride = down_stride.checked_mul(2)?;
    let gu = packed_expert_range(expert_id, gate_up_stride, gate_up_all.len())?;
    let dn = packed_expert_range(expert_id, down_stride, down_all.len())?;
    Some((&gate_up_all[gu], &down_all[dn]))
}
