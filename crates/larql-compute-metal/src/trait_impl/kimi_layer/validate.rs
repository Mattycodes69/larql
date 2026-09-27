//! Host-side validation of a layer call, and route readback helpers.

use super::super::grouped_experts::GroupedError;
use crate::shaders::kimi_layer as layer_shader;
use metal::Buffer;

#[allow(unused_imports)]
use super::*;

/// Host-side checks the router's own kernel cannot make: the sizes it is
/// about to index, and that every RESIDENT expert really lies inside the
/// banks. A non-resident selection is the kernel's business; an
/// in-bounds-looking offset that is not is this function's.
pub(super) fn validate_layer(
    w: &KimiLayerWeights<'_>,
    experts: usize,
    slots: usize,
    hidden: usize,
) -> Result<(), GroupedError> {
    let moe = match &w.ffn {
        ffn::FfnSpec::Moe(m) => m,
        ffn::FfnSpec::Dense(d) => return ffn::FfnSpec::validate_dense(d, hidden),
    };
    if experts == 0 || moe.top_k == 0 {
        return Err(GroupedError::NoExpertsSelected);
    }
    if experts > layer_shader::MAX_EXPERTS || slots > layer_shader::MAX_SLOTS {
        return Err(GroupedError::SlotCountMismatch {
            expected: layer_shader::MAX_EXPERTS.min(layer_shader::MAX_SLOTS),
            found: experts.max(slots),
        });
    }
    // Whether a shared expert exists is ONE semantic fact; the three
    // projections declaring it differently would silently drop one
    // projection's shared contribution.
    let has_shared = moe.gate.shared.is_some();
    for bank in [&moe.up, &moe.down] {
        if bank.shared.is_some() != has_shared {
            return Err(GroupedError::SharedBranchInconsistent);
        }
    }
    for bank in [&moe.gate, &moe.up, &moe.down] {
        if bank.addressing.experts() != experts {
            return Err(GroupedError::SlotCountMismatch {
                expected: experts,
                found: bank.addressing.experts(),
            });
        }
    }
    if moe.router_weight.len() != experts * hidden {
        return Err(GroupedError::SlotCountMismatch {
            expected: experts * hidden,
            found: moe.router_weight.len(),
        });
    }
    for (name, bank, n, k) in [
        (0usize, moe.gate, moe.inter, hidden),
        (1, moe.up, moe.inter, hidden),
        (2, moe.down, hidden, moe.inter),
    ] {
        // Sized at what the bank CLAIMS to be, not at bf16. Bytes that
        // are Q6_K dispatched as BF16 need more room than they have and
        // are refused here; the opposite direction is caught where the
        // bank's exact extent is known, since a shifted view over a
        // whole segment legitimately has room to spare.
        let per = bank
            .routed
            .encoding
            .matrix_bytes(n, k)
            .ok_or(GroupedError::KNotSuperblockAligned { k })?;
        // Every ADDRESSABLE expert must lie inside the routed bank. For
        // an identity bank that is every expert; for a packed one only
        // those the table names.
        //
        // An identity bank is walked expert by expert and never filtered:
        // `offset_of` answers `None` there only when `expert * stride`
        // overflows, and skipping that expert would admit exactly the
        // bank this check exists to refuse.
        for expert in 0..experts {
            let off = match (bank.addressing, bank.addressing.offset_of(expert)) {
                (_, Some(off)) => off,
                (ExpertAddressing::Table(_), None) => continue,
                (ExpertAddressing::Identity { stride, .. }, None) => {
                    return Err(GroupedError::OffsetExceedsAddressWidth {
                        slot: name,
                        offset: (expert as u64).saturating_mul(u64::from(stride)),
                    })
                }
            };
            // The device offset table and the address kernel carry `u32`.
            let Ok(device_off) = u32::try_from(off) else {
                return Err(GroupedError::OffsetExceedsAddressWidth {
                    slot: name,
                    offset: off,
                });
            };
            let need = off as usize + per;
            if need > bank.routed.bytes.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot: name,
                    offset: device_off,
                    need,
                    have: bank.routed.bytes.len(),
                });
            }
        }
        // The shared branch's own region, under its OWN encoding —
        // which need not be the routed bank's.
        if let Some(shared) = &bank.shared {
            let need = shared
                .encoding
                .matrix_bytes(n, k)
                .ok_or(GroupedError::KNotSuperblockAligned { k })?;
            if need > shared.bytes.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot: name,
                    offset: 0,
                    need,
                    have: shared.bytes.len(),
                });
            }
        }
    }
    Ok(())
}

/// Read each layer's selected expert ids, after the wait and before the
/// scratch is recycled.
pub(super) fn collect_routes(
    layers: &[KimiLayerCall<'_>],
    scratch: &[LayerScratch],
    trace: Option<&mut ExecutionTrace>,
) {
    let Some(trace) = trace else {
        return;
    };
    trace.routes.clear();
    trace.selection_scores.clear();
    trace.combine_weights.clear();
    let want_scores = trace.want_selection_scores;
    for (call, s) in layers.iter().zip(scratch) {
        let top_k = call.weights.ffn.top_k();
        // A dense layer routes nothing, and its `chosen` buffer is the
        // one-element placeholder `layer_scratch` allocates. Reading it
        // would report an expert that was never selected.
        trace.routes.push(if top_k == 0 {
            Vec::new()
        } else {
            read_u32(&s.chosen, top_k)
        });
        if want_scores {
            let experts = call.weights.ffn.experts();
            trace.selection_scores.push(if top_k == 0 {
                Vec::new()
            } else {
                crate::buffers::read_buffer_f32(&s.sel_scores, experts)
            });
            trace.combine_weights.push(if top_k == 0 {
                Vec::new()
            } else {
                crate::buffers::read_buffer_f32(&s.weights, call.weights.ffn.slots())
            });
        }
    }
}

pub(super) fn bytemuck_u32(v: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and no invalid bit patterns, and `u8`
    // has weaker alignment, so any `&[u32]` is a valid `&[u8]` of four
    // times the length for the same lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

pub(super) fn read_u32(buf: &Buffer, n: usize) -> Vec<u32> {
    let ptr = buf.contents() as *const u32;
    if ptr.is_null() {
        return vec![0; n];
    }
    // SAFETY: shared-storage buffer of at least `n * 4` bytes, read after
    // `wait_until_completed`, so no GPU work is in flight against it.
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}
