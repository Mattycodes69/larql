//! Q8 row kernels using SDOT indexing and vectors.

#[allow(unused_imports)]
use super::*;

/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98; see [`q8_row_sdot`].
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_row_asym_indexed_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + SDOT_LANES <= hi {
            lanes = vdotq_s32(
                lanes,
                vld1q_s8(codes.as_ptr().add(i)),
                vld1q_s8(qx.as_ptr().add(i)),
            );
            i += SDOT_LANES;
        }
        let mut dot = vaddvq_s32(lanes);
        while i < hi {
            dot += *codes.get_unchecked(i) as i32 * *qx.get_unchecked(i) as i32;
            i += 1;
        }
        let mut wsum = 0i32;
        for k in index_span(lo, hi) {
            wsum += *sums.get_unchecked(k) as i32;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// **CPU5-K2.** Four block-16 reductions, batched.
///
/// At `block == 16` there is exactly ONE `SDOT` per block, so the
/// cross-lane `vaddvq_s32` after it has no independent work to hide its
/// latency behind. That is why cost is SUPERLINEAR in block count —
/// 80 blocks 266 ms, 160 blocks 298 ms, 320 blocks 484 ms — an
/// instruction-level-parallelism collapse rather than extra arithmetic.
///
/// Four blocks reduce together with three pairwise adds instead of four
/// cross-lane reductions:
///
/// ```text
/// vpaddq(d0,d1) -> [d0a+d0b, d0c+d0d, d1a+d1b, d1c+d1d]
/// vpaddq(d2,d3) -> likewise
/// vpaddq(  ,  ) -> [SUM d0, SUM d1, SUM d2, SUM d3]
/// ```
///
/// **Bit-identical**: every rearrangement is in i32, where these sums are
/// exact, and the four float multiply-adds still happen in block order.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_row_asym_b16_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let blocks = in_dim / SDOT_LANES;
    let mut acc = 0.0f32;
    let mut b = 0usize;
    while b + 4 <= blocks {
        let i0 = b * SDOT_LANES;
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        let z = vdupq_n_s32(0);
        let d0 = vdotq_s32(z, w0, vld1q_s8(qx.as_ptr().add(i0)));
        let d1 = vdotq_s32(z, w1, vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)));
        let d2 = vdotq_s32(z, w2, vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)));
        let d3 = vdotq_s32(z, w3, vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)));
        let s0 = vdotq_s32(z, w0, ones);
        let s1 = vdotq_s32(z, w1, ones);
        let s2 = vdotq_s32(z, w2, ones);
        let s3 = vdotq_s32(z, w3, ones);
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        let sv = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
        acc += fold_scale[b] * vgetq_lane_s32(dv, 0) as f32
            + fold_mid[b] * vgetq_lane_s32(sv, 0) as f32;
        acc += fold_scale[b + 1] * vgetq_lane_s32(dv, 1) as f32
            + fold_mid[b + 1] * vgetq_lane_s32(sv, 1) as f32;
        acc += fold_scale[b + 2] * vgetq_lane_s32(dv, 2) as f32
            + fold_mid[b + 2] * vgetq_lane_s32(sv, 2) as f32;
        acc += fold_scale[b + 3] * vgetq_lane_s32(dv, 3) as f32
            + fold_mid[b + 3] * vgetq_lane_s32(sv, 3) as f32;
        b += 4;
    }
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        let sm = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, ones));
        acc += fold_scale[b] * d as f32 + fold_mid[b] * sm as f32;
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        acc += fold_scale[blocks] * d as f32 + fold_mid[blocks] * sm as f32;
    }
    acc
}

/// **CPU5-K3.** Block-16 rows accumulated in the VECTOR domain.
///
/// K2 batched the integer reductions but still crossed from vector to
/// scalar registers twice per block — eight `vgetq_lane_s32` and eight
/// scalar float operations per group of four blocks. At 320 blocks a row
/// that crossing is what remains of the block-16 pathology.
///
/// Here the packed i32 block sums are converted in place, multiplied by
/// vectors of scales (and offsets), and accumulated into a four-lane
/// float accumulator — **one horizontal reduction per ROW** instead of
/// per block.
///
/// **NOT bit-identical**, by design and uniquely on this ladder. K1 and
/// K2 preserved the arithmetic exactly; this reassociates the sum of
/// already-computed block contributions, so the numbers move at the
/// rounding level and the frozen quality gates must be re-established on
/// the full bank rather than inherited.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_row_b16_vector_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: Option<&[f32]>,
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let blocks = in_dim / SDOT_LANES;
    let mut acc_v = vdupq_n_f32(0.0);
    let mut b = 0usize;
    while b + 4 <= blocks {
        let i0 = b * SDOT_LANES;
        let z = vdupq_n_s32(0);
        let w0 = vld1q_s8(codes.as_ptr().add(i0));
        let w1 = vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES));
        let w2 = vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES));
        let w3 = vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES));
        let d0 = vdotq_s32(z, w0, vld1q_s8(qx.as_ptr().add(i0)));
        let d1 = vdotq_s32(z, w1, vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)));
        let d2 = vdotq_s32(z, w2, vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)));
        let d3 = vdotq_s32(z, w3, vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)));
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        // Stay in the vector domain: no lane extract, no scalar float.
        acc_v = vfmaq_f32(
            acc_v,
            vld1q_f32(fold_scale.as_ptr().add(b)),
            vcvtq_f32_s32(dv),
        );
        if let Some(mid) = fold_mid {
            let s0 = vdotq_s32(z, w0, ones);
            let s1 = vdotq_s32(z, w1, ones);
            let s2 = vdotq_s32(z, w2, ones);
            let s3 = vdotq_s32(z, w3, ones);
            let sv = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
            acc_v = vfmaq_f32(acc_v, vld1q_f32(mid.as_ptr().add(b)), vcvtq_f32_s32(sv));
        }
        b += 4;
    }
    let mut acc = vaddvq_f32(acc_v);
    // Whole blocks below a group of four, then any ragged remainder.
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        acc += fold_scale[b] * d as f32;
        if let Some(mid) = fold_mid {
            acc += mid[b] * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, ones)) as f32;
        }
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        acc += fold_scale[blocks] * d as f32;
        if let Some(mid) = fold_mid {
            acc += mid[blocks] * sm as f32;
        }
    }
    acc
}

/// The bit-identical asymmetric row, whatever the process arm.
///
/// Tests need BOTH implementations reachable in one binary: K3
/// reassociates, and a control taken across two builds would be arguing
/// with a compiler as much as with the change.
#[cfg(test)]
pub(in super::super) fn q8_row_asym_exact(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe {
            if block == SDOT_LANES {
                q8_row_asym_b16_sdot(codes, fold_scale, fold_mid, qx, in_dim)
            } else {
                q8_row_asym_sdot(codes, fold_scale, fold_mid, qx, in_dim, block)
            }
        };
    }
    q8_row_asym_portable(codes, fold_scale, fold_mid, qx, in_dim, block)
}

/// The K3 row, reachable from a test whatever the process arm.
#[cfg(test)]
pub(in super::super) fn q8_row_asym_k3(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_vector_sdot(codes, fold_scale, Some(fold_mid), qx, in_dim) };
    }
    q8_row_asym_portable(codes, fold_scale, fold_mid, qx, in_dim, SDOT_LANES)
}

/// **CPU5-K4.** K3's vector accumulation, with the weight sums LOADED
/// as a vector instead of recomputed.
///
/// K1 removed the same four correction `SDOT`s and LOST 111 ms, because
/// it fetched the index as 320 scalar `i16` reads a row — a third stream
/// of tiny dependent loads. Under K3's four-block geometry the same four
/// sums are one 64-bit load and one widen:
///
/// ```text
/// K3:  4 useful SDOT + 4 correction SDOT + 6 vpaddq + 2 cvt + 2 vfma
/// K4:  4 useful SDOT               + 3 vpaddq + 2 cvt + 2 vfma
///                                  + 1 vld1_s16 + 1 vmovl_s16
/// ```
///
/// **Bit-identical to K3**, and that is the point rather than a bonus:
/// the index holds exactly the integers the correction `SDOT`s produce,
/// so `vcvtq_f32_s32` sees the same lanes and every float operation is
/// unchanged. One Bank-1 run therefore covers both.
///
/// Requires `sums` blocked on [`SUM_BLOCK`] with the activation blocked
/// identically, so a group of four activation blocks is four consecutive
/// sums.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_row_b16_indexed_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let blocks = in_dim / SDOT_LANES;
    let mut acc_v = vdupq_n_f32(0.0);
    let mut b = 0usize;
    while b + 4 <= blocks {
        let i0 = b * SDOT_LANES;
        let z = vdupq_n_s32(0);
        let d0 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0)),
            vld1q_s8(qx.as_ptr().add(i0)),
        );
        let d1 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0 + SDOT_LANES)),
            vld1q_s8(qx.as_ptr().add(i0 + SDOT_LANES)),
        );
        let d2 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0 + 2 * SDOT_LANES)),
            vld1q_s8(qx.as_ptr().add(i0 + 2 * SDOT_LANES)),
        );
        let d3 = vdotq_s32(
            z,
            vld1q_s8(codes.as_ptr().add(i0 + 3 * SDOT_LANES)),
            vld1q_s8(qx.as_ptr().add(i0 + 3 * SDOT_LANES)),
        );
        let dv = vpaddq_s32(vpaddq_s32(d0, d1), vpaddq_s32(d2, d3));
        acc_v = vfmaq_f32(
            acc_v,
            vld1q_f32(fold_scale.as_ptr().add(b)),
            vcvtq_f32_s32(dv),
        );
        // The four correction SDOTs, replaced by one 64-bit load.
        let sv = vmovl_s16(vld1_s16(sums.as_ptr().add(b)));
        acc_v = vfmaq_f32(
            acc_v,
            vld1q_f32(fold_mid.as_ptr().add(b)),
            vcvtq_f32_s32(sv),
        );
        b += 4;
    }
    let mut acc = vaddvq_f32(acc_v);
    // **The tail must associate exactly as K3's does.** K3 adds the
    // scale term and the offset term in two separate accumulations;
    // folding them into one `acc += A + B` here is a different rounding,
    // and the bit-identity gate caught precisely that on a shape with a
    // ragged group of four.
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        acc += fold_scale[b] * d as f32;
        acc += fold_mid[b] * *sums.get_unchecked(b) as f32;
        b += 1;
    }
    let done = blocks * SDOT_LANES;
    if done < in_dim {
        let (mut d, mut sm) = (0i32, 0i32);
        for i in done..in_dim {
            let w = *codes.get_unchecked(i) as i32;
            d += w * *qx.get_unchecked(i) as i32;
            sm += w;
        }
        acc += fold_scale[blocks] * d as f32;
        acc += fold_mid[blocks] * sm as f32;
    }
    acc
}

/// The K4 row, reachable from a test whatever the process arm.
#[cfg(test)]
pub(in super::super) fn q8_row_asym_k4(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_indexed_sdot(codes, fold_scale, fold_mid, qx, sums, in_dim) };
    }
    q8_row_asym_indexed_portable(codes, fold_scale, fold_mid, qx, sums, in_dim, SDOT_LANES)
}
