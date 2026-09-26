//! Register-blocked Q8 row kernels.

use super::super::arithmetic::ScaleSpan;

#[allow(unused_imports)]
use super::*;

/// **CPU5-K5.** K3's vector accumulation with the folded scales built in
/// REGISTERS instead of in a per-row buffer.
///
/// `fold_scales` materialised a 320-entry `f32` array per output row —
/// two of them for the asymmetric arm — and the kernel then read them
/// back. That is ~320 vector load/store operations against ~320 `SDOT`s
/// per row, roughly doubling the inner loop's op count. It is not DRAM
/// traffic (2560 B lives in L1); it is load/store port pressure, which
/// is why K4 removing vector ALU work changed nothing.
///
/// The buffers are avoidable because `ascale` and `amid` are
/// ROW-INVARIANT — they describe the activation, not the row — and
/// because at `ablock` 16 with `block` 64 a group of four activation
/// blocks is exactly one weight block, so the weight scale is a constant
/// within a group:
///
/// ```text
/// was:  fold[b] = wscale[b/4] * ascale[b], stored, reloaded
/// K5:   vmulq_n_f32(vld1q_f32(&ascale[b]), wscale[b/4])
/// ```
///
/// **Bit-identical to K3 and K4**: `ws * ascale[b]` is the same f32
/// whether it goes through memory first or not, and every subsequent
/// operation is unchanged.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. `wscales` must cover
/// `in_dim / (SDOT_LANES * PER_WEIGHT_B16)` blocks.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_row_b16_register_sdot(
    codes: &[i8],
    wscales: &[f32],
    ascales: &[f32],
    amids: Option<&[f32]>,
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let blocks = in_dim / SDOT_LANES;
    let mut acc_v = vdupq_n_f32(0.0);
    let mut b = 0usize;
    while b + PER_WEIGHT_B16 <= blocks {
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
        // ONE broadcast multiply where a 320-entry buffer used to be.
        let ws = *wscales.get_unchecked(b / PER_WEIGHT_B16);
        let scale_v = vmulq_n_f32(vld1q_f32(ascales.as_ptr().add(b)), ws);
        acc_v = vfmaq_f32(acc_v, scale_v, vcvtq_f32_s32(dv));
        if let Some(mid) = amids {
            let s0 = vdotq_s32(z, w0, ones);
            let s1 = vdotq_s32(z, w1, ones);
            let s2 = vdotq_s32(z, w2, ones);
            let s3 = vdotq_s32(z, w3, ones);
            let sv = vpaddq_s32(vpaddq_s32(s0, s1), vpaddq_s32(s2, s3));
            let mid_v = vmulq_n_f32(vld1q_f32(mid.as_ptr().add(b)), ws);
            acc_v = vfmaq_f32(acc_v, mid_v, vcvtq_f32_s32(sv));
        }
        b += PER_WEIGHT_B16;
    }
    let mut acc = vaddvq_f32(acc_v);
    while b < blocks {
        let lo = b * SDOT_LANES;
        let w = vld1q_s8(codes.as_ptr().add(lo));
        let ws = *wscales.get_unchecked(b / PER_WEIGHT_B16);
        let d = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(qx.as_ptr().add(lo))));
        acc += ws * *ascales.get_unchecked(b) * d as f32;
        if let Some(mid) = amids {
            acc +=
                ws * *mid.get_unchecked(b) * vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w, ones)) as f32;
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
        let ws = *wscales.get_unchecked(blocks / PER_WEIGHT_B16);
        acc += ws * *ascales.get_unchecked(blocks) * d as f32;
        if let Some(mid) = amids {
            acc += ws * *mid.get_unchecked(blocks) * sm as f32;
        }
    }
    acc
}

/// Activation blocks inside one weight block at `ablock` 16, `block` 64.
/// The value that makes the weight scale a constant within a group.
pub(in super::super) const PER_WEIGHT_B16: usize = 4;

/// The K5 row. Its only arm is the NEON one — the fallback below is an
/// `unimplemented!()`, so every caller is an aarch64-gated test and the
/// function is gated to match rather than sitting unused (and `-D
/// warnings`-fatal) on every other target.
#[cfg(all(test, target_arch = "aarch64"))]
pub(in super::super) fn q8_row_k3_register(
    codes: &[i8],
    wscales: &[f32],
    ascales: &[f32],
    amids: Option<&[f32]>,
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_register_sdot(codes, wscales, ascales, amids, qx, in_dim) };
    }
    unimplemented!("K5 has no portable arm; the gate runs on aarch64")
}

/// The K3 SYMMETRIC row. Gated with its caller: the K5-vs-K3 parity is
/// the only consumer and that comparison exists only where SDOT does.
#[cfg(all(test, target_arch = "aarch64"))]
pub(in super::super) fn q8_row_k3_sym(
    codes: &[i8],
    fold_scale: &[f32],
    qx: &[i8],
    in_dim: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_vector_sdot(codes, fold_scale, None, qx, in_dim) };
    }
    q8_row_portable(codes, fold_scale, qx, in_dim, SDOT_LANES)
}

/// Opts OUT of CPU5-K3, back to the bit-identical K2 kernels.
///
/// Exists so the two can be compared in ONE binary: K3 reassociates, and
/// a numerical control across two builds would be arguing with a
/// compiler as much as with the change.
pub const K2_ONLY_ENV: &str = "LARQL_CPU_BIT_IDENTICAL";

/// Whether to stay on the bit-identical K2 kernels.
pub fn bit_identical_only() -> bool {
    pub(super) static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var(K2_ONLY_ENV).ok().as_deref().map(str::trim),
            Some("1") | Some("true")
        )
    })
}

/// One asymmetric Q8 row, vectorised where possible.
#[inline]
pub(in super::super) fn q8_row_asym(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: both guarded by the runtime feature check.
        return unsafe {
            if block == SDOT_LANES && !bit_identical_only() {
                q8_row_b16_vector_sdot(codes, fold_scale, Some(fold_mid), qx, in_dim)
            } else if block == SDOT_LANES {
                q8_row_asym_b16_sdot(codes, fold_scale, fold_mid, qx, in_dim)
            } else {
                q8_row_asym_sdot(codes, fold_scale, fold_mid, qx, in_dim, block)
            }
        };
    }
    q8_row_asym_portable(codes, fold_scale, fold_mid, qx, in_dim, block)
}

/// The scale a block's integer sum is multiplied by: the weight's block
/// scale times the activation's.
///
/// One entry per ACTIVATION block. Where the activation block is finer
/// than the weight block, consecutive entries share a weight scale —
/// exact, because a weight scale is constant across its own block by
/// construction.
pub(super) fn fold_scales(
    weight_scales: &[f32],
    act_scales: &[f32],
    per_weight: usize,
    into: &mut Vec<f32>,
) {
    into.clear();
    into.extend(
        act_scales
            .iter()
            .enumerate()
            .map(|(s, a)| weight_scales[s / per_weight] * *a),
    );
}

/// Whether the integer arms scale the activation per tensor or per block.
///
/// Read from the same environment string as the arm, with a `b` suffix,
/// and resolved once per process for the same reason: a bank run is one
/// process per arm and a value that could change mid-decode would make
/// the resulting distribution describe no single representation.
pub fn activation_scaling() -> ScaleSpan {
    pub(super) static SCALING: std::sync::OnceLock<ScaleSpan> = std::sync::OnceLock::new();
    *SCALING.get_or_init(|| {
        scaling_for_arm(
            std::env::var(super::super::physical::ARITHMETIC_ARM_ENV)
                .ok()
                .as_deref(),
        )
    })
}

/// The scaling an arithmetic-arm string names — the pure half of
/// [`activation_scaling`], separated so the mapping can be checked
/// without setting the process-wide variable its cache reads once.
pub(in super::super) fn scaling_for_arm(arm: Option<&str>) -> ScaleSpan {
    match arm.map(str::trim) {
        // Blocked on the WEIGHTS' boundaries, so the two scales fold
        // into one multiply per block and `SDOT` is untouched.
        Some("bf16xq8b") | Some("q8xq8b") | Some("q4xq8b") => ScaleSpan::Block(activation_block()),
        _ => ScaleSpan::Tensor,
    }
}

/// Whether this machine has the `dotprod` extension `SDOT` needs.
///
/// Baseline on every Apple M-series part, but not on all aarch64, so the
/// portable definition is a real fallback and not dead code.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(in super::super) fn has_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// **The DEFINITION** of a Q8 x Q8 row, in portable Rust.
///
/// The integer sum is EXACT — no rounding happens inside a block at all —
/// so the only floating-point in the whole row is one multiply and one
/// add per block. That is what makes the vectorised path bit-comparable
/// rather than merely close.
pub(in super::super) fn q8_row_portable(
    codes: &[i8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut sum = 0i32;
        for i in lo..hi {
            sum += codes[i] as i32 * qx[i] as i32;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// **The DEFINITION** of a Q4 x Q8 row, in portable Rust.
///
/// Byte `j` of a block carries element `j` in its low nibble and element
/// `j + half` in its high nibble, so a block is two CONTIGUOUS runs of
/// the activation rather than one interleaved one.
pub(in super::super) fn q4_row_portable(
    packed: &[u8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let half = (hi - lo) / 2;
        let mut sum = 0i32;
        for j in 0..half {
            let byte = packed[lo / 2 + j];
            sum += ((byte & NIBBLE) as i32 - Q4_BIAS) * qx[lo + j] as i32;
            sum += ((byte >> 4) as i32 - Q4_BIAS) * qx[lo + j + half] as i32;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// `SDOT`: sixteen int8 pairs into four i32 lanes, one instruction.
///
/// The widen chain Q8 x f32 spends its time on disappears entirely —
/// no `s8 -> s16 -> s32 -> f32` per element, just a load and a dot.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. Every access stays
/// inside the slices the row geometry describes.
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98 and the workspace still
/// declares `rust-version = "1.88"`. The toolchain is PINNED to 1.98 in
/// `rust-toolchain.toml` — deliberately, so local lints are CI's lints —
/// so the declared floor is already below what anything here builds
/// with. Allowed at the function rather than raised workspace-wide,
/// because bumping the manifest's MSRV is a policy decision about every
/// crate and not a side effect of adding a kernel.
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q8_row_sdot(
    codes: &[i8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
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
        let mut sum = vaddvq_s32(lanes);
        while i < hi {
            sum += *codes.get_unchecked(i) as i32 * *qx.get_unchecked(i) as i32;
            i += 1;
        }
        acc += scale * sum as f32;
    }
    acc
}

/// Q4 x Q8 through `SDOT`: mask and shift one 16-byte load into two int8
/// vectors, unbias by 8, dot each against its half of the activation.
///
/// No widening and no float anywhere in the inner loop.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`]. Every access stays
/// inside the slices the row geometry describes.
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98 and the workspace still
/// declares `rust-version = "1.88"`. The toolchain is PINNED to 1.98 in
/// `rust-toolchain.toml` — deliberately, so local lints are CI's lints —
/// so the declared floor is already below what anything here builds
/// with. Allowed at the function rather than raised workspace-wide,
/// because bumping the manifest's MSRV is a policy decision about every
/// crate and not a side effect of adding a kernel.
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn q4_row_sdot(
    packed: &[u8],
    scales: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(NIBBLE);
    let bias = vdupq_n_s8(Q4_BIAS as i8);
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let half = (hi - lo) / 2;
        let base = packed.as_ptr().add(lo / 2);
        let xbase = qx.as_ptr().add(lo);
        let mut lanes = vdupq_n_s32(0);
        let mut j = 0usize;
        while j + SDOT_LANES <= half {
            let raw = vld1q_u8(base.add(j));
            let low = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);
            let high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
            lanes = vdotq_s32(lanes, low, vld1q_s8(xbase.add(j)));
            lanes = vdotq_s32(lanes, high, vld1q_s8(xbase.add(j + half)));
            j += SDOT_LANES;
        }
        let mut sum = vaddvq_s32(lanes);
        while j < half {
            let byte = *packed.get_unchecked(lo / 2 + j);
            sum += ((byte & NIBBLE) as i32 - Q4_BIAS) * *qx.get_unchecked(lo + j) as i32;
            sum += ((byte >> 4) as i32 - Q4_BIAS) * *qx.get_unchecked(lo + j + half) as i32;
            j += 1;
        }
        acc += scale * sum as f32;
    }
    acc
}
