//! Integer-domain projections: int8 activations, int32 accumulators.
//!
//! Every kernel beside this one multiplies a compact weight by an **f32
//! activation**. CPU-4X measured what that costs and CPU-4Y measured what
//! removing it buys:
//!
//! ```text
//! BF16 x F32   420.84 ms/token   51.20 GB   121.7 GB/s   1.00x
//! Q8   x F32   332.97 ms         27.20 GB    83.4 GB/s   1.26x
//! Q8   x Q8    224.75 ms         27.20 GB   118.0 GB/s   1.87x
//! Q4   x Q8    135.10 ms         14.40 GB   106.6 GB/s   3.12x
//! ```
//!
//! **The regimes alternate.** bf16 is memory-bound; Q8 against f32 leaves
//! that regime and becomes conversion-bound, which is why halving the
//! bytes again to Q4 x f32 came back 20% SLOWER; integer arithmetic
//! removes the conversion and puts the format back on the memory wall,
//! at which point Q4's halved traffic pays again. Compressing weights
//! buys nothing until the arithmetic can consume them, and native
//! arithmetic buys nothing once bytes are the limit.
//!
//! ## What these kernels are FOR
//!
//! They are the numerical instrument for CPU-5's quality gate, and they
//! are the deployment path — deliberately the same code. Scoring a
//! *simulation* of Q4 x Q8 and then shipping a different kernel would
//! qualify something nobody runs.
//!
//! ## Why the activation is quantised INSIDE the kernel
//!
//! The activation is a property of the CALL, not of the weight slab, so
//! it wants to be quantised once in [`super::executor::CpuExecutor::project`]
//! and shared by every worker. It is not, yet, because
//! [`DenseProjector`] names only a weight representation and teaching
//! [`super::PhysicalProjectionPlan`] to name an ACTIVATION and an
//! ACCUMULATOR is a real change to what a physical plan means.
//!
//! Re-quantising per worker is therefore waste, and it is bounded waste:
//! `in_dim` operations against a slab of `out_dim/workers * in_dim`, and
//! CPU-4X priced the whole activation quantiser at **0.33% of the integer
//! path**. It is also numerically FREE — every worker sees the same full
//! `x` and so derives the same scale and the same codes, which is why the
//! row partition cannot change an answer.
//!
//! When the plan learns to say `Q4 x Q8 -> I32 -> F32`, the quantiser
//! moves up to the executor and these kernels take a
//! [`QuantisedActivation`] instead of making one.

use crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK;

mod q4_projectors;
mod q8_register;
mod q8_sdot;
pub use q4_projectors::*;
pub use q8_register::*;
use q8_sdot::*;
// Test-only scalar references the sibling test modules compare against.
#[cfg(test)]
pub(super) use q8_sdot::{q8_row_asym_exact, q8_row_asym_k3, q8_row_asym_k4};

/// The largest magnitude an int8 activation code may represent.
///
/// 127 and not 128, symmetric, for the same reason the weight quantisers
/// use it: the negative extreme would give one direction a level the
/// other lacks.
const ACT_MAX: f32 = 127.0;

/// Elements per `SDOT` instruction: sixteen int8 pairs into four i32
/// lanes.
///
/// Gated: off aarch64 the portable definitions run and nothing consumes
/// this, which `-D warnings` treats as an error on the CI targets — the
/// exact way cfg-gated code has broken this build before.
///
/// NOT `cfg`-gated, despite naming an `aarch64` instruction: it is a plain
/// integer, and `Q8xQ8::project_rows` compares against it on every target
/// before narrowing to the NEON path. Gating it made this crate fail to
/// build on x86 — the same failure the note above warns about.
pub(super) const SDOT_LANES: usize = 16;

/// The int4 bias that makes a signed code an unsigned nibble.
const Q4_BIAS: i32 = 8;

/// Low nibble mask.
const NIBBLE: u8 = 0x0f;

/// One activation vector as symmetric int8, plus the scale that restores
/// it.
///
/// **One scale for the whole vector**, not one per block. The activation
/// is read once per projection and its scale multiplies out at the very
/// end, so a blocked activation scale would buy accuracy the weights'
/// own blocking already provides and cost a multiply per block.
pub struct QuantisedActivation {
    pub codes: Vec<i8>,
    pub scale: f32,
}

/// `scale = max|x| / 127`, `code = round(x / scale)`.
///
/// A zero vector would divide by zero; 1.0 keeps its codes at zero and
/// the vector reconstructs exactly.
pub fn quantise_activation(x: &[f32]) -> QuantisedActivation {
    let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if peak > 0.0 { peak / ACT_MAX } else { 1.0 };
    let inv = 1.0 / scale;
    let codes = x
        .iter()
        .map(|v| (v * inv).round().clamp(-ACT_MAX, ACT_MAX) as i8)
        .collect();
    QuantisedActivation { codes, scale }
}

/// The same rule, once per `block` elements along the input axis.
///
/// Blocks never straddle the vector's end: a short final block takes its
/// own peak rather than borrowing a neighbour's scale.
pub fn quantise_activation_blocked(x: &[f32], block: usize) -> (Vec<i8>, Vec<f32>) {
    let blocks = x.len().div_ceil(block);
    let mut codes = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; blocks];
    for (b, (scale_slot, chunk)) in scales.iter_mut().zip(x.chunks(block)).enumerate() {
        let peak = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if peak > 0.0 { peak / ACT_MAX } else { 1.0 };
        *scale_slot = scale;
        let inv = 1.0 / scale;
        for (i, v) in chunk.iter().enumerate() {
            codes[b * block + i] = (v * inv).round().clamp(-ACT_MAX, ACT_MAX) as i8;
        }
    }
    (codes, scales)
}

/// Names the activation's scale block, independently of the weight's.
pub const ACT_BLOCK_ENV: &str = "LARQL_CPU_ACT_BLOCK";

/// The smallest activation block one `SDOT` can fill.
const SDOT_MIN: usize = 16;

/// **How many elements share one activation scale.**
///
/// Independent of the weight block, and cheap in a way the weight block
/// is not. A weight scale is paid once per block PER ROW, so halving the
/// weight block costs half a bit on every weight in the model. The
/// activation is ONE VECTOR: at `in_dim` 5120 its scales are 80 floats at
/// block 64 and 320 at block 16 — 320 B against 1.3 KB, set beside 14.4
/// GB of weights per token. Under a millionth of the traffic either way.
///
/// So the only real cost of a finer activation block is arithmetic: one
/// extra float multiply-add per sub-block against `block` integer MACs,
/// and `SDOT` itself is untouched.
///
/// **This asymmetry is the whole reason the activation is worth fixing
/// before the weights are.** CPU-5 measured a blocked-Q8[64] activation
/// against EXACT weights at KL 0.00061 bits/token, 3.8x the entire
/// accepted cost of Q8 weight quantisation — so the activation, not the
/// weight format, is what a Q4 x Q8 plan is spending its budget on.
///
/// Must DIVIDE the weight block and fill at least one `SDOT`. A value
/// that does neither is refused rather than rounded, because a
/// mismatched geometry pairs weights with another block's scale and
/// still returns finite, plausible numbers.
pub fn activation_block() -> usize {
    static BLOCK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BLOCK.get_or_init(|| {
        let weight_block = crate::format::vindex3::opplan::exec::quantise::Q8_BLOCK;
        let want = std::env::var(ACT_BLOCK_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(weight_block);
        assert!(
            want >= SDOT_MIN && want <= weight_block && weight_block.is_multiple_of(want),
            "{ACT_BLOCK_ENV}={want} must divide the weight block {weight_block} and be at \
             least {SDOT_MIN}"
        );
        want
    })
}

/// Whether the activation code carries a per-block OFFSET as well as a
/// scale.
///
/// Symmetric coding centres every block on zero and spends half its range
/// on whichever sign the block does not use. Measured on real residual
/// blocks, the step a per-block offset would save is
/// `2 * peak / (max - min)` — 1.0 for a balanced block, 2.0 for a
/// one-sided one:
///
/// ```text
/// layer   blk16 gain   frac of blocks > 1.2x
///   000      1.175           34.4%
///   016      1.307           63.1%
///   024      1.360           68.8%
/// ```
///
/// KL goes as the SQUARE of the step, so ~1.3x in step is ~1.7x in
/// logit KL — against the 1.6% that `Q8 x Q8[16]` missed G1 by.
///
/// **It costs arithmetic and no traffic.** Reconstructing
/// `x = c * scale + mid` turns the dot into
/// `scale * SUM(w*c) + mid * SUM(w)`, and the weight codes are already
/// in registers for the `SDOT`, so the second term is one more reduction
/// over data that has been loaded either way. On a path already at the
/// memory wall (121.0 GB/s) that is close to free.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ActivationCode {
    /// Codes centred on zero; one scale per block.
    #[default]
    Symmetric,
    /// One scale AND one offset per block.
    Asymmetric,
}

/// Opts into the CPU5-K1 weight-code index. Off by default: it was
/// measured SLOWER than recomputing the sums.
pub const WEIGHT_INDEX_ENV: &str = "LARQL_CPU_WEIGHT_INDEX";

/// Whether to build and consume the weight-code index. See
/// [`WEIGHT_INDEX_ENV`].
pub fn weight_index_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var(WEIGHT_INDEX_ENV)
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1") | Some("true")
        )
    })
}

/// Names the activation code. `symmetric` (default) or `asymmetric`.
pub const ACT_CODE_ENV: &str = "LARQL_CPU_ACT_CODE";

/// The activation code, resolved once per process.
pub fn activation_code() -> ActivationCode {
    static CODE: std::sync::OnceLock<ActivationCode> = std::sync::OnceLock::new();
    *CODE.get_or_init(
        || match std::env::var(ACT_CODE_ENV).ok().as_deref().map(str::trim) {
            Some("asymmetric") => ActivationCode::Asymmetric,
            _ => ActivationCode::Symmetric,
        },
    )
}

/// One activation vector as asymmetric int8: per-block scale AND offset.
///
/// `mid = (max + min) / 2`, `scale = (max - min) / 255`, and
/// `code = round((x - mid) / scale)` lands in `-128..=127` by
/// construction. A constant block has `max == min`; its sentinel scale
/// keeps every code at zero and the block reconstructs EXACTLY from the
/// offset alone, which a symmetric code cannot do.
pub fn quantise_activation_asymmetric(x: &[f32], block: usize) -> (Vec<i8>, Vec<f32>, Vec<f32>) {
    let blocks = x.len().div_ceil(block);
    let mut codes = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; blocks];
    let mut mids = vec![0.0f32; blocks];
    for (b, chunk) in x.chunks(block).enumerate() {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in chunk {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let mid = 0.5 * (hi + lo);
        let span = hi - lo;
        let scale = if span > 0.0 { span / ASYM_LEVELS } else { 1.0 };
        scales[b] = scale;
        mids[b] = mid;
        let inv = 1.0 / scale;
        for (i, v) in chunk.iter().enumerate() {
            codes[b * block + i] = ((*v - mid) * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (codes, scales, mids)
}

/// Levels an asymmetric int8 code spans: `-128..=127` is 255 steps.
const ASYM_LEVELS: f32 = 255.0;

/// **The DEFINITION** of an asymmetric Q8 x Q8 row.
///
/// `x = c * scale + mid` per block, so the row is
/// `SUM_b [ scale_b * SUM(w*c) + mid_b * SUM(w) ]`, with both scales
/// pre-multiplied by the weight's own block scale.
pub(super) fn q8_row_asym_portable(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut dot = 0i32;
        let mut wsum = 0i32;
        for i in lo..hi {
            dot += codes[i] as i32 * qx[i] as i32;
            wsum += codes[i] as i32;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// The same through `SDOT`. The weight sum is one more dot, against a
/// vector of ones — the codes are already loaded, so it buys the offset
/// term without touching memory again.
///
/// # Safety
/// Requires `dotprod`, checked by [`has_dotprod`].
#[cfg(target_arch = "aarch64")]
/// `dotprod` intrinsics are stable since 1.98; see the note on
/// [`q8_row_sdot`].
#[allow(clippy::incompatible_msrv)]
#[target_feature(enable = "dotprod")]
unsafe fn q8_row_asym_sdot(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    in_dim: usize,
    block: usize,
) -> f32 {
    use std::arch::aarch64::*;
    let ones = vdupq_n_s8(1);
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut dot_lanes = vdupq_n_s32(0);
        let mut sum_lanes = vdupq_n_s32(0);
        let mut i = lo;
        while i + SDOT_LANES <= hi {
            let w = vld1q_s8(codes.as_ptr().add(i));
            dot_lanes = vdotq_s32(dot_lanes, w, vld1q_s8(qx.as_ptr().add(i)));
            sum_lanes = vdotq_s32(sum_lanes, w, ones);
            i += SDOT_LANES;
        }
        let mut dot = vaddvq_s32(dot_lanes);
        let mut wsum = vaddvq_s32(sum_lanes);
        while i < hi {
            let w = *codes.get_unchecked(i) as i32;
            dot += w * *qx.get_unchecked(i) as i32;
            wsum += w;
            i += 1;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}

/// The code-sum indices covering activation elements `lo..hi`.
///
/// Derived from the span rather than from `block / SUM_BLOCK`: a row's
/// last activation block is short when `in_dim` is not a multiple of the
/// block, and so is its run of sums (`quantise::code_sums` cuts them per
/// row). A fixed stride read past the row there — the next row's sums on
/// the portable path, past the slice on the unchecked one.
#[inline]
fn index_span(lo: usize, hi: usize) -> std::ops::Range<usize> {
    lo / SUM_BLOCK..hi.div_ceil(SUM_BLOCK)
}

/// Refuses a row whose operands are shorter than `in_dim` declares.
///
/// The indexed kernels read `codes`, `qx` and `sums` unchecked on the
/// SDOT path, so the lengths are proven once here, per row, before any
/// of them is dereferenced.
#[inline]
fn check_indexed_row(codes: &[i8], qx: &[i8], sums: &[i16], in_dim: usize) {
    assert!(
        codes.len() >= in_dim && qx.len() >= in_dim && sums.len() >= in_dim.div_ceil(SUM_BLOCK),
        "indexed Q8 row needs {in_dim} codes, {in_dim} activation codes and {} sums; got {}, {} \
         and {}",
        in_dim.div_ceil(SUM_BLOCK),
        codes.len(),
        qx.len(),
        sums.len()
    );
}

/// The indexed asymmetric row: K4's vector index load at block 16, K1's
/// scalar-index row otherwise.
#[inline]
fn q8_row_asym_with_index(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    check_indexed_row(codes, qx, sums, in_dim);
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() && block == SDOT_LANES && !bit_identical_only() {
        // SAFETY: guarded by the runtime feature check.
        return unsafe { q8_row_b16_indexed_sdot(codes, fold_scale, fold_mid, qx, sums, in_dim) };
    }
    q8_row_asym_indexed(codes, fold_scale, fold_mid, qx, sums, in_dim, block)
}

/// **CPU5-K1.** The same row, with the weight sums READ rather than
/// recomputed.
///
/// `SUM(q)` depends only on the weight block, so recomputing it every
/// token costs a second `SDOT` and a second integer reduction per block.
/// The index costs one bit per weight (`i16` per 16 codes, exact because
/// `16 * 127 = 2032`), i.e. ~12% more compact traffic.
///
/// **Bit-identical to [`q8_row_asym`] by construction**: an i32 sum of
/// i16 sub-sums taken in order is the same integer the reduction would
/// have produced, and no float operation changes.
///
/// Written as a whole ROW rather than as a per-block helper on purpose.
/// A first version called a `q8_block_dot(&codes[lo..hi], &qx[lo..hi])`
/// per block and measured 1105 ms against the 757 ms it was meant to
/// beat — the slicing, bounds checks and call boundary per block cost
/// more than the `SDOT` it removed. At 320 blocks a row, per-block
/// abstraction is the thing being optimised away.
pub(super) fn q8_row_asym_indexed(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    check_indexed_row(codes, qx, sums, in_dim);
    #[cfg(target_arch = "aarch64")]
    if has_dotprod() {
        // SAFETY: guarded by the runtime feature check; operand lengths
        // proven by `check_indexed_row`.
        return unsafe {
            q8_row_asym_indexed_sdot(codes, fold_scale, fold_mid, qx, sums, in_dim, block)
        };
    }
    q8_row_asym_indexed_portable(codes, fold_scale, fold_mid, qx, sums, in_dim, block)
}

/// The portable definition of the indexed row.
pub(super) fn q8_row_asym_indexed_portable(
    codes: &[i8],
    fold_scale: &[f32],
    fold_mid: &[f32],
    qx: &[i8],
    sums: &[i16],
    in_dim: usize,
    block: usize,
) -> f32 {
    let mut acc = 0.0f32;
    for (b, (s, m)) in fold_scale.iter().zip(fold_mid).enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(in_dim);
        if lo >= hi {
            break;
        }
        let mut dot = 0i32;
        for i in lo..hi {
            dot += codes[i] as i32 * qx[i] as i32;
        }
        let mut wsum = 0i32;
        for k in index_span(lo, hi) {
            wsum += sums[k] as i32;
        }
        acc += s * dot as f32 + m * wsum as f32;
    }
    acc
}
