//! NVFP4 matrix-vector multiply — direct compressed execution.
//!
//! **Q2-R1.** The MXFP4 sibling ([`super::mxfp4_matvec`]) proved E2M1 can
//! be a compute format; this one changes only the *scale* geometry, which
//! is the variable VINDEX3-Q2 is testing:
//!
//! ```text
//!            elements   group   group scale   tensor scale
//! MXFP4      E2M1       32      E8M0          —
//! NVFP4      E2M1       16      E4M3          one f32
//! ```
//!
//! A weight-reconstruction sweep over Muse-Glimmer's real tensors, with
//! an equal-bit-budget control (E8M0 at group 16, also 4.5 bpw), found
//! the group size worth nothing — 0.996x on attention — and the scale
//! format worth 1.265x in relative RMS and 1.68x in worst-element error.
//! So the format under test here is specifically E4M3-scaled, and the
//! kernel keeps E2M1 decode byte-identical to the MXFP4 path so the two
//! differ in nothing else.
//!
//! ## Format
//!
//! Per output row, `groups = K / 16`. Group `g` holds:
//!   - 8 packed bytes at `packed[(row * groups + g) * 8 ..][..8]`, each
//!     carrying two 4-bit codes: **lo nibble first**, then hi.
//!   - one E4M3 scale byte at `scales[row * groups + g]`.
//!
//! and one f32 `tensor_scale` multiplies every decoded element:
//!
//! ```text
//! w[row, g*16 + i] = tensor_scale * e4m3(scale) * e2m1(code)
//! ```
//!
//! The association matters: the CPU reference folds `tensor_scale *
//! e4m3(scale)` into one step per group and multiplies the E2M1 code by
//! it, and this kernel does the same, so the two agree to fp rounding
//! rather than by luck.
//!
//! E4M3 decode follows OCP FP8 v1.0 and mirrors `quant::fp8::e4m3_to_f32`
//! exactly, including subnormals (`exp == 0` → `mant * 2^-9`) and the two
//! NaN encodings (`0x7F`, `0xFF`). Subnormals are not decorative here:
//! the tensor scale normalises the largest group to E4M3's *top*, so a
//! matrix with a wide spread of group amaxes pushes its quietest groups
//! into the subnormal range, and flushing them to zero would silently
//! delete whole groups of weights.
//!
//! ## Parallelism
//!
//! One simdgroup per output row, `ROWS_PER_TG` simdgroups per
//! threadgroup — the MXFP4 geometry unchanged, deliberately: a dispatch
//! shape that collapses threadgroup count has cost more than it saved
//! before, and this rung is an accuracy question, not a tuning one.
//!
//! Lane `l` walks groups `l, l+32, ...`, reading one contiguous 8-byte
//! group each; adjacent lanes cover 256 contiguous bytes per step. Half
//! the per-lane bytes of the MXFP4 kernel because the group is half as
//! wide, so a row of the same `K` takes the same number of steps with
//! twice the scale reads. The K reduction closes with `simd_sum`.
//!
//! Accumulation order differs from the CPU reference (which sums
//! left-to-right), so parity is a bounded-error contract, not
//! bit-equality — the same contract the MXFP4 rung established.

/// Output rows per threadgroup — one simdgroup each.
pub const ROWS_PER_TG: u64 = 4;

mod sweep;
pub use sweep::*;
/// 4 simdgroups x 32 lanes.
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint NVFP4_ROWS_PER_TG = 4;
constant uint NVFP4_GROUP_ELEMS = 16;
constant uint NVFP4_GROUP_BYTES = 8;

// ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} — sign in bit 3, then exp(2) and mantissa(1).
// Identical to MXFP4_LUT: the element grid is the shared half of the two
// formats, and Q2 is about the scale.
constant float NVFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 -> f32, matching quant::fp8::e4m3_to_f32 including subnormals and
// both NaN encodings. 1 sign, 4 exponent (bias 7), 3 mantissa; no Inf.
inline float nvfp4_e4m3(uchar b) {
    const uint sign = uint(b) >> 7;
    const uint exp  = (uint(b) >> 3) & 0xFu;
    const uint mant = uint(b) & 0x7u;
    float mag;
    if (exp == 0u) {
        // Subnormal: mant/8 * 2^-6 == mant * 2^-9. Reached routinely,
        // because the tensor scale pins the loudest group at E4M3's top
        // and pushes quiet groups down here.
        mag = float(mant) * 0.001953125f;   // 2^-9
    } else if (exp == 0xFu && mant == 0x7u) {
        mag = NAN;
    } else {
        mag = (1.0f + float(mant) * 0.125f) * exp2(float(int(exp) - 7));
    }
    return (sign != 0u) ? -mag : mag;
}

kernel void nvfp4_matvec(
    device const uchar*  Wp     [[buffer(0)]],   // packed [M, groups, 8]
    device const uchar*  Ws     [[buffer(1)]],   // scales [M, groups] E4M3
    device const float*  X      [[buffer(2)]],   // [K]
    device float*        out    [[buffer(3)]],   // [M]
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],   // one f32 for the matrix
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * NVFP4_ROWS_PER_TG + sg_id;
    if (row >= M) { return; }

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uchar* row_p = Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;

    float acc = 0.0f;

    // Lane l walks groups l, l+32, ... — one contiguous 8-byte read each,
    // 256 contiguous bytes across the simdgroup per step.
    for (uint g = lane; g < groups; g += 32u) {
        // Fold both scale levels once per group, exactly as the CPU
        // reference does, then apply to the E2M1 codes.
        const float step = Tscale * nvfp4_e4m3(row_s[g]);
        device const uchar* blk = row_p + (ulong)g * NVFP4_GROUP_BYTES;
        const uint base = g * NVFP4_GROUP_ELEMS;

        // Scalar byte loads, deliberately. A `uint2` + `float4` variant
        // measured *slower* (101.0 vs 110.3 GB/s over one layer's four
        // projections), so the compiler is already vectorising this and
        // load width is not what the kernel is short of.
        float part = 0.0f;
        for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += NVFP4_LUT[byte & 0x0Fu]         * X[base + 2u * b];
            part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * X[base + 2u * b + 1u];
        }
        acc += step * part;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[row] = acc; }
}

// ── v2: the falsified "issue-bound decode" hypothesis, retained ─────────
//
// The A-12 stage ledger priced the kernel above at 155–239 GB/s on the
// shapes that matter, where the f16 GEMV reaches 292–351 on identical
// geometry. Hypothesis: issue-bound on the decode — per group a v1 lane
// issues 8 scalar byte loads, 16 scalar X loads and 16 dynamic LUT
// lookups (a `constant float[16]` indexed by a runtime nibble is a
// constant-memory load). v2 decodes E2M1 arithmetically (magnitudes ×2
// packed as nibbles in one literal, sign ORed into bit 31), loads the 8
// code bytes as one `uint2` and X as four `float4`s.
//
// Measured (`examples/nvfp4_gemv_shapes.rs`, chained in one command
// buffer): v2 is 0.8–0.9× v1 at 8 rows/TG and 0.85–1.0× at 4 rows/TG —
// the decode is NOT the limiter; what remains is memory-level
// parallelism / per-dispatch ramp, which is the A-5 sweep (bytes per
// lane per step, rows per TG) under a stable power state. Kept as an
// explicit arm (`LARQL_NVFP4_KERNEL=v2`; default v1) under the shader
// retention policy. Numerically: the same values to fp32 rounding
// (rel_rms ~1e-7) — not bit-identical, because Metal's default fast
// math contracts the two code shapes differently;
// `tests/test_kernel_nvfp4_matvec_v2.rs` pins the tolerance.
constant uint NVFP4_V2_ROWS_PER_TG = 4;
// Magnitude × 2 for codes 0..7: {0,1,2,3,4,6,8,12}, nibble c at bits 4c.
constant uint NVFP4_MAG2_TABLE = 0xC8643210u;

inline float nvfp4_v2_decode(uint code) {
    const float mag2 = float((NVFP4_MAG2_TABLE >> ((code & 7u) << 2u)) & 0xFu);
    // Sign: code bit 3 → float bit 31. Exact; -0 for code 8.
    return as_type<float>(as_type<uint>(mag2) | ((code & 8u) << 28u));
}

kernel void nvfp4_matvec_v2(
    device const uchar*  Wp     [[buffer(0)]],   // packed [M, groups, 8]
    device const uchar*  Ws     [[buffer(1)]],   // scales [M, groups] E4M3
    device const float*  X      [[buffer(2)]],   // [K]
    device float*        out    [[buffer(3)]],   // [M]
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * NVFP4_V2_ROWS_PER_TG + sg_id;
    if (row >= M) { return; }

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uint2* row_p =
        (device const uint2*)(Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES);
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;
    device const float4* X4 = (device const float4*)X;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        // ×0.5 folded here (exact) because the table carries 2×magnitude.
        const float step = 0.5f * Tscale * nvfp4_e4m3(row_s[g]);
        const uint2 w = row_p[g];
        const float4 x0 = X4[g * 4u + 0u];
        const float4 x1 = X4[g * 4u + 1u];
        const float4 x2 = X4[g * 4u + 2u];
        const float4 x3 = X4[g * 4u + 3u];
        // Same element order as v1: byte b, lo nibble then hi.
        float part = 0.0f;
        part += nvfp4_v2_decode(w.x         & 0xFu) * x0.x;
        part += nvfp4_v2_decode((w.x >> 4u)  & 0xFu) * x0.y;
        part += nvfp4_v2_decode((w.x >> 8u)  & 0xFu) * x0.z;
        part += nvfp4_v2_decode((w.x >> 12u) & 0xFu) * x0.w;
        part += nvfp4_v2_decode((w.x >> 16u) & 0xFu) * x1.x;
        part += nvfp4_v2_decode((w.x >> 20u) & 0xFu) * x1.y;
        part += nvfp4_v2_decode((w.x >> 24u) & 0xFu) * x1.z;
        part += nvfp4_v2_decode((w.x >> 28u) & 0xFu) * x1.w;
        part += nvfp4_v2_decode(w.y         & 0xFu) * x2.x;
        part += nvfp4_v2_decode((w.y >> 4u)  & 0xFu) * x2.y;
        part += nvfp4_v2_decode((w.y >> 8u)  & 0xFu) * x2.z;
        part += nvfp4_v2_decode((w.y >> 12u) & 0xFu) * x2.w;
        part += nvfp4_v2_decode((w.y >> 16u) & 0xFu) * x3.x;
        part += nvfp4_v2_decode((w.y >> 20u) & 0xFu) * x3.y;
        part += nvfp4_v2_decode((w.y >> 24u) & 0xFu) * x3.z;
        part += nvfp4_v2_decode((w.y >> 28u) & 0xFu) * x3.w;
        acc += step * part;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[row] = acc; }
}
"#;

macro_rules! sweep_kernel {
    ($ty:ident, $name:literal, $rows:expr) => {
        sweep_kernel!($ty, $name, $rows, $rows * 32);
    };
    ($ty:ident, $name:literal, $rows:expr, $threads:expr) => {
        /// A-5 sweep arm; see `SWEEP_SHADER`.
        pub struct $ty;
        impl crate::kernels::TiledKernel for $ty {
            const KERNEL_NAME: &'static str = $name;
            const ROWS_PER_TG: u64 = $rows;
            const THREADS_PER_TG: u64 = $threads;
        }
    };
}
// A-5a arms: rows per lane ∈ {1,2,4} × LUT width; 4 simdgroups per TG,
// so rows per TG = 4·RL.
sweep_kernel!(KernelX2, "nvfp4_matvec_x2", 8, 128);
/// x2 with the pre-norm folded in (buffers 8/9/10 = Wn, eps, offset).
pub struct KernelX2N;
impl crate::kernels::TiledKernel for KernelX2N {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2n";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// x2 with the pre-norm staged in threadgroup memory (rung 2d form B).
pub struct KernelX2M;
impl crate::kernels::TiledKernel for KernelX2M {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2m";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// x2 with the residual add folded into the write (buffer 7 = R).
pub struct KernelX2R;
impl crate::kernels::TiledKernel for KernelX2R {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2r";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
sweep_kernel!(KernelX4, "nvfp4_matvec_x4", 16, 128);
sweep_kernel!(KernelX1B, "nvfp4_matvec_x1b", 4, 128);
sweep_kernel!(KernelX2B, "nvfp4_matvec_x2b", 8, 128);
sweep_kernel!(KernelX4B, "nvfp4_matvec_x4b", 16, 128);
/// seg3t: per-threadgroup segment resolution (8 rows/TG, tile-aligned).
pub struct KernelX2Seg3T;
impl crate::kernels::TiledKernel for KernelX2Seg3T {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2_seg3t";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// A-5b: segmented x2, up to three matrices in one dispatch (8 rows/TG).
pub struct KernelX2Seg3;
impl crate::kernels::TiledKernel for KernelX2Seg3 {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_x2_seg3";
    const ROWS_PER_TG: u64 = 8;
    const THREADS_PER_TG: u64 = 128;
}
/// VERIFY-N multi-RHS arms as (kernel name, activation rows), in
/// [`QuantKernels::nvfp4_matmul_pipelines`](crate::kernels::quant) order.
pub const MATMUL_ARMS: [(&str, usize); 7] = [
    ("nvfp4_matmul_x2_r1", 1),
    ("nvfp4_matmul_x2_r2", 2),
    ("nvfp4_matmul_x2_r4", 4),
    ("nvfp4_matmul_x2_r8", 8),
    ("nvfp4_matmul_x4_r4", 4),
    ("nvfp4_matmul_x4_r8", 8),
    ("nvfp4_matmul_x1_r8", 8),
];
sweep_kernel!(KernelMatmulR1, "nvfp4_matmul_x2_r1", 8, 128);
sweep_kernel!(KernelMatmulR2, "nvfp4_matmul_x2_r2", 8, 128);
sweep_kernel!(KernelMatmulR4, "nvfp4_matmul_x2_r4", 8, 128);
sweep_kernel!(KernelMatmulR8, "nvfp4_matmul_x2_r8", 8, 128);
sweep_kernel!(KernelMatmulX4R4, "nvfp4_matmul_x4_r4", 16, 128);
sweep_kernel!(KernelMatmulX4R8, "nvfp4_matmul_x4_r8", 16, 128);
sweep_kernel!(KernelMatmulX1R8, "nvfp4_matmul_x1_r8", 4, 128);
// VERIFY-N split-K tiles: 32 rows per threadgroup, 8 simdgroups over K,
// position count `R` (1..=8) at buffer 7.
sweep_kernel!(KernelMatmulSgk, "nvfp4_matmul_sgk", 32, 256);
/// The sgf arm production uses for wide verify blocks: 8 simdgroups split
/// K contiguously over 4 row-blocks — the sweep's best layer-weighted cost
/// at R=8 (`examples/nvfp4_verify_widths.rs`).
pub const SGF_PRODUCTION_ARM: &str = "nvfp4_matmul_sgf_s8b4c";
/// sgf geometry sweep arms: (name, rows per TG, threads per TG).
pub const SGF_SWEEP: [(&str, u64, u64); 12] = [
    ("nvfp4_matmul_sgf", 32, 256),
    ("nvfp4_matmul_sgf_s4b4", 32, 128),
    ("nvfp4_matmul_sgf_s8b8", 64, 256),
    ("nvfp4_matmul_sgf_s4b8", 64, 128),
    ("nvfp4_matmul_sgf_s8b4c", 32, 256),
    ("nvfp4_matmul_sgf_s2b8", 64, 64),
    ("nvfp4_matmul_sgf_s1b4", 32, 32),
    ("nvfp4_matmul_sgf_s8b2c", 16, 256),
    ("nvfp4_matmul_sgf_s8b1c", 8, 256),
    ("nvfp4_matmul_sgf_s16b4c", 32, 512),
    ("nvfp4_matmul_sgf_s16b2c", 16, 512),
    ("nvfp4_matmul_sgf_s32b1c", 8, 1024),
];
/// Positions one split-K dispatch covers at most.
pub const MATMUL_SGK_MAX_ROWS: usize = 8;
sweep_kernel!(KernelG2R4, "nvfp4_matvec_g2r4", 4);
sweep_kernel!(KernelG4R4, "nvfp4_matvec_g4r4", 4);
sweep_kernel!(KernelG1R2, "nvfp4_matvec_g1r2", 2);
sweep_kernel!(KernelG1R8, "nvfp4_matvec_g1r8", 8);
sweep_kernel!(KernelG2R2, "nvfp4_matvec_g2r2", 2);
sweep_kernel!(KernelG2R8, "nvfp4_matvec_g2r8", 8);

/// v2 geometry: 4 simdgroups per threadgroup (8 measured slower — it
/// halves the threadgroup count, the dispatch-geometry-mismatch class).
pub const V2_ROWS_PER_TG: u64 = 4;
pub const V2_THREADS_PER_TG: u64 = 128;

/// Marker for the v2 kernel-handle binding.
pub struct KernelV2;
impl crate::kernels::TiledKernel for KernelV2 {
    const KERNEL_NAME: &'static str = "nvfp4_matvec_v2";
    const ROWS_PER_TG: u64 = V2_ROWS_PER_TG;
    const THREADS_PER_TG: u64 = V2_THREADS_PER_TG;
}

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "nvfp4_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
