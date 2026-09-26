//! MSL source for the A-5 sweep, seg3 and VERIFY-N NVFP4 matvec/matmul arms.

/// The A-5 sweep arms: v1's scalar-LUT inner loop (the faster decode,
/// per the v2 falsification) at `G` groups per lane per step (bytes in
/// flight per lane: 8·G) and `R` rows per threadgroup. Lane `l` owns
/// groups `l·G .. l·G+G` then strides `32·G`, so a simdgroup still reads
/// `256·G` contiguous bytes per step. Summation order differs from v1
/// (different lane→group assignment), so parity is to fp32 rounding.
pub const SWEEP_SHADER: &str = r#"
#define NVFP4_SWEEP_KERNEL(NAME, G, R)                                            \
kernel void NAME(                                                                 \
    device const uchar*  Wp     [[buffer(0)]],                                    \
    device const uchar*  Ws     [[buffer(1)]],                                    \
    device const float*  X      [[buffer(2)]],                                    \
    device float*        out    [[buffer(3)]],                                    \
    constant uint&       M      [[buffer(4)]],                                    \
    constant uint&       K      [[buffer(5)]],                                    \
    constant float&      Tscale [[buffer(6)]],                                    \
    uint tg_id     [[threadgroup_position_in_grid]],                              \
    uint lane      [[thread_index_in_simdgroup]],                                 \
    uint sg_id     [[simdgroup_index_in_threadgroup]])                            \
{                                                                                 \
    uint row = tg_id * (R) + sg_id;                                               \
    if (row >= M) { return; }                                                     \
    const uint groups = K / NVFP4_GROUP_ELEMS;                                    \
    device const uchar* row_p = Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES; \
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;                  \
    float acc = 0.0f;                                                             \
    for (uint g0 = lane * (G); g0 < groups; g0 += 32u * (G)) {                    \
        for (uint j = 0u; j < (G); ++j) {                                         \
            const uint g = g0 + j;                                                \
            if (g >= groups) { break; }                                           \
            const float step = Tscale * nvfp4_e4m3(row_s[g]);                     \
            device const uchar* blk = row_p + (ulong)g * NVFP4_GROUP_BYTES;       \
            const uint base = g * NVFP4_GROUP_ELEMS;                              \
            float part = 0.0f;                                                    \
            for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {                       \
                const uchar byte = blk[b];                                        \
                part += NVFP4_LUT[byte & 0x0Fu]         * X[base + 2u * b];       \
                part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * X[base + 2u * b + 1u];  \
            }                                                                     \
            acc += step * part;                                                   \
        }                                                                         \
    }                                                                             \
    acc = simd_sum(acc);                                                          \
    if (lane == 0u) { out[row] = acc; }                                           \
}

// ── A-5a arms: rows per lane (X reuse) and LUT width ──────────────────
//
// The α/B fit says v1 is issue-bound at ~326 GB/s-equivalent regardless
// of geometry. Per 16-element group a v1 lane issues 16 X loads, 8 byte
// loads, 16 nibble LUT loads and 16 FMAs. Two levers on the instruction
// stream: (1) `RL` rows per lane share one set of X loads; (2) a
// byte-indexed `float2` table decodes both nibbles in one load.
// Same per-row element order as v1 → parity to fp32 rounding.
//
// Byte → (lo nibble value, hi nibble value), 256 entries, constant
// address space. Built from the same E2M1 grid as NVFP4_LUT.
constant float2 NVFP4_BYTE_LUT[256] = {
#define NVFP4_ROW(hi) \
    float2(0.0f,hi), float2(0.5f,hi), float2(1.0f,hi), float2(1.5f,hi), \
    float2(2.0f,hi), float2(3.0f,hi), float2(4.0f,hi), float2(6.0f,hi), \
    float2(-0.0f,hi), float2(-0.5f,hi), float2(-1.0f,hi), float2(-1.5f,hi), \
    float2(-2.0f,hi), float2(-3.0f,hi), float2(-4.0f,hi), float2(-6.0f,hi),
    NVFP4_ROW(0.0f) NVFP4_ROW(0.5f) NVFP4_ROW(1.0f) NVFP4_ROW(1.5f)
    NVFP4_ROW(2.0f) NVFP4_ROW(3.0f) NVFP4_ROW(4.0f) NVFP4_ROW(6.0f)
    NVFP4_ROW(-0.0f) NVFP4_ROW(-0.5f) NVFP4_ROW(-1.0f) NVFP4_ROW(-1.5f)
    NVFP4_ROW(-2.0f) NVFP4_ROW(-3.0f) NVFP4_ROW(-4.0f) NVFP4_ROW(-6.0f)
#undef NVFP4_ROW
};

// One group of one row: 16 elements against xv[0..16].
// BYTE_LUT=0: v1's two nibble lookups per byte; 1: one float2 lookup.
#define NVFP4_GROUP_DOT(part, blk, xv, BYTE_LUT)                                  \
    for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {                               \
        const uchar byte = blk[b];                                                \
        if (BYTE_LUT) {                                                           \
            const float2 w2 = NVFP4_BYTE_LUT[byte];                               \
            part += w2.x * xv[2u * b];                                            \
            part += w2.y * xv[2u * b + 1u];                                       \
        } else {                                                                  \
            part += NVFP4_LUT[byte & 0x0Fu]         * xv[2u * b];                 \
            part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * xv[2u * b + 1u];            \
        }                                                                         \
    }

// A-5b rung 2d — MEASURED SLOWER IN BOTH FORMS (2026-08-19), retained as
// arms, not wired: form A (below) α +4.4 µs and B −18% (the per-group
// weight loads in the hot loop); form B (`nvfp4_matvec_x2m`, staged in
// threadgroup memory) 117–173 GB/s — the K-sized threadgroup allocation
// collapses occupancy. A separate single-threadgroup norm dispatch is the
// cheaper structure; the ledger's ~11 µs per norm was mostly sampling
// drain (tiny stages read ~5–7 µs high under stage-boundary counters).
//
// The idea: the pre-norm folded into the GEMV. Every threadgroup
// recomputes inv_rms(X) in its prologue (K floats from cache — trivial
// against the weight stream) and applies `(Wn[i] + off) * inv` while
// loading X, so the separate single-threadgroup RMS-norm dispatch — a
// serialised ~11–16 µs latency chain, measured — disappears. The per-
// element expression is the norm kernel's (`x * (w + off) * rms`); only
// the reduction order of the sum of squares differs (128 threads, not
// 1024), so parity is to fp32 rounding, not bit.
inline float nvfp4_prenorm_inv(device const float* X, uint K, float eps,
                               uint tid, uint tg_sz, uint lane, uint sg_id,
                               threadgroup float* tg_p) {
    float partial = 0.0f;
    for (uint i = tid; i < K; i += tg_sz) { partial += X[i] * X[i]; }
    const float sg = simd_sum(partial);
    if (lane == 0u) { tg_p[sg_id] = sg; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = 0.0f;
    const uint n_sg = (tg_sz + 31u) / 32u;
    for (uint i = 0u; i < n_sg; ++i) { sum_sq += tg_p[i]; }
    return 1.0f / sqrt(sum_sq / float(K) + eps);
}

// RES=1 adds buffer(7) R and writes out[row] = R[row] + acc — the residual
// add folded into the GEMV (A-5b rung 2a), the same fp32 add as the
// residual kernel, so bit-identical to x2-then-add.
#define NVFP4_MULTIROW_KERNEL(NAME, RL, SG, BYTE_LUT) \
    NVFP4_MULTIROW_KERNEL_RN(NAME, RL, SG, BYTE_LUT, 0, 0)
#define NVFP4_MULTIROW_KERNEL_R(NAME, RL, SG, BYTE_LUT, RES) \
    NVFP4_MULTIROW_KERNEL_RN(NAME, RL, SG, BYTE_LUT, RES, 0)
// PRENORM=1 adds buffer(8) Wn, (9) eps, (10) off: X is normalised on load.
#define NVFP4_MULTIROW_KERNEL_RN(NAME, RL, SG, BYTE_LUT, RES, PRENORM)            \
kernel void NAME(                                                                 \
    device const uchar*  Wp     [[buffer(0)]],                                    \
    device const uchar*  Ws     [[buffer(1)]],                                    \
    device const float*  X      [[buffer(2)]],                                    \
    device float*        out    [[buffer(3)]],                                    \
    constant uint&       M      [[buffer(4)]],                                    \
    constant uint&       K      [[buffer(5)]],                                    \
    constant float&      Tscale [[buffer(6)]],                                    \
    device const float*  R      [[buffer(7)]],                                    \
    device const float*  Wn     [[buffer(8)]],                                    \
    constant float&      Neps   [[buffer(9)]],                                    \
    constant float&      Noff   [[buffer(10)]],                                   \
    uint tg_id     [[threadgroup_position_in_grid]],                              \
    uint tid       [[thread_index_in_threadgroup]],                               \
    uint tg_sz     [[threads_per_threadgroup]],                                   \
    uint lane      [[thread_index_in_simdgroup]],                                 \
    uint sg_id     [[simdgroup_index_in_threadgroup]])                            \
{                                                                                 \
    threadgroup float tg_p[32];                                                   \
    float inv = 1.0f;                                                             \
    if (PRENORM) {                                                                \
        /* before the row guard: every thread joins the barrier */               \
        inv = nvfp4_prenorm_inv(X, K, Neps, tid, tg_sz, lane, sg_id, tg_p);      \
    }                                                                             \
    const uint row0 = (tg_id * (SG) + sg_id) * (RL);                              \
    if (row0 >= M) { return; }                                                    \
    const uint groups = K / NVFP4_GROUP_ELEMS;                                    \
    float acc[RL];                                                                \
    for (uint r = 0u; r < (RL); ++r) { acc[r] = 0.0f; }                           \
    for (uint g = lane; g < groups; g += 32u) {                                   \
        const uint base = g * NVFP4_GROUP_ELEMS;                                  \
        float xv[NVFP4_GROUP_ELEMS];                                              \
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) {                           \
            xv[i] = (PRENORM) ? X[base + i] * (Wn[base + i] + Noff) * inv        \
                              : X[base + i];                                      \
        }                                                                         \
        for (uint r = 0u; r < (RL); ++r) {                                        \
            const uint row = row0 + r;                                            \
            if (row >= M) { break; }                                              \
            const ulong rg = (ulong)row * (ulong)groups + (ulong)g;               \
            const float step = Tscale * nvfp4_e4m3(Ws[rg]);                       \
            device const uchar* blk = Wp + rg * NVFP4_GROUP_BYTES;                \
            float part = 0.0f;                                                    \
            NVFP4_GROUP_DOT(part, blk, xv, BYTE_LUT)                              \
            acc[r] += step * part;                                                \
        }                                                                         \
    }                                                                             \
    for (uint r = 0u; r < (RL); ++r) {                                            \
        const float total = simd_sum(acc[r]);                                     \
        if (lane == 0u && row0 + r < M) {                                         \
            out[row0 + r] = (RES) ? (R[row0 + r] + total) : total;                \
        }                                                                         \
    }                                                                             \
}

NVFP4_MULTIROW_KERNEL_R(nvfp4_matvec_x2r, 2u, 4u, 0, 1)
NVFP4_MULTIROW_KERNEL_RN(nvfp4_matvec_x2n, 2u, 4u, 0, 0, 1)

// Rung 2d, form B: the normalised X is staged ONCE per threadgroup in
// threadgroup memory (`Xs`, K floats, bound dynamically — K ≤ 8160 fits
// the 32 KB limit beside the reduction scratch) and the hot loop reads
// it with no per-group weight loads. Per element the norm expression is
// the norm kernel's; the sum-of-squares order differs (128 threads).
kernel void nvfp4_matvec_x2m(
    device const uchar*  Wp     [[buffer(0)]],
    device const uchar*  Ws     [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],
    device const float*  Wn     [[buffer(8)]],
    constant float&      Neps   [[buffer(9)]],
    constant float&      Noff   [[buffer(10)]],
    threadgroup float*   Xs     [[threadgroup(0)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint tg_sz     [[threads_per_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    threadgroup float tg_p[32];
    const float inv = nvfp4_prenorm_inv(X, K, Neps, tid, tg_sz, lane, sg_id, tg_p);
    for (uint i = tid; i < K; i += tg_sz) {
        Xs[i] = X[i] * (Wn[i] + Noff) * inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tg_id * 4u + sg_id) * 2u;
    if (row0 >= M) { return; }
    const uint groups = K / NVFP4_GROUP_ELEMS;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const bool has1 = row0 + 1u < M;
    device const uchar* rp0 = Wp + (ulong)row0 * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs0 = Ws + (ulong)row0 * (ulong)groups;
    device const uchar* rp1 = rp0 + (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs1 = rs0 + groups;
    for (uint g = lane; g < groups; g += 32u) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        float xv[NVFP4_GROUP_ELEMS];
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { xv[i] = Xs[base + i]; }
        {
            const float step = Tscale * nvfp4_e4m3(rs0[g]);
            device const uchar* blk = rp0 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc0 += step * part;
        }
        if (has1) {
            const float step = Tscale * nvfp4_e4m3(rs1[g]);
            device const uchar* blk = rp1 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc1 += step * part;
        }
    }
    const float t0 = simd_sum(acc0);
    const float t1 = simd_sum(acc1);
    if (lane == 0u) {
        out[row0] = t0;
        if (has1) { out[row0 + 1u] = t1; }
    }
}
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x2,   2u, 4u, 0)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x4,   4u, 4u, 0)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x1b,  1u, 4u, 1)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x2b,  2u, 4u, 1)
NVFP4_MULTIROW_KERNEL(nvfp4_matvec_x4b,  4u, 4u, 1)

// ── A-5b: segmented x2 — up to three matrices sharing one X, one dispatch.
//
// α (the fixed per-dispatch term, ~6 µs for x2) is paid once for Q, K
// and V — or gate and up — instead of once each. Rows are numbered
// across the segments in order; every row resolves its own segment, so
// the two rows a lane owns may straddle a boundary. Per row the body is
// x2's exactly (same order, same scale fold) → bit-identical to x2.
// A segment with M = 0 is absent (gate+up uses two).
kernel void nvfp4_matvec_x2_seg3(
    device const uchar*  Wp0    [[buffer(0)]],
    device const uchar*  Ws0    [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out0   [[buffer(3)]],
    constant uint&       M0     [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Ts0    [[buffer(6)]],
    device const uchar*  Wp1    [[buffer(7)]],
    device const uchar*  Ws1    [[buffer(8)]],
    device float*        out1   [[buffer(9)]],
    constant uint&       M1     [[buffer(10)]],
    constant float&      Ts1    [[buffer(11)]],
    device const uchar*  Wp2    [[buffer(12)]],
    device const uchar*  Ws2    [[buffer(13)]],
    device float*        out2   [[buffer(14)]],
    constant uint&       M2     [[buffer(15)]],
    constant float&      Ts2    [[buffer(16)]],
    // A-5b rung 2a: optional residual folded into the write —
    // out[r] = R[r] + acc, the same fp32 add the residual kernel did,
    // so the fused form is bit-identical. Applies to segment 0 only
    // (a single-matrix dispatch such as o-proj or down).
    device const float*  R      [[buffer(17)]],
    constant uint&       has_R  [[buffer(18)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    const uint M = M0 + M1 + M2;
    const uint row0 = (tg_id * 4u + sg_id) * 2u;
    if (row0 >= M) { return; }
    const uint groups = K / NVFP4_GROUP_ELEMS;
    // Per-row segment resolution into SCALARS: a dynamically indexed
    // local array here costs ~3.7 us of fixed prologue per dispatch and
    // drags the long-K shapes (measured: seg1 α 8.9 us vs x2's 5.2, down
    // [2560,8192] 247 vs 315 GB/s) — it forces the pointers to memory.
    const uint rowA = row0;
    const uint rowB = row0 + 1u;
    const bool hasB = rowB < M;
    device const uchar* wpA; device const uchar* wsA; device float* opA; float tsA; uint lrA;
    device const uchar* wpB; device const uchar* wsB; device float* opB; float tsB; uint lrB;
    if (rowA < M0) { wpA = Wp0; wsA = Ws0; opA = out0; tsA = Ts0; lrA = rowA; }
    else if (rowA < M0 + M1) { wpA = Wp1; wsA = Ws1; opA = out1; tsA = Ts1; lrA = rowA - M0; }
    else { wpA = Wp2; wsA = Ws2; opA = out2; tsA = Ts2; lrA = rowA - M0 - M1; }
    if (rowB < M0) { wpB = Wp0; wsB = Ws0; opB = out0; tsB = Ts0; lrB = rowB; }
    else if (rowB < M0 + M1) { wpB = Wp1; wsB = Ws1; opB = out1; tsB = Ts1; lrB = rowB - M0; }
    else { wpB = Wp2; wsB = Ws2; opB = out2; tsB = Ts2; lrB = rowB - M0 - M1; }
    // Row-major bases, so the loop indexes by group only.
    device const uchar* rowpA = wpA + (ulong)lrA * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rowsA = wsA + (ulong)lrA * (ulong)groups;
    device const uchar* rowpB = wpB + (ulong)lrB * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rowsB = wsB + (ulong)lrB * (ulong)groups;
    float accA = 0.0f;
    float accB = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        float xv[NVFP4_GROUP_ELEMS];
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { xv[i] = X[base + i]; }
        {
            const float step = tsA * nvfp4_e4m3(rowsA[g]);
            device const uchar* blk = rowpA + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            accA += step * part;
        }
        if (hasB) {
            const float step = tsB * nvfp4_e4m3(rowsB[g]);
            device const uchar* blk = rowpB + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            accB += step * part;
        }
    }
    const float totalA = simd_sum(accA);
    const float totalB = simd_sum(accB);
    if (lane == 0u) {
        const bool resA = (has_R != 0u) && (rowA < M0);
        opA[lrA] = resA ? (R[lrA] + totalA) : totalA;
        if (hasB) {
            const bool resB = (has_R != 0u) && (rowB < M0);
            opB[lrB] = resB ? (R[lrB] + totalB) : totalB;
        }
    }
}

NVFP4_SWEEP_KERNEL(nvfp4_matvec_g2r4, 2u, 4u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g4r4, 4u, 4u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g1r2, 1u, 2u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g1r8, 1u, 8u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g2r2, 2u, 2u)
NVFP4_SWEEP_KERNEL(nvfp4_matvec_g2r8, 2u, 8u)


// ── seg3t: per-THREADGROUP segment resolution ──────────────────────────
//
// The row-pair resolve above prices at ~4.8 µs per dispatch on the
// gpt-oss QKV shape (238 vs a resolve-free 276 GB/s,
// `examples/qkv_seg3_probe.rs`): every simdgroup pays the 3-way branch
// chain and carries the resolved pointers in registers. Here the grid is
// tiled so each threadgroup lies wholly inside ONE segment —
// `TILE_END[s]` are prefix sums of ceil(M_s / 8) — and the resolve is
// two uniform compares. Per-row walk unchanged → bit-identical.
kernel void nvfp4_matvec_x2_seg3t(
    device const uchar*  Wp0    [[buffer(0)]],
    device const uchar*  Ws0    [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out0   [[buffer(3)]],
    constant uint&       M0     [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Ts0    [[buffer(6)]],
    device const uchar*  Wp1    [[buffer(7)]],
    device const uchar*  Ws1    [[buffer(8)]],
    device float*        out1   [[buffer(9)]],
    constant uint&       M1     [[buffer(10)]],
    constant float&      Ts1    [[buffer(11)]],
    device const uchar*  Wp2    [[buffer(12)]],
    device const uchar*  Ws2    [[buffer(13)]],
    device float*        out2   [[buffer(14)]],
    constant uint&       M2     [[buffer(15)]],
    constant float&      Ts2    [[buffer(16)]],
    device const float*  R      [[buffer(17)]],
    constant uint&       has_R  [[buffer(18)]],
    constant uint3&      TILE_END [[buffer(19)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    // Uniform per-TG segment pick: two compares, no divergence.
    device const uchar* wp;
    device const uchar* ws;
    device float*       op;
    float ts;
    uint  m;
    uint  tile0;
    bool  res;
    if (tg_id < TILE_END.x) {
        wp = Wp0; ws = Ws0; op = out0; ts = Ts0; m = M0; tile0 = 0u;
        res = has_R != 0u;
    } else if (tg_id < TILE_END.y) {
        wp = Wp1; ws = Ws1; op = out1; ts = Ts1; m = M1; tile0 = TILE_END.x;
        res = false;
    } else {
        wp = Wp2; ws = Ws2; op = out2; ts = Ts2; m = M2; tile0 = TILE_END.y;
        res = false;
    }
    const uint row0 = ((tg_id - tile0) * 4u + sg_id) * 2u;
    if (row0 >= m) { return; }
    const bool has1 = row0 + 1u < m;

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uchar* rp0 = wp + (ulong)row0 * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs0 = ws + (ulong)row0 * (ulong)groups;
    device const uchar* rp1 = rp0 + (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* rs1 = rs0 + groups;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        float xv[NVFP4_GROUP_ELEMS];
        for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { xv[i] = X[base + i]; }
        {
            const float step = ts * nvfp4_e4m3(rs0[g]);
            device const uchar* blk = rp0 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc0 += step * part;
        }
        if (has1) {
            const float step = ts * nvfp4_e4m3(rs1[g]);
            device const uchar* blk = rp1 + (ulong)g * NVFP4_GROUP_BYTES;
            float part = 0.0f;
            NVFP4_GROUP_DOT(part, blk, xv, 0)
            acc1 += step * part;
        }
    }
    const float t0 = simd_sum(acc0);
    const float t1 = simd_sum(acc1);
    if (lane == 0u) {
        op[row0] = res ? (R[row0] + t0) : t0;
        if (has1) { op[row0 + 1u] = res ? (R[row0 + 1u] + t1) : t1; }
    }
}

// ── VERIFY-N: weight-stationary multi-RHS (RL rows/lane, NR activations) ─
//
// Speculative verification evaluates NR positions against the same
// weights. Each lane decodes one group of each of its RL rows ONCE and
// dots it against NR activation rows, so the weight stream is read once
// for NR outputs instead of NR times. X is read as float4: past NR=2 the
// scalar form is bound on X-load issue (16 loads per group per RHS), and
// RL > 1 shares each X load across rows. Per row and per RHS the fold is
// x2's `acc += step * part`; the within-group sum is float4-grouped, so
// parity with x2 is to fp32 rounding.
//
// X is [NR, K] row-major; out is [NR, M] row-major (column n of the
// product at out[n * M + row]). 4 simdgroups per threadgroup.
#define NVFP4_MULTIRHS_KERNEL(NAME, RL, NR)                                       \
kernel void NAME(                                                                 \
    device const uchar*  Wp     [[buffer(0)]],                                    \
    device const uchar*  Ws     [[buffer(1)]],                                    \
    device const float*  X      [[buffer(2)]],                                    \
    device float*        out    [[buffer(3)]],                                    \
    constant uint&       M      [[buffer(4)]],                                    \
    constant uint&       K      [[buffer(5)]],                                    \
    constant float&      Tscale [[buffer(6)]],                                    \
    uint tg_id     [[threadgroup_position_in_grid]],                              \
    uint lane      [[thread_index_in_simdgroup]],                                 \
    uint sg_id     [[simdgroup_index_in_threadgroup]])                            \
{                                                                                 \
    const uint row0 = (tg_id * 4u + sg_id) * (RL);                                \
    if (row0 >= M) { return; }                                                    \
    const uint groups = K / NVFP4_GROUP_ELEMS;                                    \
    float acc[RL][NR];                                                            \
    for (uint r = 0u; r < (RL); ++r)                                              \
        for (uint n = 0u; n < (NR); ++n) { acc[r][n] = 0.0f; }                    \
    for (uint g = lane; g < groups; g += 32u) {                                   \
        const uint base = g * NVFP4_GROUP_ELEMS;                                  \
        float4 w[RL][4];                                                          \
        float step[RL];                                                           \
        for (uint r = 0u; r < (RL); ++r) {                                        \
            const uint row = row0 + r;                                            \
            if (row < M) {                                                        \
                const ulong rg = (ulong)row * (ulong)groups + (ulong)g;           \
                step[r] = Tscale * nvfp4_e4m3(Ws[rg]);                            \
                device const uchar* blk = Wp + rg * NVFP4_GROUP_BYTES;            \
                for (uint q = 0u; q < 4u; ++q) {                                  \
                    const uchar b0 = blk[2u * q];                                 \
                    const uchar b1 = blk[2u * q + 1u];                            \
                    w[r][q] = float4(NVFP4_LUT[b0 & 0x0Fu],                       \
                                     NVFP4_LUT[(b0 >> 4u) & 0x0Fu],               \
                                     NVFP4_LUT[b1 & 0x0Fu],                       \
                                     NVFP4_LUT[(b1 >> 4u) & 0x0Fu]);              \
                }                                                                 \
            } else {                                                              \
                step[r] = 0.0f;                                                   \
                for (uint q = 0u; q < 4u; ++q) { w[r][q] = float4(0.0f); }        \
            }                                                                     \
        }                                                                         \
        for (uint n = 0u; n < (NR); ++n) {                                        \
            device const float4* xp =                                             \
                (device const float4*)(X + (ulong)n * (ulong)K + base);           \
            const float4 x0 = xp[0];                                              \
            const float4 x1 = xp[1];                                              \
            const float4 x2 = xp[2];                                              \
            const float4 x3 = xp[3];                                              \
            for (uint r = 0u; r < (RL); ++r) {                                    \
                const float part = dot(w[r][0], x0) + dot(w[r][1], x1)            \
                                 + dot(w[r][2], x2) + dot(w[r][3], x3);           \
                acc[r][n] += step[r] * part;                                      \
            }                                                                     \
        }                                                                         \
    }                                                                             \
    for (uint n = 0u; n < (NR); ++n) {                                            \
        for (uint r = 0u; r < (RL); ++r) {                                        \
            const float t = simd_sum(acc[r][n]);                                  \
            if (lane == 0u && row0 + r < M) {                                     \
                out[(ulong)n * (ulong)M + row0 + r] = t;                          \
            }                                                                     \
        }                                                                         \
    }                                                                             \
}


// ── VERIFY-N sgk: split-K simdgroup-matrix tiles ──────────────────────────
//
// The multi-RHS kernels above re-read all R activation rows per 2 weight
// rows (X traffic ~2K bytes per output, 28x the NVFP4 weight bytes at
// R=8). Here a threadgroup owns NV_SGK_ROWS weight rows and splits K
// across NV_SGK_SG simdgroups (simdgroup s takes groups s, s+SG, ...), the
// shape that keeps a short-M matrix's walk parallel. Per 16-wide group
// step a simdgroup:
//   - stages X[0..8, g*16..+16] (rows >= R zero) in its private tile,
//   - dequantises its 32 rows' group g (lane = row) into a half tile,
//   - runs 2 k-substeps x 4 row-blocks of 8x8 MACs:
//        C[pos, row] += X[pos, k..k+8] . W[row, k..k+8]^T
// then all simdgroups' partial C sum through threadgroup memory and only
// the R valid position rows are written (out may be a KV-cache range).
//
// Contract: K % 16 == 0, 1 <= R <= 8. X [R, K], out [R, M], row-major.
// Parity to x2 is to fp32 rounding (the MAC's reduction order).
constant uint NV_SGK_SG = 8;
constant uint NV_SGK_ROWS = 32;
constant uint NV_SGK_BLOCKS = NV_SGK_ROWS / 8;

kernel void nvfp4_matmul_sgk(
    device const uchar*  Wp     [[buffer(0)]],
    device const uchar*  Ws     [[buffer(1)]],
    device const float*  X      [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],
    constant uint&       R      [[buffer(7)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    // Per simdgroup: a [32 rows, 16 k] weight tile and an [8, 16] X tile.
    // The weight tile is float: a half tile would round each dequantised
    // weight (step x E2M1) to 11 bits, a representation change the x2
    // path does not make.
    threadgroup float wt_all[NV_SGK_SG * NV_SGK_ROWS * NVFP4_GROUP_ELEMS];
    threadgroup float xt_all[NV_SGK_SG * 8 * NVFP4_GROUP_ELEMS];
    threadgroup float* wt = wt_all + sg_id * NV_SGK_ROWS * NVFP4_GROUP_ELEMS;
    threadgroup float* xt = xt_all + sg_id * 8 * NVFP4_GROUP_ELEMS;

    const uint row0 = tg_id * NV_SGK_ROWS;
    const uint groups = K / NVFP4_GROUP_ELEMS;
    const uint my_row = row0 + lane;
    simdgroup_float8x8 acc[NV_SGK_BLOCKS];
    for (uint b = 0u; b < NV_SGK_BLOCKS; ++b) {
        acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint g = sg_id; g < groups; g += NV_SGK_SG) {
        const uint base = g * NVFP4_GROUP_ELEMS;
        // X tile: 128 floats, 4 per lane.
        for (uint i = lane; i < 8u * NVFP4_GROUP_ELEMS; i += 32u) {
            const uint pos = i / NVFP4_GROUP_ELEMS;
            const uint kk = i % NVFP4_GROUP_ELEMS;
            xt[i] = pos < R ? X[(ulong)pos * K + base + kk] : 0.0f;
        }
        // Weight tile: lane = row, one 16-element group each.
        threadgroup float* dst = wt + lane * NVFP4_GROUP_ELEMS;
        if (my_row < M) {
            const ulong rg = (ulong)my_row * (ulong)groups + (ulong)g;
            const float step = Tscale * nvfp4_e4m3(Ws[rg]);
            device const uchar* blk = Wp + rg * NVFP4_GROUP_BYTES;
            for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {
                const uchar byte = blk[b];
                dst[2u * b]      = NVFP4_LUT[byte & 0x0Fu] * step;
                dst[2u * b + 1u] = NVFP4_LUT[(byte >> 4u) & 0x0Fu] * step;
            }
        } else {
            for (uint i = 0u; i < NVFP4_GROUP_ELEMS; ++i) { dst[i] = 0.0f; }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < NVFP4_GROUP_ELEMS; kk += 8u) {
            simdgroup_float8x8 a;
            simdgroup_load(a, xt + kk, NVFP4_GROUP_ELEMS);
            for (uint b = 0u; b < NV_SGK_BLOCKS; ++b) {
                simdgroup_float8x8 w;
                simdgroup_load(w, wt + b * 8u * NVFP4_GROUP_ELEMS + kk,
                               NVFP4_GROUP_ELEMS, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[b], a, w, acc[b]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Reduce the SG partials: reuse the weight tiles as [SG][8 pos][32 rows].
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* c = wt_all + sg_id * 8u * NV_SGK_ROWS;
    for (uint b = 0u; b < NV_SGK_BLOCKS; ++b) {
        simdgroup_store(acc[b], c + b * 8u, NV_SGK_ROWS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8u * NV_SGK_ROWS; i += NV_SGK_SG * 32u) {
        const uint pos = i / NV_SGK_ROWS;
        const uint row = row0 + i % NV_SGK_ROWS;
        if (pos < R && row < M) {
            float t = 0.0f;
            for (uint s = 0u; s < NV_SGK_SG; ++s) { t += wt_all[s * 8u * NV_SGK_ROWS + i]; }
            out[(ulong)pos * M + row] = t;
        }
    }
}


// ── VERIFY-N sgf: split-K, weights dequantised straight into fragments ──
//
// sgk's fixed cost is its weight tile's round trip through threadgroup
// memory (16 stores + reloads per lane per group), which made its time
// flat in R at ~2x one GEMV. Here the weight tile is the LEFT operand,
//   C[row, pos] += W[row, k..k+8] . X^T[k..k+8, pos]
// and each lane writes its own fragment elements directly. In the 8x8
// simdgroup-matrix layout a lane holds row fm, columns fn and fn+1 (fn
// even) — i.e. the lo and hi nibble of ONE NVFP4 byte — so a tile costs
// each lane one byte load and two LUT reads. The fragment is half: an
// E2M1 code times an E4M3 scale is exact in half (<= 5 significant bits,
// |v| <= 2688, >= 2^-10), and the f32 tensor scale is applied once at the
// output, so no weight is rounded. X keeps the masked staged tile.
//
// Same contract as sgk: K % 16 == 0, 1 <= R <= 8, R at buffer 7.
#define NVFP4_SGF_KERNEL(NAME, SGN, BLOCKS, CONTIG)                              \
kernel void NAME(                                                                 \
    device const uchar*  Wp     [[buffer(0)]],                                    \
    device const uchar*  Ws     [[buffer(1)]],                                    \
    device const float*  X      [[buffer(2)]],                                    \
    device float*        out    [[buffer(3)]],                                    \
    constant uint&       M      [[buffer(4)]],                                    \
    constant uint&       K      [[buffer(5)]],                                    \
    constant float&      Tscale [[buffer(6)]],                                    \
    constant uint&       R      [[buffer(7)]],                                    \
    uint tg_id     [[threadgroup_position_in_grid]],                              \
    uint tid       [[thread_index_in_threadgroup]],                               \
    uint lane      [[thread_index_in_simdgroup]],                                 \
    uint sg_id     [[simdgroup_index_in_threadgroup]])                            \
{                                                                                 \
    threadgroup float xt_all[(SGN) * 8 * NVFP4_GROUP_ELEMS];                      \
    threadgroup float cs[(SGN) * 8 * (BLOCKS) * 8];                               \
    threadgroup float* xt = xt_all + sg_id * 8 * NVFP4_GROUP_ELEMS;               \
    const uint ROWS = (BLOCKS) * 8u;                                              \
    const uint qid = lane / 4u;                                                   \
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);                              \
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;                           \
    const uint row0 = tg_id * ROWS;                                               \
    const uint groups = K / NVFP4_GROUP_ELEMS;                                    \
    const uint per = (groups + (SGN) - 1u) / (SGN);                               \
    const uint g_begin = (CONTIG) ? sg_id * per : sg_id;                          \
    const uint g_end = (CONTIG) ? min(groups, g_begin + per) : groups;            \
    const uint g_step = (CONTIG) ? 1u : (SGN);                                    \
    simdgroup_float8x8 acc[BLOCKS];                                               \
    for (uint b = 0u; b < (BLOCKS); ++b) {                                        \
        acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);                 \
    }                                                                             \
    for (uint g = g_begin; g < g_end; g += g_step) {                              \
        const uint base = g * NVFP4_GROUP_ELEMS;                                  \
        for (uint i = lane; i < 8u * NVFP4_GROUP_ELEMS; i += 32u) {               \
            const uint pos = i / NVFP4_GROUP_ELEMS;                               \
            xt[i] = pos < R ? X[(ulong)pos * K + base + i % NVFP4_GROUP_ELEMS]    \
                            : 0.0f;                                               \
        }                                                                         \
        float sc[BLOCKS];                                                         \
        uchar lo_byte[BLOCKS];                                                    \
        uchar hi_byte[BLOCKS];                                                    \
        for (uint b = 0u; b < (BLOCKS); ++b) {                                    \
            const uint row = row0 + b * 8u + fm;                                  \
            if (row < M) {                                                        \
                const ulong rg = (ulong)row * (ulong)groups + (ulong)g;           \
                sc[b] = nvfp4_e4m3(Ws[rg]);                                       \
                device const uchar* blk = Wp + rg * NVFP4_GROUP_BYTES;            \
                lo_byte[b] = blk[fn / 2u];                                        \
                hi_byte[b] = blk[4u + fn / 2u];                                   \
            } else {                                                              \
                sc[b] = 0.0f; lo_byte[b] = 0u; hi_byte[b] = 0u;                   \
            }                                                                     \
        }                                                                         \
        simdgroup_barrier(mem_flags::mem_threadgroup);                            \
        for (uint sub = 0u; sub < 2u; ++sub) {                                    \
            simdgroup_float8x8 xb;                                                \
            simdgroup_load(xb, xt + sub * 8u, NVFP4_GROUP_ELEMS, ulong2(0, 0), true); \
            for (uint b = 0u; b < (BLOCKS); ++b) {                                \
                const uchar byte = sub == 0u ? lo_byte[b] : hi_byte[b];           \
                simdgroup_half8x8 w;                                              \
                w.thread_elements()[0] = half(NVFP4_LUT[byte & 0x0Fu] * sc[b]);   \
                w.thread_elements()[1] = half(NVFP4_LUT[(byte >> 4u) & 0x0Fu] * sc[b]); \
                simdgroup_multiply_accumulate(acc[b], w, xb, acc[b]);             \
            }                                                                     \
        }                                                                         \
        simdgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                             \
    threadgroup float* c = cs + sg_id * 8u * ROWS;                                \
    for (uint b = 0u; b < (BLOCKS); ++b) {                                        \
        simdgroup_store(acc[b], c + b * 8u, ROWS, ulong2(0, 0), true);            \
    }                                                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    for (uint i = tid; i < 8u * ROWS; i += (SGN) * 32u) {                         \
        const uint pos = i / ROWS;                                                \
        const uint row = row0 + i % ROWS;                                         \
        if (pos < R && row < M) {                                                 \
            float t = 0.0f;                                                       \
            for (uint s = 0u; s < (SGN); ++s) { t += cs[s * 8u * ROWS + i]; }     \
            out[(ulong)pos * M + row] = Tscale * t;                               \
        }                                                                         \
    }                                                                             \
}

NVFP4_SGF_KERNEL(nvfp4_matmul_sgf, 8u, 4u, 0)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s4b4, 4u, 4u, 0)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s8b8, 8u, 8u, 0)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s4b8, 4u, 8u, 0)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s8b4c, 8u, 4u, 1)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s2b8, 2u, 8u, 0)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s1b4, 1u, 4u, 0)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s8b2c, 8u, 2u, 1)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s8b1c, 8u, 1u, 1)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s16b4c, 16u, 4u, 1)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s16b2c, 16u, 2u, 1)
NVFP4_SGF_KERNEL(nvfp4_matmul_sgf_s32b1c, 32u, 1u, 1)

NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x2_r1, 2u, 1u)
NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x2_r2, 2u, 2u)
NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x2_r4, 2u, 4u)
NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x2_r8, 2u, 8u)
NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x4_r4, 4u, 4u)
NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x4_r8, 4u, 8u)
NVFP4_MULTIRHS_KERNEL(nvfp4_matmul_x1_r8, 1u, 8u)

"#;
