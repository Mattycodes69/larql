//! Binary wire format for `POST /v1/experts/multi-layer-batch`.
//!
//! Collapses 30 per-layer HTTP requests into one per shard, eliminating the
//! per-request HTTPS overhead (~20 ms × 30 = 600 ms in the predispatch path).
//! The server runs tasks in parallel (rayon `par_iter` over tasks, nested
//! with per-expert parallelism, all on the one global rayon pool); the
//! client additionally parallelises across shards.
//!
//! Request layout (little-endian):
//!   u32  num_tasks
//!   for each task:
//!     u32  layer
//!     u32  hidden            (residual length = h_post_attn size)
//!     u32  num_experts
//!     f32[hidden]  residual
//!     u32[n]       expert_ids
//!     f32[n]       weights
//!
//! Response layout:
//!   u32  num_results
//!   for each result:
//!     u32  layer
//!     u32  hidden
//!     f32[hidden]  h2         (raw weighted sum; caller applies post-experts norm)

pub const MULTI_LAYER_BATCH_CONTENT_TYPE: &str = "application/x-larql-experts-multi-layer";

/// HTTP path served by the multi-layer batch endpoint.
pub const MULTI_LAYER_BATCH_PATH: &str = "/v1/experts/multi-layer-batch";

/// Q8K-prenormed variant: client sends `h_norm` pre-quantised to Q8_K
/// (already computed during routing — zero extra client compute).  Server
/// skips `pre_experts_norm` + `quantize_h_norm_for_q4k` and calls the
/// matvec directly.  4× smaller upload than the f32 residual path.
///
/// Request layout — same header as f32, but residual field replaced:
///   u32  num_tasks
///   for each task:
///     u32  layer
///     u32  hidden              (= n_blocks × 256)
///     u32  num_experts
///     i8[hidden]  q8k_qs       (quantised activation)
///     f32[n_blocks]  q8k_d     (per-super-block scales)
///     i16[n_blocks × 8]  q8k_sums  (precomputed sub-block sums)
///     u32[num_experts]  expert_ids
///     f32[num_experts]  weights
pub const MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE: &str = "application/x-larql-experts-multi-layer-q8k";

/// HTTP path served by the Q8K-prenormed multi-layer batch endpoint.
pub const MULTI_LAYER_BATCH_Q8K_PATH: &str = "/v1/experts/multi-layer-batch-q8k";

pub struct MultiLayerTask {
    pub layer: usize,
    pub residual: Vec<f32>,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Q8K-prenormed task: carries already-quantised h_norm so the server skips
/// normalisation and directly calls `q4k_q8k_matvec_into`.
pub struct MultiLayerTaskQ8K {
    pub layer: usize,
    pub hidden: usize,
    /// Flat i8 activation: `qs[block * 256 .. (block+1) * 256]` per block.
    pub qs: Vec<i8>,
    /// Per-super-block f32 scale: `d[block]`.
    pub d: Vec<f32>,
    /// Per-sub-block i16 sums: `sums[block * 8 + sb]`.
    pub sums: Vec<i16>,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

pub struct MultiLayerResult {
    pub layer: usize,
    pub h2: Vec<f32>,
}

pub fn encode_multi_layer_request(tasks: &[MultiLayerTask]) -> Vec<u8> {
    let cap = 4 + tasks
        .iter()
        .map(|t| 12 + t.residual.len() * 4 + t.expert_ids.len() * 8)
        .sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, tasks.len() as u32);
    for t in tasks {
        push_u32(&mut buf, t.layer as u32);
        push_u32(&mut buf, t.residual.len() as u32);
        push_u32(&mut buf, t.expert_ids.len() as u32);
        push_f32_slice(&mut buf, &t.residual);
        push_u32_slice(&mut buf, &t.expert_ids);
        push_f32_slice(&mut buf, &t.weights);
    }
    buf
}

pub fn decode_multi_layer_request(bytes: &[u8]) -> Option<Vec<MultiLayerTask>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    // Bound against the minimal 12-byte-per-task header (layer + hidden +
    // num_experts) before reserving — an attacker-controlled `n` must not
    // reach `Vec::with_capacity` directly. Mirrors q8k_wire.rs's
    // `max_possible_entries` guard (PR 104 CI).
    if n > bytes.len().saturating_sub(pos) / 12 {
        return None;
    }
    let mut tasks = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let ne = read_u32(bytes, &mut pos)? as usize;
        let residual = read_f32_slice(bytes, &mut pos, hidden)?;
        // Each expert entry needs 4 bytes now (id) + 4 bytes later (weight).
        if ne > bytes.len().saturating_sub(pos) / 8 {
            return None;
        }
        let mut expert_ids = Vec::with_capacity(ne);
        for _ in 0..ne {
            expert_ids.push(read_u32(bytes, &mut pos)?);
        }
        let mut weights = Vec::with_capacity(ne);
        for _ in 0..ne {
            weights.push(read_f32(bytes, &mut pos)?);
        }
        tasks.push(MultiLayerTask {
            layer,
            residual,
            expert_ids,
            weights,
        });
    }
    Some(tasks)
}

pub fn encode_multi_layer_response(results: &[MultiLayerResult]) -> Vec<u8> {
    let cap = 4 + results.iter().map(|r| 8 + r.h2.len() * 4).sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, results.len() as u32);
    for r in results {
        push_u32(&mut buf, r.layer as u32);
        push_u32(&mut buf, r.h2.len() as u32);
        push_f32_slice(&mut buf, &r.h2);
    }
    buf
}

pub fn decode_multi_layer_response(bytes: &[u8]) -> Option<Vec<MultiLayerResult>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    // Each result needs at least 8 bytes (layer + hidden) before its
    // payload; guard the allocation against an attacker-controlled `n`
    // (a malicious shard can send this response too — see
    // docs/audits/dec-readiness-review-2026-07-22.md §2a).
    if n > bytes.len().saturating_sub(pos) / 8 {
        return None;
    }
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let h2 = read_f32_slice(bytes, &mut pos, hidden)?;
        results.push(MultiLayerResult { layer, h2 });
    }
    Some(results)
}

// ── Q8K-prenormed wire ────────────────────────────────────────────────────────

use crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS as ELEMS_PER_Q8K_BLOCK;
const SUMS_PER_Q8K_BLOCK: usize = 8;

pub fn encode_multi_layer_request_q8k(tasks: &[MultiLayerTaskQ8K]) -> Vec<u8> {
    let cap = 4 + tasks
        .iter()
        .map(|t| {
            let nb = t.hidden / ELEMS_PER_Q8K_BLOCK;
            12 // layer + hidden + num_experts
            + t.hidden  // qs (i8)
            + nb * 4    // d (f32)
            + nb * SUMS_PER_Q8K_BLOCK * 2  // sums (i16)
            + t.expert_ids.len() * 8 // expert_ids + weights
        })
        .sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, tasks.len() as u32);
    for t in tasks {
        let nb = t.hidden / ELEMS_PER_Q8K_BLOCK;
        push_u32(&mut buf, t.layer as u32);
        push_u32(&mut buf, t.hidden as u32);
        push_u32(&mut buf, t.expert_ids.len() as u32);
        // Q8K activation
        push_i8_slice(&mut buf, &t.qs);
        push_f32_slice(&mut buf, &t.d);
        push_i16_slice(&mut buf, &t.sums);
        debug_assert_eq!(t.qs.len(), t.hidden, "qs length mismatch");
        debug_assert_eq!(t.d.len(), nb, "d length mismatch");
        debug_assert_eq!(
            t.sums.len(),
            nb * SUMS_PER_Q8K_BLOCK,
            "sums length mismatch"
        );
        // Expert routing
        push_u32_slice(&mut buf, &t.expert_ids);
        push_f32_slice(&mut buf, &t.weights);
    }
    buf
}

pub fn decode_multi_layer_request_q8k(bytes: &[u8]) -> Option<Vec<MultiLayerTaskQ8K>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    // Bound against the minimal 12-byte-per-task header before reserving
    // — same rationale as `decode_multi_layer_request` above.
    if n > bytes.len().saturating_sub(pos) / 12 {
        return None;
    }
    let mut tasks = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let ne = read_u32(bytes, &mut pos)? as usize;
        // Q8K activations quantise in 256-element super-blocks; a `hidden`
        // that is not block-aligned would silently floor to `nb` blocks and
        // desync every subsequent field offset (corrupt decode, not an
        // error). Reject it here so the handler surfaces a 400.
        if !hidden.is_multiple_of(ELEMS_PER_Q8K_BLOCK) {
            return None;
        }
        let nb = hidden / ELEMS_PER_Q8K_BLOCK;
        // Q8K activation
        let qs = read_i8_slice(bytes, &mut pos, hidden)?;
        let d = read_f32_slice(bytes, &mut pos, nb)?;
        let sums = read_i16_slice(bytes, &mut pos, nb * SUMS_PER_Q8K_BLOCK)?;
        // Expert routing
        if ne > bytes.len().saturating_sub(pos) / 8 {
            return None;
        }
        let mut expert_ids = Vec::with_capacity(ne);
        for _ in 0..ne {
            expert_ids.push(read_u32(bytes, &mut pos)?);
        }
        let mut weights = Vec::with_capacity(ne);
        for _ in 0..ne {
            weights.push(read_f32(bytes, &mut pos)?);
        }
        tasks.push(MultiLayerTaskQ8K {
            layer,
            hidden,
            qs,
            d,
            sums,
            expert_ids,
            weights,
        });
    }
    Some(tasks)
}

fn read_i8_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<i8>> {
    let end = pos.checked_add(n)?;
    if end > bytes.len() {
        return None;
    }
    // i8 and u8 share size, alignment (1) and have no invalid bit patterns,
    // so reinterpreting the byte slab is sound on every target — one bulk
    // memcpy instead of a per-element loop.
    let src: &[i8] =
        unsafe { std::slice::from_raw_parts(bytes[*pos..end].as_ptr().cast::<i8>(), n) };
    let v = src.to_vec();
    *pos = end;
    Some(v)
}

fn read_i16_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<i16>> {
    // `n` (e.g. `nb * SUMS_PER_Q8K_BLOCK`, derived from a wire `hidden`)
    // must not reach `Vec::with_capacity` unbounded — see PR 104 CI.
    // Single length check up front, then a bulk endian-correct copy.
    if n > bytes.len().saturating_sub(*pos) / 2 {
        return None;
    }
    let end = *pos + n * 2;
    let mut v = Vec::with_capacity(n);
    v.extend(
        bytes[*pos..end]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]])),
    );
    *pos = end;
    Some(v)
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Bulk little-endian append of a `f32` slice. On little-endian targets the
/// in-memory representation already IS the wire representation, so the whole
/// slice is appended with one memcpy (reinterpreting `&[f32]` as `&[u8]` is
/// sound: u8 has alignment 1 and no invalid bit patterns). Big-endian targets
/// fall back to the per-element byte-swapping loop. Wire bytes are identical
/// either way.
fn push_f32_slice(buf: &mut Vec<u8>, vals: &[f32]) {
    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), std::mem::size_of_val(vals))
        };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for &v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Bulk little-endian append of a `u32` slice (see `push_f32_slice`).
fn push_u32_slice(buf: &mut Vec<u8>, vals: &[u32]) {
    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), std::mem::size_of_val(vals))
        };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for &v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Bulk little-endian append of an `i16` slice (see `push_f32_slice`).
fn push_i16_slice(buf: &mut Vec<u8>, vals: &[i16]) {
    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), std::mem::size_of_val(vals))
        };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for &v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Bulk append of an `i8` slice — endian-independent (single bytes);
/// reinterpreting `&[i8]` as `&[u8]` is sound on every target.
fn push_i8_slice(buf: &mut Vec<u8>, vals: &[i8]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len()) };
    buf.extend_from_slice(bytes);
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let end = pos.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let v = u32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Some(v)
}

fn read_f32(bytes: &[u8], pos: &mut usize) -> Option<f32> {
    let end = pos.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let v = f32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Some(v)
}

fn read_f32_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<f32>> {
    // `n` is wire-controlled (e.g. `hidden` straight off the wire): a
    // `[n=1][layer=0][hidden=0xFFFFFFFF]` request must not reach
    // `Vec::with_capacity` unbounded (~17 GB reservation → SIGABRT).
    // See docs/audits/dec-readiness-review-2026-07-22.md §2a / PR 104 CI.
    if n > bytes.len().saturating_sub(*pos) / 4 {
        return None;
    }
    // Length was validated once above — bulk endian-correct copy, no
    // per-element bounds checks.
    let end = *pos + n * 4;
    let mut v = Vec::with_capacity(n);
    v.extend(
        bytes[*pos..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
    );
    *pos = end;
    Some(v)
}

#[cfg(test)]
mod tests;
