//! Binary wire codec for the LARQL FFN remote protocol — the SINGLE
//! source of truth for the dense walk-ffn binary frame.
//!
//! Both halves of the wire live here (ROADMAP hardening item 16):
//! - client side: [`encode_binary_request`] + the response decoders
//!   ([`decode_binary_single`], [`decode_binary_batch`], f16/i8 variants,
//!   [`decode_single_response`]);
//! - server side: [`decode_binary_request`] + the response encoders
//!   ([`encode_binary_output`], [`encode_binary_output_f16`],
//!   [`encode_binary_output_i8`], [`encode_json_full_output`]).
//!
//! `larql-server` (`routes/walk_ffn/binary.rs`) and `larql-router` import
//! these functions/constants instead of maintaining their own copies.
//! See the `super` module doc for the full binary frame layout.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod format;
pub use format::*;

pub use larql_router_protocol::walk_ffn::{BATCH_MARKER, BINARY_CT};

/// Fixed response-header size in bytes: `layer|marker (4) + seq_len|count
/// (4) + latency_ms (4)`. Shared by the single and batch response frames
/// in every dtype arm.
pub const RESPONSE_HEADER_LEN: usize = 12;

/// Fixed single-layer request-header size in bytes: `layer (4) + seq_len
/// (4) + flags (4) + top_k (4)`. Batch requests carry `marker (4) +
/// num_layers (4) + layers (4×K)` before the trailing 12 fixed bytes.
pub const REQUEST_HEADER_LEN: usize = 16;

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{what}: byte length overflow"))
}

fn checked_end(offset: usize, len: usize, total: usize, what: &str) -> Result<usize, String> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("{what}: byte range overflow"))?;
    if end > total {
        return Err(format!(
            "{what}: truncated: need {len}, have {}",
            total.saturating_sub(offset)
        ));
    }
    Ok(end)
}

fn read_u32(body: &[u8], offset: usize, what: &str) -> Result<u32, String> {
    let end = checked_end(offset, 4, body.len(), what)?;
    Ok(u32::from_le_bytes(body[offset..end].try_into().unwrap()))
}

fn validate_batch_result_count(body: &[u8], num_results: usize, what: &str) -> Result<(), String> {
    let max_results_with_headers = body.len().saturating_sub(12) / 12;
    if num_results > max_results_with_headers {
        return Err(format!(
            "{what}: declared {num_results} results but only {max_results_with_headers} headers fit"
        ));
    }
    Ok(())
}

// ── Wire types (JSON fallback) ────────────────────────────────────────────────

#[derive(Serialize)]
#[allow(dead_code)]
pub(super) struct WalkFfnHttpRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<usize>>,
    pub residual: Vec<f32>,
    pub seq_len: usize,
    pub full_output: bool,
}

#[derive(Deserialize)]
pub(super) struct WalkFfnSingleResponse {
    #[allow(dead_code)]
    pub layer: usize,
    pub output: Vec<f32>,
    #[allow(dead_code)]
    pub seq_len: usize,
}

// ── Latency profiling result ──────────────────────────────────────────────────

/// Breakdown returned by [`super::http::RemoteWalkBackend::probe_latency`].
#[derive(Debug, Clone)]
pub struct RemoteLatencyStats {
    /// Wall-clock round-trip (client-measured), averaged over `samples` calls.
    pub total_ms: f64,
    /// FFN compute time reported by the server in the binary response header.
    pub server_ms: f64,
    /// `total_ms - server_ms`: HTTP framing + TCP + serialization overhead.
    pub overhead_ms: f64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub samples: usize,
}

impl std::fmt::Display for RemoteLatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "layers={} hidden={} samples={}\n  total    {:7.2} ms\n  server   {:7.2} ms  (FFN compute)\n  overhead {:7.2} ms  (HTTP + TCP + framing)",
            self.num_layers, self.hidden_size, self.samples,
            self.total_ms, self.server_ms, self.overhead_ms,
        )
    }
}

// ── Binary codec ──────────────────────────────────────────────────────────────

/// Append the request header (`layer|marker[+layers]`, `seq_len`, `flags`,
/// `top_k`) shared by every request dtype arm (f32/f16/i8 — ADR-0009: only
/// the residual payload encoding differs between formats).
fn push_request_header(
    buf: &mut Vec<u8>,
    layer: Option<usize>,
    layers: Option<&[usize]>,
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) {
    if let Some(ls) = layers {
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(ls.len() as u32).to_le_bytes());
        for &l in ls {
            buf.extend_from_slice(&(l as u32).to_le_bytes());
        }
    } else {
        let l = layer.unwrap_or(0) as u32;
        buf.extend_from_slice(&l.to_le_bytes());
    }
    buf.extend_from_slice(&(seq_len as u32).to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&(top_k as u32).to_le_bytes());
}

/// Encode a request as binary (f32 residual payload — the historical wire).
/// `layer` and `layers` are mutually exclusive; pass `None` for the unused one.
///
/// Thin symmetric wrapper over [`encode_binary_request_as`] with
/// [`WireFormat::F32`]; kept so no existing caller changes behaviour.
pub fn encode_binary_request(
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    encode_binary_request_as(
        WireFormat::F32,
        layer,
        layers,
        residual,
        seq_len,
        full_output,
        top_k,
    )
}

/// Encode a request with the residual payload in `format` (asymmetric
/// direction codecs, DEC funnel v0.5 §3 DEC-1A: the inbound format is
/// declared by the request `Content-Type` and is independent of the
/// `Accept`-negotiated return format).
///
/// The header bytes are identical across formats; the residual payload
/// mirrors the corresponding RESPONSE encoding exactly:
/// - f32: `f32 LE` per value;
/// - f16: `u16 LE` IEEE half per value;
/// - i8: per position `[scale f32 LE][zero_point f32 LE][data i8[hidden]]`
///   with `scale = max(|x|)/127`, symmetric (`hidden = residual.len() /
///   max(seq_len, 1)`; `residual.len()` must divide evenly).
pub fn encode_binary_request_as(
    format: WireFormat,
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    use half::f16;
    let mut buf = Vec::with_capacity(16 + residual.len() * 4);
    push_request_header(&mut buf, layer, layers, seq_len, full_output, top_k);
    match format {
        WireFormat::F32 => {
            for &v in residual {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        WireFormat::F16 => {
            for &v in residual {
                buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        WireFormat::I8 => {
            let seq = seq_len.max(1);
            debug_assert!(
                residual.is_empty() || residual.len().is_multiple_of(seq),
                "i8 request: residual length {} not a multiple of seq_len {seq}",
                residual.len()
            );
            let hidden = residual.len() / seq;
            if hidden > 0 {
                for pos in 0..seq {
                    quantise_i8_position(&residual[pos * hidden..(pos + 1) * hidden], &mut buf);
                }
            }
        }
    }
    buf
}

/// Decode a binary single-layer full_output response.
/// Returns `(layer, output_floats)`.
pub fn decode_binary_single(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!("binary response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer response but got batch marker".into());
    }
    let layer = marker as usize;
    // bytes 4-7: seq_len (ignored here — caller validates against expected shape)
    // bytes 8-11: latency f32
    if !(body.len() - 12).is_multiple_of(4) {
        return Err("binary response: output byte length is not a multiple of f32".into());
    }
    let floats: Vec<f32> = body[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch full_output response.
/// Returns a map from layer → output floats.
pub fn decode_binary_batch(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "binary batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());

    // Single-layer response — accept it as a batch of 1.
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single(body)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }

    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "binary batch")?;
    // bytes 8-11: latency f32 (skip)
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);

    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "binary batch result header")?;
        let layer = read_u32(body, offset, "binary batch layer")? as usize;
        // offset+4: seq_len (skip)
        let num_floats = read_u32(body, offset + 8, "binary batch output length")? as usize;
        offset += 12;
        let bytes_needed = checked_mul(num_floats, 4, "binary batch output")?;
        let end = checked_end(offset, bytes_needed, body.len(), "binary batch output")?;
        let floats: Vec<f32> = body[offset..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset = end;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// f16 content-type constant (ADR-0009).
pub const F16_CT: &str = "application/x-larql-ffn-f16";

/// Decode a binary single-layer f16 response into f32 output.
pub fn decode_binary_single_f16(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    use half::f16;
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "f16 binary response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer f16 response but got batch marker".into());
    }
    let layer = marker as usize;
    if !(body.len() - 12).is_multiple_of(2) {
        return Err("f16 binary response: output byte length is not a multiple of f16".into());
    }
    let floats: Vec<f32> = body[12..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch f16 response into f32 outputs.
pub fn decode_binary_batch_f16(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    use half::f16;
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "f16 batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single_f16(body)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "f16 batch")?;
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);
    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "f16 batch result header")?;
        let layer = read_u32(body, offset, "f16 batch layer")? as usize;
        let num_floats = read_u32(body, offset + 8, "f16 batch output length")? as usize;
        offset += 12;
        let bytes_needed = checked_mul(num_floats, 2, "f16 batch output")?;
        let end = checked_end(offset, bytes_needed, body.len(), "f16 batch output")?;
        let floats: Vec<f32> = body[offset..end]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        offset = end;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// i8 content-type constant (ADR-0009).
pub const I8_CT: &str = "application/x-larql-ffn-i8";

/// Quantise one position into an i8 per-position block (symmetric):
/// `[scale f32 LE][zero_point f32 LE = 0][data i8[len]]` with
/// `scale = max(|x|)/127` (1.0 for an all-zero position). Shared by the
/// i8 response encoder and the i8 request encoder (same payload layout in
/// both directions — ADR-0009).
fn quantise_i8_position(vals: &[f32], buf: &mut Vec<u8>) {
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    buf.extend_from_slice(&scale.to_le_bytes());
    buf.extend_from_slice(&0.0f32.to_le_bytes()); // zero_point = 0
    for &v in vals {
        let q = (v / scale).clamp(-127.0, 127.0).round() as i8;
        buf.push(q as u8);
    }
}

/// Decode one position from an i8 per-position block.
/// Format: `[scale f32 LE][zero_point f32 LE (ignored)][data i8[hidden_size]]`
fn decode_i8_position(
    body: &[u8],
    offset: usize,
    hidden: usize,
) -> Result<(Vec<f32>, usize), String> {
    let needed = 8usize
        .checked_add(hidden)
        .ok_or_else(|| "i8: position byte length overflow".to_string())?;
    let end = checked_end(offset, needed, body.len(), "i8 position")?;
    let scale = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
    // zero_point at offset+4 is always 0.0 (symmetric), skip it
    let floats: Vec<f32> = body[offset + 8..offset + 8 + hidden]
        .iter()
        .map(|&b| (b as i8) as f32 * scale)
        .collect();
    Ok((floats, end))
}

/// Decode a binary single-layer i8 response into f32 output.
pub(crate) fn decode_binary_single_i8(
    body: &[u8],
    hidden_size: usize,
) -> Result<(usize, Vec<f32>), String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!(
            "i8 binary response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer i8 response but got batch marker".into());
    }
    let layer = marker as usize;
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    let seq_len = seq_len.max(1);
    let mut offset = 12usize;
    let total_floats = checked_mul(seq_len, hidden_size, "i8 single output")?;
    let mut all_floats = Vec::with_capacity(total_floats);
    for _ in 0..seq_len {
        let (pos_floats, next_offset) = decode_i8_position(body, offset, hidden_size)?;
        all_floats.extend(pos_floats);
        offset = next_offset;
    }
    Ok((layer, all_floats))
}

/// Decode a binary batch i8 response into f32 outputs.
pub(crate) fn decode_binary_batch_i8(
    body: &[u8],
    hidden_size: usize,
) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < RESPONSE_HEADER_LEN {
        return Err(format!("i8 batch response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single_i8(body, hidden_size)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "i8 batch")?;
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);
    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "i8 batch result header")?;
        let layer = read_u32(body, offset, "i8 batch layer")? as usize;
        let seq_len = read_u32(body, offset + 4, "i8 batch sequence length")? as usize;
        let seq_len = seq_len.max(1);
        let num_floats = read_u32(body, offset + 8, "i8 batch output length")? as usize;
        let expected_floats = checked_mul(seq_len, hidden_size, "i8 batch output")?;
        if num_floats != expected_floats {
            return Err(format!(
                "i8 batch: layer {layer} declared {num_floats} floats, expected {expected_floats}"
            ));
        }
        offset += 12;
        let mut all_floats = Vec::with_capacity(num_floats);
        for _ in 0..seq_len {
            let (pos_floats, next_offset) = decode_i8_position(body, offset, hidden_size)?;
            all_floats.extend(pos_floats);
            offset = next_offset;
        }
        out.insert(layer, all_floats);
    }
    Ok(out)
}

// ── Wire format axis ─────────────────────────────────────────────────────────

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
