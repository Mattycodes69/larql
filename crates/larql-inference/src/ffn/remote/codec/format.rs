//! Wire formats and response decoding.

#[allow(unused_imports)]
use super::*;

/// One direction's binary dtype on the dense walk-ffn wire (ADR-0009 +
/// DEC funnel v0.5 §3 DEC-1A asymmetric direction codecs). The inbound
/// residual format is declared by the request `Content-Type`; the return
/// format by `Accept` — the two are independent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WireFormat {
    /// `application/x-larql-ffn` — f32 LE (the historical wire, default).
    #[default]
    F32,
    /// `application/x-larql-ffn-f16` — IEEE half per value.
    F16,
    /// `application/x-larql-ffn-i8` — per-position symmetric i8 blocks.
    I8,
}

impl WireFormat {
    /// The content-type string for this format.
    pub fn content_type(self) -> &'static str {
        match self {
            Self::F32 => BINARY_CT,
            Self::F16 => F16_CT,
            Self::I8 => I8_CT,
        }
    }

    /// Short label (`"f32"`, `"f16"`, `"i8"`).
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I8 => "i8",
        }
    }

    /// Parse from a short label. Inverse of [`Self::label`].
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "i8" => Some(Self::I8),
            _ => None,
        }
    }

    /// Match a `Content-Type` header value to a wire format.
    ///
    /// ORDER MATTERS: [`BINARY_CT`] is a substring of both [`F16_CT`] and
    /// [`I8_CT`], so the suffixed types must be checked first — a plain
    /// `contains(BINARY_CT)` check would silently misread an f16/i8 body
    /// as f32. Uses `contains` so parameterised types (`…; v=2`) match.
    pub fn from_content_type(ct: &str) -> Option<Self> {
        if ct.contains(I8_CT) {
            Some(Self::I8)
        } else if ct.contains(F16_CT) {
            Some(Self::F16)
        } else if ct.contains(BINARY_CT) {
            Some(Self::F32)
        } else {
            None
        }
    }
}

/// Decode any single-layer binary walk-ffn response by content type.
///
/// Dispatches to the f32/f16/i8 decoder based on `content_type` and also
/// extracts the server-side `latency_ms` embedded at bytes 8-11.
/// Returns `(layer, server_latency_ms, output_f32)`.
pub fn decode_single_response(
    content_type: &str,
    body: &[u8],
    hidden_size: usize,
) -> Result<(usize, f64, Vec<f32>), String> {
    let server_ms = extract_response_latency_ms(body);
    let (layer, floats) = match content_type {
        I8_CT => decode_binary_single_i8(body, hidden_size)?,
        F16_CT => decode_binary_single_f16(body)?,
        BINARY_CT => decode_binary_single(body)?,
        other => {
            return Err(format!(
                "unsupported walk-ffn response content-type: {other}"
            ))
        }
    };
    Ok((layer, server_ms, floats))
}

/// Extract the `latency_ms` f32 embedded at bytes 8-11 of a binary response.
/// Returns 0.0 if the body is too short or the value is non-finite.
pub(in super::super) fn extract_response_latency_ms(body: &[u8]) -> f64 {
    if body.len() < RESPONSE_HEADER_LEN {
        return 0.0;
    }
    // Both single-layer and batch responses have latency_ms at offset 8.
    let v = f32::from_le_bytes(body[8..12].try_into().unwrap());
    if v.is_finite() {
        v as f64
    } else {
        0.0
    }
}

// ── Server-side half: request decode + response encode ───────────────────────
//
// Moved here from `larql-server/src/routes/walk_ffn/binary.rs` (ROADMAP
// hardening item 16) so the encoder and decoder of each wire direction
// live in one module. The server wraps these in its own error/request
// types; the byte layout and every length guard are pinned by the
// round-trip + rejection tests below.

/// A decoded binary walk-ffn request — the server-side inverse of
/// [`encode_binary_request`]. Pure wire data; the server layers its
/// JSON-request semantics (e.g. `moe_layer`, serde defaults) on top.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFfnRequest {
    /// Single-layer mode. Mutually exclusive with `layers`.
    pub layer: Option<usize>,
    /// Batch mode — multiple layers in one request.
    pub layers: Option<Vec<usize>>,
    /// Row-major flat residual, `seq_len × hidden` floats.
    pub residual: Vec<f32>,
    pub seq_len: usize,
    pub top_k: usize,
    pub full_output: bool,
}

/// Parsed request header — everything before the residual payload. Shared
/// by the f32/f16/i8 request decoders (the header layout is format-invariant).
pub(super) struct RequestHeader {
    pub(super) layer: Option<usize>,
    pub(super) layers: Option<Vec<usize>>,
    pub(super) seq_len: usize,
    pub(super) top_k: usize,
    pub(super) full_output: bool,
    /// Offset of the first residual payload byte.
    pub(super) payload_offset: usize,
}

/// Parse the shared request header (inverse of [`push_request_header`]),
/// with the same truncation and alloc-bomb guards as the original f32-only
/// decoder (PR 104 lineage).
pub(super) fn parse_request_header(body: &[u8]) -> Result<RequestHeader, String> {
    if body.len() < REQUEST_HEADER_LEN {
        return Err("binary: body too short (need ≥ 16 bytes)".into());
    }

    let first = u32::from_le_bytes(body[0..4].try_into().unwrap());

    let (layer, layers, header_end) = if first == BATCH_MARKER {
        let n = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
        let layers_end = checked_mul(n, 4, "binary batch layer indices")?
            .checked_add(8)
            .ok_or_else(|| "binary batch layer indices: byte range overflow".to_string())?;
        if body.len() < layers_end {
            return Err(format!(
                "binary batch: body too short for {n} layer indices"
            ));
        }
        let layers: Vec<usize> = (0..n)
            .map(|i| u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize)
            .collect();
        (None, Some(layers), layers_end)
    } else {
        (Some(first as usize), None, 4)
    };

    if body.len() < header_end + 12 {
        return Err("binary: truncated fixed header (seq_len/flags/top_k)".into());
    }
    let seq_len = u32::from_le_bytes(body[header_end..header_end + 4].try_into().unwrap()) as usize;
    let flags = u32::from_le_bytes(body[header_end + 4..header_end + 8].try_into().unwrap());
    let top_k =
        u32::from_le_bytes(body[header_end + 8..header_end + 12].try_into().unwrap()) as usize;

    Ok(RequestHeader {
        layer,
        layers,
        seq_len,
        top_k,
        full_output: (flags & 1) != 0,
        payload_offset: header_end + 12,
    })
}

impl RequestHeader {
    pub(super) fn into_request(self, residual: Vec<f32>) -> DecodedFfnRequest {
        DecodedFfnRequest {
            layer: self.layer,
            layers: self.layers,
            residual,
            seq_len: self.seq_len,
            top_k: self.top_k,
            full_output: self.full_output,
        }
    }
}

/// Decode a binary-format request body (inverse of [`encode_binary_request`]).
pub fn decode_binary_request(body: &[u8]) -> Result<DecodedFfnRequest, String> {
    let header = parse_request_header(body)?;
    let residual_bytes = &body[header.payload_offset..];
    if !residual_bytes.len().is_multiple_of(4) {
        return Err("binary: residual byte length is not a multiple of 4".into());
    }
    let residual: Vec<f32> = residual_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(header.into_request(residual))
}

/// Decode an f16-format request body (inverse of
/// [`encode_binary_request_as`] with [`WireFormat::F16`]). The residual is
/// widened to f32.
pub fn decode_binary_request_f16(body: &[u8]) -> Result<DecodedFfnRequest, String> {
    use half::f16;
    let header = parse_request_header(body)?;
    let residual_bytes = &body[header.payload_offset..];
    if !residual_bytes.len().is_multiple_of(2) {
        return Err("f16 binary request: residual byte length is not a multiple of f16".into());
    }
    let residual: Vec<f32> = residual_bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    Ok(header.into_request(residual))
}

/// Decode an i8-format request body (inverse of
/// [`encode_binary_request_as`] with [`WireFormat::I8`]). The per-position
/// hidden size is self-describing: `payload_len / max(seq_len, 1) − 8`
/// (each position carries an 8-byte scale/zero header) — validated against
/// the model's hidden size downstream like every other request.
pub fn decode_binary_request_i8(body: &[u8]) -> Result<DecodedFfnRequest, String> {
    let header = parse_request_header(body)?;
    let residual_bytes = &body[header.payload_offset..];
    if residual_bytes.is_empty() {
        return Ok(header.into_request(Vec::new()));
    }
    let seq = header.seq_len.max(1);
    if !residual_bytes.len().is_multiple_of(seq) {
        return Err(format!(
            "i8 binary request: payload of {} bytes is not a multiple of seq_len {seq}",
            residual_bytes.len()
        ));
    }
    let per_pos = residual_bytes.len() / seq;
    if per_pos <= 8 {
        return Err(format!(
            "i8 binary request: {per_pos} bytes per position leaves no residual data \
             after the 8-byte scale/zero header"
        ));
    }
    let hidden = per_pos - 8;
    let mut residual = Vec::with_capacity(checked_mul(seq, hidden, "i8 request residual")?);
    let mut offset = 0usize;
    for _ in 0..seq {
        let (pos_floats, next_offset) = decode_i8_position(residual_bytes, offset, hidden)?;
        residual.extend(pos_floats);
        offset = next_offset;
    }
    Ok(header.into_request(residual))
}

/// Decode a request body whose residual payload is in `format` — the
/// server-side inbound dispatch twin of [`encode_binary_request_as`].
pub fn decode_binary_request_as(
    format: WireFormat,
    body: &[u8],
) -> Result<DecodedFfnRequest, String> {
    match format {
        WireFormat::F32 => decode_binary_request(body),
        WireFormat::F16 => decode_binary_request_f16(body),
        WireFormat::I8 => decode_binary_request_i8(body),
    }
}

/// One layer's FFN output (server-side response payload).
#[derive(Debug, Clone, PartialEq)]
pub struct FfnEntry {
    pub layer: usize,
    pub output: Vec<f32>,
}

/// Typed server response consumed by all four response encoders
/// (f32/f16/i8 binary + JSON).
#[derive(Debug, Clone, PartialEq)]
pub struct FfnOutput {
    pub entries: Vec<FfnEntry>,
    pub seq_len: usize,
    pub latency_ms: f64,
}

/// Encode an [`FfnOutput`] as the binary response format.
pub fn encode_binary_output(out: &FfnOutput) -> Vec<u8> {
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + entry.output.len() * 4);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + num * 12);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] using f16 values for the residual/output arrays.
///
/// Wire layout: identical to the f32 format except every float in the output
/// arrays is a `u16` LE (IEEE 754 half-precision). Header fields (layer,
/// seq_len, latency_ms) remain f32/u32 LE. See ADR-0009.
pub fn encode_binary_output_f16(out: &FfnOutput) -> Vec<u8> {
    use half::f16;
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + entry.output.len() * 2);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let total_floats: usize = out.entries.iter().map(|e| e.output.len()).sum();
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + num * 12 + total_floats * 2);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] using i8 symmetric quantisation (ADR-0009).
///
/// Per position: `[scale f32 LE][zero_point f32 LE][data i8[hidden_size]]`.
/// `scale = max(|x|) / 127.0`, `zero_point = 0.0` (symmetric).
/// Header fields (layer, seq_len, latency_ms) remain f32/u32 LE.
pub fn encode_binary_output_i8(out: &FfnOutput) -> Vec<u8> {
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let seq = out.seq_len.max(1);
        let hidden = entry.output.len() / seq;
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + seq * (8 + hidden));
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for pos in 0..seq {
            quantise_i8_position(&entry.output[pos * hidden..(pos + 1) * hidden], &mut buf);
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(RESPONSE_HEADER_LEN + num * 16);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            let seq = out.seq_len.max(1);
            let hidden = entry.output.len() / seq;
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for pos in 0..seq {
                quantise_i8_position(&entry.output[pos * hidden..(pos + 1) * hidden], &mut buf);
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] as the existing JSON response format (unchanged
/// wire contract for JSON clients).
pub fn encode_json_full_output(out: &FfnOutput) -> serde_json::Value {
    let latency_rounded = (out.latency_ms * 10.0).round() / 10.0;
    if out.entries.len() == 1 {
        let e = &out.entries[0];
        serde_json::json!({
            "layer": e.layer,
            "output": e.output,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    } else {
        let results: Vec<serde_json::Value> = out
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "layer": e.layer,
                    "output": e.output,
                    "seq_len": out.seq_len,
                })
            })
            .collect();
        serde_json::json!({
            "results": results,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    }
}
