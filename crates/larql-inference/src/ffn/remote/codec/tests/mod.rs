//! Tests for [`super`].
//!
//! Split out of `codec.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;

fn make_single_response(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &v in output {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn make_batch_response(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, floats) in entries {
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
        for &v in floats {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

fn make_single_response_f16(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
    use half::f16;
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &v in output {
        buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    buf
}

fn make_batch_response_f16(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
    use half::f16;
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, floats) in entries {
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
        for &v in floats {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
    }
    buf
}

fn make_single_response_i8(
    layer: u32,
    seq_len: u32,
    latency: f32,
    positions: &[(f32, &[i8])],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(scale, data) in positions {
        buf.extend_from_slice(&scale.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes()); // zero_point ignored
        for &b in data {
            buf.push(b as u8);
        }
    }
    buf
}

#[allow(clippy::type_complexity)]
fn make_batch_response_i8(latency: f32, entries: &[(u32, u32, &[(f32, &[i8])])]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, seq_len, positions) in entries {
        // num_floats per the codec contract (`seq_len * hidden_size`)
        let hidden = positions.first().map(|(_, d)| d.len()).unwrap_or(0);
        let num_floats = seq_len as usize * hidden;
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&(num_floats as u32).to_le_bytes());
        for &(scale, data) in positions {
            buf.extend_from_slice(&scale.to_le_bytes());
            buf.extend_from_slice(&0.0f32.to_le_bytes());
            for &b in data {
                buf.push(b as u8);
            }
        }
    }
    buf
}

//
// Mirrors `moe_remote::multi_layer_wire`'s
// `multi_task_encode_decode_reencode_is_byte_identical`: the shared
// codec must reproduce the exact original bytes for every frame kind,
// pinning the consolidated implementation to the pre-consolidation
// wire (single + batch × f32/f16/i8 responses, single + batch
// requests).

fn reencode_request(req: &DecodedFfnRequest) -> Vec<u8> {
    encode_binary_request(
        req.layer,
        req.layers.as_deref(),
        &req.residual,
        req.seq_len,
        req.full_output,
        req.top_k,
    )
}

/// Rebuild an `FfnOutput` from a decoded layer→floats map, preserving
/// the original entry order (the decoders return `HashMap`s).
fn rebuild_output(
    map: &HashMap<usize, Vec<f32>>,
    layer_order: &[usize],
    seq_len: usize,
    latency_ms: f64,
) -> FfnOutput {
    FfnOutput {
        entries: layer_order
            .iter()
            .map(|&l| FfnEntry {
                layer: l,
                output: map[&l].clone(),
            })
            .collect(),
        seq_len,
        latency_ms,
    }
}

mod asymmetric_direction_pairs_in_return_ind;
mod decode_binary_single_i8_decode_binary_ba;
mod decode_single_response_content_type_disp;
mod f16_i8_request_encode_decode_asymmetric;
mod json_serialisation;
