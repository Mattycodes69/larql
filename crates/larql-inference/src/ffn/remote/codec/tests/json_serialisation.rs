//! JSON serialisation
//! encode_binary_request
//! decode_binary_single
//! decode_binary_batch
//! decode_binary_single_f16 + decode_binary_batch_f16

use super::*;

#[test]
fn request_serializes_with_seq_len_and_full_output() {
    let req = WalkFfnHttpRequest {
        layer: Some(3),
        layers: None,
        residual: vec![0.1, -0.2, 0.3, 0.4],
        seq_len: 2,
        full_output: true,
    };
    let v: serde_json::Value = serde_json::to_value(&req).unwrap();
    assert_eq!(v["layer"], 3);
    assert_eq!(v["seq_len"], 2);
    assert_eq!(v["full_output"], true);
    assert!(
        v.get("layers").is_none() || v["layers"].is_null(),
        "layers should not appear when None, got: {v}"
    );
    assert_eq!(v["residual"].as_array().unwrap().len(), 4);
}

#[test]
fn response_deserializes_hidden_vector() {
    let json = serde_json::json!({
        "layer": 5,
        "output": [0.1, 0.2, 0.3, 0.4, 0.5],
        "seq_len": 1,
        "latency_ms": 2.5,
    });
    let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.layer, 5);
    assert_eq!(parsed.output.len(), 5);
    assert_eq!(parsed.seq_len, 1);
}

#[test]
fn response_deserializes_multi_token_output() {
    let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let json = serde_json::json!({
        "layer": 0,
        "output": flat,
        "seq_len": 3,
    });
    let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.output.len(), 12);
    assert_eq!(parsed.seq_len, 3);
}

#[test]
fn encode_single_layer_header() {
    let residual = vec![1.0f32, 2.0, 3.0, 4.0];
    let body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
    // First u32 = layer index
    let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
    assert_eq!(layer, 7);
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(seq_len, 1);
    let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
    assert_eq!(flags & 1, 1); // full_output
    let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
    assert_eq!(top_k, 256);
    assert_eq!(body.len(), 16 + 4 * 4);
}

#[test]
fn encode_batch_header() {
    let residual = vec![0.5f32; 4];
    let body = encode_binary_request(None, Some(&[5, 20, 30]), &residual, 1, true, 512);
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    assert_eq!(marker, BATCH_MARKER);
    let num_layers = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(num_layers, 3);
    let l0 = u32::from_le_bytes(body[8..12].try_into().unwrap());
    let l1 = u32::from_le_bytes(body[12..16].try_into().unwrap());
    let l2 = u32::from_le_bytes(body[16..20].try_into().unwrap());
    assert_eq!((l0, l1, l2), (5, 20, 30));
}

#[test]
fn encode_residual_values_preserved() {
    let residual = vec![-1.5f32, 0.0, 3.25];
    let body = encode_binary_request(Some(0), None, &residual, 1, true, 8092);
    let offset = 16; // 4 header u32s × 4 bytes
    let v0 = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
    let v1 = f32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap());
    let v2 = f32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap());
    assert_eq!(v0.to_bits(), (-1.5f32).to_bits());
    assert_eq!(v1.to_bits(), 0.0f32.to_bits());
    assert!((v2 - 3.25f32).abs() < 1e-5);
}

#[test]
fn decode_single_response_correct() {
    let output = vec![1.0f32, -2.0, 3.5];
    let body = make_single_response(5, 1, 7.3, &output);
    let (layer, floats) = decode_binary_single(&body).unwrap();
    assert_eq!(layer, 5);
    assert_eq!(floats.len(), 3);
    assert!((floats[0] - 1.0).abs() < 1e-6);
    assert!((floats[1] - (-2.0)).abs() < 1e-6);
}

#[test]
fn decode_single_response_rejects_batch_marker() {
    let body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
    let result = decode_binary_single(&body);
    assert!(result.is_err());
}

#[test]
fn decode_single_response_too_short() {
    let result = decode_binary_single(&[0u8; 8]);
    assert!(result.is_err());
}

#[test]
fn decode_batch_response_correct() {
    let body = make_batch_response(15.0, &[(5, &[1.0, 2.0]), (20, &[3.0, 4.0])]);
    let map = decode_binary_batch(&body).unwrap();
    assert_eq!(map.len(), 2);
    let v5 = map.get(&5).unwrap();
    assert_eq!(v5.len(), 2);
    assert!((v5[0] - 1.0).abs() < 1e-6);
    let v20 = map.get(&20).unwrap();
    assert!((v20[1] - 4.0).abs() < 1e-6);
}

#[test]
fn decode_batch_accepts_single_response() {
    // A server returning single-layer response to a same-shard batch.
    let output = vec![7.0f32, 8.0];
    let body = make_single_response(10, 1, 5.0, &output);
    let map = decode_binary_batch(&body).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&10));
}

#[test]
fn decode_batch_truncated_returns_error() {
    let mut body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
    body.truncate(body.len() - 4); // cut off last float
    let result = decode_binary_batch(&body);
    assert!(result.is_err());
}

#[test]
fn decode_single_rejects_partial_float_payload() {
    let mut body = make_single_response(5, 1, 7.3, &[1.0]);
    body.push(0);
    let result = decode_binary_single(&body);
    assert!(result.is_err());
}

#[test]
fn decode_batch_rejects_impossible_result_count_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    let result = decode_binary_batch(&body);
    assert!(result.is_err());
}

#[test]
fn decode_batch_rejects_impossible_output_length_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    body.extend_from_slice(&5u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    let result = decode_binary_batch(&body);
    assert!(result.is_err());
}

#[test]
fn decode_batch_i8_rejects_inconsistent_output_shape() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    body.extend_from_slice(&5u32.to_le_bytes());
    body.extend_from_slice(&2u32.to_le_bytes());
    body.extend_from_slice(&3u32.to_le_bytes());
    let result = decode_binary_batch_i8(&body, 2);
    assert!(result.is_err());
}

#[test]
fn decode_single_f16_round_trip_within_quant_noise() {
    let body = make_single_response_f16(7, 1, 1.0, &[0.5, -0.25, 1.5, -2.5]);
    let (layer, floats) = decode_binary_single_f16(&body).unwrap();
    assert_eq!(layer, 7);
    assert_eq!(floats.len(), 4);
    // f16 round-trip is exact for these clean fractions.
    assert!((floats[0] - 0.5).abs() < 1e-6);
    assert!((floats[3] - (-2.5)).abs() < 1e-6);
}

#[test]
fn decode_single_f16_too_short_errors() {
    assert!(decode_binary_single_f16(&[0u8; 8]).is_err());
}

#[test]
fn decode_single_f16_rejects_batch_marker() {
    let body = make_batch_response_f16(1.0, &[(0, &[1.0])]);
    assert!(decode_binary_single_f16(&body).is_err());
}

#[test]
fn decode_single_f16_rejects_odd_payload_length() {
    let mut body = make_single_response_f16(0, 1, 0.0, &[1.0]);
    body.push(0u8); // odd byte tail
    assert!(decode_binary_single_f16(&body).is_err());
}

#[test]
fn decode_batch_f16_round_trip_two_entries() {
    let body = make_batch_response_f16(2.0, &[(3, &[1.0, 2.0]), (11, &[-1.0, 0.5])]);
    let map = decode_binary_batch_f16(&body).unwrap();
    assert_eq!(map.len(), 2);
    let v3 = map.get(&3).unwrap();
    assert!((v3[0] - 1.0).abs() < 1e-6 && (v3[1] - 2.0).abs() < 1e-6);
    let v11 = map.get(&11).unwrap();
    assert!((v11[1] - 0.5).abs() < 1e-6);
}

#[test]
fn decode_batch_f16_falls_through_to_single_when_no_marker() {
    let body = make_single_response_f16(5, 1, 1.0, &[1.0, 2.0, 3.0]);
    let map = decode_binary_batch_f16(&body).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&5));
}

#[test]
fn decode_batch_f16_too_short_errors() {
    assert!(decode_binary_batch_f16(&[0u8; 4]).is_err());
}

#[test]
fn decode_batch_f16_rejects_impossible_result_count() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    assert!(decode_binary_batch_f16(&body).is_err());
}
