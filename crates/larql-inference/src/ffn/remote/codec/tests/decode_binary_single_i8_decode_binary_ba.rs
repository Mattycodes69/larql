//! decode_binary_single_i8 + decode_binary_batch_i8
//! extract_response_latency_ms
//! Wire string consts

use super::*;

#[test]
fn decode_single_i8_round_trip_one_position() {
    let hidden = 4;
    let body = make_single_response_i8(5, 1, 1.0, &[(0.5, &[2i8, -4, 8, -8])]);
    let (layer, floats) = decode_binary_single_i8(&body, hidden).unwrap();
    assert_eq!(layer, 5);
    assert_eq!(floats, vec![1.0f32, -2.0, 4.0, -4.0]);
}

#[test]
fn decode_single_i8_round_trip_multi_position() {
    let hidden = 2;
    let body = make_single_response_i8(
        0,
        3,
        0.0,
        &[(1.0, &[10i8, 20]), (0.25, &[-4i8, 8]), (2.0, &[1i8, -1])],
    );
    let (_, floats) = decode_binary_single_i8(&body, hidden).unwrap();
    assert_eq!(floats, vec![10.0, 20.0, -1.0, 2.0, 2.0, -2.0]);
}

#[test]
fn decode_single_i8_rejects_batch_marker() {
    let body = make_batch_response_i8(1.0, &[(0, 1, &[(1.0, &[1i8])])]);
    assert!(decode_binary_single_i8(&body, 1).is_err());
}

#[test]
fn decode_single_i8_too_short_errors() {
    assert!(decode_binary_single_i8(&[0u8; 8], 4).is_err());
}

#[test]
fn decode_single_i8_zero_seq_len_treated_as_one() {
    // Codec promotes seq_len 0 → 1 to keep the per-position loop alive.
    let body = make_single_response_i8(2, 0, 0.0, &[(1.0, &[5i8])]);
    let (layer, floats) = decode_binary_single_i8(&body, 1).unwrap();
    assert_eq!(layer, 2);
    assert_eq!(floats, vec![5.0]);
}

#[test]
fn decode_single_i8_truncated_payload_errors() {
    // Position needs 8 + hidden bytes; cut the payload short.
    let mut body = make_single_response_i8(0, 1, 0.0, &[(1.0, &[1i8, 2, 3, 4])]);
    body.truncate(body.len() - 1);
    assert!(decode_binary_single_i8(&body, 4).is_err());
}

#[test]
fn decode_batch_i8_round_trip_two_layers() {
    let hidden = 2;
    let body = make_batch_response_i8(
        3.0,
        &[
            (10, 1, &[(1.0, &[7i8, -7])]),
            (20, 2, &[(0.5, &[10i8, -10]), (0.5, &[20i8, -20])]),
        ],
    );
    let map = decode_binary_batch_i8(&body, hidden).unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&10).unwrap(), &vec![7.0, -7.0]);
    assert_eq!(map.get(&20).unwrap(), &vec![5.0, -5.0, 10.0, -10.0]);
}

#[test]
fn decode_batch_i8_falls_through_to_single_when_no_marker() {
    let body = make_single_response_i8(9, 1, 0.0, &[(1.0, &[3i8, -3])]);
    let map = decode_binary_batch_i8(&body, 2).unwrap();
    assert_eq!(map.len(), 1);
    assert!(map.contains_key(&9));
}

#[test]
fn decode_batch_i8_too_short_errors() {
    assert!(decode_binary_batch_i8(&[0u8; 4], 1).is_err());
}

#[test]
fn decode_batch_i8_rejects_impossible_result_count() {
    let mut body = Vec::new();
    body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    body.extend_from_slice(&u32::MAX.to_le_bytes());
    body.extend_from_slice(&0.0f32.to_le_bytes());
    assert!(decode_binary_batch_i8(&body, 2).is_err());
}

#[test]
fn extract_latency_returns_zero_for_short_body() {
    assert_eq!(extract_response_latency_ms(&[]), 0.0);
    assert_eq!(extract_response_latency_ms(&[0u8; 11]), 0.0);
}

#[test]
fn extract_latency_reads_offset_8_as_f32() {
    // Body: layer(4) + seq_len(4) + latency(4)=8.5
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&8.5f32.to_le_bytes());
    assert!((extract_response_latency_ms(&body) - 8.5).abs() < 1e-6);
}

#[test]
fn extract_latency_returns_zero_for_non_finite() {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes());
    body.extend_from_slice(&1u32.to_le_bytes());
    body.extend_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(extract_response_latency_ms(&body), 0.0);
}

#[test]
fn binary_content_type_consts_pin_wire_strings() {
    assert_eq!(BINARY_CT, "application/x-larql-ffn");
    assert_eq!(F16_CT, "application/x-larql-ffn-f16");
    assert_eq!(I8_CT, "application/x-larql-ffn-i8");
    assert_eq!(BATCH_MARKER, 0xFFFF_FFFFu32);
}
