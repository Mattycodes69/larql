//! f16/i8 request encode/decode (asymmetric inbound direction)

use super::*;

#[test]
fn encode_request_as_f32_is_byte_identical_to_legacy_encoder() {
    // The symmetric wrapper and the format-parameterised encoder must
    // produce the same bytes — existing callers keep their wire.
    let residual = vec![1.5f32, -2.25, 0.0, f32::MIN_POSITIVE];
    assert_eq!(
        encode_binary_request_as(WireFormat::F32, Some(7), None, &residual, 2, true, 8092),
        encode_binary_request(Some(7), None, &residual, 2, true, 8092),
    );
    assert_eq!(
        encode_binary_request_as(WireFormat::F32, None, Some(&[1, 2]), &residual, 1, true, 0),
        encode_binary_request(None, Some(&[1, 2]), &residual, 1, true, 0),
    );
}

#[test]
fn f16_request_shares_header_layout_and_halves_payload() {
    let residual = vec![0.5f32, -0.25, 1.5, -2.5];
    let body = encode_binary_request_as(WireFormat::F16, Some(7), None, &residual, 1, true, 256);
    assert_eq!(body.len(), REQUEST_HEADER_LEN + residual.len() * 2);
    // Header bytes identical to the f32 frame's.
    let f32_body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
    assert_eq!(body[..REQUEST_HEADER_LEN], f32_body[..REQUEST_HEADER_LEN]);
}

#[test]
fn request_f16_encode_decode_reencode_is_byte_identical() {
    // f16-exact values so decode → f32 → re-encode reproduces the bits.
    let residual = vec![0.5f32, -0.25, 1.5, -2.5];
    // Single-layer frame.
    let encoded =
        encode_binary_request_as(WireFormat::F16, Some(7), None, &residual, 2, true, 8092);
    let decoded = decode_binary_request_f16(&encoded).unwrap();
    assert_eq!(decoded.layer, Some(7));
    assert_eq!(decoded.seq_len, 2);
    assert_eq!(decoded.top_k, 8092);
    assert!(decoded.full_output);
    assert_eq!(decoded.residual, residual);
    let reencoded = encode_binary_request_as(
        WireFormat::F16,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);

    // Batch frame.
    let encoded = encode_binary_request_as(
        WireFormat::F16,
        None,
        Some(&[3, 900]),
        &residual,
        1,
        true,
        0,
    );
    let decoded = decode_binary_request_f16(&encoded).unwrap();
    assert_eq!(decoded.layers, Some(vec![3, 900]));
    let reencoded = encode_binary_request_as(
        WireFormat::F16,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);
}

#[test]
fn request_i8_encode_decode_reencode_is_byte_identical() {
    // Fixed-point construction (mirrors the i8 response pin): per
    // position max|v| = 127 × scale with scale a power of two, so
    // quantise(dequantise(q)) == q and the recomputed scale bit-matches.
    // Two positions of hidden = 4: scale 0.5, then scale 0.25.
    let residual = vec![63.5f32, -32.0, 0.5, -63.5, 31.75, -16.0, 0.25, 8.0];
    let encoded = encode_binary_request_as(WireFormat::I8, Some(5), None, &residual, 2, true, 0);
    assert_eq!(encoded.len(), REQUEST_HEADER_LEN + 2 * (8 + 4));
    let decoded = decode_binary_request_i8(&encoded).unwrap();
    assert_eq!(decoded.layer, Some(5));
    assert_eq!(decoded.seq_len, 2);
    assert_eq!(decoded.residual, residual);
    let reencoded = encode_binary_request_as(
        WireFormat::I8,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);

    // Batch frame, one position (seq_len 1).
    let row = vec![127.0f32, -64.0, 1.0, -127.0]; // scale 1.0
    let encoded = encode_binary_request_as(WireFormat::I8, None, Some(&[0, 9]), &row, 1, true, 0);
    let decoded = decode_binary_request_i8(&encoded).unwrap();
    assert_eq!(decoded.layers, Some(vec![0, 9]));
    assert_eq!(decoded.residual, row);
    let reencoded = encode_binary_request_as(
        WireFormat::I8,
        decoded.layer,
        decoded.layers.as_deref(),
        &decoded.residual,
        decoded.seq_len,
        decoded.full_output,
        decoded.top_k,
    );
    assert_eq!(reencoded, encoded);
}

#[test]
fn request_i8_zero_seq_len_treated_as_one() {
    // Mirrors the i8 response decoder's seq_len 0 → 1 promotion.
    let row = vec![127.0f32, -64.0];
    let encoded = encode_binary_request_as(WireFormat::I8, Some(2), None, &row, 0, true, 0);
    let decoded = decode_binary_request_i8(&encoded).unwrap();
    assert_eq!(decoded.residual, row);
}

#[test]
fn request_f16_i8_decoders_share_header_guards() {
    // Truncated header / alloc-bomb guards run before any payload work
    // in every dtype arm (shared parse_request_header).
    for decode in [
        decode_binary_request_f16 as fn(&[u8]) -> Result<DecodedFfnRequest, String>,
        decode_binary_request_i8,
    ] {
        assert!(decode(&[]).is_err());
        assert!(decode(&[0u8; 8]).is_err());
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        buf.extend_from_slice(&[0u8; 12]);
        assert!(decode(&buf).is_err(), "alloc-bomb layer count");
    }
}

#[test]
fn request_f16_rejects_odd_payload_length() {
    let mut buf = encode_binary_request_as(WireFormat::F16, Some(0), None, &[1.0], 1, true, 0);
    buf.push(0u8);
    assert!(decode_binary_request_f16(&buf).is_err());
}

#[test]
fn request_i8_rejects_bad_payload_shapes() {
    // Payload not a multiple of seq_len.
    let mut two_pos = encode_binary_request_as(
        WireFormat::I8,
        Some(0),
        None,
        &[1.0, 2.0, 3.0, 4.0],
        2,
        true,
        0,
    );
    two_pos.push(0u8);
    assert!(decode_binary_request_i8(&two_pos).is_err());
    // Per-position bytes ≤ 8 (scale/zero header only, no data).
    let mut hdr_only = Vec::new();
    hdr_only.extend_from_slice(&0u32.to_le_bytes());
    hdr_only.extend_from_slice(&1u32.to_le_bytes());
    hdr_only.extend_from_slice(&1u32.to_le_bytes());
    hdr_only.extend_from_slice(&0u32.to_le_bytes());
    hdr_only.extend_from_slice(&[0u8; 8]); // exactly one empty position
    assert!(decode_binary_request_i8(&hdr_only).is_err());
    // Empty payload decodes to an empty residual (validated downstream).
    let empty = encode_binary_request_as(WireFormat::I8, Some(0), None, &[], 1, true, 0);
    assert_eq!(
        decode_binary_request_i8(&empty).unwrap().residual,
        Vec::<f32>::new()
    );
}

#[test]
fn decode_request_as_dispatches_every_format() {
    let residual = vec![0.5f32, -0.25, 1.5, -2.5];
    for fmt in [WireFormat::F32, WireFormat::F16, WireFormat::I8] {
        let body = encode_binary_request_as(fmt, Some(3), None, &residual, 1, true, 0);
        let decoded = decode_binary_request_as(fmt, &body).unwrap();
        assert_eq!(decoded.layer, Some(3));
        assert_eq!(decoded.residual.len(), residual.len());
    }
}
