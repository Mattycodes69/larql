//! decode_single_response (content-type dispatch)
//! decode_binary_request (server-side half)
//! Byte-identical wire pins (encode → decode → re-encode)
//! WireFormat (direction axis)

use super::*;

#[test]
fn decode_single_response_dispatches_f32() {
    let output = vec![1.0f32, -2.0, 3.5, 0.25];
    let body = make_single_response(3, 2, 4.5, &output);
    let (layer, server_ms, floats) = decode_single_response(BINARY_CT, &body, 2).unwrap();
    assert_eq!(layer, 3);
    assert!((server_ms - 4.5).abs() < 1e-6);
    assert_eq!(floats, output);
}

#[test]
fn decode_single_response_dispatches_f16() {
    let body = make_single_response_f16(7, 2, 2.25, &[0.5, -0.25, 1.5, -2.5]);
    let (layer, server_ms, floats) = decode_single_response(F16_CT, &body, 2).unwrap();
    assert_eq!(layer, 7);
    assert!((server_ms - 2.25).abs() < 1e-6);
    assert_eq!(floats, vec![0.5, -0.25, 1.5, -2.5]);
}

#[test]
fn decode_single_response_dispatches_i8_multi_row() {
    // seq_len = 3 rows of hidden = 2: the B>1 replay shape.
    let body = make_single_response_i8(
        0,
        3,
        1.5,
        &[(1.0, &[10i8, 20]), (0.25, &[-4i8, 8]), (2.0, &[1i8, -1])],
    );
    let (layer, server_ms, floats) = decode_single_response(I8_CT, &body, 2).unwrap();
    assert_eq!(layer, 0);
    assert!((server_ms - 1.5).abs() < 1e-6);
    assert_eq!(floats, vec![10.0, 20.0, -1.0, 2.0, 2.0, -2.0]);
}

#[test]
fn decode_single_response_rejects_unknown_content_type() {
    let body = make_single_response(0, 1, 0.0, &[1.0]);
    assert!(decode_single_response("application/json", &body, 1).is_err());
}

#[test]
fn encode_binary_request_top_k_zero_pins_l2_cache_bypass() {
    // The server's FfnL2Cache only engages when seq_len==1 && top_k>0.
    // Replay frames encode top_k=0 so repeated B=1 requests measure
    // real FFN compute, not cache hits. Pin the byte position.
    let residual = vec![0.5f32; 4];
    let body = encode_binary_request(Some(3), None, &residual, 1, true, 0);
    let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
    assert_eq!(top_k, 0);
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(seq_len, 1);
}

#[test]
fn encode_binary_request_multi_row_seq_len() {
    // B-row replay frame: residual = B × hidden floats, seq_len = B.
    let hidden = 4;
    let batch = 8;
    let rows: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.1).collect();
    let body = encode_binary_request(Some(0), None, &rows, batch, true, 0);
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(seq_len as usize, batch);
    assert_eq!(body.len(), 16 + batch * hidden * 4);
}

#[test]
fn binary_request_response_roundtrip() {
    // Encode a single-layer request, then simulate what the server echoes.
    let residual = vec![0.1f32, 0.2, 0.3, 0.4];
    let req = encode_binary_request(Some(5), None, &residual, 1, true, 8092);
    // Simulate server extracting the layer.
    let layer = u32::from_le_bytes(req[0..4].try_into().unwrap());
    assert_eq!(layer, 5);

    // Simulate server response.
    let output = vec![0.9f32, 0.8, 0.7, 0.6];
    let resp = make_single_response(layer, 1, 8.5, &output);
    let (resp_layer, floats) = decode_binary_single(&resp).unwrap();
    assert_eq!(resp_layer as u32, layer);
    assert_eq!(floats, output);
}

#[test]
fn decode_request_single_layer() {
    let body = encode_binary_request(Some(5), None, &[1.0, 2.0, 3.0, 4.0], 1, true, 8);
    let req = decode_binary_request(&body).unwrap();
    assert_eq!(req.layer, Some(5));
    assert!(req.layers.is_none());
    assert_eq!(req.seq_len, 1);
    assert_eq!(req.top_k, 8);
    assert!(req.full_output);
    assert_eq!(req.residual, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn decode_request_batch() {
    let body = encode_binary_request(None, Some(&[0, 1, 2]), &[1.0; 4], 1, true, 16);
    let req = decode_binary_request(&body).unwrap();
    assert!(req.layer.is_none());
    assert_eq!(req.layers, Some(vec![0, 1, 2]));
    assert_eq!(req.top_k, 16);
}

#[test]
fn decode_request_features_only_flag() {
    let body = encode_binary_request(Some(0), None, &[1.0; 4], 1, false, 8);
    let req = decode_binary_request(&body).unwrap();
    assert!(!req.full_output);
}

#[test]
fn decode_request_truncated_body_errors() {
    assert!(decode_binary_request(&[0u8; 8]).is_err());
    assert!(decode_binary_request(&[]).is_err());
}

#[test]
fn decode_request_batch_truncated_layers_errors() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes()); // claim 4 layers
    buf.extend_from_slice(&0u32.to_le_bytes()); // only 1
    buf.extend_from_slice(&[0u8; 4]);
    assert!(decode_binary_request(&buf).is_err());
}

#[test]
fn decode_request_batch_impossible_layer_count_errors() {
    // num_layers = u32::MAX must be rejected by a length guard before
    // any layer-index allocation (alloc-bomb bound, PR 104 lineage).
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    buf.extend_from_slice(&[0u8; 12]);
    assert!(decode_binary_request(&buf).is_err());
}

#[test]
fn decode_request_odd_residual_length_errors() {
    let mut buf = encode_binary_request(Some(0), None, &[1.0], 1, true, 8);
    buf.push(0u8); // 1 stray byte — residual not a multiple of 4
    assert!(decode_binary_request(&buf).is_err());
}

#[test]
fn request_encode_decode_reencode_is_byte_identical() {
    // Single-layer frame.
    let encoded = encode_binary_request(
        Some(7),
        None,
        &[1.5, -2.25, 0.0, f32::MIN_POSITIVE],
        2,
        true,
        8092,
    );
    let decoded = decode_binary_request(&encoded).unwrap();
    assert_eq!(reencode_request(&decoded), encoded);

    // Batch frame.
    let encoded =
        encode_binary_request(None, Some(&[3, 900, 41]), &[0.125, -3.5, 1e-20], 1, true, 0);
    let decoded = decode_binary_request(&encoded).unwrap();
    assert_eq!(reencode_request(&decoded), encoded);
}

#[test]
fn wire_format_content_types_labels_and_parse() {
    assert_eq!(WireFormat::F32.content_type(), BINARY_CT);
    assert_eq!(WireFormat::F16.content_type(), F16_CT);
    assert_eq!(WireFormat::I8.content_type(), I8_CT);
    for f in [WireFormat::F32, WireFormat::F16, WireFormat::I8] {
        assert_eq!(WireFormat::parse(f.label()), Some(f));
    }
    assert_eq!(WireFormat::parse(" f16 "), Some(WireFormat::F16));
    assert_eq!(WireFormat::parse("q8k"), None);
    assert_eq!(WireFormat::default(), WireFormat::F32);
}

#[test]
fn wire_format_from_content_type_checks_suffixed_types_first() {
    // BINARY_CT is a substring of F16_CT/I8_CT — the ordered match must
    // never misread a compressed body as f32.
    assert_eq!(WireFormat::from_content_type(F16_CT), Some(WireFormat::F16));
    assert_eq!(WireFormat::from_content_type(I8_CT), Some(WireFormat::I8));
    assert_eq!(
        WireFormat::from_content_type(BINARY_CT),
        Some(WireFormat::F32)
    );
    // Parameterised types still match.
    assert_eq!(
        WireFormat::from_content_type("application/x-larql-ffn-f16; v=2"),
        Some(WireFormat::F16)
    );
    assert_eq!(WireFormat::from_content_type("application/json"), None);
}
