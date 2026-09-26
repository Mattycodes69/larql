//! Asymmetric direction pairs (in/return independent — DEC-1A)
//! encode_json_full_output

use super::*;

#[test]
fn asymmetric_pairs_round_trip_through_production_codecs() {
    // The four asymmetric combos: request encoded in `input`, decoded by
    // the server-side request decoder; response encoded in `output`,
    // decoded by the client-side content-type dispatcher. Values are
    // exactly representable in every arm (f16-exact, i8 fixed-point
    // with power-of-two scales) so equality is exact end to end.
    let hidden = 4usize;
    let residual = vec![63.5f32, -32.0, 0.5, -63.5]; // i8 scale 0.5, f16-exact
    for (input, output) in [
        (WireFormat::F16, WireFormat::I8),
        (WireFormat::I8, WireFormat::F16),
        (WireFormat::F32, WireFormat::I8),
        (WireFormat::F16, WireFormat::F32),
    ] {
        // Inbound: client encodes in `input`, server decodes by CT.
        let req_body = encode_binary_request_as(input, Some(9), None, &residual, 1, true, 0);
        let fmt = WireFormat::from_content_type(input.content_type()).unwrap();
        assert_eq!(fmt, input, "CT ↔ format mapping is bijective");
        let req = decode_binary_request_as(fmt, &req_body).unwrap();
        assert_eq!(
            req.residual,
            residual,
            "{}/{}",
            input.label(),
            output.label()
        );

        // Return: server echoes the residual through the `output`
        // response encoder; client decodes via the CT dispatcher.
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 9,
                output: req.residual.clone(),
            }],
            seq_len: 1,
            latency_ms: 2.5,
        };
        let resp_body = match output {
            WireFormat::F32 => encode_binary_output(&out),
            WireFormat::F16 => encode_binary_output_f16(&out),
            WireFormat::I8 => encode_binary_output_i8(&out),
        };
        let (layer, server_ms, floats) =
            decode_single_response(output.content_type(), &resp_body, hidden).unwrap();
        assert_eq!(layer, 9);
        assert!((server_ms - 2.5).abs() < 1e-6);
        assert_eq!(floats, residual, "{}/{}", input.label(), output.label());
    }
}

#[test]
fn response_f32_encode_decode_reencode_is_byte_identical() {
    // Single-layer frame.
    let out = FfnOutput {
        entries: vec![FfnEntry {
            layer: 5,
            output: vec![0.12345, -9.87654, 1e-7, f32::MAX / 2.0],
        }],
        seq_len: 2,
        latency_ms: 7.5,
    };
    let encoded = encode_binary_output(&out);
    let (layer, floats) = decode_binary_single(&encoded).unwrap();
    let latency = extract_response_latency_ms(&encoded);
    let rebuilt = FfnOutput {
        entries: vec![FfnEntry {
            layer,
            output: floats,
        }],
        seq_len: 2,
        latency_ms: latency,
    };
    assert_eq!(encode_binary_output(&rebuilt), encoded);

    // Batch frame.
    let out = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 3,
                output: vec![1.5, -2.25],
            },
            FfnEntry {
                layer: 29,
                output: vec![-0.0, 7.0],
            },
        ],
        seq_len: 1,
        latency_ms: 15.0,
    };
    let encoded = encode_binary_output(&out);
    let map = decode_binary_batch(&encoded).unwrap();
    let rebuilt = rebuild_output(&map, &[3, 29], 1, extract_response_latency_ms(&encoded));
    assert_eq!(encode_binary_output(&rebuilt), encoded);
}

#[test]
fn response_f16_encode_decode_reencode_is_byte_identical() {
    // Values chosen to be exactly f16-representable so decode → f32 →
    // re-encode reproduces identical half bits.
    let out = FfnOutput {
        entries: vec![FfnEntry {
            layer: 7,
            output: vec![0.5, -0.25, 1.5, -2.5],
        }],
        seq_len: 1,
        latency_ms: 2.25,
    };
    let encoded = encode_binary_output_f16(&out);
    let (layer, floats) = decode_binary_single_f16(&encoded).unwrap();
    let rebuilt = FfnOutput {
        entries: vec![FfnEntry {
            layer,
            output: floats,
        }],
        seq_len: 1,
        latency_ms: extract_response_latency_ms(&encoded),
    };
    assert_eq!(encode_binary_output_f16(&rebuilt), encoded);

    // Batch frame.
    let out = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 3,
                output: vec![1.0, 2.0],
            },
            FfnEntry {
                layer: 11,
                output: vec![-1.0, 0.5],
            },
        ],
        seq_len: 1,
        latency_ms: 8.0,
    };
    let encoded = encode_binary_output_f16(&out);
    let map = decode_binary_batch_f16(&encoded).unwrap();
    let rebuilt = rebuild_output(&map, &[3, 11], 1, extract_response_latency_ms(&encoded));
    assert_eq!(encode_binary_output_f16(&rebuilt), encoded);
}

#[test]
fn response_i8_encode_decode_reencode_is_byte_identical() {
    // Fixed-point construction: per position, max|v| = 127 × scale with
    // scale a power of two, so quantise(dequantise(q)) == q and the
    // recomputed scale bit-matches. hidden = 4.
    let out = FfnOutput {
        entries: vec![FfnEntry {
            layer: 5,
            // scale = 63.5 / 127 = 0.5 exactly; q = [127, -64, 1, -127]
            output: vec![63.5, -32.0, 0.5, -63.5],
        }],
        seq_len: 1,
        latency_ms: 3.5,
    };
    let encoded = encode_binary_output_i8(&out);
    let (layer, floats) = decode_binary_single_i8(&encoded, 4).unwrap();
    let rebuilt = FfnOutput {
        entries: vec![FfnEntry {
            layer,
            output: floats,
        }],
        seq_len: 1,
        latency_ms: extract_response_latency_ms(&encoded),
    };
    assert_eq!(encode_binary_output_i8(&rebuilt), encoded);

    // Batch frame, two layers, seq_len 2 (two quantised positions each).
    let out = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 10,
                // pos0 scale 0.25, pos1 scale 1.0
                output: vec![31.75, -16.0, 0.25, 8.0, 127.0, -64.0, 1.0, -127.0],
            },
            FfnEntry {
                layer: 20,
                // pos0 scale 2.0, pos1 scale 0.5
                output: vec![254.0, -2.0, 128.0, 4.0, 63.5, -0.5, 32.0, -63.5],
            },
        ],
        seq_len: 2,
        latency_ms: 1.0,
    };
    let encoded = encode_binary_output_i8(&out);
    let map = decode_binary_batch_i8(&encoded, 4).unwrap();
    let rebuilt = rebuild_output(&map, &[10, 20], 2, extract_response_latency_ms(&encoded));
    assert_eq!(encode_binary_output_i8(&rebuilt), encoded);
}

#[test]
fn json_full_output_single_and_batch_shapes() {
    let single = FfnOutput {
        entries: vec![FfnEntry {
            layer: 7,
            output: vec![1.0, 2.0, 3.0],
        }],
        seq_len: 1,
        latency_ms: 4.24,
    };
    let v = encode_json_full_output(&single);
    assert_eq!(v["layer"].as_u64(), Some(7));
    assert!(v.get("results").is_none());
    assert_eq!(v["latency_ms"].as_f64(), Some(4.2)); // rounded to 0.1

    let batch = FfnOutput {
        entries: vec![
            FfnEntry {
                layer: 0,
                output: vec![1.0],
            },
            FfnEntry {
                layer: 1,
                output: vec![2.0],
            },
        ],
        seq_len: 2,
        latency_ms: 20.0,
    };
    let v = encode_json_full_output(&batch);
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[1]["layer"].as_u64(), Some(1));
    assert_eq!(results[1]["seq_len"].as_u64(), Some(2));
}
