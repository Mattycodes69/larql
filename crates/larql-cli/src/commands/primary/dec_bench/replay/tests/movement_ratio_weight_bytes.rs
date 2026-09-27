//! movement ratio + weight bytes

use super::*;

#[test]
fn summarize_reports_served_wire_including_fallback() {
    // An i8-requested arm that the server served as f32 must say so —
    // the bandwidth number belongs to what was actually on the wire.
    let point = SweepPoint {
        batch: 1,
        wire: WireSpec::Plain(WireArm::I8),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfn,
    };
    let mut s0 = sample(0, 1.0, Some(0.5));
    s0.served_wire_out = "f32".into();
    let mut s1 = sample(1, 1.0, Some(0.5));
    s1.served_wire_out = "f32".into();
    let stats = SweepPointStats {
        step_ms: vec![2.0],
        samples: vec![s0, s1],
    };
    let s = summarize(&point, &stats, None, None);
    assert_eq!(s.wire_format, "i8");
    assert_eq!(s.wire_in, "f32");
    assert_eq!(s.wire_out, "i8");
    assert_eq!(s.served_wire_in, vec!["f32".to_string()]);
    assert_eq!(s.served_wire_out, vec!["f32".to_string()]);
}

#[test]
fn summarize_pair_point_records_directions_and_split_bytes() {
    let point = SweepPoint {
        batch: 2,
        wire: WireSpec::Pair {
            input: WireFormat::F16,
            output: WireFormat::I8,
        },
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfn,
    };
    let mut s0 = sample(0, 1.0, Some(0.5));
    s0.served_wire_in = "f16".into();
    s0.served_wire_out = "i8".into();
    s0.bytes_sent = 300;
    s0.bytes_recv = 100;
    let mut s1 = s0.clone();
    s1.bytes_sent = 100;
    s1.bytes_recv = 60;
    let stats = SweepPointStats {
        step_ms: vec![2.0, 2.0],
        samples: vec![s0, s1],
    };
    let s = summarize(&point, &stats, None, None);
    assert_eq!(s.wire_format, "f16/i8");
    assert_eq!(s.wire_format_code, 112);
    assert_eq!(s.wire_in, "f16");
    assert_eq!(s.wire_out, "i8");
    assert_eq!(s.served_wire_in, vec!["f16".to_string()]);
    assert_eq!(s.served_wire_out, vec!["i8".to_string()]);
    // Per-direction byte accounting: 4 token rows (2 steps × batch 2).
    assert_eq!(s.payload_bytes_tok_in, 400.0 / 4.0);
    assert_eq!(s.payload_bytes_tok_out, 160.0 / 4.0);
    assert_eq!(
        s.payload_bytes_tok,
        s.payload_bytes_tok_in + s.payload_bytes_tok_out
    );
    // The run record carries the split fields.
    let json = serde_json::to_value(&s).unwrap();
    assert_eq!(json["wire_in"], "f16");
    assert_eq!(json["wire_out"], "i8");
    assert!(json["payload_bytes_tok_in"].is_number());
    assert!(json["payload_bytes_tok_out"].is_number());
    assert_eq!(json["served_wire_out"][0], "i8");
}

#[test]
fn wire_label_maps_known_cts_and_passes_through_unknown() {
    assert_eq!(
        wire_label_for_content_type("application/x-larql-ffn"),
        "f32"
    );
    assert_eq!(
        wire_label_for_content_type("application/x-larql-ffn-f16"),
        "f16"
    );
    assert_eq!(
        wire_label_for_content_type("application/x-larql-ffn-i8"),
        "i8"
    );
    assert_eq!(
        wire_label_for_content_type("application/x-larql-ffn-q8k-batch"),
        "q8k"
    );
    assert_eq!(wire_label_for_content_type("text/plain"), "text/plain");
}

#[test]
fn movement_ratio_basic_and_guards() {
    assert_eq!(movement_ratio(1.0, 1000.0), Some(0.001));
    assert_eq!(movement_ratio(1.0, 0.0), None);
    assert_eq!(movement_ratio(-1.0, 10.0), None);
}

#[test]
fn weight_bytes_per_token_sums_selected_layers() {
    let stats = serde_json::json!({
        "ffn_weights": {
            "per_layer_dense_bytes": [100, 200, null, 400],
        }
    });
    let (bytes, missing) = weight_bytes_per_token(&stats, &[0, 1, 2, 3]).unwrap();
    assert_eq!(bytes, 700.0);
    assert_eq!(missing, 1);

    let (bytes, missing) = weight_bytes_per_token(&stats, &[1]).unwrap();
    assert_eq!(bytes, 200.0);
    assert_eq!(missing, 0);

    assert!(weight_bytes_per_token(&serde_json::json!({}), &[0]).is_none());
}

#[test]
fn parse_layer_range_default_and_bounds() {
    assert_eq!(parse_layer_range(None, 3).unwrap(), vec![0, 1, 2]);
    assert_eq!(parse_layer_range(Some("1-2"), 4).unwrap(), vec![1, 2]);
    assert!(parse_layer_range(Some("2-1"), 4).is_err());
    assert!(parse_layer_range(Some("0-4"), 4).is_err());
    assert!(parse_layer_range(Some("x"), 4).is_err());
}
