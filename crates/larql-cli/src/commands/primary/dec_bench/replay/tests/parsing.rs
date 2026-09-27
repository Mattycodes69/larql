//! parsing
//! frames
//! summaries

use super::*;

#[test]
fn parse_wire_list_all_arms() {
    let arms = WireArm::parse_list("f32, f16,i8,q8k").unwrap();
    assert_eq!(
        arms,
        vec![WireArm::F32, WireArm::F16, WireArm::I8, WireArm::Q8k]
    );
    assert!(WireArm::parse_list("f64").is_err());
}

#[test]
fn parse_wire_spec_plain_arms_and_pairs() {
    let specs = WireSpec::parse_list("f32, f16/i8 ,i8/f16,q8k").unwrap();
    assert_eq!(
        specs,
        vec![
            WireSpec::Plain(WireArm::F32),
            WireSpec::Pair {
                input: WireFormat::F16,
                output: WireFormat::I8
            },
            WireSpec::Pair {
                input: WireFormat::I8,
                output: WireFormat::F16
            },
            WireSpec::Plain(WireArm::Q8k),
        ]
    );
    // All four asymmetric combos parse.
    for pair in ["f16/i8", "i8/f16", "f32/i8", "f16/f32"] {
        assert!(matches!(
            WireSpec::parse(pair).unwrap(),
            WireSpec::Pair { .. }
        ));
    }
}

#[test]
fn parse_wire_spec_rejects_malformed_pairs() {
    // Missing arms, extra segments, q8k in a pair, unknown dtypes.
    for bad in [
        "f16/", "/i8", "f16/i8/x", "q8k/f32", "f32/q8k", "f16//i8", "f64/i8", "/",
    ] {
        let err = WireSpec::parse(bad).unwrap_err();
        assert!(
            err.contains("invalid wire pair") || err.contains("unknown wire format"),
            "{bad:?} → {err}"
        );
    }
    // Malformed plain arm keeps the arm-level error.
    assert!(WireSpec::parse("f64").is_err());
}

#[test]
fn wire_spec_labels_codes_and_direction_labels() {
    let f16_i8 = WireSpec::Pair {
        input: WireFormat::F16,
        output: WireFormat::I8,
    };
    let i8_f16 = WireSpec::Pair {
        input: WireFormat::I8,
        output: WireFormat::F16,
    };
    assert_eq!(f16_i8.label(), "f16/i8");
    assert_eq!(i8_f16.label(), "i8/f16");
    // Pair codes: 100 + 10×in + out (f32=0, f16=1, i8=2).
    assert_eq!(f16_i8.code(), 112);
    assert_eq!(i8_f16.code(), 121);
    // Plain arms keep the historical labels and codes.
    assert_eq!(WireSpec::Plain(WireArm::F16).label(), "f16");
    assert_eq!(WireSpec::Plain(WireArm::Q8k).code(), 3);
    // Direction labels: plain arms actually send f32 request frames
    // (the historical wire), q8k rides its own endpoint both ways.
    assert_eq!(WireSpec::Plain(WireArm::F16).in_label(), "f32");
    assert_eq!(WireSpec::Plain(WireArm::F16).out_label(), "f16");
    assert_eq!(WireSpec::Plain(WireArm::F32).in_label(), "f32");
    assert_eq!(WireSpec::Plain(WireArm::Q8k).in_label(), "q8k");
    assert_eq!(WireSpec::Plain(WireArm::Q8k).out_label(), "q8k");
    assert_eq!(f16_i8.in_label(), "f16");
    assert_eq!(f16_i8.out_label(), "i8");
}

#[test]
fn wire_spec_accept_and_request_format() {
    // Plain arms: unchanged Accept, f32 request frames.
    assert_eq!(WireSpec::Plain(WireArm::I8).accept(), WireArm::I8.accept());
    assert_eq!(
        WireSpec::Plain(WireArm::I8).request_format(),
        WireFormat::F32
    );
    assert!(WireSpec::Plain(WireArm::Q8k).is_q8k());
    // Pairs: strict Accept = return arm only; request frame = input arm.
    let pair = WireSpec::Pair {
        input: WireFormat::I8,
        output: WireFormat::F16,
    };
    assert_eq!(pair.accept(), Some(larql_inference::F16_CT));
    assert_eq!(pair.request_format(), WireFormat::I8);
    assert!(!pair.is_q8k());
}

#[test]
fn wire_arm_labels_codes_accepts() {
    assert_eq!(WireArm::F32.label(), "f32");
    assert_eq!(WireArm::Q8k.code(), 3);
    assert_eq!(WireArm::F32.accept(), Some("application/x-larql-ffn"));
    assert_eq!(WireArm::F16.accept(), Some("application/x-larql-ffn-f16"));
    assert_eq!(WireArm::I8.accept(), Some("application/x-larql-ffn-i8"));
    assert_eq!(WireArm::Q8k.accept(), None, "q8k is its own endpoint");
}

#[test]
fn parse_dispatch_and_batch_lists() {
    let d = DispatchMode::parse_list("streaming,batch").unwrap();
    assert_eq!(d, vec![DispatchMode::Streaming, DispatchMode::Batch]);
    assert!(DispatchMode::parse_list("parallel").is_err());

    assert_eq!(parse_batch_list("1,8,16").unwrap(), vec![1, 8, 16]);
    assert!(parse_batch_list("0").is_err());
    assert!(parse_batch_list("x").is_err());
}

#[test]
fn expand_sweep_is_batch_major_cross_product() {
    let points = expand_sweep(
        &[1, 8],
        &[WireSpec::Plain(WireArm::F32), WireSpec::Plain(WireArm::F16)],
        &[DispatchMode::Streaming],
        EndpointKind::WalkFfn,
    )
    .unwrap();
    assert_eq!(points.len(), 4);
    assert_eq!(points[0].batch, 1);
    assert_eq!(points[1].batch, 1);
    assert_eq!(points[2].batch, 8);
    assert_eq!(points[1].wire, WireSpec::Plain(WireArm::F16));
    assert!(points.iter().all(|p| p.endpoint == Endpoint::WalkFfn));
}

#[test]
fn expand_sweep_resolves_endpoints_and_rejects_unservable_wires() {
    let points = expand_sweep(
        &[1],
        &[WireSpec::Plain(WireArm::F32), WireSpec::Plain(WireArm::Q8k)],
        &[DispatchMode::Streaming],
        EndpointKind::Experts,
    )
    .unwrap();
    assert_eq!(points[0].endpoint, Endpoint::ExpertsMultiLayer);
    assert_eq!(points[1].endpoint, Endpoint::ExpertsMultiLayerQ8k);

    let q8k_points = expand_sweep(
        &[1],
        &[WireSpec::Plain(WireArm::Q8k)],
        &[DispatchMode::Streaming],
        EndpointKind::WalkFfn,
    )
    .unwrap();
    assert_eq!(q8k_points[0].endpoint, Endpoint::WalkFfnQ8k);

    // f16/i8 on experts must fail at expansion (arg-validation) time.
    for wire in [WireArm::F16, WireArm::I8] {
        let err = expand_sweep(
            &[1],
            &[WireSpec::Plain(WireArm::F32), WireSpec::Plain(wire)],
            &[DispatchMode::Streaming],
            EndpointKind::Experts,
        )
        .unwrap_err();
        assert!(err.contains(wire.label()), "error names the bad arm: {err}");
    }
}

#[test]
fn expand_sweep_pairs_resolve_on_walk_ffn_and_reject_on_experts() {
    let pair = WireSpec::Pair {
        input: WireFormat::F16,
        output: WireFormat::I8,
    };
    // Pairs ride the dense walk-ffn endpoint.
    let points = expand_sweep(
        &[1],
        &[pair],
        &[DispatchMode::Streaming],
        EndpointKind::WalkFfn,
    )
    .unwrap();
    assert_eq!(points[0].endpoint, Endpoint::WalkFfn);
    assert_eq!(points[0].wire, pair);

    // Pairs on experts fail at expansion (arg-validation) time, before
    // any request is sent — same discipline as f16/i8 plain arms.
    let err = expand_sweep(
        &[1],
        &[pair],
        &[DispatchMode::Streaming],
        EndpointKind::Experts,
    )
    .unwrap_err();
    assert!(err.contains("f16/i8"), "error names the pair: {err}");
    assert!(err.contains("walk-ffn"), "error points at the fix: {err}");
}

#[test]
fn walk_ffn_frame_encodes_batch_rows_and_zero_top_k() {
    let hidden = 4;
    let batch = 3;
    let rows: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.5).collect();
    let frame = build_walk_ffn_frame(7, &rows, batch);
    // Header: layer, seq_len, flags, top_k — then batch×hidden f32s.
    assert_eq!(frame.len(), 16 + batch * hidden * 4);
    assert_eq!(u32::from_le_bytes(frame[0..4].try_into().unwrap()), 7);
    assert_eq!(
        u32::from_le_bytes(frame[4..8].try_into().unwrap()) as usize,
        batch
    );
    assert_eq!(
        u32::from_le_bytes(frame[12..16].try_into().unwrap()),
        0,
        "top_k must be 0 — non-zero engages the server L2 cache at seq_len==1"
    );
    // Row 1's first float lands after row 0.
    let v = f32::from_le_bytes(frame[16 + hidden * 4..20 + hidden * 4].try_into().unwrap());
    assert_eq!(v, hidden as f32 * 0.5);
}

#[test]
fn q8k_frame_carries_one_entry_per_row_same_layer() {
    let hidden = 256; // one Q8K superblock
    let batch = 3;
    let rows: Vec<f32> = (0..batch * hidden)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let frame = build_q8k_frame(9, &rows, batch, hidden);
    let entries = larql_inference::ffn::remote::decode_q8k_batch_request(&frame).unwrap();
    assert_eq!(entries.len(), batch);
    assert!(entries.iter().all(|e| e.layer_idx == 9));
    // Each entry must be that row's quantisation, not row 0 repeated.
    let q1 = quantize_x_to_q8k(&rows[hidden..2 * hidden]);
    assert_eq!(entries[1].q8k.qs, q1.qs);
    assert_eq!(entries[1].q8k.d, q1.d);
}

#[test]
fn summarize_computes_throughput_payload_and_per_layer() {
    let point = SweepPoint {
        batch: 8,
        wire: WireSpec::Plain(WireArm::F16),
        dispatch: DispatchMode::Batch,
        endpoint: Endpoint::WalkFfn,
    };
    let stats = SweepPointStats {
        step_ms: vec![10.0, 20.0],
        samples: vec![
            sample(0, 4.0, Some(3.0)),
            sample(1, 6.0, Some(5.0)),
            sample(0, 8.0, Some(6.0)),
            sample(1, 12.0, Some(9.0)),
        ],
    };
    let s = summarize(&point, &stats, None, None);
    assert_eq!(s.batch, 8);
    assert_eq!(s.endpoint, "walk-ffn");
    assert_eq!(s.endpoint_code, 0);
    assert_eq!(s.wire_format, "f16");
    assert_eq!(s.dispatch_mode, "batch");
    assert_eq!(s.steps, 2);
    assert!((s.step_ms_mean - 15.0).abs() < 1e-9);
    // tok_s = 8 rows × 1000 / 15 ms
    assert!((s.tok_s - 8.0 * 1000.0 / 15.0).abs() < 1e-6);
    // payload = 4 samples × 160 bytes / (2 steps × 8 rows)
    assert!((s.payload_bytes_tok - (4.0 * 160.0) / 16.0).abs() < 1e-9);
    assert!(s.server_ms_p50.is_some());
    assert_eq!(s.per_layer.len(), 2);
    assert_eq!(s.per_layer[0].layer, 0);
    assert!(s.per_layer[1].client_ms_p99 >= s.per_layer[1].client_ms_p50);
}

#[test]
fn summarize_handles_empty_and_no_server_ms() {
    let point = SweepPoint {
        batch: 1,
        wire: WireSpec::Plain(WireArm::Q8k),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfnQ8k,
    };
    let s = summarize(&point, &SweepPointStats::default(), None, None);
    assert_eq!(s.tok_s, 0.0);
    assert_eq!(s.payload_bytes_tok, 0.0);
    assert!(s.server_ms_p50.is_none());

    let stats = SweepPointStats {
        step_ms: vec![5.0],
        samples: vec![sample(0, 5.0, None)],
    };
    let s = summarize(&point, &stats, None, None);
    assert!(s.server_ms_p50.is_none(), "q8k has no embedded latency");
}
