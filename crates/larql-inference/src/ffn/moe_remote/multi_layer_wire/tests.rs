use super::*;

#[test]
fn request_round_trip() {
    let tasks = vec![
        MultiLayerTask {
            layer: 0,
            residual: vec![1.0, 2.0, 3.0],
            expert_ids: vec![5, 17],
            weights: vec![0.6, 0.4],
        },
        MultiLayerTask {
            layer: 7,
            residual: vec![0.5, -1.0, 2.5],
            expert_ids: vec![42],
            weights: vec![1.0],
        },
    ];
    let encoded = encode_multi_layer_request(&tasks);
    let decoded = decode_multi_layer_request(&encoded).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].layer, 0);
    assert_eq!(decoded[0].residual, vec![1.0, 2.0, 3.0]);
    assert_eq!(decoded[0].expert_ids, vec![5, 17]);
    assert_eq!(decoded[0].weights, vec![0.6, 0.4]);
    assert_eq!(decoded[1].layer, 7);
    assert_eq!(decoded[1].expert_ids, vec![42]);
}

#[test]
fn response_round_trip() {
    let results = vec![
        MultiLayerResult {
            layer: 3,
            h2: vec![0.1, 0.2, 0.3],
        },
        MultiLayerResult {
            layer: 15,
            h2: vec![-1.0, 0.0, 1.0],
        },
    ];
    let encoded = encode_multi_layer_response(&results);
    let decoded = decode_multi_layer_response(&encoded).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].layer, 3);
    assert_eq!(decoded[0].h2, vec![0.1, 0.2, 0.3]);
    assert_eq!(decoded[1].layer, 15);
}

#[test]
fn handles_truncation() {
    assert!(decode_multi_layer_request(&[]).is_none());
    assert!(decode_multi_layer_request(&[0, 0, 0, 1]).is_none()); // claims 1 task but no body
    assert!(decode_multi_layer_response(&[]).is_none());
}

#[test]
fn empty_request_round_trips_to_zero_tasks() {
    let encoded = encode_multi_layer_request(&[]);
    // Just the [u32 num_tasks=0] header.
    assert_eq!(encoded.len(), 4);
    let decoded = decode_multi_layer_request(&encoded).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn empty_response_round_trips_to_zero_results() {
    let encoded = encode_multi_layer_response(&[]);
    assert_eq!(encoded.len(), 4);
    let decoded = decode_multi_layer_response(&encoded).unwrap();
    assert!(decoded.is_empty());
}

// ── Allocation-bomb regressions (docs/audits/dec-readiness-review-2026-07-22.md §2a) ──

#[test]
fn decode_multi_layer_request_rejects_impossible_task_count_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // num_tasks
    assert!(decode_multi_layer_request(&body).is_none());
}

#[test]
fn decode_multi_layer_request_rejects_impossible_hidden_before_allocating() {
    // [n=1][layer=0][hidden=0xFFFFFFFF][num_experts=0] — 16 bytes total.
    // Before the fix, `hidden` reached `read_f32_slice`'s
    // `Vec::with_capacity` unbounded (~17 GB reservation → SIGABRT).
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_tasks
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // hidden
    body.extend_from_slice(&0u32.to_le_bytes()); // num_experts
    assert!(decode_multi_layer_request(&body).is_none());
}

#[test]
fn decode_multi_layer_request_rejects_impossible_expert_count_before_allocating() {
    // A well-formed zero-length residual, then an attacker-controlled
    // `num_experts` far beyond what the remaining body could hold.
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_tasks
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&0u32.to_le_bytes()); // hidden
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // num_experts
    assert!(decode_multi_layer_request(&body).is_none());
}

#[test]
fn decode_multi_layer_response_rejects_impossible_result_count_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // num_results
    assert!(decode_multi_layer_response(&body).is_none());
}

#[test]
fn decode_multi_layer_response_rejects_impossible_hidden_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_results
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // hidden
    assert!(decode_multi_layer_response(&body).is_none());
}

#[test]
fn decode_multi_layer_request_q8k_rejects_impossible_task_count_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // num_tasks
    assert!(decode_multi_layer_request_q8k(&body).is_none());
}

#[test]
fn decode_multi_layer_request_q8k_rejects_impossible_hidden_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_tasks
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // hidden
    body.extend_from_slice(&0u32.to_le_bytes()); // num_experts
    assert!(decode_multi_layer_request_q8k(&body).is_none());
}

#[test]
fn decode_multi_layer_request_q8k_rejects_impossible_expert_count_before_allocating() {
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_tasks
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&0u32.to_le_bytes()); // hidden (0 blocks)
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // num_experts
    assert!(decode_multi_layer_request_q8k(&body).is_none());
}

#[test]
fn request_with_zero_experts_round_trips() {
    // Skip-vote pattern: a layer that routed nothing — encoder must
    // still emit the header so the layer index isn't lost.
    let tasks = vec![MultiLayerTask {
        layer: 9,
        residual: vec![0.0, 0.0, 0.0, 0.0],
        expert_ids: vec![],
        weights: vec![],
    }];
    let encoded = encode_multi_layer_request(&tasks);
    let decoded = decode_multi_layer_request(&encoded).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].layer, 9);
    assert_eq!(decoded[0].residual.len(), 4);
    assert!(decoded[0].expert_ids.is_empty());
    assert!(decoded[0].weights.is_empty());
}

#[test]
fn truncated_response_returns_none() {
    let encoded = encode_multi_layer_response(&[MultiLayerResult {
        layer: 1,
        h2: vec![1.0; 4],
    }]);
    // Drop the last byte → truncated f32; decoder must reject.
    assert!(decode_multi_layer_response(&encoded[..encoded.len() - 1]).is_none());
}

#[test]
fn truncated_request_returns_none_at_each_field() {
    let encoded = encode_multi_layer_request(&[MultiLayerTask {
        layer: 0,
        residual: vec![1.0, 2.0],
        expert_ids: vec![0],
        weights: vec![1.0],
    }]);
    for cut in 1..encoded.len() {
        // Every prefix shorter than the full encoding must be rejected
        // — there's no valid framing to recover from.
        assert!(
            decode_multi_layer_request(&encoded[..cut]).is_none(),
            "decode succeeded on prefix len={cut}, expected None"
        );
    }
}

// ── Q8K-prenormed wire ──────────────────────────────────────────────

fn make_q8k_task(layer: usize, hidden: usize, ne: usize) -> MultiLayerTaskQ8K {
    let nb = hidden / ELEMS_PER_Q8K_BLOCK;
    MultiLayerTaskQ8K {
        layer,
        hidden,
        qs: (0..hidden)
            .map(|i| ((i % 256) as i32 - 128) as i8)
            .collect(),
        d: (0..nb).map(|i| 0.01 * (i as f32 + 1.0)).collect(),
        sums: (0..nb * SUMS_PER_Q8K_BLOCK)
            .map(|i| (i as i16) - 64)
            .collect(),
        expert_ids: (0..ne).map(|i| (i as u32) * 17).collect(),
        weights: (0..ne)
            .map(|i| 1.0 / ne.max(1) as f32 * (i as f32 + 1.0))
            .collect(),
    }
}

#[test]
fn q8k_request_round_trip_single_block() {
    let tasks = vec![make_q8k_task(3, ELEMS_PER_Q8K_BLOCK, 4)];
    let encoded = encode_multi_layer_request_q8k(&tasks);
    let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
    assert_eq!(decoded.len(), 1);
    let t = &decoded[0];
    assert_eq!(t.layer, 3);
    assert_eq!(t.hidden, ELEMS_PER_Q8K_BLOCK);
    assert_eq!(t.qs, tasks[0].qs);
    assert_eq!(t.d, tasks[0].d);
    assert_eq!(t.sums, tasks[0].sums);
    assert_eq!(t.expert_ids, tasks[0].expert_ids);
    assert_eq!(t.weights, tasks[0].weights);
}

#[test]
fn q8k_request_round_trip_multi_block_multi_task() {
    // Two tasks, different hidden sizes → both nb counts must be
    // independently respected by the decoder.
    let tasks = vec![
        make_q8k_task(0, ELEMS_PER_Q8K_BLOCK, 2),
        make_q8k_task(11, ELEMS_PER_Q8K_BLOCK * 3, 8),
    ];
    let encoded = encode_multi_layer_request_q8k(&tasks);
    let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
    assert_eq!(decoded.len(), 2);
    for (orig, got) in tasks.iter().zip(decoded.iter()) {
        assert_eq!(orig.layer, got.layer);
        assert_eq!(orig.hidden, got.hidden);
        assert_eq!(orig.qs, got.qs);
        assert_eq!(orig.d, got.d);
        assert_eq!(orig.sums, got.sums);
        assert_eq!(orig.expert_ids, got.expert_ids);
        assert_eq!(orig.weights, got.weights);
    }
}

#[test]
fn decode_multi_layer_request_q8k_rejects_non_block_aligned_hidden() {
    // hidden=300 is not a multiple of 256: the pre-fix decoder floored
    // nb = 300/256 = 1 and then read qs/d/sums at desynced offsets —
    // a silently corrupt decode. Must be rejected outright.
    let mut body = Vec::new();
    body.extend_from_slice(&1u32.to_le_bytes()); // num_tasks
    body.extend_from_slice(&0u32.to_le_bytes()); // layer
    body.extend_from_slice(&300u32.to_le_bytes()); // hidden (misaligned)
    body.extend_from_slice(&0u32.to_le_bytes()); // num_experts
    body.extend_from_slice(&vec![0u8; 300 + 4 + 16]); // qs + d + sums payload
    assert!(decode_multi_layer_request_q8k(&body).is_none());
}

#[test]
fn q8k_request_with_zero_experts_round_trips() {
    let tasks = vec![make_q8k_task(2, ELEMS_PER_Q8K_BLOCK, 0)];
    let encoded = encode_multi_layer_request_q8k(&tasks);
    let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
    assert_eq!(decoded.len(), 1);
    assert!(decoded[0].expert_ids.is_empty());
    assert!(decoded[0].weights.is_empty());
    // Activation payload still present.
    assert_eq!(decoded[0].qs.len(), ELEMS_PER_Q8K_BLOCK);
}

#[test]
fn empty_q8k_request_round_trips() {
    let encoded = encode_multi_layer_request_q8k(&[]);
    assert_eq!(encoded.len(), 4);
    let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn truncated_q8k_request_returns_none_at_each_field() {
    let encoded = encode_multi_layer_request_q8k(&[make_q8k_task(0, ELEMS_PER_Q8K_BLOCK, 1)]);
    for cut in 1..encoded.len() {
        assert!(
            decode_multi_layer_request_q8k(&encoded[..cut]).is_none(),
            "Q8K decode succeeded on prefix len={cut}, expected None"
        );
    }
}

#[test]
fn multi_task_encode_decode_reencode_is_byte_identical() {
    // Pins the bulk-copy encoders/decoders to the exact wire layout:
    // encode → decode → re-encode must reproduce the original bytes
    // for every frame kind, on a multi-task frame.
    let tasks = vec![
        MultiLayerTask {
            layer: 2,
            residual: vec![1.5, -2.25, 0.0, f32::MIN_POSITIVE],
            expert_ids: vec![3, 900, 41],
            weights: vec![0.25, 0.5, 0.25],
        },
        MultiLayerTask {
            layer: 29,
            residual: vec![-0.0, 7.0],
            expert_ids: vec![],
            weights: vec![],
        },
    ];
    let encoded = encode_multi_layer_request(&tasks);
    let decoded = decode_multi_layer_request(&encoded).unwrap();
    assert_eq!(encode_multi_layer_request(&decoded), encoded);

    let q8k_tasks = vec![
        make_q8k_task(0, ELEMS_PER_Q8K_BLOCK, 2),
        make_q8k_task(11, ELEMS_PER_Q8K_BLOCK * 2, 5),
    ];
    let encoded = encode_multi_layer_request_q8k(&q8k_tasks);
    let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
    assert_eq!(encode_multi_layer_request_q8k(&decoded), encoded);

    let results = vec![
        MultiLayerResult {
            layer: 4,
            h2: vec![0.125, -3.5, 1e-20],
        },
        MultiLayerResult {
            layer: 17,
            h2: vec![f32::MAX],
        },
    ];
    let encoded = encode_multi_layer_response(&results);
    let decoded = decode_multi_layer_response(&encoded).unwrap();
    assert_eq!(encode_multi_layer_response(&decoded), encoded);
}

// ── Timing-trailer extension (DEC-1A two-scoreboard schema) ──────────

use crate::ffn::remote::timing::{append_timing_trailer, split_timing_trailer};

#[test]
fn extended_response_splits_then_decodes_identically() {
    let results = vec![
        MultiLayerResult {
            layer: 3,
            h2: vec![0.1, 0.2, 0.3],
        },
        MultiLayerResult {
            layer: 15,
            h2: vec![-1.0, 0.0, 1.0],
        },
    ];
    let plain = encode_multi_layer_response(&results);
    let mut extended = plain.clone();
    append_timing_trailer(&mut extended, 777.0);
    assert_eq!(extended.len(), plain.len() + 8);

    let (payload, serve_us) = split_timing_trailer(&extended);
    assert_eq!(payload, plain.as_slice());
    assert_eq!(serve_us, Some(777.0));
    let decoded = decode_multi_layer_response(payload).unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[0].layer, 3);
    assert_eq!(decoded[1].h2, vec![-1.0, 0.0, 1.0]);
}

#[test]
fn non_timing_decoder_tolerates_extended_response() {
    // A non-timing-aware decoder fed the extended frame must still
    // work: it reads exactly `num_results` results and ignores
    // trailing bytes by design.
    let results = vec![MultiLayerResult {
        layer: 1,
        h2: vec![2.0, -2.0],
    }];
    let mut extended = encode_multi_layer_response(&results);
    append_timing_trailer(&mut extended, 3.5);
    let decoded = decode_multi_layer_response(&extended).unwrap();
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].layer, 1);
    assert_eq!(decoded[0].h2, vec![2.0, -2.0]);
}

#[test]
fn plain_response_split_returns_none_and_full_payload() {
    // Old-server path: an all-zero h2 tail must not look like a
    // trailer (magic can't be zero).
    let results = vec![MultiLayerResult {
        layer: 0,
        h2: vec![0.0; 4],
    }];
    let plain = encode_multi_layer_response(&results);
    let (payload, serve_us) = split_timing_trailer(&plain);
    assert_eq!(payload, plain.as_slice());
    assert_eq!(serve_us, None);
}

#[test]
fn result_count_guard_still_rejects_with_trailer_present() {
    // Guard preservation: the response alloc-bomb bound must keep
    // rejecting garbage counts with the 8-byte trailer appended.
    let mut body = Vec::new();
    body.extend_from_slice(&u32::MAX.to_le_bytes()); // num_results
    append_timing_trailer(&mut body, 1.0);
    assert!(decode_multi_layer_response(&body).is_none());
    let (payload, _) = split_timing_trailer(&body);
    assert!(decode_multi_layer_response(payload).is_none());
}

#[test]
fn extended_response_split_reencode_reappend_is_byte_identical() {
    // Byte-identity pin for the extended multi-layer frame.
    let results = vec![
        MultiLayerResult {
            layer: 4,
            h2: vec![0.125, -3.5, 1e-20],
        },
        MultiLayerResult {
            layer: 17,
            h2: vec![f32::MAX],
        },
    ];
    let mut extended = encode_multi_layer_response(&results);
    append_timing_trailer(&mut extended, 12.5);

    let (payload, serve_us) = split_timing_trailer(&extended);
    let decoded = decode_multi_layer_response(payload).unwrap();
    let mut rebuilt = encode_multi_layer_response(&decoded);
    append_timing_trailer(&mut rebuilt, serve_us.unwrap() as f32);
    assert_eq!(rebuilt, extended);
}

#[test]
fn read_i8_slice_handles_signed_bytes() {
    // i8 round-trip via u8 byte storage: 0xff (255) must surface as -1.
    let bytes = [0u8, 0x7f, 0x80, 0xff];
    let mut pos = 0;
    let v = read_i8_slice(&bytes, &mut pos, 4).unwrap();
    assert_eq!(v, vec![0i8, 127, -128, -1]);
    assert_eq!(pos, 4);
}

#[test]
fn read_i16_slice_handles_negative_values() {
    // Three i16 little-endian: 0, 32767, -1.
    let bytes = [0x00, 0x00, 0xff, 0x7f, 0xff, 0xff];
    let mut pos = 0;
    let v = read_i16_slice(&bytes, &mut pos, 3).unwrap();
    assert_eq!(v, vec![0i16, 32767, -1]);
    assert_eq!(pos, 6);
}

#[test]
fn read_helpers_reject_overruns() {
    let bytes = [0u8; 4];
    let mut pos = 0;
    // Asking for one past the end is None; pos unchanged.
    assert!(read_u32(&bytes, &mut 1).is_none());
    assert!(read_f32(&bytes, &mut 1).is_none());
    assert!(read_f32_slice(&bytes, &mut pos, 2).is_none());
    assert!(read_i8_slice(&bytes, &mut pos, 5).is_none());
    assert!(read_i16_slice(&bytes, &mut pos, 3).is_none());
}

#[test]
fn content_type_and_path_consts_pin_wire_strings() {
    // Renaming any of these breaks deployed clients/servers.
    assert_eq!(
        MULTI_LAYER_BATCH_CONTENT_TYPE,
        "application/x-larql-experts-multi-layer"
    );
    assert_eq!(MULTI_LAYER_BATCH_PATH, "/v1/experts/multi-layer-batch");
    assert_eq!(
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
        "application/x-larql-experts-multi-layer-q8k"
    );
    assert_eq!(
        MULTI_LAYER_BATCH_Q8K_PATH,
        "/v1/experts/multi-layer-batch-q8k"
    );
}
