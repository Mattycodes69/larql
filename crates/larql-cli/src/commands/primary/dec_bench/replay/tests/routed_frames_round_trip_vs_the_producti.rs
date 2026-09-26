//! routed frames (round-trip vs the production decoders)

use super::*;

#[test]
fn experts_ml_frame_round_trips_through_production_decoder() {
    let hidden = 4;
    let batch = 3;
    let rows: Vec<f32> = (0..batch * hidden).map(|i| i as f32 * 0.25).collect();
    let routing = vec![
        vec![(5u32, 0.75f32), (9u32, 0.25f32)],
        vec![(1u32, 1.0f32)],
        vec![], // sentinel-only record → zero-expert task
    ];
    let frame = build_experts_ml_frame(7, &rows, &routing, batch, hidden);
    let tasks = larql_inference::ffn::moe_remote::decode_multi_layer_request(&frame).unwrap();
    assert_eq!(tasks.len(), batch);
    for (r, t) in tasks.iter().enumerate() {
        assert_eq!(t.layer, 7);
        assert_eq!(t.residual, rows[r * hidden..(r + 1) * hidden].to_vec());
        let ids: Vec<u32> = routing[r].iter().map(|&(id, _)| id).collect();
        let wts: Vec<f32> = routing[r].iter().map(|&(_, w)| w).collect();
        assert_eq!(t.expert_ids, ids);
        assert_eq!(t.weights, wts);
    }
}

#[test]
fn experts_ml_q8k_frame_round_trips_through_production_decoder() {
    let hidden = 256; // one Q8K superblock
    let batch = 2;
    let rows: Vec<f32> = (0..batch * hidden)
        .map(|i| (i as f32 * 0.013).sin())
        .collect();
    let routing = vec![vec![(3u32, 0.6f32), (7u32, 0.4f32)], vec![(42u32, 1.0f32)]];
    let frame = build_experts_ml_q8k_frame(11, &rows, &routing, batch, hidden);
    let tasks = larql_inference::ffn::moe_remote::decode_multi_layer_request_q8k(&frame).unwrap();
    assert_eq!(tasks.len(), batch);
    for (r, t) in tasks.iter().enumerate() {
        assert_eq!(t.layer, 11);
        assert_eq!(t.hidden, hidden);
        // Each task must be that row's quantisation, not row 0 repeated.
        let q = quantize_x_to_q8k(&rows[r * hidden..(r + 1) * hidden]);
        assert_eq!(t.qs, q.qs);
        assert_eq!(t.d, q.d);
        assert_eq!(t.sums, q.sums);
        let ids: Vec<u32> = routing[r].iter().map(|&(id, _)| id).collect();
        assert_eq!(t.expert_ids, ids);
        let wts: Vec<f32> = routing[r].iter().map(|&(_, w)| w).collect();
        assert_eq!(t.weights, wts);
    }
}

#[test]
fn check_experts_response_validates_shape_and_flags_nonzero() {
    use larql_inference::ffn::moe_remote::{encode_multi_layer_response, MultiLayerResult};
    let hidden = 4;
    let mk = |vals: Vec<Vec<f32>>, layer: usize| {
        encode_multi_layer_response(
            &vals
                .into_iter()
                .map(|h2| MultiLayerResult { layer, h2 })
                .collect::<Vec<_>>(),
        )
    };
    // Healthy: right batch/layer/hidden, non-zero values.
    let body = mk(vec![vec![0.0, 1.5, 0.0, 0.0], vec![0.0; 4]], 3);
    assert!(check_experts_response(&body, 3, 2, hidden).unwrap());
    // All-zero: decodes fine but flags the audit §1a corruption class.
    let body = mk(vec![vec![0.0; 4], vec![0.0; 4]], 3);
    assert!(!check_experts_response(&body, 3, 2, hidden).unwrap());
    // Wrong batch.
    let body = mk(vec![vec![1.0; 4]], 3);
    assert!(check_experts_response(&body, 3, 2, hidden).is_err());
    // Wrong layer.
    let body = mk(vec![vec![1.0; 4], vec![1.0; 4]], 5);
    assert!(check_experts_response(&body, 3, 2, hidden).is_err());
    // Wrong hidden.
    let body = mk(vec![vec![1.0; 3], vec![1.0; 3]], 3);
    assert!(check_experts_response(&body, 3, 2, hidden).is_err());
    // Garbage bytes.
    assert!(check_experts_response(&[1, 2, 3], 3, 2, hidden).is_err());
}
