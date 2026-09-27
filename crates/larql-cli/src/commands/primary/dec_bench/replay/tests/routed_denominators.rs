//! routed denominators

use super::*;

#[test]
fn routed_weight_bytes_overlapping_and_disjoint_sets() {
    // One cell, two rows sharing one expert: naive counts 4, union 3.
    let cells = vec![vec![vec![1u32, 2], vec![2u32, 3]]];
    let d = routed_weight_bytes_per_token(&cells, 100.0, 1, 2);
    assert_eq!(d.weight_bytes_tok_naive, 4.0 * 100.0 / 2.0);
    assert_eq!(d.weight_bytes_tok_union, 3.0 * 100.0 / 2.0);

    // Disjoint sets: union == naive.
    let cells = vec![vec![vec![0u32, 1], vec![2u32, 3]]];
    let d = routed_weight_bytes_per_token(&cells, 10.0, 1, 2);
    assert_eq!(d.weight_bytes_tok_naive, d.weight_bytes_tok_union);
    assert_eq!(d.weight_bytes_tok_naive, 4.0 * 10.0 / 2.0);

    // Identical sets across rows: union collapses to one row's cost.
    let cells = vec![vec![vec![7u32, 9], vec![7u32, 9], vec![7u32, 9]]];
    let d = routed_weight_bytes_per_token(&cells, 10.0, 1, 3);
    assert_eq!(d.weight_bytes_tok_naive, 6.0 * 10.0 / 3.0);
    assert_eq!(d.weight_bytes_tok_union, 2.0 * 10.0 / 3.0);
}

#[test]
fn routed_weight_bytes_sentinel_stripped_short_records_and_multi_cell() {
    // Sentinel-stripped short records: row lengths differ (a stripped
    // zero-weight pair), and one row is empty.
    let cells = vec![
        vec![vec![1u32, 2], vec![5u32]], // step 0, layer A
        vec![vec![2u32], vec![]],        // step 1, layer A
    ];
    let d = routed_weight_bytes_per_token(&cells, 8.0, 2, 2);
    // naive = (2 + 1) + (1 + 0) = 4 experts over 2 steps × 2 rows.
    assert_eq!(d.weight_bytes_tok_naive, 4.0 * 8.0 / 4.0);
    // union = |{1,2,5}| + |{2}| = 4.
    assert_eq!(d.weight_bytes_tok_union, 4.0 * 8.0 / 4.0);

    // Degenerate: zero tokens → zeros, no NaN.
    let d = routed_weight_bytes_per_token(&[], 8.0, 0, 0);
    assert_eq!(d.weight_bytes_tok_naive, 0.0);
    assert_eq!(d.weight_bytes_tok_union, 0.0);
}

#[test]
fn routed_denominators_for_point_matches_hand_count() {
    let dir = tempfile::tempdir().unwrap();
    routed_fixture(dir.path(), 3, 2, 4);
    let pool = Pool::open(dir.path()).unwrap();
    let peb = 50.0;
    // Layer 0 only (the MoE subset); batch 3, steps 2.
    // Every row has 2 experts → naive = 2 × 3 rows × 2 steps = 12.
    // Union per (step, layer-0) cell: experts {(p+s)%3 for p} ∪ {3} —
    // step 0: {0,1,2,3} = 4; step 1: {1,2,0,3} = 4 → union total 8.
    let d = routed_denominators_for_point(&pool, &[0], 2, 3, peb).unwrap();
    assert_eq!(d.weight_bytes_tok_naive, 12.0 * peb / 6.0);
    assert_eq!(d.weight_bytes_tok_union, 8.0 * peb / 6.0);

    // Smaller batch narrows the union.
    let d1 = routed_denominators_for_point(&pool, &[0], 2, 1, peb).unwrap();
    assert_eq!(d1.weight_bytes_tok_naive, 4.0 * peb / 2.0);
    assert_eq!(d1.weight_bytes_tok_union, 4.0 * peb / 2.0);
}

#[test]
fn moe_weight_stats_parses_and_errs_loudly() {
    let stats = serde_json::json!({
        "ffn_weights": {
            "per_layer_dense_bytes": [null, null],
            "moe": {
                "num_experts": 128,
                "top_k": 4,
                "per_expert_bytes": 123456,
            }
        }
    });
    let m = moe_weight_stats(&stats).unwrap();
    assert_eq!(m.per_expert_bytes, 123456.0);
    assert_eq!(m.top_k, Some(4));
    assert_eq!(m.num_experts, Some(128));

    // Null per_expert_bytes → loud error (audit §1c).
    let stats = serde_json::json!({
        "ffn_weights": { "moe": { "per_expert_bytes": null, "top_k": 4 } }
    });
    assert!(moe_weight_stats(&stats)
        .unwrap_err()
        .contains("per_expert_bytes"));

    // No moe block (dense server) and no ffn_weights at all.
    let stats = serde_json::json!({ "ffn_weights": { "moe": null } });
    assert!(moe_weight_stats(&stats).is_err());
    assert!(moe_weight_stats(&serde_json::json!({})).is_err());
}

#[test]
fn summarize_fills_routed_denominators_and_dense_weight_bytes() {
    let mk_stats = || SweepPointStats {
        step_ms: vec![10.0],
        samples: vec![sample(0, 10.0, None)],
    };
    // Routed point: naive is the primary movement denominator.
    let point = SweepPoint {
        batch: 1,
        wire: WireSpec::Plain(WireArm::F32),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::ExpertsMultiLayer,
    };
    let routed = RoutedDenominators {
        weight_bytes_tok_naive: 1600.0,
        weight_bytes_tok_union: 400.0,
    };
    let s = summarize(&point, &mk_stats(), Some(9999.0), Some(&routed));
    assert_eq!(s.endpoint, "experts-ml");
    assert_eq!(s.endpoint_code, 2);
    assert_eq!(s.weight_bytes_tok, None, "dense denom ignored on routed");
    assert_eq!(s.weight_bytes_tok_naive, Some(1600.0));
    assert_eq!(s.weight_bytes_tok_union, Some(400.0));
    // payload = 160 B/tok; ratio uses NAIVE.
    assert_eq!(s.movement_ratio, Some(160.0 / 1600.0));
    assert_eq!(s.experts_union_frac, Some(0.25));

    // Dense point: weight_bytes_tok carried per-point, routed fields absent.
    let point = SweepPoint {
        batch: 1,
        wire: WireSpec::Plain(WireArm::F32),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfn,
    };
    let s = summarize(&point, &mk_stats(), Some(3200.0), Some(&routed));
    assert_eq!(s.weight_bytes_tok, Some(3200.0));
    assert_eq!(s.weight_bytes_tok_naive, None);
    assert_eq!(s.weight_bytes_tok_union, None);
    assert_eq!(s.experts_union_frac, None);
    assert_eq!(s.movement_ratio, Some(160.0 / 3200.0));

    // Dense point without stats: no denominator keys at all.
    let s = summarize(&point, &mk_stats(), None, None);
    assert_eq!(s.weight_bytes_tok, None);
    assert_eq!(s.movement_ratio, None);
    let json = serde_json::to_value(&s).unwrap();
    assert!(json.get("weight_bytes_tok").is_none(), "omitted when None");
    assert!(json.get("movement_ratio").is_none());
}
