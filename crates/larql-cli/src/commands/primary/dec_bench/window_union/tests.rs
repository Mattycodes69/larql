use super::*;
use std::io::Cursor;

fn rec(layer: usize, seq: usize, experts: &[u32]) -> String {
    let ids = experts
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"layer\":{layer},\"seq\":{seq},\"experts\":[[{ids}]]}}")
}

#[test]
fn shuffle_in_place_is_a_permutation_and_is_seed_deterministic() {
    let original: Vec<u32> = (0..20).collect();

    let mut a = original.clone();
    let mut seed_a = 42u64;
    shuffle_in_place(&mut a, &mut seed_a);

    let mut b = original.clone();
    let mut seed_b = 42u64;
    shuffle_in_place(&mut b, &mut seed_b);
    assert_eq!(a, b, "same seed must reproduce the same permutation");

    let mut sorted = a.clone();
    sorted.sort();
    assert_eq!(
        sorted, original,
        "shuffling must not lose or duplicate items"
    );

    let mut c = original.clone();
    let mut seed_c = 43u64;
    shuffle_in_place(&mut c, &mut seed_c);
    assert_ne!(
        a, c,
        "a different seed should (overwhelmingly likely) differ"
    );
}

#[test]
fn shuffled_control_on_all_identical_rows_is_exactly_the_same_frac_as_windowed() {
    // Every row identical: no permutation can change the union — the
    // control and the real windowed measurement must agree exactly,
    // regardless of trial count or seed. This is the equivalence-point
    // sanity check for the control path.
    let rows: Vec<Vec<u32>> = (0..10).map(|_| vec![1, 2, 3, 4]).collect();
    let frac = shuffled_control_union_frac(&rows, 2, 10.0, 50, 7).unwrap();
    // Every chunk pairs two identical 4-expert rows: union 4 / naive 8 = 0.5.
    assert!((frac - 0.5).abs() < 1e-12, "expected 0.5, got {frac}");
}

#[test]
fn shuffled_control_on_all_disjoint_rows_is_exactly_one_regardless_of_order() {
    // Every row pulls from a disjoint block of expert ids: no
    // permutation can create overlap, so union_frac must stay 1.0
    // exactly under any shuffle.
    let rows: Vec<Vec<u32>> = (0..8).map(|i| vec![i * 10, i * 10 + 1]).collect();
    let frac = shuffled_control_union_frac(&rows, 2, 10.0, 50, 99).unwrap();
    assert_eq!(frac, 1.0);
}

#[test]
fn shuffled_control_errs_on_k_zero_or_too_few_rows_or_zero_trials() {
    let rows = vec![vec![1u32], vec![2u32]];
    assert!(shuffled_control_union_frac(&rows, 0, 10.0, 5, 1).is_err());
    assert!(shuffled_control_union_frac(&rows, 5, 10.0, 5, 1).is_err());
    assert!(shuffled_control_union_frac(&rows, 1, 10.0, 0, 1).is_err());
}

#[test]
fn all_rows_for_layer_pools_across_files_and_filters_by_layer() {
    let file_a = vec![
        TraceRecord {
            layer: 0,
            seq: 0,
            experts: vec![vec![1, 2]],
        },
        TraceRecord {
            layer: 1,
            seq: 0,
            experts: vec![vec![9, 9]],
        },
    ];
    let file_b = vec![TraceRecord {
        layer: 0,
        seq: 0,
        experts: vec![vec![3, 4]],
    }];
    let rows = all_rows_for_layer(&[&file_a, &file_b], 0).unwrap();
    assert_eq!(
        rows,
        vec![vec![1, 2], vec![3, 4]],
        "layer 1 must not leak in"
    );
}

#[test]
fn all_rows_for_layer_rejects_a_multi_position_record() {
    let file = vec![TraceRecord {
        layer: 0,
        seq: 0,
        experts: vec![vec![1], vec![2]],
    }];
    assert!(all_rows_for_layer(&[&file], 0).is_err());
}

#[test]
fn shuffled_control_by_layer_covers_every_layer_seen_across_files() {
    let file_a = vec![
        TraceRecord {
            layer: 0,
            seq: 0,
            experts: vec![vec![1, 2, 3, 4]],
        },
        TraceRecord {
            layer: 0,
            seq: 1,
            experts: vec![vec![1, 2, 3, 4]],
        },
    ];
    let file_b = vec![
        TraceRecord {
            layer: 1,
            seq: 0,
            experts: vec![vec![5, 6]],
        },
        TraceRecord {
            layer: 1,
            seq: 1,
            experts: vec![vec![5, 6]],
        },
    ];
    let out = shuffled_control_by_layer(&[&file_a, &file_b], 2, 10.0, 20, 3).unwrap();
    let layers: Vec<usize> = out.iter().map(|l| l.layer).collect();
    assert_eq!(layers, vec![0, 1]);
    assert!(out.iter().all(|l| (l.union_frac - 0.5).abs() < 1e-12));
}

#[test]
fn parses_the_writer_format_and_round_trips() {
    let text = format!(
        "{}\n{}\n\n{}\n",
        rec(0, 0, &[5, 9, 18, 6]),
        rec(0, 1, &[28, 29, 6, 11]),
        rec(1, 0, &[2, 3])
    );
    let recs = parse_trace(Cursor::new(text)).unwrap();
    assert_eq!(recs.len(), 3, "blank lines must be skipped, not counted");
    assert_eq!(
        recs[0],
        TraceRecord {
            layer: 0,
            seq: 0,
            experts: vec![vec![5, 9, 18, 6]]
        }
    );
}

#[test]
fn malformed_line_is_a_loud_error_not_a_skip() {
    let err = parse_trace(Cursor::new("not json\n")).unwrap_err();
    assert!(
        err.contains("line 1"),
        "must name the offending line: {err}"
    );
}

#[test]
fn rejects_a_gap_in_seq_rather_than_windowing_across_it() {
    let text = format!("{}\n{}\n", rec(0, 0, &[1, 2]), rec(0, 2, &[3, 4]));
    let recs = parse_trace(Cursor::new(text)).unwrap();
    let err = window_union_by_layer(&recs, 1, 100.0).unwrap_err();
    assert!(err.contains("seq 1"), "must name the missing seq: {err}");
}

#[test]
fn rejects_a_duplicate_seq() {
    let text = format!("{}\n{}\n", rec(0, 0, &[1, 2]), rec(0, 0, &[3, 4]));
    let recs = parse_trace(Cursor::new(text)).unwrap();
    let err = window_union_by_layer(&recs, 1, 100.0).unwrap_err();
    assert!(err.contains("twice"), "must name the duplication: {err}");
}

#[test]
fn rejects_a_multi_position_record() {
    let line = "{\"layer\":0,\"seq\":0,\"experts\":[[1,2],[3,4]]}\n";
    let recs = parse_trace(Cursor::new(line)).unwrap();
    let err = window_union_by_layer(&recs, 1, 100.0).unwrap_err();
    assert!(
        err.contains("position(s)"),
        "must name the shape problem: {err}"
    );
}

#[test]
fn k_equals_one_is_always_union_frac_one() {
    // K=1: a single row cannot share with itself — naive == union at
    // every window, by construction, regardless of content. This is
    // the equivalence-point sanity check; k_two_disjoint_vs_shared
    // below is the disagreement-point oracle (R8) that actually
    // exercises the union arithmetic.
    let text = format!(
        "{}\n{}\n{}\n",
        rec(0, 0, &[1, 2, 3]),
        rec(0, 1, &[4, 5, 6]),
        rec(0, 2, &[7, 8, 9])
    );
    let recs = parse_trace(Cursor::new(text)).unwrap();
    let layers = window_union_by_layer(&recs, 1, 10.0).unwrap();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].n_windows, 3);
    assert_eq!(layers[0].union_frac, 1.0);
}

#[test]
fn k_two_disjoint_vs_shared_moves_the_ratio() {
    // Disjoint neighbours: union == naive (no sharing possible).
    let disjoint = format!("{}\n{}\n", rec(0, 0, &[0, 1]), rec(0, 1, &[2, 3]));
    let recs = parse_trace(Cursor::new(disjoint)).unwrap();
    let layers = window_union_by_layer(&recs, 2, 10.0).unwrap();
    assert_eq!(layers[0].union_frac, 1.0);

    // Identical neighbours: union collapses to one row's cost.
    let shared = format!("{}\n{}\n", rec(0, 0, &[0, 1]), rec(0, 1, &[0, 1]));
    let recs = parse_trace(Cursor::new(shared)).unwrap();
    let layers = window_union_by_layer(&recs, 2, 10.0).unwrap();
    assert_eq!(
        layers[0].union_frac, 0.5,
        "two identical rows halve the ratio"
    );

    // Partial overlap: naive 4, union 3.
    let partial = format!("{}\n{}\n", rec(0, 0, &[1, 2]), rec(0, 1, &[2, 3]));
    let recs = parse_trace(Cursor::new(partial)).unwrap();
    let layers = window_union_by_layer(&recs, 2, 10.0).unwrap();
    assert_eq!(layers[0].union_frac, 0.75);
}

#[test]
fn sliding_windows_pool_across_the_whole_layer() {
    // 4 positions, K=2 -> 3 overlapping windows: (0,1) (1,2) (2,3).
    // naive = 2+2+2 = 6 experts total; unions = |{0,1}|=2, |{1,2}|=2,
    // |{2,3}|=2 -> union total 6 -> union_frac 1.0 (all disjoint pairs).
    let text = format!(
        "{}\n{}\n{}\n{}\n",
        rec(0, 0, &[0]),
        rec(0, 1, &[1]),
        rec(0, 2, &[2]),
        rec(0, 3, &[3])
    );
    let recs = parse_trace(Cursor::new(text)).unwrap();
    let layers = window_union_by_layer(&recs, 2, 100.0).unwrap();
    assert_eq!(layers[0].n_windows, 3);
    assert_eq!(layers[0].union_frac, 1.0);
}

#[test]
fn errs_when_a_layer_is_shorter_than_k() {
    let text = rec(0, 0, &[1, 2]);
    let recs = parse_trace(Cursor::new(format!("{text}\n"))).unwrap();
    let err = window_union_by_layer(&recs, 4, 10.0).unwrap_err();
    assert!(err.contains("--steps"), "must point at the fix: {err}");
}

#[test]
fn errs_on_k_zero() {
    let recs = parse_trace(Cursor::new(format!("{}\n", rec(0, 0, &[1])))).unwrap();
    let err = window_union_by_layer(&recs, 0, 10.0).unwrap_err();
    assert!(err.contains("k must be"));
}

#[test]
fn summarize_spread_matches_hand_computed_percentiles() {
    let layers = vec![
        LayerWindowUnion {
            layer: 0,
            k: 4,
            n_windows: 1,
            weight_bytes_tok_naive: 100.0,
            weight_bytes_tok_union: 20.0,
            union_frac: 0.20,
        },
        LayerWindowUnion {
            layer: 1,
            k: 4,
            n_windows: 1,
            weight_bytes_tok_naive: 100.0,
            weight_bytes_tok_union: 40.0,
            union_frac: 0.40,
        },
        LayerWindowUnion {
            layer: 2,
            k: 4,
            n_windows: 1,
            weight_bytes_tok_naive: 100.0,
            weight_bytes_tok_union: 60.0,
            union_frac: 0.60,
        },
    ];
    let s = summarize_spread(&layers).unwrap();
    assert_eq!(s.n_layers, 3);
    assert!((s.union_frac_mean - 0.40).abs() < 1e-12);
    assert_eq!(s.union_frac_median, 0.40);
    assert_eq!(
        s.union_frac_p10, 0.20,
        "nearest-rank nudges low p toward the min"
    );
    assert_eq!(s.union_frac_p90, 0.60);
    assert!((s.amortisation_mean - 2.5).abs() < 1e-12);
}

#[test]
fn summarize_spread_errs_on_empty_input() {
    assert!(summarize_spread(&[]).is_err());
}

#[test]
fn summarize_spread_errs_on_mixed_k() {
    let a = LayerWindowUnion {
        layer: 0,
        k: 2,
        n_windows: 1,
        weight_bytes_tok_naive: 1.0,
        weight_bytes_tok_union: 1.0,
        union_frac: 1.0,
    };
    let mut b = a;
    b.k = 4;
    let err = summarize_spread(&[a, b]).unwrap_err();
    assert!(err.contains("mixed K"));
}

#[test]
fn format_spread_row_names_every_field() {
    let s = WindowUnionSpread {
        k: 4,
        n_layers: 24,
        union_frac_mean: 0.5,
        union_frac_median: 0.5,
        union_frac_p10: 0.4,
        union_frac_p90: 0.6,
        amortisation_mean: 2.0,
    };
    let row = format_spread_row(&s, 0.125);
    assert!(row.contains("K=4"));
    assert!(row.contains("layers=24"));
    assert!(row.contains("12.50%"));
    assert!(row.contains("2.00x"));
}
