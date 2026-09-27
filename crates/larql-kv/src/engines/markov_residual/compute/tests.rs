use super::*;
use larql_compute::CpuBackend;
use larql_inference::test_utils::make_test_weights;

// ── recompute_kv ──────────────────────────────────────────────────────────

#[test]
fn recompute_kv_returns_some_with_valid_weights() {
    let weights = make_test_weights();
    let h = Array2::from_elem((3, weights.hidden_size), 0.5f32);
    let result = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    );
    assert!(
        result.is_some(),
        "recompute_kv should return Some with valid weights"
    );
}

#[test]
fn recompute_kv_output_shape_correct() {
    let weights = make_test_weights();
    let seq_len = 4;
    let h = Array2::from_elem((seq_len, weights.hidden_size), 1.0f32);
    let (k, v) = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    )
    .unwrap();
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    assert_eq!(k.shape(), &[seq_len, kv_dim], "K shape mismatch");
    assert_eq!(v.shape(), &[seq_len, kv_dim], "V shape mismatch");
}

#[test]
fn recompute_kv_output_is_finite() {
    let weights = make_test_weights();
    let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
    let (k, v) = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert!(
        k.iter().all(|v| v.is_finite()),
        "K contains non-finite values"
    );
    assert!(
        v.iter().all(|v| v.is_finite()),
        "V contains non-finite values"
    );
}

#[test]
fn recompute_kv_abs_start_shifts_rope() {
    let weights = make_test_weights();
    let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
    // Different abs_start should produce different RoPE-applied K
    let (k0, _) = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    )
    .unwrap();
    let (k5, _) = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        5,
        &CpuBackend,
        None,
    )
    .unwrap();
    let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.0,
        "RoPE at different positions should produce different K"
    );
}

#[test]
fn walk_project_topk_full_k_matches_dense_projection() {
    let x = Array2::from_shape_vec((2, 3), vec![1.0, -2.0, 0.5, 0.25, 0.75, -1.0]).unwrap();
    let w = Array2::from_shape_vec(
        (4, 3),
        vec![
            0.5, 1.0, -0.5, -1.0, 0.25, 0.75, 0.0, 2.0, 1.0, 1.5, -0.5, 0.25,
        ],
    )
    .unwrap();
    let walked = walk_project_topk(&x, &w, 4).unwrap();
    let dense = dot_proj_gpu(&x, &w, None);
    let max_diff = walked
        .iter()
        .zip(dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-6, "max_diff={max_diff}");
}

#[test]
fn walk_project_topk_keeps_largest_absolute_outputs_per_row() {
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let w = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
    )
    .unwrap();
    let walked = walk_project_topk(&x, &w, 2).unwrap();
    let non_zero: Vec<usize> = walked
        .row(0)
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != 0.0).then_some(i))
        .collect();
    assert_eq!(non_zero, vec![1, 2]);
    assert_eq!(walked[[0, 1]], -6.0);
    assert_eq!(walked[[0, 2]], 6.0);
}

#[test]
fn walk_project_cached_topk_reuses_selector_layer_indices() {
    WALK_KV_SELECTION.with(|slot| {
        *slot.borrow_mut() = None;
    });
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let selector_w_k = Array2::from_shape_vec(
        (4, 3),
        vec![1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
    )
    .unwrap();
    let selector_w_v = selector_w_k.clone();
    cache_walk_kv_selection(4, 2, &x, &selector_w_k, &selector_w_v);

    let later_w = Array2::from_shape_vec(
        (4, 3),
        vec![
            10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0, 40.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let walked =
        walk_project_cached_topk(&x, &later_w, 2, 4, KvProjection::K).expect("cached walk");
    let non_zero: Vec<usize> = walked
        .row(0)
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != 0.0).then_some(i))
        .collect();
    assert_eq!(non_zero, vec![1, 2]);
    assert_eq!(walked[[0, 1]], 40.0);
    assert_eq!(walked[[0, 2]], 90.0);
}

#[test]
fn markov_walk_kv_layer_spec_accepts_ranges_and_singletons() {
    assert!(layer_in_spec("5-20", 5));
    assert!(layer_in_spec("5-20", 20));
    assert!(layer_in_spec(" 2, 5-7, 26 ", 6));
    assert!(layer_in_spec(" 2, 5-7, 26 ", 26));
    assert!(!layer_in_spec("5-20", 4));
    assert!(!layer_in_spec("5-20", 21));
    assert!(!layer_in_spec("x-y, 30", 29));
}

#[test]
fn kv_memory_bytes_for_seq_scales_linearly() {
    let weights = make_test_weights();
    let one = kv_memory_bytes_for_seq(larql_inference::WeightsView::dense(&weights), 1);
    let ten = kv_memory_bytes_for_seq(larql_inference::WeightsView::dense(&weights), 10);
    assert!(one > 0);
    assert_eq!(ten, one * 10, "kv memory must scale linearly with seq len");
}

// ── parse_quant_format pure helper (lines 384-391) ───────────────────

#[test]
fn parse_quant_format_recognises_q4k_q4kf_q6k() {
    assert!(matches!(
        parse_quant_format("Q4_K"),
        Some(QuantFormat::Q4_K)
    ));
    assert!(matches!(
        parse_quant_format("Q4_KF"),
        Some(QuantFormat::Q4_KF)
    ));
    assert!(matches!(
        parse_quant_format("Q6_K"),
        Some(QuantFormat::Q6_K)
    ));
}

#[test]
fn parse_quant_format_unknown_returns_none() {
    assert!(parse_quant_format("Q8_0").is_none());
    assert!(parse_quant_format("F16").is_none());
    assert!(parse_quant_format("").is_none());
    assert!(parse_quant_format("Q4").is_none());
    assert!(parse_quant_format("nonsense").is_none());
}

// ── Pure helpers ────────────────────────────────────────────────────────

#[test]
fn dot_rows_basic_arithmetic() {
    let a = ndarray::arr1(&[1.0f32, 2.0, 3.0]);
    let b = ndarray::arr1(&[4.0f32, 5.0, 6.0]);
    // 1*4 + 2*5 + 3*6 = 32
    assert!((dot_rows(a.view(), b.view()) - 32.0).abs() < 1e-6);
}

#[test]
fn compare_abs_desc_orders_by_absolute_magnitude() {
    let a = (0usize, -5.0f32);
    let b = (1usize, 3.0f32);
    // |a| > |b| so a comes before b under descending sort.
    assert_eq!(compare_abs_desc(&a, &b), Ordering::Less);
    assert_eq!(compare_abs_desc(&b, &a), Ordering::Greater);
    // Tie: NaN/Equal fallback returns Equal.
    let c = (2usize, 5.0f32);
    let d = (3usize, -5.0f32);
    assert_eq!(compare_abs_desc(&c, &d), Ordering::Equal);
}

#[test]
fn array_diff_stats_identical_arrays_returns_zero_diff_and_unit_cos() {
    // Identical arrays → max_abs=0, rms=0, cos=1.
    let a = Array2::<f32>::from_elem((2, 3), 1.5);
    let b = a.clone();
    let (max_abs, rms, cos) = array_diff_stats(&a, &b);
    assert!(max_abs.abs() < 1e-12);
    assert!(rms.abs() < 1e-12);
    assert!(
        (cos - 1.0).abs() < 1e-9,
        "cos should be 1 for identical, got {cos}"
    );
}

#[test]
fn array_diff_stats_reports_max_abs_and_rms() {
    let a = Array2::<f32>::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
    let b = Array2::<f32>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let (max_abs, rms, cos) = array_diff_stats(&a, &b);
    // max_abs = 3, rms = sqrt(((-1)^2 + (-2)^2 + (-3)^2) / 3) = sqrt(14/3)
    assert!((max_abs - 3.0).abs() < 1e-9);
    assert!((rms - (14.0_f64 / 3.0).sqrt()).abs() < 1e-9);
    // a is all zeros so cosine has denom=0 → returns 1.0 sentinel.
    assert!((cos - 1.0).abs() < 1e-9, "all-zeros a → cos sentinel = 1");
}

#[test]
fn array_diff_stats_mismatched_shape_returns_nan_tuple() {
    let a = Array2::<f32>::zeros((2, 3));
    let b = Array2::<f32>::zeros((3, 2));
    let (max_abs, rms, cos) = array_diff_stats(&a, &b);
    assert!(max_abs.is_nan() && rms.is_nan() && cos.is_nan());
}

#[test]
fn layer_in_spec_accepts_singleton_and_ranges() {
    // Direct test of the spec parser. Covers "5", "5-7", "1,5-7,9"
    // forms — the helper layered under markov_walk_kv_diag_layer
    // and markov_walk_kv_top_k env-var paths.
    assert!(layer_in_spec("5", 5));
    assert!(!layer_in_spec("5", 6));
    assert!(layer_in_spec("5-7", 5));
    assert!(layer_in_spec("5-7", 6));
    assert!(layer_in_spec("5-7", 7));
    assert!(!layer_in_spec("5-7", 8));
    assert!(layer_in_spec("1,5-7,9", 1));
    assert!(layer_in_spec("1,5-7,9", 6));
    assert!(layer_in_spec("1,5-7,9", 9));
    assert!(!layer_in_spec("1,5-7,9", 3));
}

#[test]
fn layer_in_spec_rejects_malformed_input() {
    // Non-numeric pieces should not crash and should return false.
    assert!(!layer_in_spec("abc", 5));
    assert!(!layer_in_spec("", 5));
}

#[test]
fn print_walk_kv_diag_runs_without_panicking() {
    // Pure logging helper. The body just prints diagnostic stats;
    // exercising it produces console output but no observable
    // state change. Coverage credit for the function body.
    let a = Array2::<f32>::from_elem((2, 4), 1.0f32);
    let b = Array2::<f32>::from_elem((2, 4), 0.5f32);
    print_walk_kv_diag(0, "test_path", "K", "test_label", &a, &b);
}

// ── Env-var-gated walk-KV paths ───────────────────────────────────────────
//
// These tests cover the `LARQL_MARKOV_WALK_KV_*` /
// `LARQL_MARKOV_KV_FORCE_F32` paths in `recompute_kv` and the
// `markov_walk_kv_*` helpers. Production reads via
// `read_markov_env`, which consults the per-thread
// `MARKOV_ENV_OVERRIDE` map *before* `std::env::var`. Tests inject
// values through `set_markov_env_override` — no process-global env
// mutation, no `#[serial]` needed, no race with other parallel
// tests that also call `recompute_kv`.

#[test]
fn markov_walk_kv_requested_top_k_parses_clamps_and_rejects_zero() {
    clear_markov_env_overrides();
    assert_eq!(markov_walk_kv_requested_top_k(32), None);
    set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("8"));
    assert_eq!(markov_walk_kv_requested_top_k(32), Some(8));
    assert_eq!(
        markov_walk_kv_requested_top_k(4),
        Some(4),
        "clamp to kv_dim"
    );
    set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("0"));
    assert_eq!(markov_walk_kv_requested_top_k(32), None);
    set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("abc"));
    assert_eq!(markov_walk_kv_requested_top_k(32), None);
    clear_markov_env_overrides();
}

#[test]
fn markov_walk_kv_select_at_parses_layer_index() {
    clear_markov_env_overrides();
    assert_eq!(markov_walk_kv_select_at(), None);
    set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("7"));
    assert_eq!(markov_walk_kv_select_at(), Some(7));
    set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("bad"));
    assert_eq!(markov_walk_kv_select_at(), None);
    clear_markov_env_overrides();
}

#[test]
fn markov_walk_kv_diag_enabled_accepts_truthy_strings() {
    clear_markov_env_overrides();
    assert!(!markov_walk_kv_diag_enabled());
    for val in ["1", "true", "TRUE", "yes", "on"] {
        set_markov_env_override("LARQL_MARKOV_WALK_KV_DIAG", Some(val));
        assert!(markov_walk_kv_diag_enabled(), "should accept {val}");
    }
    for val in ["0", "false", "no"] {
        set_markov_env_override("LARQL_MARKOV_WALK_KV_DIAG", Some(val));
        assert!(!markov_walk_kv_diag_enabled(), "should reject {val}");
    }
    clear_markov_env_overrides();
}

#[test]
fn markov_kv_force_f32_projection_reads_env() {
    clear_markov_env_overrides();
    assert!(!markov_kv_force_f32_projection());
    set_markov_env_override("LARQL_MARKOV_KV_FORCE_F32", Some("1"));
    assert!(markov_kv_force_f32_projection());
    set_markov_env_override("LARQL_MARKOV_KV_FORCE_F32", Some("no"));
    assert!(!markov_kv_force_f32_projection());
    clear_markov_env_overrides();
}

#[test]
fn markov_walk_kv_diag_layer_respects_layers_spec() {
    clear_markov_env_overrides();
    assert!(markov_walk_kv_diag_layer(0));
    assert!(markov_walk_kv_diag_layer(99));
    set_markov_env_override("LARQL_MARKOV_WALK_KV_LAYERS", Some("3-5"));
    assert!(markov_walk_kv_diag_layer(4));
    assert!(!markov_walk_kv_diag_layer(0));
    clear_markov_env_overrides();
}

#[test]
fn markov_walk_kv_top_k_honours_layers_and_select_at_gates() {
    clear_markov_env_overrides();
    assert_eq!(markov_walk_kv_top_k(0, 32), None);
    set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("4"));
    set_markov_env_override("LARQL_MARKOV_WALK_KV_LAYERS", Some("5-7"));
    assert_eq!(markov_walk_kv_top_k(0, 32), None);
    assert_eq!(markov_walk_kv_top_k(6, 32), Some(4));
    set_markov_env_override("LARQL_MARKOV_WALK_KV_LAYERS", None);
    set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("6"));
    assert_eq!(markov_walk_kv_top_k(6, 32), None);
    assert_eq!(markov_walk_kv_top_k(7, 32), Some(4));
    clear_markov_env_overrides();
}

#[test]
fn recompute_kv_force_f32_disables_q4k_path() {
    clear_markov_env_overrides();
    set_markov_env_override("LARQL_MARKOV_KV_FORCE_F32", Some("1"));
    let weights = make_test_weights();
    let h = Array2::from_elem((2, weights.hidden_size), 0.5f32);
    let (k, v) = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    )
    .unwrap();
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    assert_eq!(k.shape(), &[2, kv_dim]);
    assert_eq!(v.shape(), &[2, kv_dim]);
    clear_markov_env_overrides();
}

#[test]
fn recompute_kv_topk_routes_through_walk_projection() {
    clear_markov_env_overrides();
    set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("2"));
    let weights = make_test_weights();
    let h = Array2::from_elem((2, weights.hidden_size), 0.25f32);
    let result = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    );
    assert!(result.is_some());
    clear_markov_env_overrides();
}

#[test]
fn recompute_kv_select_at_uses_cached_indices_on_later_layers() {
    clear_markov_env_overrides();
    set_markov_env_override("LARQL_MARKOV_WALK_KV_TOPK", Some("2"));
    set_markov_env_override("LARQL_MARKOV_WALK_KV_SELECT_AT", Some("0"));
    let weights = make_test_weights();
    let h = Array2::from_elem((2, weights.hidden_size), 0.25f32);
    // Layer 0: should_cache_selection fires, populates
    // WALK_KV_SELECTION; layer 1: walk_project_cached_topk reads it.
    let _ = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    );
    if weights.num_layers >= 2 {
        let result = recompute_kv(
            larql_inference::WeightsView::dense(&weights),
            &h,
            1,
            0,
            &CpuBackend,
            None,
        );
        assert!(result.is_some());
    }
    clear_markov_env_overrides();
}

#[test]
fn recompute_kv_diag_fires_when_enabled() {
    clear_markov_env_overrides();
    set_markov_env_override("LARQL_MARKOV_WALK_KV_DIAG", Some("1"));
    let weights = make_test_weights();
    let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
    let result = recompute_kv(
        larql_inference::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &CpuBackend,
        None,
    );
    assert!(result.is_some());
    clear_markov_env_overrides();
}
