use super::*;

// ── Wire helpers ─────────────────────────────────────────────────────────

#[test]
fn encode_decode_f32_le_round_trips() {
    let values = vec![1.0_f32, -2.5, 0.0, 4.25];
    let bytes = encode_f32_le(&values);
    assert_eq!(bytes.len(), values.len() * 4);
    let back = decode_f32_le(&bytes).unwrap();
    assert_eq!(back, values);
}

#[test]
fn decode_f32_le_rejects_odd_byte_lengths() {
    let err = decode_f32_le(&[0u8; 7]).unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

// ── Pure math ────────────────────────────────────────────────────────────

#[test]
fn l2_normalize_unit_vector_is_idempotent() {
    let v = vec![1.0f32, 0.0, 0.0];
    let n = l2_normalize(&v);
    assert!((n[0] - 1.0).abs() < 1e-6);
    assert_eq!(n[1], 0.0);
}

#[test]
fn l2_normalize_zero_vector_returns_zero() {
    // NORM_EPS guards the divide; result has all-zero numerator.
    let v = vec![0.0f32; 4];
    let n = l2_normalize(&v);
    assert!(n.iter().all(|x| *x == 0.0));
}

#[test]
fn l2_normalize_rows_normalizes_each_row_independently() {
    let rows = vec![1.0, 0.0, 0.0, 3.0, 4.0, 0.0];
    let n = l2_normalize_rows(&rows, 2, 3);
    // row 0 already unit
    assert!((n[0] - 1.0).abs() < 1e-6);
    // row 1: |v| = 5, expect (0.6, 0.8, 0.0)
    assert!((n[3] - 0.6).abs() < 1e-6);
    assert!((n[4] - 0.8).abs() < 1e-6);
    assert_eq!(n[5], 0.0);
}

#[test]
fn cosine_similarities_match_dot_product_for_normed_inputs() {
    let rows = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let q = vec![1.0, 0.0, 0.0];
    let sims = cosine_similarities(&rows, &q, 2, 3);
    assert!((sims[0] - 1.0).abs() < 1e-6);
    assert!(sims[1].abs() < 1e-6);
}

#[test]
fn argmax_handles_empty_input() {
    assert_eq!(argmax(&[]), (0, 0.0));
    assert_eq!(argmax(&[0.5, -0.2, 0.9, 0.1]), (2, 0.9));
}

#[test]
fn weighted_topk_average_falls_back_to_uniform_when_all_negative() {
    let sims = vec![-0.5, -0.3, -0.1];
    let outputs = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]; // 3 rows of d=2
    let avg = weighted_topk_average(&sims, &outputs, 3, 2);
    // Uniform avg: (1+2+3)/3 = 2.0; (0+0+0)/3 = 0.0
    assert!((avg[0] - 2.0).abs() < 1e-6);
    assert!((avg[1] - 0.0).abs() < 1e-6);
}

#[test]
fn weighted_topk_average_uses_positive_cosine_weights() {
    let sims = vec![0.9, 0.1, -0.5];
    // d=1 outputs: [10, 20, 100]
    let outputs = vec![10.0, 20.0, 100.0];
    let avg = weighted_topk_average(&sims, &outputs, 3, 1);
    // weights from positive sims: [0.9, 0.1, 0.0] / 1.0
    let expected = 10.0 * 0.9 + 20.0 * 0.1 + 100.0 * 0.0;
    assert!((avg[0] - expected).abs() < 1e-5);
}

// ── Cache + lookup ───────────────────────────────────────────────────────

fn cache_with_two_entries(d: usize, tau: f32) -> ShardCache {
    // Layer 26 with two entries at d=4:
    //   row 0: [1, 0, 0, 0] → output [10, 20, 30, 40]
    //   row 1: [0, 1, 0, 0] → output [-1, -2, -3, -4]
    let inputs_normed = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let outputs = vec![10.0, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0];
    let mut cache = ShardCache::new(tau);
    cache
        .seed_from_normed(26, inputs_normed, outputs, 2, d)
        .unwrap();
    cache
}

#[test]
fn knn_lookup_hit_returns_argmax_output_when_k_is_one() {
    let cache = cache_with_two_entries(4, 0.97);
    // Query close to row 0 → expect outputs[0..4].
    let out = cache.knn_lookup(26, &[1.0, 0.0, 0.0, 0.0], 1, 0.97);
    let mlp = out.mlp_out.expect("hit");
    assert_eq!(mlp, vec![10.0, 20.0, 30.0, 40.0]);
    assert!((out.best_sim - 1.0).abs() < 1e-6);
}

#[test]
fn knn_lookup_miss_when_below_tau() {
    let cache = cache_with_two_entries(4, 0.97);
    // Query orthogonal to both rows → best_sim ≈ 0 < tau.
    let out = cache.knn_lookup(26, &[0.0, 0.0, 1.0, 0.0], 1, 0.97);
    assert!(out.mlp_out.is_none());
    assert!(out.best_sim < 0.97);
}

#[test]
fn knn_lookup_unknown_layer_is_a_miss() {
    let cache = cache_with_two_entries(4, 0.97);
    let out = cache.knn_lookup(99, &[1.0, 0.0, 0.0, 0.0], 1, 0.97);
    assert!(out.mlp_out.is_none());
    assert_eq!(out.best_sim, 0.0);
}

#[test]
fn knn_lookup_dim_mismatch_is_a_miss() {
    let cache = cache_with_two_entries(4, 0.97);
    let out = cache.knn_lookup(26, &[1.0, 0.0, 0.0], 1, 0.97);
    assert!(out.mlp_out.is_none());
}

#[test]
fn knn_lookup_k_greater_than_one_averages_top_k() {
    // Build a cache where the top-2 are tied at cos = 0.7071…
    //   row 0: [1, 0] output [10, 0]
    //   row 1: [0, 1] output [ 0, 10]
    // Query [1/√2, 1/√2] hits both with equal weight; average is
    // (5, 5).
    let mut cache = ShardCache::new(0.5);
    cache
        .seed_from_normed(
            0,
            vec![1.0, 0.0, 0.0, 1.0],
            vec![10.0, 0.0, 0.0, 10.0],
            2,
            2,
        )
        .unwrap();
    let q = vec![1.0 / 2f32.sqrt(), 1.0 / 2f32.sqrt()];
    let out = cache.knn_lookup(0, &q, 2, 0.5);
    let mlp = out.mlp_out.expect("hit");
    assert!((mlp[0] - 5.0).abs() < 1e-5);
    assert!((mlp[1] - 5.0).abs() < 1e-5);
}

#[test]
fn tau_override_can_force_hit_or_miss() {
    let cache = cache_with_two_entries(4, 0.5);
    // Query at cos = 0.7071 to both rows after normalization
    // (proportional to [1, 1, 0, 0]).
    let q = vec![1.0, 1.0, 0.0, 0.0];
    // tau = 0.5 → hit.
    assert!(cache.knn_lookup(26, &q, 1, 0.5).mlp_out.is_some());
    // tau = 0.99 → miss even though argmax is the same.
    assert!(cache.knn_lookup(26, &q, 1, 0.99).mlp_out.is_none());
}

#[test]
fn insert_layer_validates_shape() {
    let mut cache = ShardCache::new(0.97);
    let err = cache
        .insert_layer(0, &[1.0, 0.0], vec![1.0, 0.0, 0.0], 1, 2)
        .unwrap_err();
    assert!(matches!(err, CacheError::OutputShape { .. }));

    let err = cache
        .insert_layer(0, &[1.0, 0.0, 0.0], vec![1.0, 0.0], 1, 2)
        .unwrap_err();
    assert!(matches!(err, CacheError::InputShape { .. }));

    let err = cache.insert_layer(0, &[], vec![], 0, 0).unwrap_err();
    assert!(matches!(err, CacheError::ZeroDim));
}

#[test]
fn insert_layer_normalizes_unit_inputs() {
    let mut cache = ShardCache::new(0.97);
    cache
        .insert_layer(7, &[3.0, 4.0], vec![1.0, 1.0], 1, 2)
        .unwrap();
    // Direction-equal query at the same row → hit at cos = 1.
    let out = cache.knn_lookup(7, &[6.0, 8.0], 1, 0.97);
    assert!(out.mlp_out.is_some());
    assert!((out.best_sim - 1.0).abs() < 1e-5);
}

#[test]
fn accessors_report_cache_shape() {
    let mut cache = ShardCache::new(0.5);
    assert!(cache.is_empty());
    cache
        .insert_layer(0, &[1.0, 0.0], vec![1.0, 1.0], 1, 2)
        .unwrap();
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.layer_size(0), Some(1));
    assert_eq!(cache.layer_size(99), None);
    assert!((cache.tau() - 0.5).abs() < 1e-6);
}

#[test]
fn cache_error_display_includes_lengths() {
    let e = CacheError::InputShape { got: 3, want: 4 };
    let s = format!("{e}");
    assert!(s.contains("3") && s.contains("4"));
    let e = CacheError::OutputShape { got: 2, want: 6 };
    let s = format!("{e}");
    assert!(s.contains("2") && s.contains("6"));
    let s = format!("{}", CacheError::ZeroDim);
    assert!(s.contains("d must"));
}

// ── gRPC handler (exercised in-process) ──────────────────────────────────

#[tokio::test]
async fn grpc_query_returns_hit_on_matching_vector() {
    let cache = Arc::new(RwLock::new(cache_with_two_entries(4, 0.97)));
    let svc = ShardGrpcService::from_cache(cache);
    let req = Request::new(ShardQuery {
        layer_id: 26,
        k: 1,
        query_vec: encode_f32_le(&[1.0, 0.0, 0.0, 0.0]),
        tau_override: 0.0,
    });
    let resp = svc.query(req).await.unwrap().into_inner();
    assert!(resp.hit);
    let mlp = decode_f32_le(&resp.mlp_out).unwrap();
    assert_eq!(mlp, vec![10.0, 20.0, 30.0, 40.0]);
    assert!((resp.best_sim - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn grpc_query_returns_miss_when_below_tau() {
    let cache = Arc::new(RwLock::new(cache_with_two_entries(4, 0.97)));
    let svc = ShardGrpcService::from_cache(cache);
    let req = Request::new(ShardQuery {
        layer_id: 26,
        k: 1,
        query_vec: encode_f32_le(&[0.0, 0.0, 1.0, 0.0]),
        tau_override: 0.0,
    });
    let resp = svc.query(req).await.unwrap().into_inner();
    assert!(!resp.hit);
    assert!(resp.mlp_out.is_empty());
    assert!(resp.best_sim < 0.97);
}

#[tokio::test]
async fn grpc_tau_override_takes_precedence() {
    let cache = Arc::new(RwLock::new(cache_with_two_entries(4, 0.5)));
    let svc = ShardGrpcService::from_cache(cache);
    // Query [1, 1, 0, 0] hits at cos = 0.7071. tau_override = 0.99 → miss.
    let req = Request::new(ShardQuery {
        layer_id: 26,
        k: 1,
        query_vec: encode_f32_le(&[1.0, 1.0, 0.0, 0.0]),
        tau_override: 0.99,
    });
    let resp = svc.query(req).await.unwrap().into_inner();
    assert!(!resp.hit);
}

#[tokio::test]
async fn grpc_rejects_malformed_query_bytes() {
    let cache = Arc::new(RwLock::new(ShardCache::new(0.97)));
    let svc = ShardGrpcService::from_cache(cache);
    let req = Request::new(ShardQuery {
        layer_id: 0,
        k: 1,
        query_vec: vec![0u8; 7], // not a multiple of 4
        tau_override: 0.0,
    });
    let err = svc.query(req).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

// ── Vindex source ────────────────────────────────────────────────────────

/// Smoke-test the vindex enum variant. Constructs an empty
/// `PatchedVindex` (no gate / down weights loaded) so every
/// `gate_knn` returns `[]` and the source reports a clean miss
/// without panicking. End-to-end FFN-row lookups need a fully
/// loaded vindex which is exercised by the production deploy and
/// `larql-server`'s integration tests against real models.
#[tokio::test]
async fn vindex_source_returns_miss_when_index_is_empty() {
    use larql_vindex::PatchedVindex;
    let base = larql_vindex::VectorIndex::new(
        vec![None, None, None], // 3 layers, no gate vectors
        vec![None, None, None], // no down_meta
        3,
        8, // hidden_size — must match query length
    );
    let patched = Arc::new(RwLock::new(PatchedVindex::new(base)));
    let source = ShardSource::vindex(patched, 0.97);
    let lookup = source.lookup(0, &[0.0f32; 8], 1, 0.97).await;
    assert!(lookup.mlp_out.is_none(), "empty vindex must miss");
    assert_eq!(lookup.best_sim, 0.0);
}

#[tokio::test]
async fn vindex_source_default_tau_is_constructor_arg() {
    use larql_vindex::PatchedVindex;
    let base = larql_vindex::VectorIndex::new(vec![None], vec![None], 1, 4);
    let patched = Arc::new(RwLock::new(PatchedVindex::new(base)));
    let source = ShardSource::vindex(patched, 0.42);
    assert!((source.default_tau().await - 0.42).abs() < 1e-6);
}

/// Vindex source with a patched gate vector but no down weights
/// wired: `gate_knn` returns a high-cosine hit, but
/// `ffn_row_into` falls through and returns false because the
/// empty base has no down storage to read from. Exercises the
/// "matched gate but missing down row → clean miss with best_sim
/// preserved for telemetry" branch — the same path a production
/// shard would take if an operator inserts gate-only patches.
#[tokio::test]
async fn vindex_source_reports_miss_when_down_row_unavailable() {
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};

    let base = VectorIndex::new(vec![None], vec![None], 1, 4);
    let mut patched = PatchedVindex::new(base);
    let meta = FeatureMeta {
        top_token: "test".into(),
        top_token_id: 0,
        c_score: 1.0,
        top_k: vec![TopKEntry {
            token: "test".into(),
            token_id: 0,
            logit: 1.0,
        }],
    };
    patched.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], meta);

    let source = ShardSource::vindex(Arc::new(RwLock::new(patched)), 0.5);
    let lookup = source.lookup(0, &[1.0, 0.0, 0.0, 0.0], 1, 0.5).await;
    assert!(lookup.mlp_out.is_none(), "no down storage → miss");
    // The match still surfaced before the down lookup failed —
    // best_sim reflects the gate cosine, useful for diagnosing
    // mis-wired caches.
    assert!(lookup.best_sim >= 0.99, "got best_sim={}", lookup.best_sim);
}

#[tokio::test]
async fn vindex_source_misses_when_below_tau() {
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};

    let base = VectorIndex::new(vec![None], vec![None], 1, 4);
    let mut patched = PatchedVindex::new(base);
    let meta = FeatureMeta {
        top_token: "x".into(),
        top_token_id: 0,
        c_score: 1.0,
        top_k: vec![TopKEntry {
            token: "x".into(),
            token_id: 0,
            logit: 1.0,
        }],
    };
    patched.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], meta);

    let source = ShardSource::vindex(Arc::new(RwLock::new(patched)), 0.99);
    // Query orthogonal to the patched gate → best_sim ≈ 0 < tau.
    let lookup = source.lookup(0, &[0.0, 1.0, 0.0, 0.0], 1, 0.99).await;
    assert!(lookup.mlp_out.is_none());
    assert!(lookup.best_sim < 0.99);
}

#[tokio::test]
async fn vindex_source_k_gt_one_exercises_weighted_average_path() {
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, PatchedVindex, VectorIndex};

    let base = VectorIndex::new(vec![None], vec![None], 1, 4);
    let mut patched = PatchedVindex::new(base);
    let meta = |i: u32| FeatureMeta {
        top_token: format!("f{i}"),
        top_token_id: i,
        c_score: 1.0,
        top_k: vec![TopKEntry {
            token: format!("f{i}"),
            token_id: i,
            logit: 1.0,
        }],
    };
    patched.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], meta(0));
    patched.insert_feature(0, 1, vec![0.0, 1.0, 0.0, 0.0], meta(1));

    let source = ShardSource::vindex(Arc::new(RwLock::new(patched)), 0.3);
    // Query at 45° hits both features at cos ≈ 0.7071. ffn_row_into
    // falls through on both (no down storage) → miss. This still
    // exercises the k > 1 weighted-average branch ahead of the
    // failing row lookup.
    let q = [1.0 / 2f32.sqrt(), 1.0 / 2f32.sqrt(), 0.0, 0.0];
    let lookup = source.lookup(0, &q, 2, 0.3).await;
    assert!(lookup.mlp_out.is_none());
    assert!(lookup.best_sim > 0.3);
}

// ── ShardCache::seed_from_normed validation branches ─────────────────────

#[test]
fn seed_from_normed_validates_shape() {
    let mut cache = ShardCache::new(0.5);
    let err = cache
        .seed_from_normed(0, vec![1.0, 0.0], vec![1.0, 0.0, 0.0], 1, 2)
        .unwrap_err();
    assert!(matches!(err, CacheError::OutputShape { .. }));

    let err = cache
        .seed_from_normed(0, vec![1.0, 0.0, 0.0], vec![1.0, 0.0], 1, 2)
        .unwrap_err();
    assert!(matches!(err, CacheError::InputShape { .. }));

    let err = cache.seed_from_normed(0, vec![], vec![], 0, 0).unwrap_err();
    assert!(matches!(err, CacheError::ZeroDim));
}

#[test]
fn shard_source_constructors_are_callable() {
    let cache = Arc::new(RwLock::new(ShardCache::new(0.5)));
    // Just confirm both constructors compile and the variants
    // round-trip — pattern matching on the enum keeps the variants
    // honest if someone re-orders them later.
    match ShardSource::cache(Arc::clone(&cache)) {
        ShardSource::Cache(_) => {}
        ShardSource::Vindex(_, _) => panic!("expected Cache"),
    }
}
