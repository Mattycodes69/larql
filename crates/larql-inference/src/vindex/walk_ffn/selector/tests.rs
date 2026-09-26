use super::*;
use crate::test_utils::{
    attach_feature_major_f32_to_test_vindex, make_test_q4k_vindex_with_model_gate,
    make_test_q4k_weights, make_test_vindex, make_test_weights,
};
use crate::vindex::{WalkFfn, WalkFfnConfig};
use larql_vindex::{
    FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides,
    QuantizedFfnAccess,
};

/// Index whose batched gate scores contain a NaN — models a
/// corrupted / overflowed gate projection.
struct NanGateScoresIndex {
    n_features: usize,
}

impl GateLookup for NanGateScoresIndex {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        (0..top_k.min(self.n_features)).map(|i| (i, 1.0)).collect()
    }
    fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
        None
    }
    fn num_features(&self, _layer: usize) -> usize {
        self.n_features
    }
    fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> {
        let mut scores = vec![1.0f32; self.n_features];
        scores[1] = f32::NAN;
        Array2::from_shape_vec((1, self.n_features), scores).ok()
    }
}
impl PatchOverrides for NanGateScoresIndex {}
impl NativeFfnAccess for NanGateScoresIndex {}
impl QuantizedFfnAccess for NanGateScoresIndex {}
impl Fp4FfnAccess for NanGateScoresIndex {}

/// The NaN contract (2026-07-30 review, low tier): a NaN gate score
/// reaching the selector's top-K must PANIC — the previous
/// `partial_cmp(..).unwrap_or(Equal)` silently scrambled the
/// selection, splitting the contract from `top_k_by_abs` (which
/// deliberately panics) on the production gate-KNN path.
#[test]
#[should_panic(expected = "NaN in selector weight")]
fn joint_gate_knn_panics_on_nan_gate_score() {
    let weights = make_test_weights();
    let index = NanGateScoresIndex { n_features: 4 };
    let walk = crate::vindex::WalkFfn::new(&weights, &index, 2);
    let residual = Array1::from_vec(vec![0.02_f32; weights.hidden_size]);
    walk.joint_gate_knn(0, &residual, 2, FeatureSelector::GateOnly);
}

// ── Q4K-fixture selection chain ─────────────────────────────────────

/// Top-K used by the joint-selector tests — small enough that the
/// expected set is unambiguous on 256 random-valued features.
const JOINT_TEST_K: usize = 4;

fn residual_ramp(hidden: usize) -> Array1<f32> {
    Array1::from_vec((0..hidden).map(|i| (i as f32 + 1.0) * 0.02).collect())
}

/// The raw batched gate scores for layer 0 — the ground truth every
/// joint criterion must return scores from.
fn raw_gate_scores(index: &larql_vindex::VectorIndex, residual: &Array1<f32>) -> Vec<f32> {
    let x = Array2::from_shape_vec((1, residual.len()), residual.to_vec()).unwrap();
    index
        .gate_scores_batch_backend(0, &x, None)
        .expect("Q4K fixture must produce batched gate scores")
        .row(0)
        .to_vec()
}

/// Expected top-K feature set under a per-feature weight vector.
fn expected_top_k(weights_by_feat: &[f32], k: usize) -> std::collections::HashSet<usize> {
    let mut idx: Vec<usize> = (0..weights_by_feat.len()).collect();
    idx.sort_by(|&a, &b| {
        weights_by_feat[b]
            .partial_cmp(&weights_by_feat[a])
            .expect("test weights are NaN-free")
    });
    idx.truncate(k);
    idx.into_iter().collect()
}

/// Check a joint-selector result: K hits, unique in-range features,
/// scores are the RAW gate scores (not the joint weights), and the
/// selected set is exactly the top-K by the given joint weight.
fn assert_joint_selection(hits: &[(usize, f32)], raw: &[f32], joint_weight: &[f32], label: &str) {
    assert_eq!(hits.len(), JOINT_TEST_K, "{label}: wrong hit count");
    let feats: std::collections::HashSet<usize> = hits.iter().map(|(f, _)| *f).collect();
    assert_eq!(feats.len(), JOINT_TEST_K, "{label}: duplicate features");
    for &(f, g) in hits {
        assert!(f < raw.len(), "{label}: feature {f} out of range");
        assert_eq!(
            g, raw[f],
            "{label}: selector must return the RAW gate score for {f}"
        );
    }
    assert_eq!(
        feats,
        expected_top_k(joint_weight, JOINT_TEST_K),
        "{label}: selected set is not the top-K by the joint criterion"
    );
}

/// `down_row_norms` / `up_row_norms` must equal the row norms of the
/// dequantised Q4K cache — the selectors rank on these, so a stride
/// bug here silently reorders every joint selection (the H1 victim
/// class). Second call must serve the cached Arc.
#[test]
fn row_norm_caches_match_dequant_cache_norms() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
    );
    let hidden = weights.hidden_size;
    let n = index.num_features(0);

    for (component, norms, again) in [
        (
            2usize,
            walk.down_row_norms_pub(0),
            walk.down_row_norms_pub(0),
        ),
        (1usize, walk.up_row_norms_pub(0), walk.up_row_norms_pub(0)),
    ] {
        let norms = norms.expect("Q4K fixture must yield row norms");
        let again = again.expect("second call");
        assert!(
            std::sync::Arc::ptr_eq(&norms, &again),
            "component {component}: second call must hit the lazy cache"
        );
        assert_eq!(norms.len(), n);
        let cache = index
            .kquant_ffn_layer(0, component)
            .expect("dequant cache available");
        for feat in 0..n {
            let row = &cache[feat * hidden..(feat + 1) * hidden];
            let expected: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert_eq!(
                norms[feat], expected,
                "component {component}, feature {feat}: norm mismatch"
            );
        }
    }
}

/// Q4K arm of `compute_full_up_scores`: the batched Q4K matmul must
/// agree with the per-row fused Q4K dot (`ffn_row_dot`) — two
/// independent kernels over the same bytes.
#[test]
fn compute_full_up_scores_q4k_matches_per_row_dots() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
    );
    let residual = residual_ramp(weights.hidden_size);
    let scores = walk
        .compute_full_up_scores_pub(0, &residual)
        .expect("Q4K fixture must yield up scores");
    assert_eq!(scores.len(), index.num_features(0));
    let x_slice = residual.as_slice().unwrap();
    for (feat, &s) in scores.iter().enumerate() {
        let row_dot = index
            .ffn_row_dot(0, 1, feat, x_slice)
            .expect("per-row Q4K dot");
        assert!(
            (s - row_dot).abs() <= 1e-4 * row_dot.abs().max(1.0),
            "feature {feat}: batched {s} vs per-row {row_dot}"
        );
    }
}

/// Native-f32 arm of `compute_full_up_scores`: BLAS batched dot vs a
/// hand-computed per-row dot over the same up tensor.
#[test]
fn compute_full_up_scores_native_matches_manual_dots() {
    let weights = make_test_weights();
    let mut index = make_test_vindex(&weights);
    attach_feature_major_f32_to_test_vindex(&weights, &mut index);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
    );
    let residual = residual_ramp(weights.hidden_size);
    let scores = walk
        .compute_full_up_scores_pub(0, &residual)
        .expect("native fixture must yield up scores");
    let up = &weights.tensors[&weights.arch.ffn_up_key(0)];
    assert_eq!(scores.len(), weights.intermediate_size);
    for (feat, &s) in scores.iter().enumerate() {
        let manual: f32 = (0..weights.hidden_size)
            .map(|h| up[[feat, h]] * residual[h])
            .sum();
        assert!(
            (s - manual).abs() <= 1e-5 * manual.abs().max(1.0),
            "feature {feat}: BLAS {s} vs manual {manual}"
        );
    }
}

/// Every joint criterion, driven end-to-end on the Q4K fixture:
/// the selection must be exactly the top-K by that criterion's
/// weight (computed independently here from the raw scores + norms),
/// and the returned scores must be the raw gate scores.
#[test]
fn joint_gate_knn_ranks_by_each_criterion_on_q4k_fixture() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
    );
    let residual = residual_ramp(weights.hidden_size);
    let raw = raw_gate_scores(&index, &residual);
    let dn = walk.down_row_norms_pub(0).expect("down norms");
    let un = walk.up_row_norms_pub(0).expect("up norms");
    let us = walk
        .compute_full_up_scores_pub(0, &residual)
        .expect("up scores");
    // Gemma-3 fixture → GeluTanh activation, matching the selector's
    // `use_gelu` branch for ActXUpScoreXDownNorm.
    assert!(weights.arch.gate_up_is_gelu_tanh());

    let cases: Vec<(FeatureSelector, Vec<f32>, &str)> = vec![
        (
            FeatureSelector::GateOnly,
            raw.iter().map(|g| g.abs()).collect(),
            "GateOnly",
        ),
        (
            FeatureSelector::GateXDownNorm,
            raw.iter()
                .enumerate()
                .map(|(i, g)| g.abs() * dn[i])
                .collect(),
            "GateXDownNorm",
        ),
        (
            FeatureSelector::GateXUpDownNorm,
            raw.iter()
                .enumerate()
                .map(|(i, g)| g.abs() * dn[i] * un[i])
                .collect(),
            "GateXUpDownNorm",
        ),
        (
            FeatureSelector::GateXUpScore,
            raw.iter()
                .enumerate()
                .map(|(i, g)| g.abs() * us[i].abs())
                .collect(),
            "GateXUpScore",
        ),
        (
            FeatureSelector::ActXUpScoreXDownNorm,
            raw.iter()
                .enumerate()
                .map(|(i, &g)| (crate::ffn::gelu_tanh(g) * us[i]).abs() * dn[i])
                .collect(),
            "ActXUpScoreXDownNorm",
        ),
    ];
    for (kind, joint_weight, label) in cases {
        let hits = walk.joint_gate_knn(0, &residual, JOINT_TEST_K, kind);
        assert_joint_selection(&hits, &raw, &joint_weight, label);
    }
    assert_eq!(
        walk.selector_fallback_count(),
        0,
        "no joint call may have silently degraded to GateOnly (M10)"
    );
}

/// `Random` needs no determinism assertion — but it must return
/// exactly K unique valid features carrying their raw gate scores.
#[test]
fn joint_gate_knn_random_returns_k_valid_features_with_raw_scores() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
    );
    let residual = residual_ramp(weights.hidden_size);
    let raw = raw_gate_scores(&index, &residual);
    let hits = walk.joint_gate_knn(0, &residual, JOINT_TEST_K, FeatureSelector::Random);
    assert_eq!(hits.len(), JOINT_TEST_K);
    let feats: std::collections::HashSet<usize> = hits.iter().map(|(f, _)| *f).collect();
    assert_eq!(feats.len(), JOINT_TEST_K, "random draw must be unique");
    for &(f, g) in &hits {
        assert!(f < raw.len());
        assert_eq!(g, raw[f], "random hits still carry the raw gate score");
    }
}

/// `pool_restricted_gate_knn`: top-K by |gate| WITHIN the pool only,
/// out-of-range pool entries filtered, raw scores returned, empty
/// pool short-circuits.
#[test]
fn pool_restricted_gate_knn_respects_the_pool() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex_with_model_gate(&weights);
    let walk = WalkFfn::from_config(
        &weights,
        &index,
        WalkFfnConfig::sparse(weights.num_layers, JOINT_TEST_K),
    );
    let residual = residual_ramp(weights.hidden_size);
    let raw = raw_gate_scores(&index, &residual);

    assert!(walk
        .pool_restricted_gate_knn(0, &residual, 2, &[])
        .is_empty());

    let pool = [3usize, 250, 17, 99, usize::MAX]; // MAX must be filtered
    let k = 2;
    let hits = walk.pool_restricted_gate_knn(0, &residual, k, &pool);
    assert_eq!(hits.len(), k);
    // Expected: top-2 by |raw| among the valid pool entries.
    let mut valid: Vec<usize> = pool.iter().copied().filter(|&f| f < raw.len()).collect();
    valid.sort_by(|&a, &b| {
        raw[b]
            .abs()
            .partial_cmp(&raw[a].abs())
            .expect("NaN-free fixture")
    });
    let expected: std::collections::HashSet<usize> = valid[..k].iter().copied().collect();
    let got: std::collections::HashSet<usize> = hits.iter().map(|(f, _)| *f).collect();
    assert_eq!(
        got, expected,
        "must be the top-{k} by |gate| within the pool"
    );
    for &(f, g) in &hits {
        assert!(pool.contains(&f), "hit {f} escaped the pool");
        assert_eq!(g, raw[f]);
    }
}

/// Index without batched gate scores: GateOnly falls back to the
/// production chain WITHOUT counting a selector fallback (the label
/// is still truthful), while Random — a joint-null selector — must
/// count and trace the degradation (M10).
struct NoBatchIndex {
    n_features: usize,
}

impl GateLookup for NoBatchIndex {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        (0..top_k.min(self.n_features))
            .map(|i| (i, 1.0 / (i as f32 + 1.0)))
            .collect()
    }
    fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
        None
    }
    fn num_features(&self, _layer: usize) -> usize {
        self.n_features
    }
}
impl PatchOverrides for NoBatchIndex {}
impl NativeFfnAccess for NoBatchIndex {}
impl QuantizedFfnAccess for NoBatchIndex {}
impl Fp4FfnAccess for NoBatchIndex {}

#[test]
fn gate_only_fallback_is_not_counted_as_degradation() {
    let weights = make_test_weights();
    let index = NoBatchIndex { n_features: 8 };
    let walk = WalkFfn::new(&weights, &index, 4).with_dispatch_trace();
    let residual = residual_ramp(weights.hidden_size);

    let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::GateOnly);
    assert_eq!(hits.len(), 4, "production chain still selects");
    assert_eq!(
        walk.selector_fallback_count(),
        0,
        "GateOnly fallback IS the production selector — not a lie"
    );
    assert!(walk.take_dispatch_trace().is_empty());

    let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::Random);
    assert_eq!(hits.len(), 4);
    assert_eq!(
        walk.selector_fallback_count(),
        1,
        "Random degrading to GateOnly invalidates the null — count it"
    );
    let trace = walk.take_dispatch_trace();
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].path, "selector:fallback");
}

/// Batched scores with the wrong width (≠ num_features) must not be
/// trusted — the selector falls back to the production chain.
struct WrongWidthIndex {
    n_features: usize,
}

impl GateLookup for WrongWidthIndex {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        (0..top_k.min(self.n_features)).map(|i| (i, 0.5)).collect()
    }
    fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
        None
    }
    fn num_features(&self, _layer: usize) -> usize {
        self.n_features
    }
    fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> {
        // One column short — a truncated/corrupt gate projection.
        Some(Array2::zeros((1, self.n_features - 1)))
    }
}
impl PatchOverrides for WrongWidthIndex {}
impl NativeFfnAccess for WrongWidthIndex {}
impl QuantizedFfnAccess for WrongWidthIndex {}
impl Fp4FfnAccess for WrongWidthIndex {}

#[test]
fn joint_gate_knn_rejects_wrong_width_batch_scores() {
    let weights = make_test_weights();
    let index = WrongWidthIndex { n_features: 8 };
    let walk = WalkFfn::new(&weights, &index, 4);
    let residual = residual_ramp(weights.hidden_size);
    let hits = walk.joint_gate_knn(0, &residual, 4, FeatureSelector::GateOnly);
    // Fallback chain serves gate_knn's hits, whose scores are 0.5.
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|&(_, g)| g == 0.5));
}
