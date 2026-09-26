//! CONSTRUCTION
//! FEATURE LOOKUP
//! GATE KNN
//! WALK
//! MUTATION

use super::*;

#[test]
fn new_index_has_correct_dimensions() {
    let idx = test_index();
    assert_eq!(idx.num_layers, 2);
    assert_eq!(idx.hidden_size, 4);
}

#[test]
fn loaded_layers() {
    let idx = test_index();
    assert_eq!(idx.loaded_layers(), vec![0, 1]);
}

#[test]
fn num_features_per_layer() {
    let idx = test_index();
    assert_eq!(idx.num_features(0), 3);
    assert_eq!(idx.num_features(1), 3);
    assert_eq!(idx.num_features(99), 0); // out of range
}

#[test]
fn total_counts() {
    let idx = test_index();
    assert_eq!(idx.total_gate_vectors(), 6); // 3 + 3
    assert_eq!(idx.total_down_meta(), 5); // 3 + 2 (one None)
}

#[test]
fn feature_meta_lookup() {
    let idx = test_index();
    let meta = idx.feature_meta(0, 0).unwrap();
    assert_eq!(meta.top_token, "Paris");
    assert_eq!(meta.top_token_id, 100);
    assert!((meta.c_score - 0.95).abs() < 0.01);
}

#[test]
fn feature_meta_none_for_missing() {
    let idx = test_index();
    assert!(idx.feature_meta(1, 1).is_none()); // explicitly None
    assert!(idx.feature_meta(99, 0).is_none()); // out of range layer
    assert!(idx.feature_meta(0, 99).is_none()); // out of range feature
}

#[test]
fn down_meta_at_returns_slice() {
    let idx = test_index();
    let metas = idx.down_meta_at(0).unwrap();
    assert_eq!(metas.len(), 3);
    assert!(metas[0].is_some());
    assert!(metas[1].is_some());
    assert!(metas[2].is_some());

    let metas1 = idx.down_meta_at(1).unwrap();
    assert!(metas1[1].is_none()); // the gap
}

#[test]
fn gate_knn_finds_best_match() {
    let idx = test_index();

    // Query along dim 0 → should match feature 0 at layer 0
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0); // feature 0
    assert!((hits[0].1 - 1.0).abs() < 0.01); // dot product = 1.0
}

#[test]
fn gate_knn_top_k_ordering() {
    let idx = test_index();

    // Query with components in dim 0 and dim 1
    let query = Array1::from_vec(vec![0.8, 0.6, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 3);

    assert_eq!(hits.len(), 3);
    // Feature 0 (dim 0): dot = 0.8
    // Feature 1 (dim 1): dot = 0.6
    // Feature 2 (dim 2): dot = 0.0
    assert_eq!(hits[0].0, 0); // highest
    assert_eq!(hits[1].0, 1);
}

#[test]
fn gate_knn_empty_for_missing_layer() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(99, &query, 5);
    assert!(hits.is_empty());
}

#[test]
fn walk_across_layers() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = idx.walk(&query, &[0, 1], 2);

    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 fires (dim 0 = 1.0)
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].feature, 0);
    assert_eq!(hits[0].meta.top_token, "Paris");

    // Layer 1: feature 1 fires (dim 0 contributes 0.5)
    let (layer1, hits1) = &trace.layers[1];
    assert_eq!(*layer1, 1);
    assert!(!hits1.is_empty());
}

#[test]
fn walk_skips_features_without_meta() {
    let idx = test_index();
    // Query that activates feature 1 at layer 1 (which has no metadata)
    let query = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let trace = idx.walk(&query, &[1], 3);

    // Feature 1 at layer 1 has None metadata — should be filtered out
    let (_, hits) = &trace.layers[0];
    for hit in hits {
        assert_ne!(hit.feature, 1); // feature 1 should not appear
    }
}

#[test]
fn set_feature_meta() {
    let mut idx = test_index();
    assert!(idx.feature_meta(1, 1).is_none());

    let meta = make_meta("London", 300, 0.85);
    idx.set_feature_meta(1, 1, meta);

    let loaded = idx.feature_meta(1, 1).unwrap();
    assert_eq!(loaded.top_token, "London");
    assert_eq!(loaded.top_token_id, 300);
}

#[test]
fn delete_feature_meta() {
    let mut idx = test_index();
    assert!(idx.feature_meta(0, 0).is_some());

    idx.delete_feature_meta(0, 0);
    assert!(idx.feature_meta(0, 0).is_none());
}

#[test]
fn find_free_feature() {
    let mut idx = test_index();

    // Layer 0: all 3 features have metadata → returns weakest (lowest c_score)
    // Scores: Paris=0.95, French=0.88, Europe=0.75 → weakest is Europe at F2
    let slot = idx.find_free_feature(0).unwrap();
    assert_eq!(slot, 2); // Europe has lowest c_score

    // Layer 1: feature 1 is None → returns empty slot first
    assert_eq!(idx.find_free_feature(1), Some(1));

    // Delete one in layer 0 → returns the now-empty slot
    idx.delete_feature_meta(0, 2);
    assert_eq!(idx.find_free_feature(0), Some(2));
}

#[test]
fn set_gate_vector() {
    let mut idx = test_index();
    let new_vec = Array1::from_vec(vec![0.0, 0.0, 0.0, 9.9]);
    idx.set_gate_vector(0, 1, &new_vec);

    // Query along dim 3 should now match feature 1 at layer 0
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = idx.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 1); // feature 1
    assert!((hits[0].1 - 9.9).abs() < 0.01);
}

#[test]
fn mutation_does_not_affect_other_features() {
    let mut idx = test_index();

    // Mutate feature 0
    idx.set_feature_meta(0, 0, make_meta("Modified", 999, 0.5));

    // Feature 1 should be unchanged
    let meta1 = idx.feature_meta(0, 1).unwrap();
    assert_eq!(meta1.top_token, "French");
}
