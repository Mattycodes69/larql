//! GateIndex trait tests
//! GATE WALK (BLAS gemv path)

use super::*;

#[test]
fn gate_index_trait_on_vector_index() {
    let index = test_index();
    let gi: &dyn GateIndex = &index;

    // num_features
    assert_eq!(gi.num_features(0), 3);
    assert_eq!(gi.num_features(1), 3);

    // feature_meta
    assert_eq!(gi.feature_meta(0, 0).unwrap().top_token, "Paris");
    assert_eq!(gi.feature_meta(1, 0).unwrap().top_token, "Berlin");
    assert!(gi.feature_meta(0, 99).is_none());

    // gate_knn
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = gi.gate_knn(0, &query, 3);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0, 0); // feature 0 responds to dim 0
}

#[test]
fn gate_index_trait_on_patched_vindex() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    // Insert a strong feature that should dominate KNN
    patched.insert_feature(
        0,
        2,
        vec![0.0, 0.0, 0.0, 100.0],
        make_meta("Inserted", 55, 5.0),
    );
    // Delete feature 0 (Paris)
    patched.delete_feature(0, 0);

    let gi: &dyn GateIndex = &patched;

    // feature_meta sees the insert
    assert_eq!(gi.feature_meta(0, 2).unwrap().top_token, "Inserted");
    // feature_meta sees the delete
    assert!(gi.feature_meta(0, 0).is_none());
    // base features still visible
    assert_eq!(gi.feature_meta(0, 1).unwrap().top_token, "French");

    // gate_knn sees the inserted feature
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = gi.gate_knn(0, &query, 5);
    assert_eq!(hits[0].0, 2); // inserted feature dominates
                              // gate_knn excludes the deleted feature
    assert!(hits.iter().all(|(f, _)| *f != 0));
}

#[test]
fn gate_index_patched_walk_sees_mutations() {
    // This is the core test: walk (used by WALK and DESCRIBE) sees patches.
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    // Before mutation: walk should find "Paris" at layer 0
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace_before = patched.walk(&query, &[0], 5);
    let layer0_before = &trace_before.layers[0].1;
    assert!(layer0_before.iter().any(|h| h.meta.top_token == "Paris"));

    // Insert a dominating feature
    patched.insert_feature(
        0,
        2,
        vec![100.0, 0.0, 0.0, 0.0],
        make_meta("NewCity", 77, 9.0),
    );
    // Delete Paris
    patched.delete_feature(0, 0);

    // After mutation: walk should find "NewCity", not "Paris"
    let trace_after = patched.walk(&query, &[0], 5);
    let layer0_after = &trace_after.layers[0].1;
    assert!(layer0_after.iter().any(|h| h.meta.top_token == "NewCity"));
    assert!(!layer0_after.iter().any(|h| h.meta.top_token == "Paris"));
}

#[test]
fn gate_index_dynamic_dispatch_matches_direct() {
    // Verify trait dispatch produces identical results to direct method calls
    let index = test_index();
    let gi: &dyn GateIndex = &index;

    let query = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);

    let direct = index.gate_knn(0, &query, 3);
    let via_trait = gi.gate_knn(0, &query, 3);

    assert_eq!(direct.len(), via_trait.len());
    for (d, t) in direct.iter().zip(via_trait.iter()) {
        assert_eq!(d.0, t.0);
        assert!((d.1 - t.1).abs() < 1e-6);
    }
}

#[test]
fn gate_walk_matches_gate_knn() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.5, 0.0, 0.0]);

    let knn = idx.gate_knn(0, &query, 3);
    let walk = idx.gate_walk(0, &query, 3).unwrap();

    // Same features, same order
    assert_eq!(knn.len(), walk.len());
    for (k, w) in knn.iter().zip(walk.iter()) {
        assert_eq!(k.0, w.0, "feature index mismatch");
        assert!(
            (k.1 - w.1).abs() < 1e-5,
            "score mismatch: {} vs {}",
            k.1,
            w.1
        );
    }
}

#[test]
fn gate_walk_returns_none_for_empty_layer() {
    let idx = VectorIndex::new(vec![None], vec![None], 1, 4);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    assert!(idx.gate_walk(0, &query, 5).is_none());
}
