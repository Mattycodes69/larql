//! Q4 GATE KNN
//! LM HEAD KNN
//! HNSW INTEGRATION

use super::*;

#[test]
fn gate_knn_q4_produces_results() {
    let hidden = 256;
    let features = 64;
    let gate: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let gate_arr = Array2::from_shape_vec((features, hidden), gate.clone()).unwrap();
    let idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);

    let q4_data = larql_compute::cpu::q4::quantize_q4_0(&gate);
    let backend = larql_compute::cpu_backend();
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.01).sin());

    // Simulate Q4 scoring path (same logic as gate_knn_q4)
    let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(query.as_slice().unwrap());
    let scores = backend
        .q4_matvec(&q4_data, &q8_x, &q8_scales, features, hidden)
        .unwrap();
    assert_eq!(scores.len(), features);
    assert!(
        scores.iter().any(|&v| v.abs() > 0.01),
        "Q4 should produce nonzero scores"
    );

    // f32 KNN for comparison
    let f32_hits = idx.gate_knn(0, &query, 5);
    assert_eq!(f32_hits.len(), 5);

    // Q4 top-1 should usually match f32 top-1 (same dominant feature)
    let mut q4_indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    q4_indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    assert_eq!(
        q4_indexed[0].0, f32_hits[0].0,
        "Q4 top-1 should match f32 top-1"
    );
}

#[test]
fn gate_knn_q4_method_works() {
    use larql_compute::cpu::q4::quantize_q4_0;

    let hidden = 256;
    let features = 64;
    let gate_f32: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&gate_f32);
    let gate_arr = Array2::from_shape_vec((features, hidden), gate_f32).unwrap();

    // Build index with gate vectors and manually set Q4 data
    let mut idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);

    // Save Q4 to temp file, then load
    let dir = std::env::temp_dir().join("larql_test_q4_gate");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("gate_vectors_q4.bin"), &q4_data).unwrap();
    idx.load_gate_vectors_q4(&dir).unwrap();
    assert!(idx.has_gate_q4());

    // Now call gate_knn_q4
    let backend = larql_compute::cpu_backend();
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.01).sin());
    let hits = idx.gate_knn_q4(0, &query, 5, backend.as_ref()).unwrap();
    assert_eq!(hits.len(), 5);
    assert!(
        hits[0].1.abs() > hits[4].1.abs(),
        "results should be sorted by abs score"
    );

    // Compare with f32 KNN
    let f32_hits = idx.gate_knn(0, &query, 5);
    assert_eq!(hits[0].0, f32_hits[0].0, "Q4 top-1 should match f32 top-1");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gate_q4_data_returns_correct_bytes() {
    use larql_compute::cpu::q4::quantize_q4_0;

    let hidden = 256;
    let features = 32;
    let gate_f32: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&gate_f32);
    let gate_arr = Array2::from_shape_vec((features, hidden), gate_f32).unwrap();

    let mut idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);

    let dir = std::env::temp_dir().join("larql_test_q4_data");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("gate_vectors_q4.bin"), &q4_data).unwrap();
    idx.load_gate_vectors_q4(&dir).unwrap();

    let loaded = idx.gate_q4_data(0).unwrap();
    assert_eq!(loaded.len(), q4_data.len());
    assert_eq!(loaded, q4_data.as_slice());

    // Out of range returns None
    assert!(idx.gate_q4_data(99).is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn lm_head_knn_returns_top_k() {
    let hidden = 4;
    let vocab = 8;

    // Build a small lm_head: [vocab, hidden]
    let mut lm_head = vec![0.0f32; vocab * hidden];
    // Token 0 responds to dim 0
    lm_head[0] = 10.0;
    // Token 3 responds to dim 1
    lm_head[3 * hidden + 1] = 5.0;
    // Token 7 responds to dim 2
    lm_head[7 * hidden + 2] = 3.0;

    // Write to temp file
    let dir = std::env::temp_dir().join("larql_test_lm_head");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let lm_bytes: Vec<u8> = lm_head.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(dir.join(LM_HEAD_BIN), &lm_bytes).unwrap();

    let mut idx = VectorIndex::new(vec![None], vec![None], 1, hidden);
    idx.load_lm_head(&dir).unwrap();
    assert!(idx.has_lm_head());

    // Query aligned with dim 0 → token 0 should win
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.lm_head_knn(&query, 3);
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].0, 0, "token 0 should be top-1 for dim 0 query");
    assert!(
        hits[0].1 > hits[1].1,
        "results should be sorted by score desc"
    );

    // Query aligned with dim 1 → token 3 should win
    let query = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let hits = idx.lm_head_knn(&query, 3);
    assert_eq!(hits[0].0, 3, "token 3 should be top-1 for dim 1 query");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hnsw_enable_disable() {
    let idx = test_index();
    assert!(!idx.is_hnsw_enabled());

    idx.enable_hnsw(100);
    assert!(idx.is_hnsw_enabled());

    idx.disable_hnsw();
    assert!(!idx.is_hnsw_enabled());
}

#[test]
fn hnsw_knn_produces_valid_results() {
    let hidden = 128;
    let features = 500;
    let gate = Array2::from_shape_fn((features, hidden), |(r, c)| {
        let s = (r * hidden + c) as u64;
        let h = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (h >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0
    });
    let idx = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.1).sin());

    // HNSW via VectorIndex integration
    idx.enable_hnsw(100);
    let hnsw = idx.gate_knn(0, &query, 10);
    idx.disable_hnsw();

    // HNSW should return valid, non-empty results with features in range
    assert_eq!(hnsw.len(), 10, "HNSW should return requested top-K");
    for (feat, score) in &hnsw {
        assert!(*feat < features, "feature index out of range");
        assert!(score.is_finite(), "score should be finite");
    }
    // Results should be sorted by absolute score descending
    for w in hnsw.windows(2) {
        assert!(
            w[0].1.abs() >= w[1].1.abs(),
            "results should be sorted by |score| desc"
        );
    }
}
