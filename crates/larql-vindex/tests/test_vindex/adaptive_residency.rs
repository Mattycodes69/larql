//! ADAPTIVE RESIDENCY

use super::*;

#[test]
fn residency_pin_and_evict() {
    use larql_vindex::{LayerState, ResidencyManager};

    let mut rm = ResidencyManager::new(10, 4, 256, vec![32, 32, 32, 32]);
    assert_eq!(rm.num_pinned(), 0);
    assert_eq!(rm.state(0), LayerState::Cold);

    rm.mark_q4_available();
    assert_eq!(rm.state(0), LayerState::MmapQ4);

    // Pin layer 0 (32 features × 256 hidden / 32 * 18 = 4608 bytes)
    let fake_q4 = vec![0u8; 4608];
    assert!(rm.pin_layer(0, &fake_q4));
    assert_eq!(rm.state(0), LayerState::Pinned);
    assert_eq!(rm.num_pinned(), 1);
    assert!(rm.pinned_q4(0).is_some());

    // Pin again is a no-op
    assert!(rm.pin_layer(0, &fake_q4));
    assert_eq!(rm.num_pinned(), 1);

    // Evict
    rm.evict_layer(0);
    assert_eq!(rm.state(0), LayerState::MmapQ4);
    assert_eq!(rm.num_pinned(), 0);
    assert!(rm.pinned_q4(0).is_none());
}

#[test]
fn residency_budget_enforcement() {
    use larql_vindex::ResidencyManager;

    // Budget: 5 KB = 5120 bytes. Each layer's Q4 = 4608 bytes. Can fit 1, not 2.
    let mut rm = ResidencyManager::new(0, 2, 256, vec![32, 32]);
    // 0 MB budget — nothing should pin
    let data = vec![0u8; 4608];
    assert!(!rm.pin_layer(0, &data));
    assert_eq!(rm.num_pinned(), 0);

    // Use raw bytes to test budget: 4608 bytes per layer, budget just under 2 layers
    // We need a budget in MB that fits 1 layer but not 2.
    // 4608 * 2 = 9216 bytes. Create a manager and pin with exact byte checks.
    let _rm2 = ResidencyManager::new(1, 2, 256, vec![32, 32]); // 1 MB budget
                                                               // 1 MB >> 9216 bytes, so both will fit. Instead test with large layers.
                                                               // Use features=4096 so each layer is 4096*256/32*18 = 589,824 bytes = 0.56 MB
    let big_features = 4096;
    let big_data = vec![0u8; big_features * 256 / 32 * 18]; // ~576 KB
    let mut rm3 = ResidencyManager::new(1, 3, 256, vec![big_features; 3]); // 1 MB budget
    assert!(rm3.pin_layer(0, &big_data)); // ~576 KB, fits
    assert!(!rm3.pin_layer(1, &big_data)); // ~1152 KB total, exceeds 1 MB
    assert_eq!(rm3.num_pinned(), 1);
}

#[test]
fn residency_auto_pin_fills_budget() {
    use larql_vindex::ResidencyManager;

    let layers = 8;
    let features = 32;
    let hidden = 256;
    let layer_features = vec![features; layers];
    let q4_per_layer = features * hidden / 32 * 18; // 4608 bytes

    // Budget for 4 layers
    let budget_mb = 1; // 1 MB >> 4608 * 8 = 36 KB, so all fit
    let mut rm = ResidencyManager::new(budget_mb, layers, hidden, layer_features);
    rm.mark_q4_available();

    // Record accesses — layers 2, 5 are hot
    for _ in 0..100 {
        rm.record_access(2);
    }
    for _ in 0..50 {
        rm.record_access(5);
    }

    let pinned = rm.auto_pin(|_| Some(vec![0u8; q4_per_layer]));
    assert_eq!(pinned, layers); // budget fits all

    // Hottest layers should be pinned
    assert!(rm.pinned_q4(2).is_some());
    assert!(rm.pinned_q4(5).is_some());
}

#[test]
fn residency_pin_range() {
    use larql_vindex::ResidencyManager;

    let layers = 10;
    let features = 32;
    let hidden = 256;
    let q4_per_layer = features * hidden / 32 * 18;

    let mut rm = ResidencyManager::new(1, layers, hidden, vec![features; layers]);
    rm.mark_q4_available();

    // Pin knowledge band L3-L7
    let pinned = rm.pin_range(3, 8, |_| Some(vec![0u8; q4_per_layer]));
    assert_eq!(pinned, 5);
    assert!(rm.pinned_q4(3).is_some());
    assert!(rm.pinned_q4(7).is_some());
    assert!(rm.pinned_q4(2).is_none()); // not in range
    assert!(rm.pinned_q4(8).is_none()); // not in range
}

#[test]
fn residency_summary() {
    use larql_vindex::ResidencyManager;

    let mut rm = ResidencyManager::new(1, 4, 256, vec![32; 4]);
    rm.mark_q4_available();
    rm.pin_layer(0, &vec![0u8; 4608]);

    let s = rm.summary();
    assert!(s.contains("1 pinned"));
    assert!(s.contains("3 mmap"));
    assert!(s.contains("0 cold"));
}

#[test]
fn adaptive_gate_knn_uses_pinned() {
    use larql_compute::cpu::q4::quantize_q4_0;
    use larql_vindex::ResidencyManager;

    let hidden = 256;
    let features = 64;
    let gate_f32: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&gate_f32);
    let gate_arr = Array2::from_shape_vec((features, hidden), gate_f32).unwrap();

    let idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);
    let backend = larql_compute::cpu_backend();
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.01).sin());

    let mut rm = ResidencyManager::new(10, 1, hidden, vec![features]);
    rm.mark_q4_available();
    rm.pin_layer(0, &q4_data);

    // Adaptive dispatch should use pinned path
    let hits = idx.gate_knn_adaptive(0, &query, 5, &mut rm, backend.as_ref());
    assert_eq!(hits.len(), 5);

    // Should match f32 brute-force top-1
    let f32_hits = idx.gate_knn(0, &query, 5);
    assert_eq!(
        hits[0].0, f32_hits[0].0,
        "pinned Q4 top-1 should match f32 top-1"
    );
}
