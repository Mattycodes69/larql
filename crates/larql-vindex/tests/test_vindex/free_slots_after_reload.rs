//! On a vindex loaded from disk, free-slot search sees the metadata written
//! since the load, so successive inserts claim successive slots instead of
//! overwriting one another.

use super::*;

/// Slots the reloaded layer leaves empty.
const FREE_SLOTS: usize = 3;
const OCCUPIED_SLOTS: usize = 1;
const HIDDEN: usize = 4;

fn reloaded_with_free_slots(dir: &std::path::Path) -> VectorIndex {
    let features = OCCUPIED_SLOTS + FREE_SLOTS;
    let mut gate = Array2::<f32>::zeros((features, HIDDEN));
    gate[[0, 0]] = 10.0;
    let mut meta = vec![None; features];
    meta[0] = Some(make_meta("Paris", 100, 0.95));
    let idx = VectorIndex::new(vec![Some(gate)], vec![Some(meta)], 1, HIDDEN);

    let layers = idx.save_gate_vectors(dir).unwrap();
    idx.save_down_meta(dir).unwrap();
    let config = VindexConfig {
        version: 2,
        model: "free-slots".into(),
        family: "test".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: HIDDEN,
        intermediate_size: features,
        vocab_size: 200,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers,
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    VectorIndex::save_config(&config, dir).unwrap();
    let tok =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok).unwrap();
    VectorIndex::load_vindex(dir, &mut larql_vindex::SilentLoadCallbacks).unwrap()
}

#[test]
fn successive_inserts_after_a_reload_claim_distinct_slots() {
    let dir = tempfile::tempdir().unwrap();
    let mut idx = reloaded_with_free_slots(dir.path());
    let mut claimed = Vec::new();
    for n in 0..FREE_SLOTS {
        let slot = idx.find_free_feature(0).expect("a free slot remains");
        idx.set_feature_meta(0, slot, make_meta("inserted", 110 + n as u32, 0.5));
        claimed.push(slot);
    }
    let mut unique = claimed.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        FREE_SLOTS,
        "each insert must claim its own slot: {claimed:?}"
    );
    assert!(
        !claimed.contains(&0),
        "the occupied slot is never handed out: {claimed:?}"
    );
}
