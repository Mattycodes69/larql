use super::*;
use tempfile::TempDir;

// ── helpers ─────────────────────────────────────────────────────────

/// Write a minimal valid index.json into `dir`.
fn write_minimal_index_json(dir: &std::path::Path, num_layers: usize, hidden: usize) {
    let json = serde_json::json!({
        "version": 2,
        "model": "test/unit",
        "family": "llama",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": 4,
        "vocab_size": 16,
        "embed_scale": 1.0,
        "layers": [],
        "down_top_k": 5,
        "has_model_weights": false,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none"
    });
    std::fs::write(dir.join("index.json"), json.to_string()).unwrap();
}

// ── load_vindex_config ──────────────────────────────────────────────

#[test]
fn load_vindex_config_parses_valid_json() {
    let dir = TempDir::new().unwrap();
    write_minimal_index_json(dir.path(), 2, 8);
    let cfg = load_vindex_config(dir.path()).unwrap();
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.hidden_size, 8);
    assert_eq!(cfg.model, "test/unit");
    assert_eq!(cfg.family, "llama");
}

/// A VINDEX3 `index.json` carrying **every** field `VindexConfig` needs.
///
/// The real one omits `intermediate_size`, which is the only reason the
/// ungated v1 entry points refused it — an accident of field overlap, not
/// a decision. This fixture removes that accident so the test measures the
/// generation gate itself. Without the gate, these parse cleanly and the
/// v1 loader proceeds against a layout whose weights are not there.
fn write_v3_index_json_that_would_parse_as_v1(dir: &Path) {
    let json = serde_json::json!({
        "version": 3,
        "model": "test/v3",
        "family": "gemma4",
        "num_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 4,
        "vocab_size": 16,
        "embed_scale": 1.0,
        "layers": [],
        "down_top_k": 5,
        "has_model_weights": false,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none",
        "moe_manifest": "moe_manifest.json",
        "segments": { "routed/layer_000": 1 }
    });
    std::fs::write(dir.join("index.json"), json.to_string()).unwrap();
}

#[test]
fn every_v1_entry_point_refuses_a_v3_container_by_generation() {
    // The regression this guards: each of these reads `index.json` for
    // itself, and `walk_cmd` reaches `load_vindex_embeddings` *before* the
    // gated `load_vindex_config`. If any one of them loses its gate, a
    // VINDEX3 directory is served against v1 offsets rather than refused.
    let dir = TempDir::new().unwrap();
    write_v3_index_json_that_would_parse_as_v1(dir.path());

    for (entry, err) in [
        ("load_vindex_config", load_vindex_config(dir.path()).err()),
        (
            "load_vindex_embeddings",
            load_vindex_embeddings(dir.path()).err(),
        ),
        (
            "load_vindex_with_range",
            VectorIndex::load_vindex_with_range(dir.path(), &mut crate::SilentLoadCallbacks, None)
                .err(),
        ),
    ] {
        let err = err.unwrap_or_else(|| panic!("{entry} accepted a VINDEX3 container"));
        let msg = err.to_string();
        assert!(
            msg.contains("VINDEX3") || msg.contains("VINDEX2 loader"),
            "{entry} refused for the wrong reason — the message must name \
             the generation, not a missing field. Got: {msg}"
        );
    }
}

#[test]
fn load_vindex_config_missing_file_errors() {
    let dir = TempDir::new().unwrap();
    let result = load_vindex_config(dir.path());
    assert!(result.is_err());
}

#[test]
fn load_vindex_config_malformed_json_errors() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("index.json"), b"{not valid json}").unwrap();
    let result = load_vindex_config(dir.path());
    assert!(result.is_err());
}

// ── load_feature_labels ─────────────────────────────────────────────

#[test]
fn load_feature_labels_compact_format() {
    let dir = TempDir::new().unwrap();
    let jsonl = r#"{"l":0,"f":0,"t":"Paris"}
{"l":0,"f":1,"t":"French"}
{"l":1,"f":0,"t":"Berlin"}
"#;
    let path = dir.path().join(DOWN_META_JSONL);
    std::fs::write(&path, jsonl).unwrap();
    let labels = load_feature_labels(&path).unwrap();
    assert_eq!(labels.len(), 3);
    assert_eq!(labels[&(0, 0)], "Paris");
    assert_eq!(labels[&(0, 1)], "French");
    assert_eq!(labels[&(1, 0)], "Berlin");
}

#[test]
fn load_feature_labels_full_format() {
    let dir = TempDir::new().unwrap();
    let jsonl = r#"{"layer":2,"feature":5,"top_token":"Spain"}
"#;
    let path = dir.path().join(DOWN_META_JSONL);
    std::fs::write(&path, jsonl).unwrap();
    let labels = load_feature_labels(&path).unwrap();
    assert_eq!(labels[&(2, 5)], "Spain");
}

#[test]
fn load_feature_labels_skips_header_lines() {
    let dir = TempDir::new().unwrap();
    let jsonl = r#"{"_header":true,"version":1}
{"l":0,"f":0,"t":"Rome"}
"#;
    let path = dir.path().join(DOWN_META_JSONL);
    std::fs::write(&path, jsonl).unwrap();
    let labels = load_feature_labels(&path).unwrap();
    assert_eq!(labels.len(), 1);
    assert_eq!(labels[&(0, 0)], "Rome");
}

#[test]
fn load_feature_labels_skips_blank_lines() {
    let dir = TempDir::new().unwrap();
    let jsonl = "  \n{\"l\":0,\"f\":0,\"t\":\"Tokyo\"}\n\n";
    let path = dir.path().join(DOWN_META_JSONL);
    std::fs::write(&path, jsonl).unwrap();
    let labels = load_feature_labels(&path).unwrap();
    assert_eq!(labels.len(), 1);
}

#[test]
fn load_feature_labels_missing_file_errors() {
    let result = load_feature_labels(std::path::Path::new("/no/such/file.jsonl"));
    assert!(result.is_err());
}

#[test]
fn load_feature_labels_empty_file_returns_empty_map() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("empty.jsonl");
    std::fs::write(&path, b"").unwrap();
    let labels = load_feature_labels(&path).unwrap();
    assert!(labels.is_empty());
}

// ── VectorIndex::load_vindex — minimal fixture ──────────────────────

/// Write a zero-byte gate_vectors.bin and a matching index.json
/// for a model with no features (all-zero slices). This lets us test
/// `load_vindex` without running the full extract pipeline.
fn write_minimal_loadable_vindex(dir: &std::path::Path, num_layers: usize, hidden: usize) {
    // Empty gate_vectors.bin (0 features per layer → 0 bytes)
    std::fs::write(dir.join("gate_vectors.bin"), b"").unwrap();
    let json = serde_json::json!({
        "version": 2,
        "model": "test/unit",
        "family": "llama",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": 4,
        "vocab_size": 16,
        "embed_scale": 1.0,
        "layers": [],   // no layers → gate_slices all-zero
        "down_top_k": 5,
        "has_model_weights": false,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none"
    });
    std::fs::write(dir.join("index.json"), json.to_string()).unwrap();
}

#[test]
fn load_vindex_missing_dir_errors() {
    let mut cb = crate::index::SilentLoadCallbacks;
    let result = VectorIndex::load_vindex(std::path::Path::new("/nonexistent/vindex"), &mut cb);
    assert!(result.is_err());
}

#[test]
fn load_vindex_missing_index_json_errors() {
    let dir = TempDir::new().unwrap();
    // No index.json written
    let mut cb = crate::index::SilentLoadCallbacks;
    let result = VectorIndex::load_vindex(dir.path(), &mut cb);
    assert!(result.is_err());
}

#[test]
fn load_vindex_minimal_fixture_succeeds() {
    let dir = TempDir::new().unwrap();
    write_minimal_loadable_vindex(dir.path(), 3, 8);
    let mut cb = crate::index::SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
    assert_eq!(index.num_layers, 3);
    assert_eq!(index.hidden_size, 8);
}

/// Regression (review 2026-07-30 H4): an index.json layer entry
/// with `layer >= num_layers` used to panic on
/// `gate_slices[info.layer] = …`. Malformed manifests must surface
/// as `VindexError::Parse`.
#[test]
fn load_vindex_out_of_range_layer_entry_errors() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("gate_vectors.bin"), vec![0u8; 32]).unwrap();
    let json = serde_json::json!({
        "version": 2,
        "model": "test/unit",
        "family": "llama",
        "num_layers": 2,
        "hidden_size": 8,
        "intermediate_size": 4,
        "vocab_size": 16,
        "embed_scale": 1.0,
        // layer 5 >= num_layers 2 — corrupt/hand-edited manifest
        "layers": [
            {"layer": 5, "num_features": 1, "offset": 0, "length": 32}
        ],
        "down_top_k": 5,
        "has_model_weights": false,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none"
    });
    std::fs::write(dir.path().join("index.json"), json.to_string()).unwrap();
    let mut cb = crate::index::SilentLoadCallbacks;
    let err = match VectorIndex::load_vindex(dir.path(), &mut cb) {
        Ok(_) => panic!("out-of-range layer entry must error, not load"),
        Err(e) => e,
    };
    match err {
        VindexError::Parse(msg) => {
            assert!(msg.contains('5'), "should name the layer: {msg}");
            assert!(msg.contains('2'), "should name the bound: {msg}");
        }
        other => panic!("expected Parse error, got {other:?}"),
    }
}

/// Direct check of the shared bounds helper — also covers the
/// `synthesize_gate_from_q4k` path, which needs a full Q4K fixture
/// to reach end-to-end but calls the same helper first.
#[test]
fn check_layer_in_bounds_rejects_out_of_range() {
    assert!(check_layer_in_bounds(0, 2).is_ok());
    assert!(check_layer_in_bounds(1, 2).is_ok());
    let err = check_layer_in_bounds(2, 2).expect_err("layer == num_layers");
    assert!(err.to_string().contains("out of range"), "{err}");
    assert!(check_layer_in_bounds(usize::MAX, 0).is_err());
}

#[test]
fn load_vindex_with_range_sets_layer_range() {
    let dir = TempDir::new().unwrap();
    write_minimal_loadable_vindex(dir.path(), 4, 8);
    let mut cb = crate::index::SilentLoadCallbacks;
    let index = VectorIndex::load_vindex_with_range(dir.path(), &mut cb, Some((1, 3))).unwrap();
    assert!(index.is_layer_owned(1));
    assert!(index.is_layer_owned(2));
    assert!(!index.is_layer_owned(0));
    assert!(!index.is_layer_owned(3));
}

mod embedding_adoption;
mod synth_gate;
