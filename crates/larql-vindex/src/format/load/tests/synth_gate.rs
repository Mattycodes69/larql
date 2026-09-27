//! A Q4K vindex extracted without `gate_vectors.bin` carries its gate
//! only once, in `interleaved_kquant.bin`; the loader synthesises the f16
//! gate mmap from those slices. These tests pin the synthesis and every
//! way a malformed manifest refuses rather than panics.

use super::*;
use crate::format::filenames::{INTERLEAVED_KQUANT_BIN, INTERLEAVED_KQUANT_MANIFEST_JSON};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;

const LAYERS: usize = 2;
const HIDDEN: usize = 16;
const FEATURES: usize = 2;
/// The format tag the registry resolves for Q4_K.
const Q4K_TAG: &str = "Q4_K";
/// Gate, up, down: the manifest's per-layer entry order.
const ENTRIES_PER_LAYER: usize = FFN_COMPONENTS_PER_LAYER;
/// Q4_K plus the f16 round trip stays well inside this on a unit ramp.
const Q4K_TOLERANCE: f32 = 0.1;

/// The gate weights layer `layer` stores: a ramp in [-1, 1) offset by
/// layer, so the two layers' slices are distinguishable.
fn gate_values(layer: usize) -> Vec<f32> {
    let n = FEATURES * HIDDEN;
    (0..n)
        .map(|i| (i as f32 / n as f32) * 2.0 - 1.0 + layer as f32 * 0.01)
        .collect()
}

fn quantised(values: &[f32]) -> Vec<u8> {
    let mut padded = values.to_vec();
    padded.resize(
        values.len().div_ceil(K_QUANT_BLOCK_ELEMS) * K_QUANT_BLOCK_ELEMS,
        0.0,
    );
    quantize_q4_k(&padded)
}

fn write_index(dir: &Path) {
    let layers: Vec<_> = (0..LAYERS)
        .map(|layer| {
            serde_json::json!({
                "layer": layer, "num_features": FEATURES, "offset": 0, "length": 0
            })
        })
        .collect();
    let json = serde_json::json!({
        "version": 2,
        "model": "test/synth",
        "family": "llama",
        "num_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": FEATURES,
        "vocab_size": 4,
        "embed_scale": 1.0,
        "layers": layers,
        "down_top_k": 5,
        "has_model_weights": false,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none"
    });
    std::fs::write(dir.join(INDEX_JSON), json.to_string()).unwrap();
}

/// Writes the Q4K file and returns the well-formed manifest over it, one
/// entry per (layer, component), each component a copy of the layer's gate.
fn write_kquant(dir: &Path) -> Vec<serde_json::Value> {
    let mut bytes = Vec::new();
    let mut manifest = Vec::new();
    for layer in 0..LAYERS {
        let slice = quantised(&gate_values(layer));
        for _ in 0..ENTRIES_PER_LAYER {
            manifest.push(serde_json::json!({
                "offset": bytes.len(), "length": slice.len(), "format": Q4K_TAG
            }));
            bytes.extend_from_slice(&slice);
        }
    }
    std::fs::write(dir.join(INTERLEAVED_KQUANT_BIN), bytes).unwrap();
    manifest
}

fn write_manifest(dir: &Path, manifest: &[serde_json::Value]) {
    std::fs::write(
        dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON),
        serde_json::Value::Array(manifest.to_vec()).to_string(),
    )
    .unwrap();
}

/// A vindex whose manifest is `edit`ed before it is written.
fn fixture(edit: impl FnOnce(&mut Vec<serde_json::Value>)) -> TempDir {
    let dir = TempDir::new().unwrap();
    write_index(dir.path());
    let mut manifest = write_kquant(dir.path());
    edit(&mut manifest);
    write_manifest(dir.path(), &manifest);
    dir
}

fn load_err(dir: &Path) -> String {
    match VectorIndex::load_vindex(dir, &mut crate::index::SilentLoadCallbacks) {
        Ok(_) => panic!("a malformed k-quant gate must refuse"),
        Err(e) => e.to_string(),
    }
}

/// The gate entry of `layer` in the manifest.
fn gate_entry(layer: usize) -> usize {
    layer * ENTRIES_PER_LAYER
}

#[test]
fn a_missing_gate_file_is_synthesised_from_the_kquant_gate_slices() {
    let dir = fixture(|_| {});
    let index =
        VectorIndex::load_vindex(dir.path(), &mut crate::index::SilentLoadCallbacks).unwrap();
    for layer in 0..LAYERS {
        assert_eq!(index.num_features(layer), FEATURES);
        let want = gate_values(layer);
        for feature in 0..FEATURES {
            let row = index.gate_vector(layer, feature).unwrap();
            assert_eq!(row.len(), HIDDEN);
            for (got, want) in row.iter().zip(&want[feature * HIDDEN..]) {
                assert!((got - want).abs() < Q4K_TOLERANCE, "{got} vs {want}");
            }
        }
    }
}

#[test]
fn a_layer_range_synthesises_only_the_owned_layers() {
    let dir = fixture(|_| {});
    let owned = (1, LAYERS);
    let index = VectorIndex::load_vindex_with_range(
        dir.path(),
        &mut crate::index::SilentLoadCallbacks,
        Some(owned),
    )
    .unwrap();
    assert_eq!(index.num_features(0), 0, "an unowned layer gets no slice");
    assert_eq!(index.num_features(1), FEATURES);
    assert!(index.gate_vector(1, 0).is_some());
}

#[test]
fn a_kquant_file_without_its_manifest_refuses() {
    let dir = TempDir::new().unwrap();
    write_index(dir.path());
    write_kquant(dir.path());
    assert!(load_err(dir.path()).contains("manifest missing"));
}

#[test]
fn a_manifest_without_a_layers_gate_entry_refuses() {
    let dir = fixture(|m| m.truncate(gate_entry(1)));
    assert!(load_err(dir.path()).contains("missing gate entry for layer 1"));
}

#[test]
fn a_gate_entry_without_a_format_refuses() {
    let dir = fixture(|m| {
        m[gate_entry(0)].as_object_mut().unwrap().remove("format");
    });
    assert!(load_err(dir.path()).contains("missing `format`"));
}

#[test]
fn an_unknown_format_tag_refuses() {
    let dir = fixture(|m| m[gate_entry(0)]["format"] = "Q9_Z".into());
    assert!(load_err(dir.path()).contains("unknown format tag"));
}

#[test]
fn an_overflowing_gate_extent_refuses() {
    let dir = fixture(|m| {
        m[gate_entry(0)]["offset"] = u64::MAX.into();
        m[gate_entry(0)]["length"] = 1.into();
    });
    assert!(load_err(dir.path()).contains("overflow"));
}

#[test]
fn a_gate_extent_past_the_file_refuses() {
    let dir = fixture(|m| m[gate_entry(1)]["offset"] = (u32::MAX as u64).into());
    assert!(load_err(dir.path()).contains("exceeds mmap length"));
}

#[test]
fn a_gate_slice_too_short_to_dequantise_refuses() {
    let dir = fixture(|m| m[gate_entry(0)]["length"] = 1.into());
    assert!(load_err(dir.path()).contains("dequantize layer 0"));
}

#[test]
fn an_out_of_range_layer_refuses_before_synthesis() {
    let dir = fixture(|_| {});
    let path = dir.path().join(INDEX_JSON);
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    json["layers"][0]["layer"] = LAYERS.into();
    std::fs::write(&path, json.to_string()).unwrap();
    assert!(load_err(dir.path()).contains("out of range"));
}
