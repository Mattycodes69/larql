//! DOWN VECTOR OVERRIDES (used by COMPILE INTO VINDEX baker)

use super::*;

#[test]
fn down_overrides_starts_empty() {
    let idx = test_index();
    assert!(idx.down_overrides().is_empty());
}

#[test]
fn set_down_vector_records_override() {
    let mut idx = test_index();
    let v = vec![0.1, 0.2, 0.3, 0.4];
    idx.set_down_vector(0, 1, v.clone());

    assert_eq!(idx.down_overrides().len(), 1);
    let stored = idx.down_overrides().get(&(0, 1)).unwrap();
    assert_eq!(stored, &v);

    // Trait method exposes the same data.
    let via_trait = idx.down_override(0, 1).unwrap();
    assert_eq!(via_trait, v.as_slice());

    // Inherent singular accessor returns the same slice.
    let via_inherent = idx.down_override_at(0, 1).unwrap();
    assert_eq!(via_inherent, v.as_slice());

    // Missing key returns None on both forms.
    assert!(idx.down_override_at(0, 2).is_none());
    assert!(idx.down_override(0, 2).is_none());

    // has_overrides_at reflects the layer.
    assert!(idx.has_overrides_at(0));
    assert!(!idx.has_overrides_at(1));
}

#[test]
fn patched_vindex_down_override_at_forwards_to_base() {
    let mut idx = test_index();
    let v = vec![1.0, 2.0, 3.0, 4.0];
    idx.set_down_vector(1, 0, v.clone());
    let patched = larql_vindex::PatchedVindex::new(idx);

    // Singular forwarder mirrors the inherent base accessor.
    let via_patched = patched.down_override_at(1, 0).unwrap();
    assert_eq!(via_patched, v.as_slice());
    assert!(patched.down_override_at(0, 0).is_none());

    // The plural collection accessor still returns everything.
    assert_eq!(patched.down_overrides().len(), 1);
}

#[test]
fn down_overrides_overwrite_in_place() {
    let mut idx = test_index();
    idx.set_down_vector(0, 1, vec![1.0; 4]);
    idx.set_down_vector(0, 1, vec![9.0; 4]);
    let stored = idx.down_overrides().get(&(0, 1)).unwrap();
    assert_eq!(stored, &vec![9.0; 4]);
    assert_eq!(idx.down_overrides().len(), 1);
}

#[test]
fn patched_vindex_exposes_base_down_overrides() {
    let mut idx = test_index();
    idx.set_down_vector(1, 0, vec![5.0, 5.0, 5.0, 5.0]);
    let patched = larql_vindex::PatchedVindex::new(idx);
    assert_eq!(patched.down_overrides().len(), 1);
    assert!(patched.down_overrides().contains_key(&(1, 0)));
}

#[test]
fn patched_vindex_overrides_gate_at_returns_inserted_gate() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Before insert: nothing.
    assert!(patched.overrides_gate_at(0, 0).is_none());

    let gate = vec![0.5, 0.5, 0.5, 0.5];
    let meta = make_meta("Inserted", 42, 0.99);
    patched.insert_feature(0, 0, gate.clone(), meta);

    let read = patched.overrides_gate_at(0, 0).unwrap();
    assert_eq!(read, gate.as_slice());

    // Other slots remain absent.
    assert!(patched.overrides_gate_at(0, 1).is_none());
    assert!(patched.overrides_gate_at(1, 0).is_none());
}
