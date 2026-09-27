//! `SmallMatrix`'s element counts, which callers use to check a shape
//! without knowing the stored dtype.

use super::*;

#[test]
fn len_counts_elements_whatever_the_stored_dtype() {
    let f = [1.0f32, 2.0, 3.0];
    let m = SmallMatrix::F32(&f);
    assert_eq!(m.len(), 3);
    assert!(!m.is_empty());
    assert_eq!(m.exact_len(), Some(3));

    // Three bf16 codes are six bytes: `len` counts codes, not bytes.
    let b = [0u8; 3 * BF16_BYTES];
    let m = SmallMatrix::Bf16(&b);
    assert_eq!(m.len(), 3);
    assert!(!m.is_empty());
    assert_eq!(m.exact_len(), Some(3));
}

#[test]
fn empty_matrices_are_empty_in_either_dtype() {
    assert!(SmallMatrix::F32(&[]).is_empty());
    assert!(SmallMatrix::Bf16(&[]).is_empty());
}

/// A half code rounds down in `len` — which is exactly why the shape
/// checks go through `exact_len`, which refuses it.
#[test]
fn a_partial_bf16_code_rounds_down_in_len_and_is_refused_by_exact_len() {
    let b = [0u8; 2 * BF16_BYTES + 1];
    let m = SmallMatrix::Bf16(&b);
    assert_eq!(m.len(), 2);
    assert_eq!(m.exact_len(), None);
}
