//! Stored bytes are viewed as `f32`s only when that is sound.

use larql_vindex::mmap_util::{f32_vec, f32_view};

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A buffer whose payload starts `shift` bytes past an aligned base.
fn shifted(values: &[f32], shift: usize) -> (Vec<u8>, usize) {
    let mut buf = vec![0u8; shift + values.len() * 4 + 8];
    let base = buf.as_ptr().align_offset(std::mem::align_of::<f32>());
    let start = base + shift;
    buf[start..start + values.len() * 4].copy_from_slice(&le_bytes(values));
    (buf, start)
}

#[test]
fn aligned_whole_floats_are_viewed_in_place() {
    let values = [1.0f32, -2.5, 3.25];
    let (buf, start) = shifted(&values, 0);
    let view = f32_view(&buf[start..start + 12]).unwrap();
    assert_eq!(view, values);
}

#[test]
fn misaligned_bytes_are_refused_not_read() {
    let values = [1.0f32, 2.0];
    let (buf, start) = shifted(&values, 1);
    assert!(f32_view(&buf[start..start + 8]).is_none());
}

#[test]
fn a_ragged_tail_is_refused() {
    let (buf, start) = shifted(&[1.0f32, 2.0], 0);
    assert!(f32_view(&buf[start..start + 7]).is_none());
}

#[test]
fn owned_decode_handles_misaligned_input() {
    let values = [0.5f32, -7.0, 1e-3];
    let (buf, start) = shifted(&values, 3);
    assert_eq!(f32_vec(&buf[start..start + 12]), values);
    let (buf, start) = shifted(&values, 0);
    assert_eq!(f32_vec(&buf[start..start + 12]), values);
}
