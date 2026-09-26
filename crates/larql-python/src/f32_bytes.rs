//! Safe conversion between Python `bytes` and `f32` buffers.
//!
//! A `bytes` object carries no alignment guarantee for `f32`, so its
//! contents are decoded element by element instead of cast in place.
//! Native byte order matches `numpy.ndarray.tobytes()`.

const F32_BYTES: usize = std::mem::size_of::<f32>();

/// Decode `bytes` as native-endian `f32`s. The caller has already checked
/// that the length is the expected multiple of four.
pub(crate) fn f32s_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(F32_BYTES)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Encode `f32`s as native-endian bytes, in iteration (logical) order.
pub(crate) fn f32s_to_bytes<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u8> {
    values.into_iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_from_an_unaligned_offset() {
        let values = [1.5f32, -2.25, 0.0, f32::MAX];
        let mut buf = vec![0u8];
        buf.extend(f32s_to_bytes(&values));
        assert_eq!(f32s_from_bytes(&buf[1..]), values);
    }
}
