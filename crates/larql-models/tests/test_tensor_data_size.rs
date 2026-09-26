//! `tensor_data_size` turns a header-declared element count into a byte
//! count; malformed counts must be refused, not floored or wrapped.

use larql_models::quant::ggml::{
    tensor_data_size, K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, TYPE_F32, TYPE_Q4_K,
};

#[test]
fn whole_blocks_size_exactly() {
    let n = 3 * K_QUANT_BLOCK_ELEMS;
    assert_eq!(
        tensor_data_size(TYPE_Q4_K, n).unwrap(),
        3 * Q4_K_BLOCK_BYTES
    );
    assert_eq!(tensor_data_size(TYPE_F32, 5).unwrap(), 5 * 4);
}

#[test]
fn partial_block_is_refused_not_floored() {
    let err = tensor_data_size(TYPE_Q4_K, K_QUANT_BLOCK_ELEMS + 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not a multiple"), "{err}");
}

#[test]
fn overflowing_byte_size_is_refused_not_wrapped() {
    let err = tensor_data_size(TYPE_F32, usize::MAX)
        .unwrap_err()
        .to_string();
    assert!(err.contains("overflows"), "{err}");
}
