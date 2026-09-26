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

mod every_type {
    use larql_models::quant::ggml::*;

    /// (type, block elements, block bytes) for every block-quantised type.
    const BLOCKED: &[(u32, usize, usize)] = &[
        (TYPE_Q4_0, LEGACY_BLOCK_ELEMS, Q4_0_BLOCK_BYTES),
        (TYPE_Q4_1, LEGACY_BLOCK_ELEMS, Q4_1_BLOCK_BYTES),
        (TYPE_Q5_0, LEGACY_BLOCK_ELEMS, Q5_0_BLOCK_BYTES),
        (TYPE_Q5_1, LEGACY_BLOCK_ELEMS, Q5_1_BLOCK_BYTES),
        (TYPE_Q8_0, LEGACY_BLOCK_ELEMS, Q8_0_BLOCK_BYTES),
        (TYPE_Q3_K, K_QUANT_BLOCK_ELEMS, Q3_K_BLOCK_BYTES),
        (TYPE_Q4_K, K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES),
        (TYPE_Q5_K, K_QUANT_BLOCK_ELEMS, Q5_K_BLOCK_BYTES),
        (TYPE_Q6_K, K_QUANT_BLOCK_ELEMS, Q6_K_BLOCK_BYTES),
    ];

    #[test]
    fn every_block_type_sizes_whole_blocks_and_has_a_name() {
        for &(ty, elems, bytes) in BLOCKED {
            assert_eq!(
                tensor_data_size(ty, 2 * elems).unwrap(),
                2 * bytes,
                "{}",
                type_name(ty)
            );
            assert!(
                tensor_data_size(ty, elems + 1).is_err(),
                "{}",
                type_name(ty)
            );
            assert_ne!(type_name(ty), "unknown");
        }
        for ty in [
            TYPE_F32, TYPE_F16, TYPE_BF16, TYPE_Q2_K, TYPE_TQ1_0, TYPE_TQ2_0, TYPE_I2_S,
        ] {
            assert!(!type_name(ty).is_empty());
        }
        assert_eq!(tensor_data_size(TYPE_F16, 3).unwrap(), 6);
        assert_eq!(tensor_data_size(TYPE_BF16, 3).unwrap(), 6);
    }

    #[test]
    fn nvfp4_and_i2_s_size_through_their_own_rules() {
        assert!(tensor_data_size(TYPE_NVFP4, 64).unwrap() > 0);
        assert!(tensor_data_size(TYPE_I2_S, 8).is_ok());
        let err = tensor_data_size(TYPE_I2_S, 6).unwrap_err().to_string();
        assert!(err.contains("multiple of 4"), "{err}");
    }

    #[test]
    fn an_unknown_type_is_refused_by_size_and_by_decode() {
        const UNKNOWN: u32 = 999;
        assert!(tensor_data_size(UNKNOWN, 1).is_err());
        assert!(dequantize(&[], UNKNOWN, 0).is_err());
    }

    #[test]
    fn k_quant_rows_pad_to_whole_blocks() {
        assert_eq!(k_quant_padded_cols(1), K_QUANT_BLOCK_ELEMS);
        assert_eq!(
            k_quant_padded_cols(K_QUANT_BLOCK_ELEMS),
            K_QUANT_BLOCK_ELEMS
        );
        assert_eq!(
            k_quant_padded_cols(K_QUANT_BLOCK_ELEMS + 1),
            2 * K_QUANT_BLOCK_ELEMS
        );
    }

    #[test]
    fn zeroed_k_quant_blocks_decode_to_zeros() {
        for (ty, bytes) in [(TYPE_Q3_K, Q3_K_BLOCK_BYTES), (TYPE_Q5_K, Q5_K_BLOCK_BYTES)] {
            let out = dequantize(&vec![0u8; bytes], ty, K_QUANT_BLOCK_ELEMS).unwrap();
            assert_eq!(out.len(), K_QUANT_BLOCK_ELEMS);
            assert!(out.iter().all(|v| *v == 0.0), "{}", type_name(ty));
        }
    }

    #[test]
    fn a_short_half_precision_payload_is_refused() {
        assert!(dequantize(&[0u8; 3], TYPE_F16, 2).is_err());
        assert_eq!(dequantize(&[0u8; 4], TYPE_BF16, 2).unwrap(), vec![0.0, 0.0]);
    }
}
