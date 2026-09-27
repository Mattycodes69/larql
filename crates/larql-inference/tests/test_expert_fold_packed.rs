//! Checked byte slicing of the packed BF16 expert table
//! (`ffn::expert_fold::packed`).

use larql_inference::ffn::expert_fold::packed::{
    packed_bf16_expert, packed_expert_range, PACKED_BF16_BYTES,
};

const HIDDEN: usize = 4;
const INTER: usize = 3;
const EXPERTS: usize = 2;

fn tables() -> (Vec<u8>, Vec<u8>) {
    let down = HIDDEN * INTER * PACKED_BF16_BYTES;
    let gate_up = 2 * down;
    let gu: Vec<u8> = (0..gate_up * EXPERTS).map(|i| i as u8).collect();
    let dn: Vec<u8> = (0..down * EXPERTS)
        .map(|i| (i as u8).wrapping_add(100))
        .collect();
    (gu, dn)
}

#[test]
fn range_of_each_expert_in_table() {
    assert_eq!(packed_expert_range(0, 10, 20), Some(0..10));
    assert_eq!(packed_expert_range(1, 10, 20), Some(10..20));
}

#[test]
fn range_past_table_is_none() {
    assert_eq!(packed_expert_range(2, 10, 20), None);
    assert_eq!(packed_expert_range(1, 10, 19), None);
}

#[test]
fn range_with_overflowing_offset_is_none() {
    // 2^60 * 16 == 2^64 wraps to 0 without checked arithmetic.
    assert_eq!(packed_expert_range(1 << 60, 16, 64), None);
    // start fits but start + stride overflows.
    assert_eq!(packed_expert_range(1, usize::MAX, usize::MAX), None);
}

#[test]
fn bf16_expert_slices_each_expert() {
    let (gu, dn) = tables();
    let down = HIDDEN * INTER * PACKED_BF16_BYTES;
    for e in 0..EXPERTS {
        let (g, d) = packed_bf16_expert(&gu, &dn, e, HIDDEN, INTER).expect("in table");
        assert_eq!(g, &gu[e * 2 * down..(e + 1) * 2 * down]);
        assert_eq!(d, &dn[e * down..(e + 1) * down]);
    }
}

#[test]
fn bf16_expert_outside_either_table_is_none() {
    let (gu, dn) = tables();
    assert!(packed_bf16_expert(&gu, &dn, EXPERTS, HIDDEN, INTER).is_none());
    // Gate/up table holds both experts, down table only the first.
    let short_down = &dn[..HIDDEN * INTER * PACKED_BF16_BYTES];
    assert!(packed_bf16_expert(&gu, short_down, 1, HIDDEN, INTER).is_none());
    assert!(packed_bf16_expert(&gu, &dn, 1 << 60, HIDDEN, INTER).is_none());
}

#[test]
fn bf16_expert_with_overflowing_shape_is_none() {
    assert!(packed_bf16_expert(&[], &[], 0, usize::MAX, 2).is_none());
}
