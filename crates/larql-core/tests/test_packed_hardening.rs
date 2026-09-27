//! Packed-format deserialisation must refuse crafted lengths before
//! allocating for them.

use larql_core::*;

/// Byte range of the header's string-table offset (`u64`, little-endian).
const STRING_TABLE_OFFSET_FIELD: std::ops::Range<usize> = 24..32;
/// Width of a string-table entry's length prefix.
const STRING_LEN_PREFIX: usize = 4;

fn packed_with_one_edge() -> Vec<u8> {
    let mut graph = Graph::new();
    graph.add_edge(Edge::new("A", "rel", "B"));
    to_packed_bytes(&graph).unwrap()
}

fn string_table_offset(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[STRING_TABLE_OFFSET_FIELD].try_into().unwrap()) as usize
}

#[test]
fn string_entry_longer_than_file_is_refused_not_allocated() {
    let mut bytes = packed_with_one_edge();
    let at = string_table_offset(&bytes);
    bytes[at..at + STRING_LEN_PREFIX].copy_from_slice(&u32::MAX.to_le_bytes());

    let err = from_packed_bytes(&bytes).unwrap_err().to_string();
    assert!(err.contains("declares"), "unexpected error: {err}");
}

#[test]
fn string_entry_one_byte_past_end_is_refused() {
    let mut bytes = packed_with_one_edge();
    let at = string_table_offset(&bytes);
    let remaining = bytes.len() - at - STRING_LEN_PREFIX;
    let too_long = u32::try_from(remaining + 1).unwrap();
    bytes[at..at + STRING_LEN_PREFIX].copy_from_slice(&too_long.to_le_bytes());

    assert!(from_packed_bytes(&bytes).is_err());
}

#[test]
fn well_formed_string_table_still_round_trips() {
    let bytes = packed_with_one_edge();
    let graph = from_packed_bytes(&bytes).unwrap();
    assert_eq!(graph.edge_count(), 1);
}
