//! A compiled bank is mapped by its ledger's layout, not by a segment
//! header, and an empty region reports no resident bytes rather than
//! asking the OS about a zero-length range.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::super::compile::{CompilationLedger, OperandSeal};
use super::super::physical::PhysicalStore;

const STORE_ID: &str = "compiled-bank";
const OBJECT: &str = "target.decoder_stack";
const TENSOR: &str = "0.mlp.up_proj.weight";
const ENCODING: &str = "nvfp4";
/// Bytes before the sealed operand, so the ledger's offset is load-bearing.
const LEADING: usize = 8;
const OPERAND_LEN: usize = 16;

fn ledger() -> CompilationLedger {
    let seal = OperandSeal {
        object: OBJECT.to_string(),
        tensor: TENSOR.to_string(),
        encoding: ENCODING.to_string(),
        source_hash: String::new(),
        target_hash: String::new(),
        target_offset: LEADING as u64,
        target_len: OPERAND_LEN as u64,
    };
    CompilationLedger {
        map_name: ENCODING.to_string(),
        sealed: BTreeMap::from([(format!("{OBJECT}\u{1}{TENSOR}"), seal)]),
    }
}

#[test]
fn a_compiled_bank_is_addressed_by_its_ledger() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bank.bin");
    let bytes: Vec<u8> = (0..(LEADING + OPERAND_LEN) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let store = Arc::new(PhysicalStore::map_compiled(STORE_ID, &path, &ledger()).unwrap());
    assert_eq!(store.id(), STORE_ID);
    assert!(store.holds(TENSOR));
    let region = store.whole(TENSOR).unwrap();
    assert_eq!(region.bytes(), &bytes[LEADING..]);
    assert_eq!(region.store_id(), STORE_ID);
}

#[test]
fn a_missing_compiled_bank_is_an_io_error() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("absent.bin");
    assert!(PhysicalStore::map_compiled(STORE_ID, &missing, &ledger()).is_err());
}

#[test]
fn an_empty_region_has_no_resident_bytes() {
    let store = Arc::new(PhysicalStore::owned(
        STORE_ID,
        vec![0u8; OPERAND_LEN],
        BTreeMap::new(),
    ));
    let empty = store.span(0, 0).unwrap();
    assert!(empty.is_empty());
    #[cfg(unix)]
    assert_eq!(empty.resident_bytes(), Some(0));
}
