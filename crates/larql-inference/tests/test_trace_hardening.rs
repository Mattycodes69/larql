//! A context store's header is untrusted: counts it declares beyond the
//! format's own bounds are refused at open, not indexed.

use larql_inference::trace::{ContextStore, ContextTier, ContextWriter};

/// Byte offset of `n_critical` in the 128-byte header:
/// magic(4) version(4) hidden(4) n_layers(4) window(4) tier(1).
const N_CRITICAL_OFFSET: usize = 21;

fn written_store(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("ctx.ctxt");
    let writer =
        ContextWriter::create(&path, 8, 4, 16, ContextTier::FfnDeltas, &[1, 2], 4).unwrap();
    writer.finish().unwrap()
}

#[test]
fn a_well_formed_store_opens() {
    let dir = tempfile::tempdir().unwrap();
    let store = ContextStore::open(&written_store(dir.path())).unwrap();
    assert_eq!(store.critical_layers(), vec![1, 2]);
}

#[test]
fn more_critical_layers_than_the_format_holds_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = written_store(dir.path());
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[N_CRITICAL_OFFSET] = 200;
    std::fs::write(&path, bytes).unwrap();
    let err = ContextStore::open(&path).err().expect("must refuse");
    assert!(err.to_string().contains("critical layers"), "{err}");
}
