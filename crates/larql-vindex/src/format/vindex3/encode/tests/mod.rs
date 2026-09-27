//! Encode → inspect round trip: the G3 gate, on the fixture system.

use std::io::{Read, Seek, SeekFrom};

use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::encode::{
    encode_system, encode_system_unenforced, SEGMENTS_DIR, SYSTEM_GRAPH_JSON,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense, payload_pattern,
};

fn glimmer_system() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    Vec<(String, larql_models::inventory::ArchitectureInventory)>,
) {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    (target_dir, drafter_dir, named)
}

//
// A container is two agreeing descriptions of the same bytes — the
// directory in `index.json` and each segment's own header. Inspection
// exists to notice when they stop agreeing, so every disagreement it can
// name needs a case where it actually names it. These tamper with the
// directory only: the segments stay exactly as written, so any defect
// reported is a real disagreement and not a corrupted fixture.

/// Encode the dense fixture, then rewrite `index.json` through `edit`.
fn tampered_directory(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("only-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system_unenforced(&named, out.path()).unwrap();

    let index_path = out.path().join(crate::format::filenames::INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    edit(&mut index);
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    // Rendered as Debug: `InspectionDefect` carries its message as a
    // payload rather than implementing Display, and the message is what
    // these tests are about.
    inspect_container(out.path(), false)
        .unwrap()
        .defects
        .iter()
        .map(|d| format!("{d:?}"))
        .collect()
}

/// The first representation entry's id, for tests that need to name one.
fn first_representation(index: &serde_json::Value) -> String {
    index["representations"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone()
}

//
// `ArtifactSource` is the one place the encoder touches bytes it did not
// write. Every refusal below is about a shard that parses far enough to
// look usable and is not: the guards exist so a malformed checkpoint is
// named at open, not discovered as a wrong byte range halfway through a
// multi-gigabyte encode.

/// Write a `.safetensors` shard: 8-byte LE header length, header, payload.
fn write_shard(path: &std::path::Path, header: &str, payload: &[u8]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(payload).unwrap();
    f.flush().unwrap();
}

fn open_err(dir: &std::path::Path) -> String {
    use crate::format::vindex3::encode::source::ArtifactSource;
    match ArtifactSource::open(dir) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("opened a shard that should have been refused"),
    }
}

mod checkpoint_rs_the_shared_one_checkpoint;
mod directory_segment_coherence_what_inspect;
mod reading_an_untrusted_checkpoint;
