//! `shard_archive` — the deterministic Mode B tar and its content hash.
//!
//! The content hash is only meaningful if the donor's announce and the
//! donor's `/v1/shard` stream produce byte-identical archives, and if the
//! bytes depend on file names and contents alone.

use std::path::Path;

use sha2::{Digest, Sha256};
use tempfile::TempDir;

use larql_server::shard_archive::{
    is_sha256_hex, sha256_hex, shard_content_sha256, write_shard_tar, SHARD_SHA256_HEX_LEN,
};

fn write(dir: &Path, rel: &str, bytes: &[u8]) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// A small vindex-shaped tree, created in `order`.
fn tree(order: &[&str]) -> TempDir {
    let dir = TempDir::new().unwrap();
    for rel in order {
        write(dir.path(), rel, rel.as_bytes());
    }
    dir
}

const MTIME_TICK: std::time::Duration = std::time::Duration::from_millis(20);
const FILES: [&str; 4] = ["index.json", "layers/l0.bin", "layers/l1.bin", "z.bin"];

#[test]
fn content_hash_is_the_sha256_of_the_streamed_archive() {
    let dir = tree(&FILES);
    let archive = write_shard_tar(dir.path(), Vec::new()).unwrap();
    let hash = shard_content_sha256(dir.path()).unwrap();
    assert_eq!(hash, sha256_hex(Sha256::new_with_prefix(&archive)));
    assert!(is_sha256_hex(&hash));
    // Streaming the same directory twice gives the same bytes.
    assert_eq!(archive, write_shard_tar(dir.path(), Vec::new()).unwrap());
}

#[test]
fn hash_ignores_creation_order_and_mtime() {
    let forward = tree(&FILES);
    // Past filesystem mtime granularity, so the two trees' mtimes differ.
    std::thread::sleep(MTIME_TICK);
    let mut reversed = FILES;
    reversed.reverse();
    let backward = tree(&reversed);
    assert_eq!(
        shard_content_sha256(forward.path()).unwrap(),
        shard_content_sha256(backward.path()).unwrap()
    );
}

#[test]
fn hash_changes_with_file_bytes_and_names() {
    let base = tree(&FILES);
    let base_hash = shard_content_sha256(base.path()).unwrap();

    let edited = tree(&FILES);
    write(edited.path(), "layers/l1.bin", b"different");
    assert_ne!(shard_content_sha256(edited.path()).unwrap(), base_hash);

    let renamed = tree(&FILES);
    std::fs::rename(renamed.path().join("z.bin"), renamed.path().join("y.bin")).unwrap();
    assert_ne!(shard_content_sha256(renamed.path()).unwrap(), base_hash);
}

#[test]
fn archive_round_trips_with_relative_paths() {
    let dir = tree(&FILES);
    let archive = write_shard_tar(dir.path(), Vec::new()).unwrap();
    let out = TempDir::new().unwrap();
    tar::Archive::new(archive.as_slice())
        .unpack(out.path())
        .unwrap();
    for rel in FILES {
        assert_eq!(std::fs::read(out.path().join(rel)).unwrap(), rel.as_bytes());
    }
    let names: Vec<String> = tar::Archive::new(archive.as_slice())
        .entries()
        .unwrap()
        .map(|e| e.unwrap().path().unwrap().display().to_string())
        .collect();
    assert!(
        names
            .iter()
            .all(|n| !n.starts_with("./") && !n.starts_with('/')),
        "entries must be relative: {names:?}"
    );
}

#[test]
fn missing_directory_is_an_error() {
    let dir = TempDir::new().unwrap();
    assert!(shard_content_sha256(&dir.path().join("absent")).is_err());
}

#[cfg(unix)]
#[test]
fn symlink_cycle_is_refused_not_recursed_forever() {
    let dir = tree(&["index.json"]);
    std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();
    let err = shard_content_sha256(dir.path()).expect_err("cycle must be refused");
    assert!(err.to_string().contains("symlink cycle"), "got: {err}");
}

#[test]
fn sha256_hex_shape_is_checked() {
    let valid = "a".repeat(SHARD_SHA256_HEX_LEN);
    assert!(is_sha256_hex(&valid));
    assert!(is_sha256_hex(&sha256_hex(Sha256::new())));
    assert!(!is_sha256_hex(""));
    assert!(!is_sha256_hex("0123456789abcdef"), "identity-hash length");
    assert!(
        !is_sha256_hex(&"A".repeat(SHARD_SHA256_HEX_LEN)),
        "uppercase"
    );
    assert!(!is_sha256_hex(&"g".repeat(SHARD_SHA256_HEX_LEN)), "non-hex");
    assert!(!is_sha256_hex(&"a".repeat(SHARD_SHA256_HEX_LEN + 1)));
}
