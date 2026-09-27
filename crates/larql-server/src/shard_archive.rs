//! The Mode B shard archive — one deterministic tar encoding of a vindex
//! directory, shared by the donor that streams it (`GET /v1/shard/...`) and
//! the donor's announce, which publishes its SHA-256 as the shard's CONTENT
//! hash (`AnnounceMsg.shard_sha256`).
//!
//! Both sides must produce byte-identical output for verification to mean
//! anything, so the encoding fixes every input the filesystem would otherwise
//! choose: entries are visited in sorted name order (not `read_dir` order),
//! paths are relative with no `./` prefix, and headers use
//! [`tar::HeaderMode::Deterministic`] (fixed mtime, uid/gid 0, normalised
//! permissions). The hash therefore changes only when a file's name or bytes
//! change.
//!
//! The CONTENT hash is deliberately distinct from the IDENTITY hash
//! (`announce::vindex_identity_hash`, carried as `AnnounceMsg.vindex_hash`):
//! identity names which model a server holds, content names the exact bytes
//! a receiver must end up with.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Length of a lowercase-hex SHA-256 digest (32 bytes, two hex chars each).
pub const SHARD_SHA256_HEX_LEN: usize = 64;

/// `true` when `s` is a lowercase-hex SHA-256 digest — the only shape a
/// content hash may take on the wire.
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == SHARD_SHA256_HEX_LEN && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Lowercase-hex encoding of a finished SHA-256 digest.
pub fn sha256_hex(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Write the deterministic shard tar of `dir` into `writer` and return the
/// writer once the archive is finished.
pub fn write_shard_tar<W: Write>(dir: &Path, writer: W) -> io::Result<W> {
    let mut builder = tar::Builder::new(writer);
    builder.mode(tar::HeaderMode::Deterministic);
    // Follow symlinks rather than archiving them; vindex directories
    // sometimes resolve through cache-style symlinks.
    builder.follow_symlinks(true);
    let root = std::fs::canonicalize(dir)?;
    append_sorted(&mut builder, &root, Path::new(""), &mut vec![root.clone()])?;
    builder.into_inner()
}

/// Append `dir`'s entries in sorted name order, recursing into
/// subdirectories. `ancestors` holds the canonical path of every directory on
/// the current walk: `follow_symlinks` would turn a link back to one of them
/// into unbounded recursion, so that is refused as a cycle.
fn append_sorted<W: Write>(
    builder: &mut tar::Builder<W>,
    dir: &Path,
    rel: &Path,
    ancestors: &mut Vec<PathBuf>,
) -> io::Result<()> {
    let mut entries: Vec<(OsString, PathBuf)> = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| (e.file_name(), e.path())))
        .collect::<io::Result<_>>()?;
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, path) in entries {
        let rel_path = rel.join(&name);
        // `metadata` (not `symlink_metadata`) follows links, matching the
        // builder's `follow_symlinks(true)`.
        if std::fs::metadata(&path)?.is_dir() {
            let canonical = std::fs::canonicalize(&path)?;
            if ancestors.contains(&canonical) {
                return Err(io::Error::other(format!(
                    "symlink cycle in shard tree: {} resolves to its ancestor {}",
                    rel_path.display(),
                    canonical.display()
                )));
            }
            builder.append_dir(&rel_path, &canonical)?;
            ancestors.push(canonical.clone());
            append_sorted(builder, &canonical, &rel_path, ancestors)?;
            ancestors.pop();
        } else {
            builder.append_path_with_name(&path, &rel_path)?;
        }
    }
    Ok(())
}

/// `io::Write` sink that hashes everything written and discards it.
struct HashingSink(Sha256);

impl Write for HashingSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The CONTENT hash of `dir`: lowercase-hex SHA-256 of [`write_shard_tar`]'s
/// output. Reads every byte of the directory once, so callers on an async
/// runtime run it under `spawn_blocking`.
pub fn shard_content_sha256(dir: &Path) -> io::Result<String> {
    let sink = write_shard_tar(dir, HashingSink(Sha256::new()))?;
    Ok(sha256_hex(sink.0))
}
