//! CONTINUATION-CODEC-2's checkpoint store and its file names.
//!
//! W2 erratum: the as-run store named an arm's scores `n{n}-{arm}.json`
//! and its kept logits `n{n}-{arm}.f32` + index `n{n}-{arm}.json` — the
//! same file for arm C, so C's scores were overwritten by its logits index
//! (notes, W2). Logits now live under their own base; the as-run layout is
//! kept as a named reader so the W2 checkpoints are read exactly as
//! written.

use std::path::{Path, PathBuf};

use serde::{de::DeserializeOwned, Serialize};

use super::*;

const AUTHORITIES: &str = "authorities.json";
const F32_BYTES: usize = std::mem::size_of::<f32>();

/// An arm's per-position scores for a rung.
pub(super) fn scores_name(n: usize, arm: &str) -> String {
    format!("n{n}-{arm}.json")
}

/// The base of an arm's kept decode logits (`{base}.f32` + `{base}.json`).
pub(super) fn logits_base(n: usize, arm: &str) -> String {
    format!("n{n}-{arm}.logits")
}

/// W2's as-run logits base, whose index collided with arm C's scores.
pub(super) fn w2_as_run_logits_base(n: usize, arm: &str) -> String {
    format!("n{n}-{arm}")
}

pub(super) struct Checkpoints {
    dir: PathBuf,
}

impl Checkpoints {
    /// Open `dir` for a run, stamping it with `stamp` or refusing it if it
    /// carries another.
    pub(super) fn open(dir: &Path, stamp: &Value) -> Self {
        std::fs::create_dir_all(dir).unwrap();
        let store = Self {
            dir: dir.to_path_buf(),
        };
        match store.read::<Value>(AUTHORITIES) {
            Some(held) if &held != stamp => panic!(
                "{} holds checkpoints from other authorities; a resume must not mix them. \
                 Held: {held}\nNow: {stamp}",
                dir.display()
            ),
            Some(_) => eprintln!("resuming in {}", dir.display()),
            None => store.write(AUTHORITIES, stamp),
        }
        store
    }

    /// Open a finished run's directory for reading only; its stamp is the
    /// run's authorities.
    pub(super) fn open_as_run(dir: &Path) -> (Self, Value) {
        let store = Self {
            dir: dir.to_path_buf(),
        };
        let stamp = store
            .read::<Value>(AUTHORITIES)
            .unwrap_or_else(|| panic!("{} has no authorities stamp", dir.display()));
        (store, stamp)
    }

    pub(super) fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub(super) fn read<T: DeserializeOwned>(&self, name: &str) -> Option<T> {
        let raw = std::fs::read_to_string(self.path(name)).ok()?;
        Some(serde_json::from_str(&raw).expect("a checkpoint written by this run"))
    }

    /// Temp file, then rename: a checkpoint is whole or absent.
    fn write_bytes(&self, name: &str, bytes: &[u8]) {
        let tmp = self.path(&format!("{name}.tmp"));
        std::fs::write(&tmp, bytes).unwrap();
        std::fs::rename(&tmp, self.path(name)).unwrap();
    }

    pub(super) fn write<T: Serialize>(&self, name: &str, value: &T) {
        self.write_bytes(
            name,
            serde_json::to_string_pretty(value).unwrap().as_bytes(),
        );
    }

    /// Decode logits: raw f32 LE, then the index that makes them count
    /// (written last, so a stop between leaves no index).
    pub(super) fn write_logits(&self, base: &str, rows: &[Vec<f32>]) {
        let width = rows.first().map_or(0, Vec::len);
        let mut bytes = Vec::with_capacity(rows.len() * width * F32_BYTES);
        for x in rows.iter().flatten() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.write_bytes(&format!("{base}.f32"), &bytes);
        self.write(
            &format!("{base}.json"),
            &json!({"rows": rows.len(), "width": width}),
        );
    }

    pub(super) fn read_logits(&self, base: &str) -> Option<Vec<Vec<f32>>> {
        let index: Value = self.read(&format!("{base}.json"))?;
        let (rows, width) = (
            index["rows"].as_u64()? as usize,
            index["width"].as_u64()? as usize,
        );
        let bytes = std::fs::read(self.path(&format!("{base}.f32"))).ok()?;
        assert_eq!(
            bytes.len(),
            rows * width * F32_BYTES,
            "{base}: truncated logits"
        );
        let values: Vec<f32> = bytes
            .chunks_exact(F32_BYTES)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Some(values.chunks_exact(width).map(<[f32]>::to_vec).collect())
    }

    /// Every file in the store, sorted.
    pub(super) fn files(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}
