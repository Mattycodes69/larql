//! Coverage for the f32 BLAS fallback path (`lm_head_knn`) plus the
//! `Stride32Mode` env-var dispatch and the early-return guards on
//! the f16 / Q4_K backend paths. The backend happy paths (Q4_K matvec,
//! stride-32, f16 GEMV) are in `backend_path_tests.rs`.
use ndarray::Array1;

use super::*;
use crate::format::filenames::LM_HEAD_BIN;

/// Build a `VectorIndex` with a synthetic `lm_head.bin` mmap'd in.
/// `lm_head` is row-major `[vocab, hidden]` f32.
fn vindex_with_lm_head(vocab: usize, hidden: usize, data: &[f32]) -> VectorIndex {
    assert_eq!(data.len(), vocab * hidden);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join(LM_HEAD_BIN);
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&path, &bytes).unwrap();

    let mut v = VectorIndex::empty(1, hidden);
    v.load_lm_head(tmp.path())
        .expect("synthetic lm_head loads cleanly");
    // Hold the tempdir for the test lifetime — the mmap's underlying
    // file stays open via memmap2 even if the dir is dropped on Linux,
    // but on macOS the file needs to outlive the mmap.
    std::mem::forget(tmp);
    assert_eq!(v.vocab_size, vocab);
    v
}

#[test]
fn lm_head_knn_returns_empty_when_no_mmap() {
    let v = VectorIndex::empty(1, 4);
    let q = Array1::from_vec(vec![1.0_f32; 4]);
    assert!(v.lm_head_knn(&q, 5).is_empty());
}

#[test]
fn lm_head_knn_returns_empty_when_vocab_zero() {
    // Force lm_head_mmap to be Some but vocab_size = 0.
    let mut v = VectorIndex::empty(1, 4);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join(LM_HEAD_BIN);
    std::fs::write(&path, [0u8; 16]).unwrap();
    v.load_lm_head(tmp.path()).unwrap();
    // Override vocab_size to 0 to hit the `vocab == 0` guard.
    v.vocab_size = 0;
    std::mem::forget(tmp);
    let q = Array1::from_vec(vec![1.0_f32; 4]);
    assert!(v.lm_head_knn(&q, 5).is_empty());
}

#[test]
fn lm_head_knn_returns_empty_when_mmap_too_small() {
    // Manually construct: tiny lm_head mmap with vocab=99 declared
    // → required bytes vastly exceeds actual mmap size.
    let mut v = VectorIndex::empty(1, 4);
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join(LM_HEAD_BIN);
    let bytes: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
    std::fs::write(&path, &bytes).unwrap();
    v.load_lm_head(tmp.path()).unwrap();
    // Manually inflate vocab to force a too-small check.
    v.vocab_size = 99;
    std::mem::forget(tmp);
    let q = Array1::from_vec(vec![1.0_f32; 4]);
    assert!(v.lm_head_knn(&q, 5).is_empty());
}

#[test]
fn lm_head_knn_returns_top_k_in_descending_score_order() {
    // lm_head is [4, 2] — 4 vocab tokens × 2 hidden dims.
    // Each row is a "token's projection vector". The query
    // selects which row scores highest.
    let lm_head: Vec<f32> = vec![
        1.0, 0.0, // token 0: aligned with [1, 0]
        0.5, 0.5, // token 1: 45° in xy
        0.0, 1.0, // token 2: aligned with [0, 1]
        -1.0, -1.0, // token 3: anti-aligned with [1, 1]
    ];
    let v = vindex_with_lm_head(4, 2, &lm_head);
    // Query [1, 1] → scores: [1.0, 1.0, 1.0, -2.0]
    let q = Array1::from_vec(vec![1.0_f32, 1.0]);
    let hits = v.lm_head_knn(&q, 3);
    assert_eq!(hits.len(), 3);
    // Top 3 should be tokens 0, 1, 2 (in some order — they all tie at 1.0).
    let mut ids: Vec<u32> = hits.iter().map(|(id, _)| *id).collect();
    ids.sort();
    assert_eq!(ids, vec![0, 1, 2]);
    // Lowest score (token 3 = -2.0) is excluded.
    assert!(hits.iter().all(|(id, _)| *id != 3));
}

#[test]
fn lm_head_knn_top_k_zero_returns_full_unsorted_pairs() {
    // The truncate guard requires k > 0; with k = 0 we return the
    // full indexed set sorted by score (no truncation). This pins
    // the actual contract — callers passing top_k=0 should treat
    // it as "give me everything".
    let lm_head: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v = vindex_with_lm_head(2, 2, &lm_head);
    let q = Array1::from_vec(vec![1.0_f32, 0.0]);
    let hits = v.lm_head_knn(&q, 0);
    assert_eq!(hits.len(), 2);
}

#[test]
fn lm_head_knn_top_k_larger_than_vocab_returns_all() {
    let lm_head: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v = vindex_with_lm_head(2, 2, &lm_head);
    let q = Array1::from_vec(vec![1.0_f32, 0.0]);
    let hits = v.lm_head_knn(&q, 99);
    assert_eq!(hits.len(), 2);
}

// ── lm_head_stride32_mode ──

/// RAII override for `LARQL_LM_HEAD_STRIDE32` via the larql-compute
/// thread-local override (NOT `std::env::set_var`, which races the
/// concurrent `getenv` every parallel decode test does → SIGSEGVs
/// libc). Thread-local, so no `#[serial]` needed; cleared on drop.
struct EnvSet;
impl EnvSet {
    fn set(value: Option<&str>) -> Self {
        larql_compute::options::set_env_override(ENV_LM_HEAD_STRIDE32, value);
        Self
    }
}
impl Drop for EnvSet {
    fn drop(&mut self) {
        larql_compute::options::clear_fast_path_overrides();
    }
}

#[test]
fn stride32_mode_unset_falls_back() {
    let _g = EnvSet::set(None);
    assert_eq!(lm_head_stride32_mode(), Stride32Mode::Fallback);
}

#[test]
fn stride32_mode_truthy_values_select_first() {
    for val in ["1", "true", "on", "yes"] {
        let _g = EnvSet::set(Some(val));
        assert_eq!(lm_head_stride32_mode(), Stride32Mode::First, "value {val}");
    }
}

#[test]
fn stride32_mode_falsy_values_select_disabled() {
    for val in ["0", "false", "off", "no"] {
        let _g = EnvSet::set(Some(val));
        assert_eq!(
            lm_head_stride32_mode(),
            Stride32Mode::Disabled,
            "value {val}"
        );
    }
}

#[test]
fn stride32_mode_unknown_value_falls_back() {
    let _g = EnvSet::set(Some("maybe"));
    assert_eq!(lm_head_stride32_mode(), Stride32Mode::Fallback);
}

// ── Backend dispatch coverage ─────────────────────────────────
//
// These tests exercise the `lm_head_knn_backend` /
// `lm_head_knn_backend_skip_q4k` paths on a `CpuBackend`. The
// Cpu backend has Q4 support (`supports_quant(Q4_K)` is true), so the Q4
// stride-32 / matvec branches fire when Q4 bytes are present.
// f32 fallback fires when neither Q4 nor f16 is loaded.

#[test]
fn lm_head_knn_backend_falls_back_to_f32_when_no_q4_or_f16() {
    // Stride-32 disabled so the backend tries f16 (none) → f32.
    let _g = EnvSet::set(Some("0"));
    let lm_head: Vec<f32> = vec![
        1.0, 0.0, //
        0.0, 1.0, //
    ];
    let v = vindex_with_lm_head(2, 2, &lm_head);
    let cpu = larql_compute::CpuBackend;
    let q = Array1::from_vec(vec![1.0_f32, 0.0]);
    let hits = v.lm_head_knn_backend(&q, 1, &cpu);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0, "token 0 [1, 0] best matches query [1, 0]");
}

#[test]
fn lm_head_knn_backend_skip_q4k_falls_back_to_f32() {
    let _g = EnvSet::set(Some("0"));
    let lm_head: Vec<f32> = vec![
        1.0, 0.0, //
        0.0, 1.0, //
    ];
    let v = vindex_with_lm_head(2, 2, &lm_head);
    let cpu = larql_compute::CpuBackend;
    let q = Array1::from_vec(vec![0.0_f32, 1.0]);
    let hits = v.lm_head_knn_backend_skip_q4k(&q, 1, &cpu);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 1);
}

/// `lm_head_stride32_backend_hits` short-circuits when no Q4 mmap
/// is loaded on the storage façade. Exercised through the public
/// `lm_head_knn_backend` with stride32 enabled.
#[test]
fn lm_head_stride32_path_returns_none_without_q4_mmap() {
    let _g = EnvSet::set(Some("1"));
    let lm_head: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v = vindex_with_lm_head(2, 2, &lm_head);
    // Storage has no lm_head_q4 — stride32 returns None, falls
    // through to f16 (none) → f32, returns the f32 dot.
    let cpu = larql_compute::CpuBackend;
    let q = Array1::from_vec(vec![1.0_f32, 0.0]);
    let hits = v.lm_head_knn_backend(&q, 1, &cpu);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0);
}

/// `lm_head_f16_backend_hits` returns None when vocab_size is 0.
/// We force the path by populating the f16 mmap then setting
/// vocab_size = 0 — the early-return on `vocab > 0` fires.
#[test]
fn lm_head_f16_path_returns_none_when_vocab_zero() {
    let _g = EnvSet::set(Some("0")); // skip stride32

    // f16-encoded 2×2 lm_head.
    let f16_bytes = larql_models::quant::half::encode_f16(&[1.0, 0.0, 0.0, 1.0]);
    let mut anon = memmap2::MmapOptions::new()
        .len(f16_bytes.len())
        .map_anon()
        .unwrap();
    anon.copy_from_slice(&f16_bytes);
    let mmap = anon.make_read_only().unwrap();

    let mut v = VectorIndex::empty(1, 2);
    v.set_lm_head_f16_mmap(std::sync::Arc::new(mmap));
    // Force vocab_size = 0 so the f16 path bails on the inner guard.
    v.vocab_size = 0;
    let cpu = larql_compute::CpuBackend;
    let q = Array1::from_vec(vec![1.0_f32, 0.0]);
    // f16 returns None (vocab=0), f32 fallback also vocab=0 → empty.
    let hits = v.lm_head_knn_backend(&q, 1, &cpu);
    assert!(hits.is_empty());
}
