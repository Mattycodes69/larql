//! The backend lm_head paths on `CpuBackend`, which supports Q4_K: the
//! Q4_K matvec head (unpadded and block-padded rows), the stride-32
//! diagnostic head, and the f16 GEMV head — each must pick the token whose
//! row matches the query.

use ndarray::Array1;

use super::*;
use crate::format::filenames::LM_HEAD_KQUANT_BIN;
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;

const VOCAB: usize = 4;
/// The token whose row the query points along.
const TARGET: usize = 2;
/// A Q4_K row is one block; a narrower model pads its rows up to it.
const PADDED_HIDDEN: usize = Q4_K_BLOCK_ELEMS / 2;

/// `[VOCAB, cols]` rows, each a scaled one-hot along its own token index,
/// quantised to Q4_K and loaded as the store.
fn q4k_head(hidden: usize) -> VectorIndex {
    let cols = Q4_K_BLOCK_ELEMS;
    let mut rows = vec![0.0f32; VOCAB * cols];
    for t in 0..VOCAB {
        rows[t * cols + t] = 1.0;
    }
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), quantize_q4_k(&rows)).unwrap();
    let mut v = VectorIndex::empty(1, hidden);
    v.vocab_size = VOCAB;
    v.load_lm_head_kquant(tmp.path())
        .expect("synthetic Q4_K lm_head loads");
    // Same reasoning as the sibling fixtures: the mmap must outlive the file.
    std::mem::forget(tmp);
    v
}

fn query(hidden: usize) -> Array1<f32> {
    let mut q = vec![0.0f32; hidden];
    q[TARGET] = 1.0;
    Array1::from_vec(q)
}

#[test]
fn the_q4k_head_picks_the_matching_row_at_every_k() {
    let v = q4k_head(Q4_K_BLOCK_ELEMS);
    let cpu = larql_compute::CpuBackend;
    for k in [1, VOCAB] {
        let hits = v.lm_head_knn_backend(&query(Q4_K_BLOCK_ELEMS), k, &cpu);
        assert_eq!(hits[0].0 as usize, TARGET, "k={k}: {hits:?}");
    }
}

#[test]
fn a_block_padded_q4k_head_reads_a_zero_padded_query() {
    let v = q4k_head(PADDED_HIDDEN);
    let cpu = larql_compute::CpuBackend;
    let hits = v.lm_head_knn_backend(&query(PADDED_HIDDEN), 1, &cpu);
    assert_eq!(hits[0].0 as usize, TARGET, "{hits:?}");
}

#[test]
fn a_store_that_does_not_divide_into_rows_is_not_read_at_a_guessed_stride() {
    let tmp = tempfile::tempdir().unwrap();
    // One byte short of whole rows: no stride explains it.
    std::fs::write(
        tmp.path().join(LM_HEAD_KQUANT_BIN),
        vec![0u8; VOCAB * 144 - 1],
    )
    .unwrap();
    let mut v = VectorIndex::empty(1, Q4_K_BLOCK_ELEMS);
    v.vocab_size = VOCAB;
    v.load_lm_head_kquant(tmp.path())
        .expect("the store loads; its shape is judged at use");
    std::mem::forget(tmp);
    let cpu = larql_compute::CpuBackend;
    // Falls through the Q4_K head to the (absent) f16 and f32 heads.
    assert!(v
        .lm_head_knn_backend(&query(Q4_K_BLOCK_ELEMS), 1, &cpu)
        .is_empty());
}

/// A backend that serves the stride-32 and f16 heads CPU does not, each
/// scoring `TARGET` highest. It exercises the dispatch and top-k assembly
/// around those kernels; their arithmetic is the Metal backend's to test.
/// `fused` selects whether the fused top-k kernels answer or only the
/// full-readback `f16_gemv` does.
struct ServingHead {
    fused: bool,
}

fn target_scores(n: usize) -> Vec<f32> {
    (0..n)
        .map(|t| if t == TARGET { 1.0 } else { 0.0 })
        .collect()
}

impl larql_compute::MatMul for ServingHead {
    fn matmul(
        &self,
        _a: ndarray::ArrayView2<f32>,
        _b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        unreachable!("the lm_head paths never matmul")
    }
    fn matmul_transb(
        &self,
        _a: ndarray::ArrayView2<f32>,
        _b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        unreachable!("the lm_head paths never matmul")
    }
    fn f16_gemv_topk1(&self, _w: &[u8], _x: &[f32], _n: usize, _k: usize) -> Option<(u32, f32)> {
        self.fused.then_some((TARGET as u32, 1.0))
    }
    fn f16_gemv_topk(
        &self,
        _w: &[u8],
        _x: &[f32],
        n: usize,
        _k: usize,
        top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        self.fused
            .then(|| VectorIndex::top_k_sorted(target_scores(n), top_k))
    }
    fn f16_gemv(&self, _w: &[u8], _x: &[f32], n: usize, _k: usize) -> Option<Vec<f32>> {
        Some(target_scores(n))
    }
}

impl larql_compute::QuantMatVec for ServingHead {
    fn supports_quant(&self, format: larql_compute::QuantFormat) -> bool {
        matches!(format, larql_compute::QuantFormat::Q4_K)
    }
    fn q4k_matvec_stride32(
        &self,
        _q: &[u8],
        _x: &[f32],
        rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(target_scores(rows))
    }
}

impl larql_compute::DecodeBackend for ServingHead {}

impl larql_compute::ComputeBackend for ServingHead {
    fn name(&self) -> &str {
        "serving-head"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn supports(&self, _cap: larql_compute::Capability) -> bool {
        false
    }
}

#[test]
fn the_stride32_diagnostic_head_serves_the_skip_q4k_chain() {
    let _on = larql_compute::options::ScopedEnvOverride::set(ENV_LM_HEAD_STRIDE32, Some("1"));
    let v = q4k_head(Q4_K_BLOCK_ELEMS);
    let hits =
        v.lm_head_knn_backend_skip_q4k(&query(Q4_K_BLOCK_ELEMS), 1, &ServingHead { fused: true });
    assert_eq!(hits[0].0 as usize, TARGET, "{hits:?}");
}

/// A tied-embedding f16 head: `[VOCAB, VOCAB]` rows.
fn f16_head() -> VectorIndex {
    let hidden = VOCAB;
    let rows = target_scores(VOCAB * hidden);
    let bytes = larql_models::quant::half::encode_f16(&rows);
    let mut anon = memmap2::MmapOptions::new()
        .len(bytes.len())
        .map_anon()
        .unwrap();
    anon.copy_from_slice(&bytes);
    let mut v = VectorIndex::empty(1, hidden);
    v.set_lm_head_f16_mmap(std::sync::Arc::new(anon.make_read_only().unwrap()));
    v.vocab_size = VOCAB;
    v
}

#[test]
fn the_f16_head_serves_greedy_and_top_k_through_fused_or_full_readback() {
    let _off = larql_compute::options::ScopedEnvOverride::set(ENV_LM_HEAD_STRIDE32, Some("0"));
    let v = f16_head();
    for fused in [true, false] {
        for k in [1, VOCAB] {
            let hits = v.lm_head_knn_backend_skip_q4k(&query(VOCAB), k, &ServingHead { fused });
            assert_eq!(hits[0].0 as usize, TARGET, "fused={fused} k={k}: {hits:?}");
        }
    }
}
