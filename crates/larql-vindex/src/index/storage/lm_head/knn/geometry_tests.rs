//! Cover for [`VectorIndex::lm_head_q4k_geometry`], the accessor the
//! fused decode head reads its store's shape from.
//!
//! It exists so the GPU head and the CPU `q4k_row_query` path agree on
//! one answer about stride. Its refusals therefore matter more than its
//! successes: a store this cannot explain must come back `None` so the
//! caller runs the unfused head, because the alternative is a full-vocab
//! matvec issued at a guessed row width — which produces logits rather
//! than an error.

use super::*;
use crate::format::filenames::LM_HEAD_KQUANT_BIN;
use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

/// A vindex carrying `vocab` rows of `cols` Q4_K elements. `cols` is the
/// STORED width, which may exceed the model's `hidden` — that padding is
/// the case gpt-oss actually hits (2880 stored at 3072).
fn vindex_with_q4k_lm_head(vocab: usize, cols: usize, hidden: usize) -> VectorIndex {
    let row_bytes = cols / Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES;
    write_store(vocab, hidden, &vec![0u8; vocab * row_bytes])
}

fn write_store(vocab: usize, hidden: usize, bytes: &[u8]) -> VectorIndex {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join(LM_HEAD_KQUANT_BIN), bytes).unwrap();
    let mut v = VectorIndex::empty(1, hidden);
    v.vocab_size = vocab;
    v.load_lm_head_kquant(tmp.path())
        .expect("synthetic Q4_K lm_head loads");
    // The mmap outlives the tempdir handle on macOS only if the file is
    // not unlinked; same reasoning as the sibling fixture above.
    std::mem::forget(tmp);
    v
}

#[test]
fn reports_vocab_and_stored_row_width_for_an_unpadded_store() {
    let v = vindex_with_q4k_lm_head(8, 256, 256);
    let (bytes, vocab, cols) = v
        .lm_head_q4k_geometry(256)
        .expect("a well-formed store is explicable");
    assert_eq!((vocab, cols), (8, 256));
    assert_eq!(bytes.len(), 8 * Q4_K_BLOCK_BYTES);
}

/// The production shape: rows stored wider than the model's hidden.
/// `cols` must report the STORED width, because that is the stride the
/// matvec walks; reporting `hidden` would read every row misaligned.
#[test]
fn reports_the_padded_width_not_the_hidden_size() {
    let v = vindex_with_q4k_lm_head(4, 512, 300);
    let (_, vocab, cols) = v.lm_head_q4k_geometry(300).expect("padded store");
    assert_eq!((vocab, cols), (4, 512));
}

#[test]
fn refuses_when_there_is_no_q4k_store() {
    let v = VectorIndex::empty(1, 256);
    assert!(v.lm_head_q4k_geometry(256).is_none());
}

#[test]
fn refuses_a_zero_vocab_or_zero_hidden() {
    let mut v = vindex_with_q4k_lm_head(8, 256, 256);
    assert!(v.lm_head_q4k_geometry(0).is_none(), "hidden 0");
    v.vocab_size = 0;
    assert!(v.lm_head_q4k_geometry(256).is_none(), "vocab 0");
}

/// Bytes that do not divide evenly by the row count describe no
/// rectangular store at all.
#[test]
fn refuses_a_length_that_does_not_divide_by_vocab() {
    let v = write_store(3, 256, &vec![0u8; 2 * Q4_K_BLOCK_BYTES + 7]);
    assert!(v.lm_head_q4k_geometry(256).is_none());
}

/// A row length that is not a whole number of super-blocks cannot be
/// walked at the Q4_K stride, whatever it is.
#[test]
fn refuses_a_row_that_is_not_whole_super_blocks() {
    let v = write_store(4, 256, &vec![0u8; 4 * (Q4_K_BLOCK_BYTES + 1)]);
    assert!(v.lm_head_q4k_geometry(256).is_none());
}

/// A store narrower than the query cannot consume it. Reporting these
/// dimensions anyway would run the matvec off the end of every row.
#[test]
fn refuses_a_store_narrower_than_the_hidden_state() {
    let v = vindex_with_q4k_lm_head(4, 256, 512);
    assert!(v.lm_head_q4k_geometry(512).is_none());
}

/// The accessor and the CPU path must not disagree about a store both
/// can read — they are the two consumers of one geometry.
#[test]
fn agrees_with_the_cpu_row_query_on_the_same_store() {
    const VOCAB: usize = 6;
    const HIDDEN: usize = 256;
    let v = vindex_with_q4k_lm_head(VOCAB, HIDDEN, HIDDEN);
    let (bytes, _, cols) = v.lm_head_q4k_geometry(HIDDEN).expect("explicable");
    let x = vec![0.5f32; HIDDEN];
    let (cpu_cols, _) =
        q4k_row_query(bytes, &x, VOCAB, HIDDEN).expect("the CPU path reads the same store");
    assert_eq!(
        cols, cpu_cols,
        "the fused head and the unfused path would walk different strides"
    );
}
