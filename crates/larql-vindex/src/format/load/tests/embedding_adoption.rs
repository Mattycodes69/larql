//! The loader's embedding reads: the f16 `embeddings.bin` is adopted as
//! the tied LM head only when no separate head ships, and
//! `load_vindex_embeddings` detects the stored width from the file size.

use super::*;

/// The fixture `write_minimal_loadable_vindex` declares.
const LAYERS: usize = 2;
const HIDDEN: usize = 8;
const VOCAB: usize = 16;
const F16_WIDTH: usize = 2;
const F32_WIDTH: usize = 4;
/// A distinctive embedding value that survives f16 exactly.
const EMBED_VALUE: f32 = 0.5;

fn loadable(dir: &Path) {
    write_minimal_loadable_vindex(dir, LAYERS, HIDDEN);
}

fn write_embeddings(dir: &Path, width: usize) {
    let values = vec![EMBED_VALUE; VOCAB * HIDDEN];
    let bytes = match width {
        F16_WIDTH => larql_models::quant::half::encode_f16(&values),
        F32_WIDTH => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        other => panic!("no fixture encoding for width {other}"),
    };
    std::fs::write(dir.join(EMBEDDINGS_BIN), bytes).unwrap();
}

fn load(dir: &Path) -> VectorIndex {
    VectorIndex::load_vindex(dir, &mut crate::index::SilentLoadCallbacks).unwrap()
}

#[test]
fn an_f16_embedding_table_is_adopted_as_the_tied_lm_head() {
    let dir = TempDir::new().unwrap();
    loadable(dir.path());
    write_embeddings(dir.path(), F16_WIDTH);
    let index = load(dir.path());
    assert!(index.has_lm_head_f16(), "tied f16 embeddings adopted");
    assert_eq!(index.vocab_size, VOCAB);
}

#[test]
fn a_separate_lm_head_file_blocks_the_embedding_adoption() {
    let dir = TempDir::new().unwrap();
    loadable(dir.path());
    write_embeddings(dir.path(), F16_WIDTH);
    std::fs::write(dir.path().join(LM_HEAD_BIN), b"").unwrap();
    let index = load(dir.path());
    assert!(
        !index.has_lm_head_f16(),
        "an untied head is never the embed"
    );
}

#[test]
fn an_f32_embedding_table_is_not_adopted_as_an_f16_head() {
    let dir = TempDir::new().unwrap();
    loadable(dir.path());
    write_embeddings(dir.path(), F32_WIDTH);
    assert!(!load(dir.path()).has_lm_head_f16());
}

#[test]
fn embeddings_decode_at_the_width_their_size_implies() {
    for width in [F16_WIDTH, F32_WIDTH] {
        let dir = TempDir::new().unwrap();
        loadable(dir.path());
        write_embeddings(dir.path(), width);
        let (embed, scale) = load_vindex_embeddings(dir.path()).unwrap();
        assert_eq!(embed.dim(), (VOCAB, HIDDEN), "width {width}");
        assert!(embed.iter().all(|v| *v == EMBED_VALUE), "width {width}");
        assert_eq!(scale, 1.0);
    }
}

#[test]
fn an_embedding_table_of_the_wrong_size_refuses() {
    let dir = TempDir::new().unwrap();
    loadable(dir.path());
    std::fs::write(dir.path().join(EMBEDDINGS_BIN), [0u8; F16_WIDTH]).unwrap();
    assert!(load_vindex_embeddings(dir.path()).is_err());
}

#[test]
fn a_legacy_jsonl_down_meta_is_read_when_no_binary_ships() {
    let dir = TempDir::new().unwrap();
    loadable(dir.path());
    std::fs::write(
        dir.path().join(DOWN_META_JSONL),
        "{\"layer\":0,\"feature\":0,\"top_token\":\"Paris\",\"top_token_id\":1}\n",
    )
    .unwrap();
    let index = load(dir.path());
    let meta = index.feature_meta(0, 0).expect("legacy down meta loaded");
    assert_eq!(meta.top_token, "Paris");
}

#[test]
fn a_missing_tokenizer_is_a_parse_error() {
    let dir = TempDir::new().unwrap();
    assert!(matches!(
        load_vindex_tokenizer(dir.path()),
        Err(VindexError::Parse(_))
    ));
}
