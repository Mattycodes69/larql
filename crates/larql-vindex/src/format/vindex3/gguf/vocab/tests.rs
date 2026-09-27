use serde_json::json;

use std::path::Path;

use super::{qwen35_vocab, vocab_pre, VOCAB_PRE_BY_REGEX};
use larql_models::loading::gguf::GgufValue;

fn regex_of(id: &str) -> &'static str {
    VOCAB_PRE_BY_REGEX
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, r)| *r)
        .expect("id in table")
}

/// The pre_tokenizer a Qwen-family `tokenizer.json` declares: a split
/// regex followed by byte-level encoding.
fn sequence(regex: &str) -> serde_json::Value {
    json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
        ]
    })
}

#[test]
fn qwen35_regex_with_combining_marks_maps_to_qwen35() {
    // Verbatim from the Qwen3.8 container's tokenizer.json.
    let declared = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    assert_eq!(vocab_pre(&sequence(declared)).unwrap(), "qwen35");
}

#[test]
fn qwen2_regex_maps_to_qwen2() {
    assert_eq!(vocab_pre(&sequence(regex_of("qwen2"))).unwrap(), "qwen2");
}

#[test]
fn bare_split_is_read_too() {
    let bare = json!({"type": "Split", "pattern": {"Regex": regex_of("qwen35")}});
    assert_eq!(vocab_pre(&bare).unwrap(), "qwen35");
}

#[test]
fn unknown_regex_is_refused() {
    let err = vocab_pre(&sequence(r"\s+")).unwrap_err().to_string();
    assert!(err.contains("matches no llama.cpp pre-tokenizer"), "{err}");
}

#[test]
fn pre_tokenizer_without_a_split_is_refused() {
    for pre in [
        json!(null),
        json!({"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}),
        json!({"type": "Sequence", "pretokenizers": [{"type": "ByteLevel"}]}),
        json!({"type": "Split", "pattern": {"String": " "}}),
    ] {
        let err = vocab_pre(&pre).unwrap_err().to_string();
        assert!(err.contains("declares no split regex"), "{pre}: {err}");
    }
}

fn write_container(pre_tokenizer: serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = json!({
        "pre_tokenizer": pre_tokenizer,
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
        "added_tokens": [{"content": "<eos>", "id": 2, "special": true}]
    });
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"eos_token": "<eos>"}).to_string(),
    )
    .unwrap();
    dir
}

#[test]
fn vocab_table_carries_the_derived_pre_id() {
    let dir = write_container(sequence(regex_of("qwen35")));
    let table = qwen35_vocab(dir.path(), 4).unwrap();
    let pre = table
        .entries
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.pre")
        .map(|(_, v)| v);
    assert!(matches!(pre, Some(GgufValue::String(s)) if s == "qwen35"));
    assert_eq!((table.tokens, table.padded, table.merges), (3, 1, 1));
}

#[test]
fn vocab_table_refuses_an_unknown_pre_tokenizer() {
    let dir = write_container(sequence(r"\w+"));
    assert!(qwen35_vocab(dir.path(), 4).is_err());
}

fn write_tokenizer(dir: &Path, vocab: &[(&str, u64)], added: &[(&str, u64, bool)]) {
    let vocab_map: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .map(|(c, id)| (c.to_string(), serde_json::json!(id)))
        .collect();
    let added_list: Vec<serde_json::Value> = added
        .iter()
        .map(|(c, id, s)| serde_json::json!({"content": c, "id": id, "special": s}))
        .collect();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::json!({
            "pre_tokenizer": sequence(regex_of("qwen2")),
            "model": {"type": "BPE", "vocab": vocab_map, "merges": ["a b", ["b", "c"]]},
            "added_tokens": added_list,
        })
        .to_string(),
    )
    .unwrap();
}

/// The table is padded to the MODEL's vocabulary, and says so.
#[test]
fn tokens_pad_to_the_declared_vocabulary_and_types_follow_the_files() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(
        dir.path(),
        &[("a", 0), ("b", 1), ("c", 2)],
        &[("<eos>", 3, true), ("<fim>", 4, false)],
    );
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::json!({"eos_token": "<eos>", "add_bos_token": false}).to_string(),
    )
    .unwrap();

    let table = qwen35_vocab(dir.path(), 8).unwrap();
    assert_eq!((table.tokens, table.padded), (5, 3));
    assert_eq!((table.control, table.user_defined), (1, 1));
    let get = |k: &str| {
        table
            .entries
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
    };
    let GgufValue::Array(tokens) = get("tokenizer.ggml.tokens").unwrap() else {
        panic!()
    };
    assert_eq!(tokens.len(), 8);
    assert_eq!(tokens[4], GgufValue::String("<fim>".into()));
    assert_eq!(
        tokens[7],
        GgufValue::String("[PAD7]".into()),
        "the gap is explicit"
    );
    let GgufValue::Array(types) = get("tokenizer.ggml.token_type").unwrap() else {
        panic!()
    };
    assert_eq!(types[0], GgufValue::I32(1), "vocab tokens are NORMAL");
    assert_eq!(
        types[3],
        GgufValue::I32(3),
        "special added tokens are CONTROL"
    );
    assert_eq!(
        types[4],
        GgufValue::I32(4),
        "non-special added are USER_DEFINED"
    );
    assert_eq!(types[7], GgufValue::I32(5), "padding is UNUSED");
    // Merges: verbatim string, and the two-element spelling joined.
    let GgufValue::Array(merges) = get("tokenizer.ggml.merges").unwrap() else {
        panic!()
    };
    assert_eq!(merges[0], GgufValue::String("a b".into()));
    assert_eq!(merges[1], GgufValue::String("b c".into()));
    // The eos resolved through the named token, not a guess.
    assert_eq!(get("tokenizer.ggml.eos_token_id"), Some(GgufValue::U32(3)));
    assert_eq!(
        get("tokenizer.ggml.add_bos_token"),
        Some(GgufValue::Bool(false))
    );
}

/// A tokenizer that defines ids beyond the model's vocabulary is a
/// tokenizer for a different model.
#[test]
fn an_id_beyond_the_vocabulary_refuses() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path(), &[("a", 0)], &[("<x>", 9, true)]);
    let err = qwen35_vocab(dir.path(), 4).unwrap_err().to_string();
    assert!(err.contains("different model"), "{err}");
}

/// No eos anywhere is a refusal, not a default.
#[test]
fn a_missing_eos_refuses_rather_than_guessing() {
    let dir = tempfile::tempdir().unwrap();
    write_tokenizer(dir.path(), &[("a", 0)], &[]);
    let err = qwen35_vocab(dir.path(), 2).unwrap_err().to_string();
    assert!(err.contains("eos"), "{err}");
    // And the generation config alone is enough to resolve it.
    std::fs::write(
        dir.path().join("generation_config.json"),
        serde_json::json!({"eos_token_id": [1, 0]}).to_string(),
    )
    .unwrap();
    let table = qwen35_vocab(dir.path(), 2).unwrap();
    let eos = table
        .entries
        .iter()
        .find(|(k, _)| k == "tokenizer.ggml.eos_token_id");
    assert_eq!(eos.map(|(_, v)| v.clone()), Some(GgufValue::U32(1)));
}

/// A container whose `tokenizer.json` is `tokenizer` verbatim, with an
/// eos named so only the tokenizer itself can fail.
fn container_with(tokenizer: serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        json!({"eos_token": "b"}).to_string(),
    )
    .unwrap();
    dir
}

/// A well-formed tokenizer, to break one field at a time.
fn good_tokenizer() -> serde_json::Value {
    json!({
        "pre_tokenizer": sequence(regex_of("qwen35")),
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": ["a b"]},
    })
}

fn refusal(tokenizer: serde_json::Value) -> String {
    let dir = container_with(tokenizer);
    qwen35_vocab(dir.path(), 4).unwrap_err().to_string()
}

#[test]
fn a_container_without_tokenizer_json_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let err = qwen35_vocab(dir.path(), 4).unwrap_err().to_string();
    assert!(err.contains("no tokenizer.json"), "{err}");
}

#[test]
fn unparseable_tokenizer_json_refuses_naming_the_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), "{not json").unwrap();
    let err = qwen35_vocab(dir.path(), 4).unwrap_err().to_string();
    assert!(err.contains("tokenizer.json"), "{err}");
}

#[test]
fn a_non_bpe_model_refuses() {
    let mut t = good_tokenizer();
    t["model"]["type"] = json!("Unigram");
    assert!(refusal(t).contains("`Unigram`"));
}

#[test]
fn a_vocab_that_is_not_an_object_refuses() {
    let mut t = good_tokenizer();
    t["model"]["vocab"] = json!([["a", 0]]);
    assert!(refusal(t).contains("model.vocab is not an object"));
}

#[test]
fn a_vocab_id_that_is_not_an_integer_refuses() {
    let mut t = good_tokenizer();
    t["model"]["vocab"] = json!({"a": "zero"});
    assert!(refusal(t).contains("vocab id for `a`"));
}

#[test]
fn an_id_claimed_twice_refuses() {
    let mut t = good_tokenizer();
    t["added_tokens"] = json!([{"content": "<x>", "id": 1, "special": true}]);
    assert!(refusal(t).contains("claimed twice"));
}

#[test]
fn an_added_token_without_an_id_refuses() {
    let mut t = good_tokenizer();
    t["added_tokens"] = json!([{"content": "<x>", "special": true}]);
    assert!(refusal(t).contains("added token `<x>` has no id"));
}

#[test]
fn missing_merges_refuse() {
    let mut t = good_tokenizer();
    t["model"].as_object_mut().unwrap().remove("merges");
    assert!(refusal(t).contains("model.merges missing"));
}

#[test]
fn a_malformed_merge_pair_refuses() {
    let mut t = good_tokenizer();
    t["model"]["merges"] = json!([["a", 1]]);
    assert!(refusal(t).contains("malformed merge pair"));
}

#[test]
fn a_merge_that_is_neither_string_nor_pair_refuses() {
    let mut t = good_tokenizer();
    t["model"]["merges"] = json!([7]);
    assert!(refusal(t).contains("malformed merge entry"));
}

/// With no named tokens, bos/pad/eos come from generation_config.json.
#[test]
fn special_ids_fall_back_to_generation_config() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        good_tokenizer().to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("generation_config.json"),
        json!({"eos_token_id": 1, "bos_token_id": 0, "pad_token_id": 1}).to_string(),
    )
    .unwrap();
    let table = qwen35_vocab(dir.path(), 4).unwrap();
    let id = |key: &str| {
        table
            .entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(id("tokenizer.ggml.eos_token_id"), Some(GgufValue::U32(1)));
    assert_eq!(id("tokenizer.ggml.bos_token_id"), Some(GgufValue::U32(0)));
    assert_eq!(id("tokenizer.ggml.padding_token_id"), Some(GgufValue::U32(1)));
}
