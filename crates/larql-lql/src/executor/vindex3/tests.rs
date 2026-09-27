use super::declared_bos_token;

fn container_with(name: &str, body: serde_json::Value) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(name), body.to_string()).unwrap();
    dir
}

/// The primary declaration: HF's `generation_config.json`.
#[test]
fn generation_config_declares_the_bos_token() {
    let dir = container_with(
        "generation_config.json",
        serde_json::json!({"bos_token_id": 2}),
    );
    assert_eq!(declared_bos_token(dir.path()), Some(2));
}

/// `tokenizer_config.json` answers when the generation config does
/// not — both arrive with the capability snapshot, and older
/// checkpoints carry only the latter.
#[test]
fn tokenizer_config_answers_when_generation_config_is_silent() {
    let dir = container_with(
        "tokenizer_config.json",
        serde_json::json!({"bos_token_id": 7, "tokenizer_class": "GPT2Tokenizer"}),
    );
    assert_eq!(declared_bos_token(dir.path()), Some(7));

    // …and the generation config wins when both speak.
    std::fs::write(
        dir.path().join("generation_config.json"),
        serde_json::json!({"bos_token_id": 2}).to_string(),
    )
    .unwrap();
    assert_eq!(declared_bos_token(dir.path()), Some(2));
}

/// A container that declares nothing — including one predating the
/// capability snapshot — encodes exactly as it did before.
#[test]
fn a_container_declaring_nothing_yields_no_bos() {
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(declared_bos_token(empty.path()), None);

    let silent = container_with("generation_config.json", serde_json::json!({"top_k": 64}));
    assert_eq!(declared_bos_token(silent.path()), None);
}

/// Malformed carried config is not a hard failure: the fact is
/// absent, the container still binds.
#[test]
fn malformed_config_is_treated_as_no_declaration() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("generation_config.json"), "{not json").unwrap();
    assert_eq!(declared_bos_token(dir.path()), None);
}
