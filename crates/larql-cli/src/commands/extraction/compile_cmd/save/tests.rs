use super::*;
use serde_json::json;

#[test]
fn a_plain_config_is_kept_verbatim_whatever_the_family() {
    let llama = json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"});
    assert_eq!(text_only_config(&llama), llama);
}

#[test]
fn a_wrapper_is_unwrapped_to_its_text_config_with_the_text_class() {
    let wrapper = json!({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {"model_type": "gemma3_text", "hidden_size": 8},
    });
    let out = text_only_config(&wrapper);
    assert_eq!(out["model_type"], "gemma3_text");
    assert_eq!(out["architectures"], json!(["Gemma3ForCausalLM"]));
    assert_eq!(out["hidden_size"], 8);
    assert!(out.get("text_config").is_none());
}

#[test]
fn declared_text_facts_win_and_unknown_ones_are_not_invented() {
    let wrapper = json!({
        "architectures": ["SomeVisionModel"],
        "tie_word_embeddings": false,
        "text_config": {"architectures": ["OwnCausalLM"]},
    });
    let out = text_only_config(&wrapper);
    assert_eq!(out["architectures"], json!(["OwnCausalLM"]));
    assert_eq!(out["tie_word_embeddings"], false);

    let unknown = json!({"architectures": ["SomeVisionModel"], "text_config": {}});
    assert!(text_only_config(&unknown).get("architectures").is_none());
}
