//! Omitted config fields resolve to the family's own `transformers` class
//! default, taken from its registry row — never to another family's.

use crate::defaults::{ROPE_BASE_DEFAULT, ROPE_BASE_GEMMA};
use crate::detect::*;

fn parse(config: serde_json::Value) -> crate::config::ModelConfig {
    detect_from_json(&config).config().clone()
}

fn bare(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "model_type": model_type,
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "intermediate_size": 8192,
    })
}

#[test]
fn undeclared_kv_heads_mean_multi_head_attention() {
    let mut config = bare("llama");
    config["num_attention_heads"] = serde_json::json!(32);
    let c = parse(config);
    assert_eq!(c.num_q_heads, 32);
    assert_eq!(
        c.num_kv_heads, 32,
        "LlamaConfig: kv heads default to query heads"
    );
}

#[test]
fn undeclared_heads_are_absent_for_families_without_a_class_default() {
    let c = parse(bare("llama"));
    assert_eq!(
        c.num_q_heads, 0,
        "absence must reach validation, not become 8"
    );
}

#[test]
fn gemma1_uses_gemma_config_defaults() {
    let c = parse(bare("gemma"));
    assert_eq!((c.num_q_heads, c.num_kv_heads, c.head_dim), (16, 16, 256));
    assert_eq!(c.rope_base, ROPE_BASE_DEFAULT);
}

#[test]
fn gemma2_uses_gemma2_config_defaults() {
    let c = parse(bare("gemma2"));
    assert_eq!((c.num_q_heads, c.num_kv_heads, c.head_dim), (8, 4, 256));
    assert_eq!(
        c.rope_base, ROPE_BASE_DEFAULT,
        "Gemma2Config rope_theta is 10 000"
    );
}

#[test]
fn gemma3_and_gemma4_use_the_gemma3_rope_base() {
    for model_type in ["gemma3", "gemma3_text", "gemma4"] {
        let c = parse(bare(model_type));
        assert_eq!(c.rope_base, ROPE_BASE_GEMMA, "{model_type}");
        assert_eq!(c.head_dim, 256, "{model_type}");
    }
}

#[test]
fn declared_values_win_over_family_defaults() {
    let mut config = bare("gemma3");
    config["num_attention_heads"] = serde_json::json!(4);
    config["num_key_value_heads"] = serde_json::json!(1);
    config["head_dim"] = serde_json::json!(128);
    config["rope_theta"] = serde_json::json!(5000.0);
    let c = parse(config);
    assert_eq!((c.num_q_heads, c.num_kv_heads, c.head_dim), (4, 1, 128));
    assert_eq!(c.rope_base, 5000.0);
}

#[test]
fn gpt2_derives_its_ffn_width_and_others_do_not() {
    let mut gpt2 = bare("gpt2");
    gpt2.as_object_mut().unwrap().remove("intermediate_size");
    assert_eq!(parse(gpt2).intermediate_size, 4 * 2048);

    let mut llama = bare("llama");
    llama.as_object_mut().unwrap().remove("intermediate_size");
    assert_eq!(parse(llama).intermediate_size, 0);
}
