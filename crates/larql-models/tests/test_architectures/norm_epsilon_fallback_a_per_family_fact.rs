//! Norm-epsilon fallback — a per-family fact, pinned per family

use super::*;

/// `transformers`' config classes disagree about the `rms_norm_eps` class
/// default, and a checkpoint that omits the field is served at *its own*
/// class default. One crate-wide fallback answered for everyone, and OLMoE —
/// which ships no `rms_norm_eps` at all — ran every norm of every layer an
/// order of magnitude tight. See `docs/k3-funnel.md` §4.8.
///
/// Every family in the support table is listed, so adding an architecture
/// without deciding its value fails here rather than silently inheriting the
/// majority. The values are read off the `transformers` config classes, not
/// guessed.
#[test]
fn norm_eps_fallback_is_pinned_per_family() {
    const EPS_1E6: f32 = 1e-6;
    const EPS_1E5: f32 = 1e-5;
    let cases: [(serde_json::Value, f32); 9] = [
        (
            serde_json::json!({"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "mistral", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "qwen2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 8, "num_experts_per_tok": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "gemma2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E6,
        ),
        (
            serde_json::json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}}),
            EPS_1E6,
        ),
        // The 1e-5 families. OLMoE is the one that actually fires in practice.
        (
            serde_json::json!({"model_type": "olmoe", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": false}),
            EPS_1E5,
        ),
        (
            serde_json::json!({"model_type": "gpt_oss", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 8, "num_experts_per_tok": 2}),
            EPS_1E5,
        ),
        (
            serde_json::json!({"model_type": "starcoder2", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2}),
            EPS_1E5,
        ),
    ];

    for (config, expected) in &cases {
        let arch = detect_from_json(config);
        assert_eq!(
            arch.default_norm_eps(),
            *expected,
            "{}: fallback epsilon must match its own transformers config class \
             default, not the majority's",
            arch.family()
        );
        // With no eps in the config, `norm_eps()` must *be* the fallback —
        // otherwise declaring the fallback changes nothing that runs.
        assert_eq!(arch.norm_eps(), *expected, "{}", arch.family());
    }
}

/// An explicit `rms_norm_eps` always wins over the family fallback: the
/// fallback exists for absence, and must never override a stated value.
#[test]
fn explicit_norm_eps_overrides_the_family_fallback() {
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "olmoe", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4,
        "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": false,
        "rms_norm_eps": 1e-7,
    }));
    assert_eq!(arch.default_norm_eps(), 1e-5, "the fallback is unchanged");
    assert_eq!(arch.norm_eps(), 1e-7, "but the config is what runs");
}
