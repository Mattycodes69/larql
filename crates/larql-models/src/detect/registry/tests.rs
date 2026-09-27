//! Cross-checks the registry against real `detect_from_json` dispatch.
//! Per-type unit tests for the pieces that don't need dispatch at all
//! live next to those types (`pattern.rs`, `entry.rs`, `table.rs`).

use super::*;
use crate::detect::detect_from_json;

const GENERIC_FAMILY: &str = "generic";

fn probe_for(pattern: ModelTypeMatch) -> String {
    match pattern {
        ModelTypeMatch::Exact(s) => s.to_string(),
        // Exercises the prefix behaviour, not just the bare prefix string,
        // in case a row's pattern is accidentally narrowed to exact.
        ModelTypeMatch::Prefix(p) => format!("{p}-probe-suffix"),
    }
}

#[test]
fn find_architecture_returns_none_for_unknown_model_types() {
    assert!(find_architecture("totally-unknown-arch").is_none());
    assert!(find_architecture("").is_none());
}

#[test]
fn find_architecture_returns_some_for_every_registered_pattern() {
    for entry in ARCHITECTURE_REGISTRY {
        for &pattern in entry.patterns {
            let probe = probe_for(pattern);
            let found = find_architecture(&probe);
            assert!(
                found.is_some(),
                "probe {probe:?} (from {}) should resolve via find_architecture",
                entry.model_type
            );
        }
    }
}

/// Every pattern this registry declares must reach a real architecture
/// through `detect_from_json` — never the `GenericArch` fallback. Catches a
/// row shadowed by an earlier, broader pattern.
#[test]
fn registry_entries_are_honored_by_detect_from_json() {
    for entry in ARCHITECTURE_REGISTRY {
        for &pattern in entry.patterns {
            let probe = probe_for(pattern);
            let config = serde_json::json!({ "model_type": probe });
            let arch = detect_from_json(&config);
            assert_ne!(
                arch.family(),
                GENERIC_FAMILY,
                "registry entry {} claims to match {probe:?}, but detect_from_json fell back to GenericArch",
                entry.model_type
            );
        }
    }
}

#[test]
fn unregistered_model_type_falls_back_to_generic_in_both_registry_and_dispatch() {
    let probe = "totally-unknown-arch";
    assert!(find_architecture(probe).is_none());

    let config = serde_json::json!({ "model_type": probe });
    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), GENERIC_FAMILY);
}

/// A row's GGUF alias must resolve back to that same row; otherwise a GGUF
/// export would be detected as a different family than it declares.
#[test]
fn every_gguf_alias_resolves_to_its_own_row() {
    for entry in ARCHITECTURE_REGISTRY {
        for &(gguf, hf) in entry.gguf.aliases {
            assert_eq!(
                gguf_model_type(gguf),
                hf,
                "alias {gguf} of {}",
                entry.model_type
            );
            let resolved = find_gguf_architecture(gguf).map(|e| e.model_type);
            assert_eq!(
                resolved,
                Some(entry.model_type),
                "GGUF {gguf} resolves to {resolved:?}, not its declaring row {}",
                entry.model_type
            );
        }
    }
}

/// No GGUF spelling may be claimed by two rows.
#[test]
fn gguf_aliases_are_unique() {
    let mut seen: Vec<&str> = ARCHITECTURE_REGISTRY
        .iter()
        .flat_map(|e| e.gguf.aliases.iter().map(|(gguf, _)| *gguf))
        .collect();
    let before = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), before, "a GGUF alias is declared twice");
}

#[test]
fn unaliased_gguf_architecture_is_its_own_model_type() {
    assert_eq!(gguf_model_type("llama"), "llama");
    assert_eq!(
        gguf_model_type("totally-unknown-arch"),
        "totally-unknown-arch"
    );
    assert!(find_gguf_architecture("totally-unknown-arch").is_none());
}

/// Gemma 1 is served by the Gemma 2 architecture but must not inherit the
/// Gemma 2+ GGUF key layout.
#[test]
fn gemma1_shares_the_architecture_but_not_the_gguf_key_layout() {
    let gemma1 = find_gguf_architecture("gemma").unwrap();
    let gemma2 = find_gguf_architecture("gemma2").unwrap();
    assert!(gemma1.gguf.key_replacements.is_empty());
    assert!(!gemma2.gguf.key_replacements.is_empty());

    let family = |model_type: &str| {
        detect_from_json(&serde_json::json!({ "model_type": model_type }))
            .family()
            .to_string()
    };
    assert_eq!(family("gemma"), family("gemma2"));
}

/// Every declared band split partitions its stack in order, and sits on
/// the row its own `model_type` resolves to.
#[test]
fn layer_band_splits_are_ordered_and_on_their_own_row() {
    for entry in ARCHITECTURE_REGISTRY {
        for split in entry.layer_bands {
            assert!(
                split.syntax_last < split.knowledge_last
                    && split.knowledge_last + 1 < split.num_layers,
                "{split:?} does not leave three non-empty bands"
            );
            assert_eq!(
                find_architecture(split.model_type).map(|e| e.model_type),
                Some(entry.model_type),
                "{split:?} is declared on {} but resolves elsewhere",
                entry.model_type
            );
            assert_eq!(
                find_layer_band_split(split.model_type, split.num_layers),
                Some(split)
            );
        }
    }
}

#[test]
fn layer_band_lookup_is_exact() {
    assert!(find_layer_band_split("gemma3", 34).is_some());
    assert!(find_layer_band_split("gemma3-clone", 34).is_none());
    assert!(find_layer_band_split("gemma3", 35).is_none());
}

/// Gemma 4 declares its own turn format, and only it carries a default
/// system prompt and a template fallback.
#[test]
fn gemma4_declares_its_own_chat_format() {
    use super::ChatFormat;
    assert_eq!(
        find_architecture("gemma4").and_then(|e| e.chat_format),
        Some(ChatFormat::Gemma4Turns)
    );
    assert_eq!(
        find_architecture("gemma3").and_then(|e| e.chat_format),
        Some(ChatFormat::GemmaTurns)
    );
    let template = ChatFormat::Gemma4Turns.fallback_template().unwrap();
    assert!(template.contains("<|turn>") && !template.contains("<|channel>"));
    assert!(ChatFormat::Gemma4Turns.default_system_prompt().is_some());
    for other in [
        ChatFormat::GemmaTurns,
        ChatFormat::MistralInst,
        ChatFormat::Llama3Headers,
        ChatFormat::ChatMl,
    ] {
        assert_eq!(other.fallback_template(), None, "{other:?}");
        assert_eq!(other.default_system_prompt(), None, "{other:?}");
    }
}
