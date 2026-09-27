//! Compliance gates + layer-band assignments.
//!
//! - `ComplianceGate` — the self-policing fp4/fp8 quality gate
//!   applied at extract time.
//! - `LayerBands` — per-layer-band classifications (syntax /
//!   knowledge / output) used by DESCRIBE and label matching.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup. `ComplianceGate` carries a `Precision` (defined
//! in the sibling `quantization` module).

use serde::{Deserialize, Serialize};

use super::quantization::Precision;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceGate {
    pub threshold_ratio: f32,
    pub min_compliant_fraction: f32,
    pub fallback_precision: Precision,
}

/// Fewest layers the proportional estimate will band.
pub const MIN_BANDED_LAYERS: usize = 8;
/// The proportional estimate's shares, in fifths of the stack.
const FIFTHS: usize = 5;
const SYNTAX_FIFTHS: usize = 2;
const KNOWLEDGE_FIFTHS: usize = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBands {
    /// Syntax/morphological band (e.g., [0, 13] for Gemma 3 4B).
    pub syntax: (usize, usize),
    /// Knowledge/factual band (e.g., [14, 27] for Gemma 3 4B).
    pub knowledge: (usize, usize),
    /// Output/formatting band (e.g., [28, 33] for Gemma 3 4B).
    pub output: (usize, usize),
}

impl LayerBands {
    /// Layer bands for a model: the split its family declares for exactly
    /// this `model_type` and depth (`larql_models` architecture layer), or
    /// a proportional estimate. `None` below [`MIN_BANDED_LAYERS`].
    pub fn for_family(family: &str, num_layers: usize) -> Option<Self> {
        let last = num_layers.saturating_sub(1);
        if let Some(split) = larql_models::detect::find_layer_band_split(family, num_layers) {
            return Some(Self {
                syntax: (0, split.syntax_last),
                knowledge: (split.syntax_last + 1, split.knowledge_last),
                output: (split.knowledge_last + 1, last),
            });
        }
        if num_layers < MIN_BANDED_LAYERS {
            return None;
        }
        // ~40% syntax, ~40% knowledge, ~20% output.
        let syntax_end = num_layers * SYNTAX_FIFTHS / FIFTHS;
        let knowledge_end = num_layers * (SYNTAX_FIFTHS + KNOWLEDGE_FIFTHS) / FIFTHS;
        Some(Self {
            syntax: (0, syntax_end.saturating_sub(1)),
            knowledge: (syntax_end, knowledge_end.saturating_sub(1)),
            output: (knowledge_end, last),
        })
    }

    /// Check which band a layer belongs to.
    pub fn band_for_layer(&self, layer: usize) -> &'static str {
        if layer >= self.syntax.0 && layer <= self.syntax.1 {
            "syntax"
        } else if layer >= self.knowledge.0 && layer <= self.knowledge.1 {
            "knowledge"
        } else if layer >= self.output.0 && layer <= self.output.1 {
            "output"
        } else {
            "unknown"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma3_34_layer_bands() {
        let b = LayerBands::for_family("gemma3", 34).unwrap();
        assert_eq!(b.syntax, (0, 13));
        assert_eq!(b.knowledge, (14, 27));
        assert_eq!(b.output, (28, 33));
    }

    #[test]
    fn llama_32_layer_bands() {
        let b = LayerBands::for_family("llama", 32).unwrap();
        assert_eq!(b.syntax, (0, 12));
        assert_eq!(b.knowledge, (13, 25));
        assert_eq!(b.output, (26, 31));
    }

    #[test]
    fn unknown_family_with_sufficient_layers_uses_fallback() {
        let b = LayerBands::for_family("custom_model", 20);
        assert!(b.is_some(), "should fall back to fraction-based estimate");
        let b = b.unwrap();
        // Bands partition [0, 19] into syntax/knowledge/output
        assert!(b.syntax.1 < b.knowledge.0);
        assert!(b.knowledge.1 < b.output.0);
        assert_eq!(b.output.1, 19);
    }

    #[test]
    fn unknown_family_does_not_inherit_known_bands_by_string_prefix() {
        // The fallback path is layer-count driven, not name driven.
        // String prefixes like "gemma3-finetune" or "llama-clone" must
        // NOT pick up the curated bands for the canonical family.
        let gemma = LayerBands::for_family("gemma3", 34).unwrap();
        let llama = LayerBands::for_family("llama", 32).unwrap();

        let gemma_lookalike = LayerBands::for_family("gemma3-clone", 34).unwrap();
        assert_ne!(
            gemma_lookalike.knowledge, gemma.knowledge,
            "fallback must not inherit canonical gemma3 knowledge band by name prefix"
        );

        let llama_lookalike = LayerBands::for_family("llamafied", 32).unwrap();
        assert_ne!(
            llama_lookalike.knowledge, llama.knowledge,
            "fallback must not inherit canonical llama knowledge band by name prefix"
        );

        // The fraction-based fallback is structurally distinct: 2/5 syntax,
        // 4/5 knowledge cutoff. For 32 layers that's syntax=(0, 11),
        // knowledge=(12, 24), which is one layer off from canonical llama.
        assert_eq!(llama_lookalike.syntax, (0, 11));
        assert_eq!(llama_lookalike.knowledge, (12, 24));
    }

    #[test]
    fn too_few_layers_returns_none() {
        assert!(LayerBands::for_family("gpt2", 4).is_none());
        assert!(LayerBands::for_family("tiny", 1).is_none());
    }

    #[test]
    fn band_for_layer_gemma3() {
        let b = LayerBands::for_family("gemma3", 34).unwrap();
        assert_eq!(b.band_for_layer(0), "syntax");
        assert_eq!(b.band_for_layer(13), "syntax");
        assert_eq!(b.band_for_layer(14), "knowledge");
        assert_eq!(b.band_for_layer(27), "knowledge");
        assert_eq!(b.band_for_layer(28), "output");
        assert_eq!(b.band_for_layer(33), "output");
    }

    #[test]
    fn band_for_layer_out_of_range_is_unknown() {
        let b = LayerBands {
            syntax: (0, 5),
            knowledge: (6, 10),
            output: (11, 15),
        };
        assert_eq!(b.band_for_layer(99), "unknown");
    }

    #[test]
    fn layer_bands_serde_round_trip() {
        let b = LayerBands::for_family("gemma3", 34).unwrap();
        let j = serde_json::to_string(&b).unwrap();
        let back: LayerBands = serde_json::from_str(&j).unwrap();
        assert_eq!(back.syntax, b.syntax);
        assert_eq!(back.knowledge, b.knowledge);
        assert_eq!(back.output, b.output);
    }

    #[test]
    fn compliance_gate_serde_round_trip() {
        use crate::config::quantization::Precision;
        let gate = ComplianceGate {
            threshold_ratio: 16.0,
            min_compliant_fraction: 0.99,
            fallback_precision: Precision::Fp8,
        };
        let j = serde_json::to_string(&gate).unwrap();
        let back: ComplianceGate = serde_json::from_str(&j).unwrap();
        assert_eq!(back.threshold_ratio, 16.0);
        assert_eq!(back.min_compliant_fraction, 0.99);
        assert_eq!(back.fallback_precision, Precision::Fp8);
    }

    #[test]
    fn gpt2_12_layer_bands() {
        let b = LayerBands::for_family("gpt2", 12).unwrap();
        assert_eq!(b.syntax, (0, 4));
        assert_eq!(b.knowledge, (5, 9));
        assert_eq!(b.output, (10, 11));
    }
}
