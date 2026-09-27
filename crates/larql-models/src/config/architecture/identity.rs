//! The [`ArchitectureCore`] slice of [`super::ModelArchitecture`].

use crate::config::ModelConfig;

/// Identity: which family this is, the config it was built from, and
/// whether that config is valid for it.
pub trait ArchitectureCore: Send + Sync {
    /// Model family name (e.g., "gemma3", "llama").
    fn family(&self) -> &str;

    /// Parsed model configuration.
    fn config(&self) -> &ModelConfig;

    /// BOS token to prepend before inference when the tokenizer's
    /// `post_processor` doesn't already add one.
    ///
    /// Gemma 4's shipped `tokenizer.json` leaves BOS out of the
    /// `TemplateProcessing.single` template (unlike Gemma 2/3), so
    /// `tokenizer.encode(prompt, true)` returns tokens without BOS and
    /// the model sees a broken sequence. Architectures that need BOS
    /// return `Some(id)` here and callers prepend it if the encoding
    /// doesn't already start with it.
    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// Declared context bound (`max_position_embeddings`).
    fn max_position_embeddings(&self) -> Option<usize> {
        self.config().max_position_embeddings
    }

    /// Target-model layers whose hidden states this artifact consumes —
    /// present only on drafter-style checkpoints (`target_layer_ids`).
    fn target_layer_ids(&self) -> Option<&[usize]> {
        self.config().target_layer_ids.as_deref()
    }

    /// Multi-modal contract for this architecture, if any.
    ///
    /// Returns `None` for text-only models (the default). Architectures
    /// that accept images / audio override to return a stable reference
    /// to a `MultiModalProtocol` impl describing their placeholder
    /// convention, expected encoder family, token budget, and scaling
    /// rules.
    ///
    /// Phase 0: every existing impl uses the default `None`. Adding
    /// `Some(...)` returns is a Phase 1+ concern, gated on the
    /// `EmbeddingPlan` consumer landing in `larql-compute`.
    fn multimodal(&self) -> Option<&dyn crate::multimodal::MultiModalProtocol> {
        None
    }
}
