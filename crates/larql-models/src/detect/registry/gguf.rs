//! How a family's GGUF export is translated back to its HF shape.
//!
//! llama.cpp names architectures and tensors its own way, and some families
//! carry facts in GGUF metadata that the flat HF mapping cannot express. Each
//! registry row owns those differences for its family, so the GGUF loader
//! holds no family names of its own.

use crate::loading::gguf::GgufFile;

use super::{find_architecture, ArchitectureEntry, ARCHITECTURE_REGISTRY};

/// Writes family-specific HF config keys derived from a GGUF file.
pub type GgufConfigHook = fn(&GgufFile, &mut serde_json::Value);

/// One family's GGUF translation rules.
#[derive(Clone, Copy, Debug)]
pub struct GgufTranslation {
    /// `(gguf_architecture, hf_model_type)` spellings that differ. A GGUF
    /// architecture not listed anywhere is taken as its own `model_type`.
    pub aliases: &'static [(&'static str, &'static str)],
    /// Tensor-name replacements applied before the shared GGUF→HF table, for
    /// families whose layer layout the shared table gets wrong.
    pub key_replacements: &'static [(&'static str, &'static str)],
    /// Family-specific config keys the flat metadata mapping cannot express.
    pub config: Option<GgufConfigHook>,
}

impl GgufTranslation {
    /// A family whose GGUF export needs no translation beyond the shared one.
    pub const NONE: Self = Self {
        aliases: &[],
        key_replacements: &[],
        config: None,
    };
}

/// The HF `model_type` a GGUF `general.architecture` denotes.
pub fn gguf_model_type(gguf_architecture: &str) -> &str {
    ARCHITECTURE_REGISTRY
        .iter()
        .flat_map(|entry| entry.gguf.aliases)
        .find(|(gguf, _)| *gguf == gguf_architecture)
        .map_or(gguf_architecture, |(_, hf)| hf)
}

/// The registry row a GGUF `general.architecture` resolves to, if any.
pub fn find_gguf_architecture(gguf_architecture: &str) -> Option<&'static ArchitectureEntry> {
    find_architecture(gguf_model_type(gguf_architecture))
}
