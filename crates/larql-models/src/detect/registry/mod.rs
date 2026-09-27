//! Architecture registry — declarative metadata for every `model_type`
//! [`super::detect_from_json`] recognises.
//!
//! This table *is* the dispatch: `detect_from_json` parses the config,
//! finds the first matching row, and calls its constructor. Adding a family
//! is one `architectures/<family>.rs` plus one row here.
//!
//! The consumer this exists for: Vindex Factory's PR-time architecture
//! gate (docs/vindex-factory.md §15.2) — "does the pinned larql release
//! understand this `model_type`, and if so what does it support" —
//! answerable without loading a model.
//!
//! One file per concept: [`pattern`] (how a string is matched),
//! [`attention`] (attention-family vocabulary), [`construct`] (how a row
//! builds its architecture), [`defaults`] (what an omitted config field
//! means for the family), [`gguf`] (how a row's GGUF export maps back),
//! [`entry`] (one row's shape), [`table`] (the data itself).

mod attention;
mod chat;
mod construct;
mod defaults;
mod entry;
mod gguf;
mod layer_bands;
mod pattern;
mod table;

pub use attention::AttentionKind;
pub use chat::ChatFormat;
pub use construct::ArchitectureConstructor;
pub use defaults::{ConfigDefaults, IntermediateSize};
pub use entry::{ArchitectureEntry, ComponentRole};
pub use gguf::{find_gguf_architecture, gguf_model_type, GgufConfigHook, GgufTranslation};
pub use layer_bands::{find_layer_band_split, LayerBandSplit};
pub use pattern::ModelTypeMatch;
pub use table::{ARCHITECTURE_REGISTRY, LLAMA_FAMILY};

/// Look up the registry entry for a `model_type` string, first match wins.
/// `None` means `detect_from_json` falls back to `GenericArch` — i.e. this
/// `model_type` isn't supported yet.
pub fn find_architecture(model_type: &str) -> Option<&'static ArchitectureEntry> {
    ARCHITECTURE_REGISTRY.iter().find(|e| e.matches(model_type))
}

#[cfg(test)]
mod tests;
