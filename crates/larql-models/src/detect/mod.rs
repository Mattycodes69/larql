//! Auto-detect model architecture from `config.json`.
//!
//! The module is organised so each concern lives in its own file:
//! - [`config_io`] reads `config.json` from disk and enforces presence of
//!   topology fields that have no defensible architecture-class default.
//! - [`parser`] turns a parsed JSON value into a [`ModelConfig`], honouring
//!   both multimodal nesting (`text_config`) and flat layouts.
//! - [`registry`] is the family table: the one place a `model_type` is
//!   bound to a concrete [`ModelArchitecture`].
//! - This file owns [`ModelError`] and the public entry points.

use std::path::Path;

use crate::architectures::generic::GenericArch;
use crate::config::ModelArchitecture;
use crate::validation::ConfigValidationError;

mod config_io;
mod parser;
pub mod registry;

use config_io::{
    config_path, read_config_json, require_config_fields, CONFIG_FILE_NAME, CONFIG_KEY_TEXT_CONFIG,
};
use parser::parse_model_config;

pub use registry::{
    find_architecture, find_gguf_architecture, find_layer_band_split, gguf_model_type,
    ArchitectureConstructor, ArchitectureEntry, AttentionKind, ChatFormat, GgufConfigHook,
    GgufTranslation, LayerBandSplit, ModelTypeMatch, ARCHITECTURE_REGISTRY,
};

/// Error from model detection/config parsing.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("not a directory: {0}")]
    NotADirectory(std::path::PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(std::path::PathBuf),
    #[error("config validation failed: {0:?}")]
    ConfigValidation(Vec<ConfigValidationError>),
    #[error(
        "{CONFIG_FILE_NAME} not found at {0:?} — \
         architecture cannot be inferred from safetensors alone; \
         copy {CONFIG_FILE_NAME} from the source model into this directory"
    )]
    ConfigMissing(std::path::PathBuf),
    #[error(
        "{CONFIG_FILE_NAME} at {path:?} is missing required field(s): {missing:?} \
         (checked under top level and `{CONFIG_KEY_TEXT_CONFIG}`)"
    )]
    ConfigFieldsMissing {
        path: std::path::PathBuf,
        missing: Vec<&'static str>,
    },
}

/// Read `config.json` from a model directory and return the architecture.
///
/// Errors with [`ModelError::ConfigMissing`] when the directory has no
/// `config.json`, and with [`ModelError::ConfigFieldsMissing`] when the
/// file exists but lacks topology fields without a defensible
/// architecture-class default. This prevents the silent fallback-to-defaults
/// path from inventing a wrong topology and then panicking deep inside the
/// extract pipeline (issue #22).
pub fn detect_architecture(model_dir: &Path) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let config_path = config_path(model_dir);
    let config_json = read_config_json(&config_path)?;
    require_config_fields(&config_json, &config_path)?;
    Ok(detect_from_json(&config_json))
}

/// Read `config.json` from a model directory, detect the architecture, and validate it.
pub fn detect_architecture_validated(
    model_dir: &Path,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_architecture(model_dir)?;
    validate_detected_architecture(arch)
}

/// Detect architecture from an already-parsed `config.json` value.
///
/// Infallible by design: callers building an in-memory config for tests
/// or programmatic loads (e.g. GGUF-derived configs) can keep terse setup
/// and rely on [`ModelArchitecture::validate`] downstream to catch any
/// missing required fields.
pub fn detect_from_json(config: &serde_json::Value) -> Box<dyn ModelArchitecture> {
    let model_config = parse_model_config(config);
    match find_architecture(&model_config.model_type) {
        Some(entry) => (entry.construct)(model_config, config),
        None => Box::new(GenericArch::from_config(model_config)),
    }
}

/// Detect architecture from an already-parsed `config.json` value and validate it.
pub fn detect_from_json_validated(
    config: &serde_json::Value,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_from_json(config);
    validate_detected_architecture(arch)
}

pub(crate) fn validate_detected_architecture(
    arch: Box<dyn ModelArchitecture>,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    match arch.validate() {
        Ok(()) => Ok(arch),
        Err(errors) => Err(ModelError::ConfigValidation(errors)),
    }
}

#[cfg(test)]
mod tests;
