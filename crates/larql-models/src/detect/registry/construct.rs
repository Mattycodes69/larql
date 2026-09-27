//! How a registry row builds its architecture from a parsed config.

use crate::architectures::moss_tts_realtime::{MossTtsRealtimeArch, MOSS_TTS_REALTIME_MODEL_TYPE};
use crate::config::{ModelArchitecture, ModelConfig};
use crate::detect::config_io::CONFIG_KEY_LANGUAGE_CONFIG;
use crate::detect::parser::parse_model_config;

/// Builds a family's architecture from its parsed [`ModelConfig`]. The raw
/// `config.json` is passed alongside for families whose real config is
/// nested under a key the generic parser does not descend into.
pub type ArchitectureConstructor =
    fn(ModelConfig, &serde_json::Value) -> Box<dyn ModelArchitecture>;

/// MOSS-TTS-Realtime — a stock Qwen3 backbone nested under
/// `language_config`, whose output is a hidden state for a side-loaded audio
/// depth transformer, never text. The nested object is a complete Qwen3
/// config carrying its own `model_type: "qwen3"`, so it is parsed directly
/// and rebranded; the flat fallback keeps in-memory test configs terse.
pub(super) fn moss_tts_realtime(
    _flat: ModelConfig,
    raw: &serde_json::Value,
) -> Box<dyn ModelArchitecture> {
    let nested = raw.get(CONFIG_KEY_LANGUAGE_CONFIG).unwrap_or(raw);
    let mut config = parse_model_config(nested);
    config.model_type = MOSS_TTS_REALTIME_MODEL_TYPE.to_string();
    Box::new(MossTtsRealtimeArch::from_config(config))
}
