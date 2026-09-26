//! The `larql slice --preset` vocabulary — one definition shared by the CLI
//! (which expands a preset into files) and the Vindex Factory (which
//! validates recipes and estimates bytes). Each consumer maps a preset to its
//! own parts; only the names and aliases live here, so a new preset is a
//! compile error in every consumer that has not decided what it means.

use std::fmt;
use std::str::FromStr;

/// The name a recipe uses for the unsliced extract output. Not a slice
/// preset: nothing is removed.
pub const UNSLICED_PRESET: &str = "full";

/// A named subset of a vindex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SlicePreset {
    /// 2-tier client: attention + embeddings locally, FFN remote.
    Client,
    /// 3-tier attention client: attention only (ADR-0008).
    Attention,
    /// Embed server: embeddings + tokenizer (ADR-0008).
    Embed,
    /// FFN server.
    Server,
    /// DESCRIBE / WALK / SELECT only; no forward pass.
    Browse,
    /// MoE router weights only (ADR-0003).
    Router,
    /// MoE expert server.
    ExpertServer,
    /// Every part.
    All,
}

impl SlicePreset {
    /// Every preset, in documentation order.
    pub const ALL: &'static [Self] = &[
        Self::Client,
        Self::Attention,
        Self::Embed,
        Self::Server,
        Self::Browse,
        Self::Router,
        Self::ExpertServer,
        Self::All,
    ];

    /// The canonical spelling.
    pub fn name(self) -> &'static str {
        self.spellings()[0]
    }

    /// Every accepted spelling, canonical first.
    pub fn spellings(self) -> &'static [&'static str] {
        match self {
            Self::Client => &["client"],
            Self::Attention => &["attn", "attention"],
            Self::Embed => &["embed", "embed-server"],
            Self::Server => &["server", "ffn", "ffn-service"],
            Self::Browse => &["browse"],
            Self::Router => &["router"],
            Self::ExpertServer => &["expert-server", "expert_server", "moe-server"],
            Self::All => &["all"],
        }
    }

    /// The canonical names, comma-separated, for error messages.
    pub fn known_names() -> String {
        Self::ALL
            .iter()
            .map(|p| p.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// A preset name no [`SlicePreset`] spells.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "unknown preset '{got}'. Expected one of: {}",
    SlicePreset::known_names()
)]
pub struct UnknownSlicePreset {
    /// The name as given.
    pub got: String,
}

impl FromStr for SlicePreset {
    type Err = UnknownSlicePreset;

    /// Case-insensitive; accepts every alias.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        Self::ALL
            .iter()
            .copied()
            .find(|p| p.spellings().contains(&lower.as_str()))
            .ok_or_else(|| UnknownSlicePreset { got: s.to_string() })
    }
}

impl fmt::Display for SlicePreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
