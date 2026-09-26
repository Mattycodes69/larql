//! DESCRIBE layer bands a family has set for particular depths.

use super::ARCHITECTURE_REGISTRY;

/// Layer bands for one exact `model_type` at one depth: syntax is
/// `[0, syntax_last]`, knowledge `(syntax_last, knowledge_last]`, output
/// the remaining layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerBandSplit {
    /// The exact `model_type` the split was set for.
    pub model_type: &'static str,
    /// The depth it applies to.
    pub num_layers: usize,
    /// Last layer of the syntax band.
    pub syntax_last: usize,
    /// Last layer of the knowledge band.
    pub knowledge_last: usize,
}

/// The split a family declares for exactly this `model_type` and depth.
/// Matching is exact: a lookalike name gets no family's curated bands.
pub fn find_layer_band_split(
    model_type: &str,
    num_layers: usize,
) -> Option<&'static LayerBandSplit> {
    ARCHITECTURE_REGISTRY
        .iter()
        .flat_map(|entry| entry.layer_bands)
        .find(|s| s.model_type == model_type && s.num_layers == num_layers)
}
