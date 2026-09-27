//! Structured DESCRIBE for library callers.
//!
//! The `DESCRIBE` statement renders text; a binding (the Python
//! `Vindex.describe`) needs the same edges as data. Both run the band
//! resolution, walk, content filters and relation labelling defined in
//! `executor::query::describe`, so a binding cannot drift from LQL.

use larql_vindex::ndarray::Array1;
use larql_vindex::{LayerBands, VectorIndex, VindexConfig};

pub use crate::ast::{DescribeMode, LayerBand};
use crate::executor::query::describe::collect::{describe_collect_edges, describe_scan_layers};
use crate::executor::query::describe::format::resolve_label;
use crate::executor::tuning::{
    DESCRIBE_MAX_EDGES_BRIEF, DESCRIBE_MAX_EDGES_VERBOSE, DESCRIBE_WALK_TOP_K,
};
use crate::relations::RelationClassifier;

/// One DESCRIBE edge: a target token an entity's features point at.
#[derive(Debug, Clone, PartialEq)]
pub struct DescribedEdge {
    /// Relation label, when the classifier knows the strongest feature.
    pub relation: Option<String>,
    /// `true` when the label was confirmed by a probe, not a cluster.
    pub is_probe: bool,
    pub target: String,
    /// Strongest gate score across the scanned layers.
    pub gate_score: f32,
    /// Layer and feature of the strongest hit.
    pub layer: usize,
    pub feature: usize,
    /// Feature-metadata confidence of the strongest hit.
    pub confidence: f32,
    /// Every scanned layer this target appeared in.
    pub layers: Vec<usize>,
    /// Number of feature hits folded into this edge.
    pub count: usize,
    /// Secondary content tokens from the strongest feature.
    pub also: Vec<String>,
}

/// The band a DESCRIBE band keyword names (`syntax`, `knowledge`,
/// `output`, `all`), case-insensitively.
pub fn band_from_name(name: &str) -> Option<LayerBand> {
    match name.to_ascii_lowercase().as_str() {
        "syntax" => Some(LayerBand::Syntax),
        "knowledge" => Some(LayerBand::Knowledge),
        "output" => Some(LayerBand::Output),
        "all" => Some(LayerBand::All),
        _ => None,
    }
}

/// How many edges DESCRIBE shows per band in `mode`.
pub fn edge_cap(mode: DescribeMode) -> usize {
    match mode {
        DescribeMode::Brief => DESCRIBE_MAX_EDGES_BRIEF,
        DescribeMode::Verbose | DescribeMode::Raw => DESCRIBE_MAX_EDGES_VERBOSE,
    }
}

/// The layer bands DESCRIBE uses for `config`: declared bands, else the
/// family's, else every band spans the whole model.
pub fn layer_bands(config: &VindexConfig) -> LayerBands {
    crate::executor::query::resolve_bands(config)
}

/// DESCRIBE `entity` against `index`, given its query embedding. Edges are
/// sorted by descending gate score.
pub fn describe_edges(
    index: &VectorIndex,
    config: &VindexConfig,
    classifier: Option<&RelationClassifier>,
    entity: &str,
    query: &Array1<f32>,
    band: Option<LayerBand>,
) -> Vec<DescribedEdge> {
    let bands = layer_bands(config);
    let layers = describe_scan_layers(&bands, &index.loaded_layers(), band, None);
    let trace = index.walk(query, &layers, DESCRIBE_WALK_TOP_K);
    describe_collect_edges(&trace, entity)
        .into_iter()
        .map(|edge| {
            let (label, is_probe, _) = resolve_label(classifier, &edge);
            DescribedEdge {
                relation: (!label.is_empty()).then_some(label),
                is_probe,
                target: edge.original,
                gate_score: edge.gate,
                layer: edge.best_layer,
                feature: edge.best_feature,
                confidence: edge.best_confidence,
                layers: edge.layers,
                count: edge.count,
                also: edge.also,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
