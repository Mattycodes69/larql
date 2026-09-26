//! Relation type classifier for DESCRIBE edges.
//!
//! Uses discovered cluster centres from the vindex (computed during build).
//! Falls back to embedding-direction heuristics if no clusters are available.

use larql_inference::ndarray::{Array1, Array2};
use larql_inference::tokenizers::Tokenizer;
use larql_vindex::clustering::ClusterResult;
use larql_vindex::format::filenames::{
    FEATURE_CLUSTERS_JSONL, FEATURE_LABELS_JSON, RELATION_CLUSTERS_JSON,
};

/// Classifies edges into relation types using discovered clusters
/// or embedding-space direction matching.
pub struct RelationClassifier {
    /// Discovered clusters from the vindex (if available).
    clusters: Option<ClusterResult>,
    /// Per-feature cluster assignments: (layer, feature) → cluster_id.
    feature_assignments: std::collections::HashMap<(usize, usize), usize>,
    /// Probe-confirmed per-feature labels (highest priority).
    probe_labels: std::collections::HashMap<(usize, usize), String>,
    /// Number of probe-confirmed labels.
    probe_count: usize,
}

impl RelationClassifier {
    /// Build a classifier from discovered clusters + probe labels in a vindex directory.
    /// Returns Some even if only probe labels exist (no clusters needed).
    pub fn from_vindex(vindex_path: &std::path::Path) -> Option<Self> {
        let clusters_path = vindex_path.join(RELATION_CLUSTERS_JSON);
        let assignments_path = vindex_path.join(FEATURE_CLUSTERS_JSONL);
        let probe_labels_path = vindex_path.join(FEATURE_LABELS_JSON);

        // Clusters are optional — probe labels work without them
        let clusters: Option<ClusterResult> = std::fs::read_to_string(&clusters_path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok());

        let mut feature_assignments = std::collections::HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&assignments_path) {
            for line in text.lines() {
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
                    let layer = obj["l"].as_u64().unwrap_or(0) as usize;
                    let feat = obj["f"].as_u64().unwrap_or(0) as usize;
                    let cluster = obj["c"].as_u64().unwrap_or(0) as usize;
                    feature_assignments.insert((layer, feat), cluster);
                }
            }
        }

        // Load probe-confirmed per-feature labels (highest priority)
        let mut probe_labels = std::collections::HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&probe_labels_path) {
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(map) = obj.as_object() {
                    for (key, value) in map {
                        if let Some(rel) = value.as_str() {
                            // Parse "L{layer}_F{feature}" key
                            let parts: Vec<&str> = key.split('_').collect();
                            if parts.len() == 2 {
                                if let (Some(layer), Some(feat)) = (
                                    parts[0]
                                        .strip_prefix('L')
                                        .and_then(|s| s.parse::<usize>().ok()),
                                    parts[1]
                                        .strip_prefix('F')
                                        .and_then(|s| s.parse::<usize>().ok()),
                                ) {
                                    probe_labels.insert((layer, feat), rel.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Return None only if we have nothing at all
        if clusters.is_none() && feature_assignments.is_empty() && probe_labels.is_empty() {
            return None;
        }

        let probe_count = probe_labels.len();

        Some(Self {
            clusters,
            feature_assignments,
            probe_labels,
            probe_count,
        })
    }

    /// Get the relation label for a feature at (layer, feature_index).
    /// Probe-confirmed labels take priority over cluster-assigned labels.
    pub fn label_for_feature(&self, layer: usize, feature: usize) -> Option<&str> {
        // Tier 1: probe-confirmed label (ground truth)
        if let Some(label) = self.probe_labels.get(&(layer, feature)) {
            return Some(label.as_str());
        }
        // Tier 2: cluster-assigned label
        let clusters = self.clusters.as_ref()?;
        let &cluster_id = self.feature_assignments.get(&(layer, feature))?;
        clusters.labels.get(cluster_id).map(|s| s.as_str())
    }

    /// Get the cluster ID for a feature.
    pub fn cluster_for_feature(&self, layer: usize, feature: usize) -> Option<usize> {
        self.feature_assignments.get(&(layer, feature)).copied()
    }

    /// Get cluster info (label, count, top tokens).
    pub fn cluster_info(&self, cluster_id: usize) -> Option<(&str, usize, &[String])> {
        let clusters = self.clusters.as_ref()?;
        let label = clusters.labels.get(cluster_id)?;
        let count = clusters.counts.get(cluster_id).copied().unwrap_or(0);
        let tops = clusters
            .top_tokens
            .get(cluster_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        Some((label, count, tops))
    }

    /// Distinct known relation labels (cluster labels + probe labels), with
    /// empty / `unknown` filtered out. The candidate set the FR3 relation
    /// resolver builds residual keys for. Order is unspecified.
    pub fn relation_labels(&self) -> Vec<String> {
        let mut seen = std::collections::BTreeSet::new();
        if let Some(clusters) = self.clusters.as_ref() {
            for l in &clusters.labels {
                seen.insert(l.clone());
            }
        }
        for l in self.probe_labels.values() {
            seen.insert(l.clone());
        }
        seen.into_iter()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty() && !t.eq_ignore_ascii_case("unknown")
            })
            .collect()
    }

    /// Known relation labels ranked by how many features carry each (probe
    /// labels per-feature, cluster labels by cluster size), most-common first,
    /// keeping the top `top_n`. The frequency-ordered analogue of
    /// [`Self::relation_labels`] (which is alphabetical).
    ///
    /// FR3b's explicit classifier needs the *meaningful* relations in its
    /// candidate set — "language", "capital", "currency". An alphabetical
    /// top-N silently drops them (e.g. keeps a rare early-alphabet label while
    /// cutting "language"), which made "mother tongue" fail to resolve while a
    /// niche relation survived. Ranking by frequency keeps the relations that
    /// actually populate the index and lets the `none` escape work (rare labels
    /// fall out, so an out-of-domain word finds no plausible match).
    pub fn relation_labels_ranked(&self, top_n: usize) -> Vec<String> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        if let Some(clusters) = self.clusters.as_ref() {
            for (i, l) in clusters.labels.iter().enumerate() {
                *counts.entry(l.clone()).or_insert(0) +=
                    clusters.counts.get(i).copied().unwrap_or(0);
            }
        }
        for l in self.probe_labels.values() {
            *counts.entry(l.clone()).or_insert(0) += 1;
        }
        let mut ranked: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(l, _)| {
                let t = l.trim();
                !t.is_empty() && !t.eq_ignore_ascii_case("unknown")
            })
            .collect();
        // Most-common first; alphabetical tie-break for determinism.
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_n);
        ranked.into_iter().map(|(l, _)| l).collect()
    }

    /// Check whether a feature's label is probe-confirmed (vs cluster-assigned).
    pub fn is_probe_label(&self, layer: usize, feature: usize) -> bool {
        self.probe_labels.contains_key(&(layer, feature))
    }

    /// Number of probe-confirmed feature labels.
    pub fn num_probe_labels(&self) -> usize {
        self.probe_count
    }

    /// Number of discovered clusters.
    pub fn num_clusters(&self) -> usize {
        self.clusters.as_ref().map(|c| c.k).unwrap_or(0)
    }

    /// Whether this classifier has discovered clusters (vs empty).
    pub fn has_clusters(&self) -> bool {
        self.clusters.is_some() && self.num_clusters() > 0
    }

    /// Classify a direction vector against the stored cluster centres.
    /// Returns (cluster_id, label, cosine_similarity).
    pub fn classify_direction(&self, direction: &Array1<f32>) -> Option<(usize, &str, f32)> {
        let clusters = self.clusters.as_ref()?;
        let (cluster_id, sim) =
            larql_vindex::clustering::classify_direction(direction, &clusters.centres);
        let label = clusters
            .labels
            .get(cluster_id)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        Some((cluster_id, label, sim))
    }

    /// Find the cluster ID for a relation label.
    /// Tries exact match, then normalised variants (hyphens, underscores, spaces all equivalent).
    pub fn cluster_for_relation(&self, relation: &str) -> Option<usize> {
        let clusters = self.clusters.as_ref()?;
        let norm = normalise_relation(relation);

        // Exact match
        for (i, label) in clusters.labels.iter().enumerate() {
            if label.eq_ignore_ascii_case(relation) {
                return Some(i);
            }
        }
        // Normalised match (hyphens, underscores, spaces all equivalent)
        for (i, label) in clusters.labels.iter().enumerate() {
            if normalise_relation(label) == norm {
                return Some(i);
            }
        }
        // Substring match
        for (i, label) in clusters.labels.iter().enumerate() {
            let label_norm = normalise_relation(label);
            if label_norm.contains(&norm) || norm.contains(&label_norm) {
                return Some(i);
            }
        }
        None
    }

    /// Get the cluster centre vector for a relation label.
    pub fn cluster_centre_for_relation(&self, relation: &str) -> Option<Vec<f32>> {
        let cluster_id = self.cluster_for_relation(relation)?;
        let clusters = self.clusters.as_ref()?;
        clusters.centres.get(cluster_id).cloned()
    }

    /// Find the typical layer for a relation by scanning probe labels and cluster assignments.
    /// Returns the most common layer for features with this relation.
    pub fn typical_layer_for_relation(&self, relation: &str) -> Option<usize> {
        let norm = normalise_relation(relation);
        let mut layer_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        // Check probe labels
        for (&(layer, _), label) in &self.probe_labels {
            let label_norm = normalise_relation(label);
            if label_norm == norm || label_norm.contains(&norm) || norm.contains(&label_norm) {
                *layer_counts.entry(layer).or_default() += 1;
            }
        }

        // If no probe matches, check cluster assignments
        if layer_counts.is_empty() {
            if let Some(cluster_id) = self.cluster_for_relation(relation) {
                for (&(layer, _), &cid) in &self.feature_assignments {
                    if cid == cluster_id {
                        *layer_counts.entry(layer).or_default() += 1;
                    }
                }
            }
        }

        layer_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(layer, _)| layer)
    }
}

/// Normalise a relation name: lowercase, replace hyphens/underscores/spaces with a single space.
fn normalise_relation(s: &str) -> String {
    s.to_lowercase()
        .replace(['-', '_'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Get the averaged embedding for a token string (public for executor use).
pub fn token_embedding_pub(
    text: &str,
    embed: &Array2<f32>,
    embed_scale: f32,
    tokenizer: &Tokenizer,
) -> Option<Array1<f32>> {
    let encoding = tokenizer.encode(text, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return None;
    }

    let hidden = embed.shape()[1];
    let mut avg = Array1::<f32>::zeros(hidden);
    for &id in ids {
        if (id as usize) < embed.shape()[0] {
            avg += &embed.row(id as usize).mapv(|v| v * embed_scale);
        }
    }
    avg /= ids.len() as f32;
    Some(avg)
}

#[cfg(test)]
mod tests;
