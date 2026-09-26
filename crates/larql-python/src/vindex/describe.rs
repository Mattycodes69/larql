//! Vindex: DESCRIBE, relations and clusters (through `larql_lql::describe`).

use numpy::{IntoPyArray, PyArray1};

#[allow(unused_imports)]
use super::*;

#[pymethods]
impl PyVindex {
    // ══════════════════════════════════════════════
    //  DESCRIBE — knowledge edge discovery
    // ══════════════════════════════════════════════

    /// Describe an entity: find all knowledge edges.
    ///
    /// Returns a list of DescribeEdge objects with relation labels,
    /// targets, gate scores, layer info, and secondary tokens.
    ///
    /// Args:
    ///     entity: Entity name ("France", "Einstein")
    ///     band: "knowledge" (default), "syntax", "output", or "all"
    ///     verbose: Include cluster labels (not just probe-confirmed)
    #[pyo3(signature = (entity, band="knowledge", verbose=false))]
    pub(super) fn describe(
        &self,
        entity: &str,
        band: &str,
        verbose: bool,
    ) -> PyResult<Vec<PyDescribeEdge>> {
        use larql_lql::describe::{band_from_name, describe_edges, edge_cap, DescribeMode};

        let band = band_from_name(band).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown band {band:?}: expected syntax, knowledge, output or all"
            ))
        })?;
        let query = self.compute_embed(entity)?;
        let mode = if verbose {
            DescribeMode::Verbose
        } else {
            DescribeMode::Brief
        };
        let edges = describe_edges(
            &self.index,
            &self.config,
            self.classifier.as_ref(),
            entity,
            &query,
            Some(band),
        );
        Ok(edges
            .into_iter()
            .take(edge_cap(mode))
            .map(|e| PyDescribeEdge {
                source: match (&e.relation, e.is_probe) {
                    (None, _) => "none",
                    (Some(_), true) => "probe",
                    (Some(_), false) => "cluster",
                }
                .to_string(),
                relation: e.relation,
                target: e.target,
                gate_score: e.gate_score,
                layer: e.layer,
                feature: e.feature,
                confidence: e.confidence,
                also: e.also,
            })
            .collect())
    }

    // ══════════════════════════════════════════════
    //  Relations & Clusters
    // ══════════════════════════════════════════════

    /// List all known relation types with counts and cluster info.
    pub(super) fn relations(&self) -> Vec<PyRelation> {
        let rc = match &self.classifier {
            Some(rc) if rc.has_clusters() => rc,
            _ => return Vec::new(),
        };

        let mut rels = Vec::new();
        for i in 0..rc.num_clusters() {
            if let Some((label, count, tops)) = rc.cluster_info(i) {
                if is_path_like_label(label) {
                    continue;
                }
                rels.push(PyRelation {
                    name: label.to_string(),
                    cluster_id: i,
                    count,
                    top_tokens: tops.to_vec(),
                });
            }
        }

        rels.sort_by_key(|r| std::cmp::Reverse(r.count));
        rels
    }

    /// Get the cluster centre vector for a relation type as numpy array.
    /// Returns None if the relation is not found.
    pub(super) fn cluster_centre<'py>(
        &self,
        py: Python<'py>,
        relation: &str,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let rc = match &self.classifier {
            Some(rc) => rc,
            None => return Ok(None),
        };
        Ok(rc
            .cluster_centre_for_relation(relation)
            .map(|v| v.into_pyarray(py)))
    }

    /// Get the typical layer for a relation type.
    pub(super) fn typical_layer(&self, relation: &str) -> Option<usize> {
        self.classifier
            .as_ref()?
            .typical_layer_for_relation(relation)
    }

    /// Check if entity has an edge with the given relation.
    #[pyo3(signature = (entity, relation=None))]
    pub(super) fn has_edge(&self, entity: &str, relation: Option<&str>) -> PyResult<bool> {
        let edges = self.describe(entity, "knowledge", false)?;
        Ok(match relation {
            Some(r) => edges.iter().any(|e| {
                e.relation
                    .as_deref()
                    .map(|l| l.eq_ignore_ascii_case(r))
                    .unwrap_or(false)
            }),
            None => !edges.is_empty(),
        })
    }

    /// Get the target token for an entity+relation pair.
    /// Returns None if not found.
    #[pyo3(signature = (entity, relation))]
    pub(super) fn get_target(&self, entity: &str, relation: &str) -> PyResult<Option<String>> {
        let edges = self.describe(entity, "knowledge", false)?;
        Ok(edges
            .iter()
            .find(|e| {
                e.relation
                    .as_deref()
                    .map(|l| l.eq_ignore_ascii_case(relation))
                    .unwrap_or(false)
            })
            .map(|e| e.target.clone()))
    }
}
