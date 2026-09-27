//! Vindex: INSERT and related mutation.
//!
//! Every mutation takes the index write lock inside `Python::detach`, so a
//! mutation is atomic with respect to concurrent Python threads without
//! relying on the GIL. INSERT in particular finds a free slot and fills it
//! in one critical section: two threads can never claim the same slot.

use larql_vindex::FeatureMeta;
use ndarray::Array1;

#[allow(unused_imports)]
use super::*;

/// Weight of the entity embedding in a synthesised gate vector; the
/// relation's cluster centre supplies the remainder.
const ENTITY_GATE_WEIGHT: f32 = 0.7;
const CLUSTER_GATE_WEIGHT: f32 = 1.0 - ENTITY_GATE_WEIGHT;
/// Features sampled to estimate a layer's typical gate-vector norm.
const NORM_SAMPLE_FEATURES: usize = 100;

#[pymethods]
impl PyVindex {
    // ══════════════════════════════════════════════
    //  Mutation — INSERT
    // ══════════════════════════════════════════════

    /// Insert a knowledge edge: synthesise gate vector and write to index.
    ///
    /// Gate vector = entity_embed * 0.7 + cluster_centre * 0.3, normalised to
    /// match existing layer magnitudes. Returns (layer, feature).
    #[pyo3(signature = (entity, relation, target, layer=None, confidence=0.8))]
    pub(super) fn insert(
        &self,
        py: Python<'_>,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<usize>,
        confidence: f32,
    ) -> PyResult<(usize, usize)> {
        py.detach(|| {
            let entity_embed = self.compute_embed(entity)?;
            let target_layer = self.insert_layer(relation, layer);
            let gate_vec = self.synthesise_gate(relation, entity_embed);
            let meta = self.single_token_meta(target, confidence)?;

            let mut index = crate::sync::write(&self.index);
            let gate_vec = normalise_to_layer(&index, target_layer, gate_vec);
            let feature = index.find_free_feature(target_layer).ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "No free feature slot at layer {}",
                    target_layer
                ))
            })?;
            index.set_gate_vector(target_layer, feature, &gate_vec);
            index.set_feature_meta(target_layer, feature, meta);
            Ok((target_layer, feature))
        })
    }

    /// Low-level: find a free feature slot at a layer.
    pub(super) fn find_free_feature(&self, py: Python<'_>, layer: usize) -> Option<usize> {
        self.read_index(py, |index| index.find_free_feature(layer))
    }

    /// Low-level: set a gate vector directly. For constellation insert experiments.
    pub(super) fn set_gate_vector(
        &self,
        py: Python<'_>,
        layer: usize,
        feature: usize,
        vector: Vec<f32>,
    ) -> PyResult<()> {
        let arr = Array1::from_vec(vector);
        self.write_index(py, |index| index.set_gate_vector(layer, feature, &arr));
        Ok(())
    }

    /// Low-level: set a custom down vector override for a feature.
    /// During inference, this vector is used instead of the model's down weight row.
    pub(super) fn set_down_vector(
        &self,
        py: Python<'_>,
        layer: usize,
        feature: usize,
        vector: Vec<f32>,
    ) -> PyResult<()> {
        self.write_index(py, |index| index.set_down_vector(layer, feature, vector));
        Ok(())
    }

    /// Low-level: set a custom up vector override for a feature.
    /// During inference, this vector is used instead of the model's up weight row.
    pub(super) fn set_up_vector(
        &self,
        py: Python<'_>,
        layer: usize,
        feature: usize,
        vector: Vec<f32>,
    ) -> PyResult<()> {
        self.write_index(py, |index| index.set_up_vector(layer, feature, vector));
        Ok(())
    }

    /// Low-level: set feature metadata directly.
    #[pyo3(signature = (layer, feature, top_token, c_score=0.9))]
    pub(super) fn set_feature_meta(
        &self,
        py: Python<'_>,
        layer: usize,
        feature: usize,
        top_token: &str,
        c_score: f32,
    ) -> PyResult<()> {
        let meta = self.single_token_meta(top_token, c_score)?;
        self.write_index(py, |index| index.set_feature_meta(layer, feature, meta));
        Ok(())
    }

    /// Delete edges matching an entity (and optionally a relation).
    #[pyo3(signature = (entity, relation=None, layer=None))]
    pub(super) fn delete(
        &self,
        py: Python<'_>,
        entity: &str,
        relation: Option<&str>,
        layer: Option<usize>,
    ) -> PyResult<usize> {
        // DESCRIBE and delete under one write lock, so the edges removed are
        // exactly the edges found.
        self.write_index(py, |index| {
            let edges = self.describe_in(index, entity, "all", true)?;
            let mut deleted = 0;
            for edge in &edges {
                if let Some(r) = relation {
                    if edge
                        .relation
                        .as_deref()
                        .map(|l| !l.eq_ignore_ascii_case(r))
                        .unwrap_or(true)
                    {
                        continue;
                    }
                }
                if layer.is_some_and(|l| edge.layer != l) {
                    continue;
                }
                index.delete_feature_meta(edge.layer, edge.feature);
                deleted += 1;
            }
            Ok(deleted)
        })
    }
}

impl PyVindex {
    /// Target layer for an INSERT: the explicit hint, else the relation's
    /// typical layer, else the middle of the knowledge band.
    fn insert_layer(&self, relation: &str, layer: Option<usize>) -> usize {
        layer
            .or_else(|| {
                self.classifier
                    .as_ref()?
                    .typical_layer_for_relation(relation)
            })
            .unwrap_or_else(|| {
                if let Some(ref b) = self.config.layer_bands {
                    (b.knowledge.0 + b.knowledge.1) / 2
                } else {
                    self.config.num_layers * 3 / 5
                }
            })
    }

    /// Blend the entity embedding with the relation's cluster centre when
    /// one of matching width exists.
    fn synthesise_gate(&self, relation: &str, entity_embed: Array1<f32>) -> Array1<f32> {
        let centre = self
            .classifier
            .as_ref()
            .and_then(|rc| rc.cluster_centre_for_relation(relation))
            .filter(|c| c.len() == self.config.hidden_size);
        match centre {
            Some(centre) => {
                &entity_embed * ENTITY_GATE_WEIGHT + &Array1::from_vec(centre) * CLUSTER_GATE_WEIGHT
            }
            None => entity_embed,
        }
    }

    /// Feature metadata naming `token` (first sub-token id) with `score`.
    fn single_token_meta(&self, token: &str, score: f32) -> PyResult<FeatureMeta> {
        let encoding = self
            .tokenizer
            .encode(token, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let token_id = encoding.get_ids().first().copied().unwrap_or(0);
        Ok(FeatureMeta {
            top_token: token.to_string(),
            top_token_id: token_id,
            c_score: score,
            top_k: vec![larql_models::TopKEntry {
                token: token.to_string(),
                token_id,
                logit: score,
            }],
        })
    }
}

/// Scale `gate_vec` to the mean norm of a sample of the layer's existing
/// gate vectors; unchanged when the layer has none to sample.
fn normalise_to_layer(index: &VectorIndex, layer: usize, mut gate_vec: Array1<f32>) -> Array1<f32> {
    let sample_count = index.num_features(layer).min(NORM_SAMPLE_FEATURES);
    let norms: Vec<f32> = (0..sample_count)
        .filter_map(|f| index.gate_vector(layer, f))
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .filter(|n| *n > 0.0)
        .collect();
    if norms.is_empty() {
        return gate_vec;
    }
    let avg_norm = norms.iter().sum::<f32>() / norms.len() as f32;
    let my_norm: f32 = gate_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if my_norm > 0.0 {
        gate_vec *= avg_norm / my_norm;
    }
    gate_vec
}
