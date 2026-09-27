//! Vindex: INSERT and related mutation.

use larql_vindex::FeatureMeta;
use ndarray::Array1;

#[allow(unused_imports)]
use super::*;

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
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<usize>,
        confidence: f32,
    ) -> PyResult<(usize, usize)> {
        let entity_embed = self.compute_embed(entity)?;

        // Determine target layer
        let target_layer = layer
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
            });

        // Synthesise gate vector
        let mut gate_vec = if let Some(ref rc) = self.classifier {
            if let Some(centre) = rc.cluster_centre_for_relation(relation) {
                if centre.len() == self.config.hidden_size {
                    let centre_arr = Array1::from_vec(centre);
                    &entity_embed * 0.7 + &centre_arr * 0.3
                } else {
                    entity_embed.clone()
                }
            } else {
                entity_embed.clone()
            }
        } else {
            entity_embed.clone()
        };

        // Normalise to match layer magnitudes (sample first 100 features)
        let sample_count = self.index.num_features(target_layer).min(100);
        if sample_count > 0 {
            let mut norm_sum = 0.0f32;
            let mut norm_count = 0usize;
            for f in 0..sample_count {
                if let Some(v) = self.index.gate_vector(target_layer, f) {
                    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if n > 0.0 {
                        norm_sum += n;
                        norm_count += 1;
                    }
                }
            }
            if norm_count > 0 {
                let avg_norm = norm_sum / norm_count as f32;
                let my_norm: f32 = gate_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if my_norm > 0.0 {
                    gate_vec *= avg_norm / my_norm;
                }
            }
        }

        // Find a free feature slot
        let feature = self.index.find_free_feature(target_layer).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "No free feature slot at layer {}",
                target_layer
            ))
        })?;

        // Tokenize target for metadata
        let target_encoding = self
            .tokenizer
            .encode(target, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let target_ids = target_encoding.get_ids();
        let target_token_id = target_ids.first().copied().unwrap_or(0);

        let meta = FeatureMeta {
            top_token: target.to_string(),
            top_token_id: target_token_id,
            c_score: confidence,
            top_k: vec![larql_models::TopKEntry {
                token: target.to_string(),
                token_id: target_token_id,
                logit: confidence,
            }],
        };

        // Write to index
        self.index.set_gate_vector(target_layer, feature, &gate_vec);
        self.index.set_feature_meta(target_layer, feature, meta);

        Ok((target_layer, feature))
    }

    /// Low-level: find a free feature slot at a layer.
    pub(super) fn find_free_feature(&self, layer: usize) -> Option<usize> {
        self.index.find_free_feature(layer)
    }

    /// Low-level: set a gate vector directly. For constellation insert experiments.
    pub(super) fn set_gate_vector(
        &mut self,
        layer: usize,
        feature: usize,
        vector: Vec<f32>,
    ) -> PyResult<()> {
        let arr = Array1::from_vec(vector);
        self.index.set_gate_vector(layer, feature, &arr);
        Ok(())
    }

    /// Low-level: set a custom down vector override for a feature.
    /// During inference, this vector is used instead of the model's down weight row.
    pub(super) fn set_down_vector(
        &mut self,
        layer: usize,
        feature: usize,
        vector: Vec<f32>,
    ) -> PyResult<()> {
        self.index.set_down_vector(layer, feature, vector);
        Ok(())
    }

    /// Low-level: set a custom up vector override for a feature.
    /// During inference, this vector is used instead of the model's up weight row.
    pub(super) fn set_up_vector(
        &mut self,
        layer: usize,
        feature: usize,
        vector: Vec<f32>,
    ) -> PyResult<()> {
        self.index.set_up_vector(layer, feature, vector);
        Ok(())
    }

    /// Low-level: set feature metadata directly.
    #[pyo3(signature = (layer, feature, top_token, c_score=0.9))]
    pub(super) fn set_feature_meta(
        &mut self,
        layer: usize,
        feature: usize,
        top_token: &str,
        c_score: f32,
    ) -> PyResult<()> {
        let token_encoding = self
            .tokenizer
            .encode(top_token, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let token_ids = token_encoding.get_ids();
        let token_id = token_ids.first().copied().unwrap_or(0);

        let meta = FeatureMeta {
            top_token: top_token.to_string(),
            top_token_id: token_id,
            c_score,
            top_k: vec![larql_models::TopKEntry {
                token: top_token.to_string(),
                token_id,
                logit: c_score,
            }],
        };
        self.index.set_feature_meta(layer, feature, meta);
        Ok(())
    }

    /// Delete edges matching an entity (and optionally a relation).
    #[pyo3(signature = (entity, relation=None, layer=None))]
    pub(super) fn delete(
        &mut self,
        entity: &str,
        relation: Option<&str>,
        layer: Option<usize>,
    ) -> PyResult<usize> {
        // Find matching features via describe
        let edges = self.describe(entity, "all", true)?;
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
            if let Some(l) = layer {
                if edge.layer != l {
                    continue;
                }
            }
            self.index.delete_feature_meta(edge.layer, edge.feature);
            deleted += 1;
        }

        Ok(deleted)
    }
}
