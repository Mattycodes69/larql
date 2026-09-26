//! Vindex: embeddings, gate vectors, KNN / walk and feature metadata.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::types::PyDict;

#[allow(unused_imports)]
use super::*;

#[pymethods]
impl PyVindex {
    // ══════════════════════════════════════════════
    //  Embeddings
    // ══════════════════════════════════════════════

    /// Embed entity text as a scaled numpy array.
    /// Multi-token entities are averaged (e.g., "John Coyle" averages both tokens).
    pub(super) fn embed<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = self.compute_embed(text)?;
        Ok(arr.to_vec().into_pyarray(py))
    }

    /// Tokenize text and return all token IDs.
    pub(super) fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub(super) fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.tokenizer
            .decode(&ids, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get the raw embedding for a token ID (unscaled).
    pub(super) fn embedding<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let id = token_id as usize;
        if id >= self.embeddings.shape()[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Token ID {} out of range",
                token_id
            )));
        }
        Ok(self.embeddings.row(id).to_vec().into_pyarray(py))
    }

    /// Get the full embedding matrix as numpy (vocab_size, hidden_size).
    pub(super) fn embedding_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (rows, cols) = (self.embeddings.shape()[0], self.embeddings.shape()[1]);
        let data = if let Some(slice) = self.embeddings.as_slice() {
            slice.to_vec()
        } else {
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                data.extend(self.embeddings.row(r).iter());
            }
            data
        };
        let arr = ndarray::Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    // ══════════════════════════════════════════════
    //  Gate vectors
    // ══════════════════════════════════════════════

    /// Get a single gate vector as numpy array (hidden_size,).
    pub(super) fn gate_vector<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        feature: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.index
            .gate_vector(layer, feature)
            .map(|v| v.into_pyarray(py))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "No gate vector at L{}:F{}",
                    layer, feature
                ))
            })
    }

    /// Get all gate vectors at a layer as numpy (num_features, hidden_size).
    pub(super) fn gate_vectors<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (data, rows, cols) = self.index.gate_vectors_flat(layer).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("No gate vectors at layer {}", layer))
        })?;
        let arr = ndarray::Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    // ══════════════════════════════════════════════
    //  KNN & Walk
    // ══════════════════════════════════════════════

    /// Gate KNN: find top-K features at a layer by dot product with a query vector.
    /// Returns list of (feature_index, score) tuples.
    #[pyo3(signature = (layer, query_vector, top_k=10))]
    pub(super) fn gate_knn(
        &self,
        layer: usize,
        query_vector: Vec<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let arr = Array1::from_vec(query_vector);
        self.index.gate_knn(layer, &arr, top_k)
    }

    /// Walk: gate KNN across multiple layers with a raw residual vector.
    /// Returns list of WalkHit objects.
    #[pyo3(signature = (residual, layers=None, top_k=5))]
    pub(super) fn walk(
        &self,
        residual: Vec<f32>,
        layers: Option<Vec<usize>>,
        top_k: usize,
    ) -> Vec<PyWalkHit> {
        let arr = Array1::from_vec(residual);
        let layer_list = layers.unwrap_or_else(|| self.index.loaded_layers());
        let trace = self.index.walk(&arr, &layer_list, top_k);
        trace
            .layers
            .into_iter()
            .flat_map(|(_, hits)| hits.into_iter().map(PyWalkHit::from))
            .collect()
    }

    /// Convenience: embed entity text and walk across layers.
    /// Like walk() but takes a string instead of a raw vector.
    #[pyo3(signature = (entity, layers=None, top_k=5))]
    pub(super) fn entity_walk(
        &self,
        entity: &str,
        layers: Option<Vec<usize>>,
        top_k: usize,
    ) -> PyResult<Vec<PyWalkHit>> {
        let arr = self.compute_embed(entity)?;
        let layer_list = layers.unwrap_or_else(|| self.index.loaded_layers());
        let trace = self.index.walk(&arr, &layer_list, top_k);
        Ok(trace
            .layers
            .into_iter()
            .flat_map(|(_, hits)| hits.into_iter().map(PyWalkHit::from))
            .collect())
    }

    /// Convenience: embed entity and do gate KNN at a layer.
    #[pyo3(signature = (entity, layer, top_k=10))]
    pub(super) fn entity_knn(
        &self,
        entity: &str,
        layer: usize,
        top_k: usize,
    ) -> PyResult<Vec<(usize, f32)>> {
        let arr = self.compute_embed(entity)?;
        Ok(self.index.gate_knn(layer, &arr, top_k))
    }

    // ══════════════════════════════════════════════
    //  Feature metadata
    // ══════════════════════════════════════════════

    /// Look up metadata for a specific feature. Returns FeatureMeta or None.
    pub(super) fn feature_meta(&self, layer: usize, feature: usize) -> Option<PyFeatureMeta> {
        self.index
            .feature_meta(layer, feature)
            .map(|m| PyFeatureMeta { inner: m })
    }

    /// Get feature metadata as a dict (for quick inspection in notebooks).
    pub(super) fn feature<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        feature: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let meta = match self.index.feature_meta(layer, feature) {
            Some(m) => m,
            None => return Ok(None),
        };
        let dict = PyDict::new(py);
        dict.set_item("layer", layer)?;
        dict.set_item("feature", feature)?;
        dict.set_item("top_token", &meta.top_token)?;
        dict.set_item("top_token_id", meta.top_token_id)?;
        dict.set_item("c_score", meta.c_score)?;
        let top_k: Vec<(&str, u32, f32)> = meta
            .top_k
            .iter()
            .map(|t| (t.token.as_str(), t.token_id, t.logit))
            .collect();
        dict.set_item("top_k", top_k)?;
        Ok(Some(dict))
    }

    /// Get the relation label for a feature (probe or cluster-assigned).
    pub(super) fn feature_label(&self, layer: usize, feature: usize) -> Option<String> {
        self.classifier
            .as_ref()?
            .label_for_feature(layer, feature)
            .map(|s| s.to_string())
    }
}
