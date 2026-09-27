//! Vindex: config / stats and INFER.

use numpy::{IntoPyArray, PyArray1};
use pyo3::types::PyDict;

#[allow(unused_imports)]
use super::*;

#[pymethods]
impl PyVindex {
    // ══════════════════════════════════════════════
    //  Config / Stats
    // ══════════════════════════════════════════════

    /// Return vindex stats as a dict.
    pub(super) fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("model", &self.config.model)?;
        dict.set_item("family", &self.config.family)?;
        dict.set_item("num_layers", self.config.num_layers)?;
        dict.set_item("hidden_size", self.config.hidden_size)?;
        dict.set_item("intermediate_size", self.config.intermediate_size)?;
        dict.set_item("vocab_size", self.config.vocab_size)?;
        dict.set_item("embed_scale", self.config.embed_scale)?;
        dict.set_item("dtype", self.config.dtype.to_string())?;
        let (total_gate_vectors, total_down_meta, is_mmap) = self.read_index(py, |index| {
            (
                index.total_gate_vectors(),
                index.total_down_meta(),
                index.is_mmap(),
            )
        });
        dict.set_item("total_gate_vectors", total_gate_vectors)?;
        dict.set_item("total_down_meta", total_down_meta)?;
        dict.set_item("is_mmap", is_mmap)?;
        if let Some(ref rc) = self.classifier {
            dict.set_item("num_clusters", rc.num_clusters())?;
            dict.set_item("num_probe_labels", rc.num_probe_labels())?;
        }

        if let Some(ref bands) = self.config.layer_bands {
            let bands_dict = PyDict::new(py);
            bands_dict.set_item("syntax", (bands.syntax.0, bands.syntax.1))?;
            bands_dict.set_item("knowledge", (bands.knowledge.0, bands.knowledge.1))?;
            bands_dict.set_item("output", (bands.output.0, bands.output.1))?;
            dict.set_item("layer_bands", bands_dict)?;
        }

        Ok(dict)
    }

    /// Layer bands (syntax, knowledge, output) if available.
    pub(super) fn layer_bands<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &self.config.layer_bands {
            None => Ok(None),
            Some(bands) => {
                let dict = PyDict::new(py);
                dict.set_item("syntax", (bands.syntax.0, bands.syntax.1))?;
                dict.set_item("knowledge", (bands.knowledge.0, bands.knowledge.1))?;
                dict.set_item("output", (bands.output.0, bands.output.1))?;
                Ok(Some(dict))
            }
        }
    }

    // ══════════════════════════════════════════════
    //  INFER — full forward pass with walk FFN
    // ══════════════════════════════════════════════

    /// Run inference: full forward pass with vindex walk FFN.
    ///
    /// Model weights are mmap'd on first call and reused — zero-copy, fast.
    /// Subsequent calls reuse the cached weights (OS page cache warms up).
    ///
    /// Routes through `larql_inference::infer_patched`, which is also the
    /// entry point for the LQL `SELECT ... INFER` executor — the two paths
    /// produce byte-identical top-k predictions on any vindex. See ADR 0001
    /// (`docs/adr/0001-python-lql-infer-parity.md`).
    ///
    /// Args:
    ///     prompt: input text
    ///     top_k_predictions: number of top predictions to return (default 5)
    ///
    /// Returns:
    ///     List of (token, probability) tuples
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    pub(super) fn infer(
        &self,
        py: Python<'_>,
        prompt: &str,
        top_k_predictions: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        self.with_walk_model(py, |infer_state, index| {
            let encoding = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let result = infer_state.inference.infer_patched(
                &self.tokenizer,
                index,
                self.knn_store.as_ref(),
                &token_ids,
                top_k_predictions,
                &larql_inference::KnnRouteMode::from_env(),
            );
            Ok(result.predictions)
        })
    }

    /// Layers that have at least one entry in the L0 KnnStore.
    ///
    /// Empty if the vindex has no `knn_store.bin` or it loaded as empty.
    /// Used by measurement scripts that probe stored-key cosines against
    /// held-out residuals without running the override themselves.
    pub(super) fn knn_layers(&self) -> Vec<usize> {
        self.knn_store
            .as_ref()
            .map(|s| s.layers())
            .unwrap_or_default()
    }

    /// Total number of entries across all layers in the L0 KnnStore.
    pub(super) fn knn_len(&self) -> usize {
        self.knn_store.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    /// Top-k cosine-similarity query against the L0 KnnStore at a single
    /// layer. Returns `(entity, relation, target_token, cosine)` tuples
    /// sorted descending by cosine.
    ///
    /// `residual` is the query vector — L2-normalisation is handled inside
    /// `query_knn`. Typical usage: capture residuals via `infer_trace`, then
    /// probe each layer in `knn_layers()` to measure the negative-mass
    /// distribution of held-out prompts against stored keys.
    #[pyo3(signature = (residual, layer, k=2))]
    pub(super) fn knn_query(
        &self,
        py: Python<'_>,
        residual: numpy::PyReadonlyArray1<f32>,
        layer: usize,
        k: usize,
    ) -> PyResult<Vec<(String, String, String, f32)>> {
        let store = match self.knn_store.as_ref() {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let query = residual
            .as_slice()
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("residual must be contiguous: {e}"))
            })?
            .to_vec();
        let hits = py.detach(|| store.query_knn(layer, &query, k));
        Ok(hits
            .into_iter()
            .map(|(entry, cos)| {
                (
                    entry.entity.clone(),
                    entry.relation.clone(),
                    entry.target_token.clone(),
                    cos,
                )
            })
            .collect())
    }

    /// Per-fact target-delta optimisation (MEMIT phase 3).
    ///
    /// Returns (delta_array, baseline_loss, final_loss). Currently only
    /// install_layer = n_layers-1 is supported; mid-layer backward
    /// through attention+FFN is pending.
    #[pyo3(signature = (prompt, target, install_layer, steps=60, lr=0.5, kl_weight=0.0625))]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn optimise_target_delta<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        target: &str,
        install_layer: usize,
        steps: usize,
        lr: f32,
        kl_weight: f32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, f32)> {
        let (delta, baseline_loss, final_loss) = self.with_walk_model(py, |infer_state, _| {
            let prompt_enc = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
            let target_spaced = format!(" {target}");
            let target_enc = self
                .tokenizer
                .encode(target_spaced.as_str(), false)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let target_id: u32 = target_enc.get_ids().first().copied().unwrap_or(0);

            let opts = larql_inference::TargetDeltaOpts {
                steps,
                lr,
                kl_weight,
                normalise: false,
            };
            let result = larql_inference::forward::target_delta::optimise_target_delta(
                infer_state.inference.as_weights(),
                &prompt_ids,
                target_id,
                install_layer,
                opts,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

            Ok((
                result.delta.to_vec(),
                result.baseline_loss,
                result.final_loss,
            ))
        })?;
        let delta_np = numpy::PyArray1::from_vec(py, delta);
        Ok((delta_np, baseline_loss, final_loss))
    }

    /// Run inference and capture per-layer residuals — the actual query
    /// vectors the walk FFN's `gate_knn` operates on at each layer
    /// (post-attention, post-RMSNorm, last-token position).
    ///
    /// Routes through `larql_inference::infer_patched` — same pipeline as
    /// `infer()` and the LQL `SELECT ... INFER` executor, so the returned
    /// predictions match those surfaces byte-for-byte (ADR 0001).
    ///
    /// Residuals are returned as `(layer, array)` tuples because the walk
    /// FFN only emits residuals for layers with vindex features — positional
    /// indexing does not correspond to layer number. Iterate:
    ///
    ///     for layer, r in residuals:
    ///         ...
    ///
    /// Returns:
    ///   (predictions, residuals) where
    ///     predictions: list of (token, probability) tuples
    ///     residuals:   list of (layer_index, (hidden_size,) numpy array)
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    #[allow(clippy::type_complexity)]
    pub(super) fn infer_trace<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        top_k_predictions: usize,
    ) -> PyResult<(Vec<(String, f64)>, Vec<(usize, Bound<'py, PyArray1<f32>>)>)> {
        let result = self.with_walk_model(py, |infer_state, index| {
            let encoding = self
                .tokenizer
                .encode(prompt, true)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            Ok(infer_state.inference.infer_patched(
                &self.tokenizer,
                index,
                self.knn_store.as_ref(),
                &token_ids,
                top_k_predictions,
                &larql_inference::KnnRouteMode::from_env(),
            ))
        })?;

        let residuals: Vec<(usize, Bound<'py, PyArray1<f32>>)> = result
            .residuals
            .into_iter()
            .map(|(layer, vec)| (layer, ndarray::Array1::from_vec(vec).into_pyarray(py)))
            .collect();

        Ok((result.predictions, residuals))
    }

    /// Find features whose down weight vectors project toward a target token.
    ///
    /// For each feature at the given layers, computes:
    ///   score = lm_head[token_id] · down_weight[layer, feature]
    ///
    /// Returns list of (layer, feature, score, top_token) sorted by score descending.
    /// Only returns features with score > 0.
    #[pyo3(signature = (target, layers=None, top_k=20))]
    pub(super) fn find_features_by_target(
        &self,
        py: Python<'_>,
        target: &str,
        layers: Option<Vec<usize>>,
        top_k: usize,
    ) -> PyResult<Vec<(usize, usize, f32, String)>> {
        self.with_walk_model(py, |infer_state, index| {
            let weights = infer_state.inference.as_weights();

            let encoding = self
                .tokenizer
                .encode(target, false)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let token_ids = encoding.get_ids();
            if token_ids.is_empty() {
                return Ok(vec![]);
            }
            let target_id = token_ids[0] as usize;
            let lm_head_row = weights.lm_head.row(target_id);

            let scan_layers = layers.unwrap_or_else(|| index.loaded_layers());
            let mut results: Vec<(usize, usize, f32, String)> = Vec::new();

            for &layer in &scan_layers {
                let arch = &*weights.arch;
                let down_key = arch.ffn_down_key(layer);
                let down_weights = match weights.tensors.get(&down_key) {
                    Some(w) => w,
                    None => continue,
                };
                let num_features = down_weights.shape()[0];

                for feat in 0..num_features {
                    let down_row = down_weights.row(feat);
                    let score: f32 = lm_head_row
                        .iter()
                        .zip(down_row.iter())
                        .map(|(a, b)| a * b)
                        .sum();

                    if score > 0.0 {
                        let token = index
                            .feature_meta(layer, feat)
                            .map(|m| m.top_token.clone())
                            .unwrap_or_default();
                        results.push((layer, feat, score, token));
                    }
                }
            }

            results.sort_by(|a, b| b.2.total_cmp(&a.2));
            results.truncate(top_k);
            Ok(results)
        })
    }

    pub(super) fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Vindex(model='{}', layers={}, hidden={}, features={})",
            self.config.model,
            self.config.num_layers,
            self.config.hidden_size,
            self.read_index(py, |index| index.total_gate_vectors())
        )
    }
}
