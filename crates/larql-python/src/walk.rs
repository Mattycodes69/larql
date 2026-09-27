//! WalkModel — model weights + vindex for walk FFN inference.
//!
//! Weights come from the shared `larql_vindex::load_model_weights` loader,
//! the same one the CLI uses, so Python sees identical tensors and the
//! same header validation. The gate KNN index stays mmap'd in `VectorIndex`.
//!
//! Threading: a WalkModel is immutable after load (`frozen`), so every
//! forward pass, capture and KNN runs inside `Python::detach` without a
//! lock. Python inputs are copied into owned Rust values before the GIL is
//! released and Python outputs are built after it is re-acquired.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;
use std::sync::Arc;

use larql_inference::ffn::FfnBackend;
use larql_inference::{predict_with_ffn, ModelWeights, WalkFfn};
use larql_vindex::{load_vindex_tokenizer, tokenizers, SilentLoadCallbacks, VectorIndex};

use crate::f32_bytes::{f32s_from_bytes, f32s_to_bytes};
use crate::trace_py;

mod interp;
mod lens;

const F32_BYTES: usize = std::mem::size_of::<f32>();

// ── InferState: lazy-loaded inference weights for vindex.infer() ──

/// Format-aware model weights, reusable across infer() calls.
/// Created lazily on first infer(), held by PyVindex.
pub struct InferState {
    pub inference: larql_inference::InferenceWeights,
}

impl InferState {
    pub fn load(dir: &Path, config: &larql_vindex::VindexConfig) -> Result<Self, String> {
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let inference = larql_inference::InferenceWeights::load(dir, config, &mut cb)
            .map_err(|e| e.to_string())?;
        Ok(Self { inference })
    }
}

// ── Python class ──

#[pyclass(name = "WalkModel", frozen)]
pub struct PyWalkModel {
    weights: Arc<ModelWeights>,
    index: VectorIndex,
    tokenizer: Arc<tokenizers::Tokenizer>,
    top_k: usize,
    path: String,
}

#[pymethods]
impl PyWalkModel {
    /// Load a walk model from a vindex directory.
    ///
    /// Weights load through the shared vindex loader; the gate KNN
    /// index stays mmap'd, so only the pages a walk touches are paged in.
    #[new]
    #[pyo3(signature = (path, top_k=8192))]
    fn new(py: Python<'_>, path: &str, top_k: usize) -> PyResult<Self> {
        py.detach(|| Self::load(path, top_k))
    }

    /// Run full forward pass with walk FFN. Returns [(token, probability)].
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    fn predict(
        &self,
        py: Python<'_>,
        prompt: &str,
        top_k_predictions: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        py.detach(|| {
            let token_ids = self.encode(prompt)?;
            let walk_ffn = self.walk_ffn();
            let result = predict_with_ffn(
                &self.weights,
                &self.tokenizer,
                &token_ids,
                top_k_predictions,
                &walk_ffn,
            );
            Ok(result.predictions)
        })
    }

    /// Run walk FFN for a single layer.
    ///
    /// Accepts raw f32 bytes (from MLX memoryview), returns raw f32 bytes.
    /// No numpy: MLX → bytes → Rust → bytes → MLX.
    fn ffn_layer<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        x_bytes: &[u8],
        seq_len: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let hidden = self.weights.hidden_size;
        let floats = decode_positions(x_bytes, seq_len, hidden)?;
        let out_bytes = py.detach(|| {
            let x_arr = ndarray::Array2::from_shape_vec((seq_len, hidden), floats)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let output = self.walk_ffn().forward(layer, &x_arr);
            Ok::<_, PyErr>(f32s_to_bytes(&output))
        })?;
        Ok(PyBytes::new(py, &out_bytes))
    }

    /// Feature selection only — returns indices for MLX sparse matmul.
    ///
    /// Runs gate KNN on the vindex for each sequence position, returns the
    /// union of top-K feature indices. MLX uses these to gather rows and
    /// do the matmul on Metal GPU.
    ///
    /// Args:
    ///     layer: layer index
    ///     x_bytes: raw f32 bytes (seq_len × hidden) from MLX
    ///     seq_len: number of sequence positions
    ///     top_k: features to select per position (default: self.top_k)
    ///
    /// Returns:
    ///     List of feature indices (sorted, deduplicated union across positions)
    #[pyo3(signature = (layer, x_bytes, seq_len, top_k=None))]
    fn gate_select(
        &self,
        py: Python<'_>,
        layer: usize,
        x_bytes: &[u8],
        seq_len: usize,
        top_k: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        let (indices, _) = self.gate_select_scored(py, layer, x_bytes, seq_len, top_k)?;
        Ok(indices)
    }

    /// Feature selection returning indices and gate scores.
    ///
    /// Like gate_select but also returns the max gate score per feature
    /// (useful for debugging / weighted sparse FFN).
    #[pyo3(signature = (layer, x_bytes, seq_len, top_k=None))]
    fn gate_select_scored(
        &self,
        py: Python<'_>,
        layer: usize,
        x_bytes: &[u8],
        seq_len: usize,
        top_k: Option<usize>,
    ) -> PyResult<(Vec<usize>, Vec<f32>)> {
        let hidden = self.weights.hidden_size;
        let floats = decode_positions(x_bytes, seq_len, hidden)?;
        let k = top_k.unwrap_or(self.top_k);

        let pairs = py.detach(|| {
            let mut best: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
            for row in floats.chunks_exact(hidden) {
                let arr = ndarray::Array1::from_vec(row.to_vec());
                for (idx, score) in self.index.gate_knn(layer, &arr, k) {
                    let entry = best.entry(idx).or_insert(0.0f32);
                    if score.abs() > entry.abs() {
                        *entry = score;
                    }
                }
            }
            let mut pairs: Vec<(usize, f32)> = best.into_iter().collect();
            pairs.sort_unstable_by_key(|(idx, _)| *idx);
            pairs
        });
        Ok(pairs.into_iter().unzip())
    }

    #[getter]
    fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    #[getter]
    fn intermediate_size(&self) -> usize {
        self.weights.intermediate_size
    }

    #[getter]
    fn top_k(&self) -> usize {
        self.top_k
    }

    /// Capture a complete residual stream trace.
    ///
    /// Runs a full forward pass through WalkFfn, recording the residual,
    /// attn_delta, and post-attention ffn_delta at every layer. Returns a
    /// ResidualTrace object.
    ///
    /// Args:
    ///     prompt: Input text
    ///     positions: "last" (default) or "all"
    ///
    /// Example:
    ///     t = walk_model.trace("The capital of France is")
    ///     t.answer_trajectory("Paris")
    #[pyo3(signature = (prompt, positions="last"))]
    fn trace(
        &self,
        py: Python<'_>,
        prompt: &str,
        positions: &str,
    ) -> PyResult<trace_py::PyResidualTrace> {
        py.detach(|| {
            trace_py::capture_trace_with_ffn(
                &self.weights,
                &self.tokenizer,
                prompt,
                positions,
                &self.walk_ffn(),
            )
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "WalkModel(path='{}', layers={}, hidden={}, top_k={})",
            self.path, self.weights.num_layers, self.weights.hidden_size, self.top_k
        )
    }
}

impl PyWalkModel {
    /// Load weights, index and tokenizer (Rust-callable; no GIL needed).
    fn load(path: &str, top_k: usize) -> PyResult<Self> {
        let dir = std::path::Path::new(path);

        let mut load_cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir, &mut load_cb)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let weights = larql_vindex::load_model_weights(dir, &mut load_cb)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tokenizer = load_vindex_tokenizer(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            weights: Arc::new(weights),
            index,
            tokenizer: Arc::new(tokenizer),
            top_k,
            path: path.to_string(),
        })
    }

    /// Tokenize a prompt to ids, raising a Python ValueError on failure.
    fn encode(&self, prompt: &str) -> PyResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The walk FFN over this model's weights and gate index.
    fn walk_ffn(&self) -> WalkFfn<'_> {
        WalkFfn::new(&self.weights, &self.index, self.top_k)
    }
}

/// Decode `seq_len × hidden` native-endian f32s from Python bytes, checking
/// the length first.
fn decode_positions(x_bytes: &[u8], seq_len: usize, hidden: usize) -> PyResult<Vec<f32>> {
    let expected = seq_len
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(F32_BYTES));
    if expected != Some(x_bytes.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Expected {} bytes ({}x{}xf32), got {}",
            seq_len.saturating_mul(hidden).saturating_mul(F32_BYTES),
            seq_len,
            hidden,
            x_bytes.len()
        )));
    }
    Ok(f32s_from_bytes(x_bytes))
}
