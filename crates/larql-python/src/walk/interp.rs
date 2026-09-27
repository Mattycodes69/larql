//! WalkModel: mechanistic-interp surface (lazarus parity) — captures, ablations, steering, patching.
//!
//! Every forward pass here runs inside `Python::detach`; numpy inputs are
//! copied out before, and the result dicts are built after.

use larql_inference::forward::{
    capture_donor_state_with_ffn, patch_and_trace_with_ffn, trace_forward_attn_only_capture_pre_o,
    trace_forward_attn_only_with_head_zero, trace_forward_full_hooked, AttnZeroHook, CompositeHook,
    FFNZeroHook, LayerHook, RecordHook, SteerHook, ZeroAblateHook,
};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::types::PyDict;
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

/// Post-layer residual matrices keyed by layer.
type LayerMatrices = HashMap<usize, Array2<f32>>;
/// Last-token residual vectors, one per captured layer.
type LayerVectors = Vec<(usize, Vec<f32>)>;

#[pymethods]
impl PyWalkModel {
    // ── Mechanistic interp surface (lazarus parity) ────────────────────────
    //
    // These methods mirror the chuk-mcp-lazarus tool surface. They run a
    // forward pass with a `LayerHook` registered and return numpy tensors
    // ready for Python-side analysis.

    /// Tokenize then capture last-token residual at each requested layer.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (hidden_size,)`.
    #[pyo3(signature = (prompt, layers))]
    pub(super) fn capture_residuals<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let captured = py.detach(|| self.record_post_layer(prompt, &layers, None))?;
        // Last-token row only — matches the convention everywhere else in
        // larql_inference. Full matrix available via `forward_with_capture`
        // if a caller needs every position.
        let last_rows: LayerVectors = captured
            .into_iter()
            .map(|(layer, mat)| (layer, mat.row(mat.nrows() - 1).to_vec()))
            .collect();
        vectors_dict(py, last_rows)
    }

    /// Run a forward pass with a [`RecordHook`] and return the **full**
    /// `(seq_len, hidden_size)` post-layer residual at each requested
    /// layer. Larger than `capture_residuals` — only call when you need
    /// per-position activations (patching, full causal trace).
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (seq_len, hidden_size)`.
    #[pyo3(signature = (prompt, layers))]
    pub(super) fn forward_with_capture<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let captured = py.detach(|| self.record_post_layer(prompt, &layers, None))?;
        matrices_dict(py, captured)
    }

    /// Run a forward pass with the **FFN sublayer skipped at every layer**
    /// (the FFN computation still runs but its addition to the residual stream
    /// is discarded) and capture the resulting post-layer residual at each
    /// requested layer. Used for attention-vs-FFN decomposition: this gives
    /// the "attention-only contribution" residual stream.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (seq_len, hidden_size)`.
    #[pyo3(signature = (prompt, layers))]
    pub(super) fn forward_attn_only<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let captured = py.detach(|| {
            let mut ffn_zero = FFNZeroHook::for_layers(0..self.weights.num_layers);
            self.record_post_layer(prompt, &layers, Some(&mut ffn_zero))
        })?;
        matrices_dict(py, captured)
    }

    /// Run a forward pass with the **attention sublayer skipped at every
    /// layer** (the attention computation still runs but its addition to the
    /// residual stream is discarded) and capture the resulting post-layer
    /// residual at each requested layer. Used for attention-vs-FFN
    /// decomposition: this gives the "FFN-only contribution" residual stream.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (seq_len, hidden_size)`.
    #[pyo3(signature = (prompt, layers))]
    pub(super) fn forward_ffn_only<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let captured = py.detach(|| {
            let mut attn_zero = AttnZeroHook::for_layers(0..self.weights.num_layers);
            self.record_post_layer(prompt, &layers, Some(&mut attn_zero))
        })?;
        matrices_dict(py, captured)
    }

    /// Attn-only forward with **per-layer pre-W_O head zeroing**.
    ///
    /// Like `forward_attn_only` (FFN/PLE/scalar skipped at every layer) but at
    /// each layer L listed in `head_zeros` the listed query-head slices are
    /// zeroed before the W_O projection. With an empty `head_zeros` list this
    /// matches `forward_attn_only` exactly (useful as a self-consistency
    /// check).
    ///
    /// `head_zeros` is a list of `(layer, [head_idx, ...])` tuples.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (seq_len, hidden_size)`.
    #[pyo3(signature = (prompt, head_zeros, layers))]
    pub(super) fn forward_attn_only_head_zero<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        head_zeros: Vec<(usize, Vec<usize>)>,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let head_zero_map: HashMap<usize, Vec<usize>> = head_zeros.into_iter().collect();
        let captured = py.detach(|| {
            let token_ids = self.encode(prompt)?;
            Ok::<_, PyErr>(trace_forward_attn_only_with_head_zero(
                &self.weights,
                &token_ids,
                &layers,
                &head_zero_map,
            ))
        })?;
        matrices_dict(py, captured)
    }

    /// Attn-only forward returning **pre-W_O per-head outputs** at each
    /// requested layer. FFN/PLE/scalar are skipped at every layer (matching
    /// `forward_attn_only` semantics). The returned array has shape
    /// `(seq_len, num_q_heads * head_dim)` per layer; slice per-head as
    /// `pre_o[:, h * head_dim : (h + 1) * head_dim]`.
    ///
    /// Use `num_q_heads_for_layer(L)` and `head_dim_for_layer(L)` to get the
    /// strides.
    #[pyo3(signature = (prompt, layers))]
    pub(super) fn forward_attn_only_capture_pre_o<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let captured = py.detach(|| {
            let token_ids = self.encode(prompt)?;
            Ok::<_, PyErr>(trace_forward_attn_only_capture_pre_o(
                &self.weights,
                &token_ids,
                &layers,
            ))
        })?;
        matrices_dict(py, captured)
    }

    /// Returns the number of query heads at the given layer (Gemma 3 4B has 8).
    pub(super) fn num_q_heads_for_layer(&self, layer: usize) -> PyResult<usize> {
        Ok(self.weights.arch.num_q_heads_for_layer(layer))
    }

    /// Returns the per-head dimension at the given layer (Gemma 3 4B uses 256).
    pub(super) fn head_dim_for_layer(&self, layer: usize) -> PyResult<usize> {
        Ok(self.weights.arch.head_dim_for_layer(layer))
    }

    /// Returns the W_O slice for a specific head at a specific layer.
    /// Shape: (head_dim, hidden_size). Used for projecting per-head pre-W_O
    /// outputs into residual space.
    pub(super) fn w_o_for_head<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        head: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let head_dim = self.weights.arch.head_dim_for_layer(layer);
        let key = self.weights.arch.attn_o_key(layer);
        let w_o = self
            .weights
            .tensors
            .get(&key)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("missing {key}")))?;
        let start = head * head_dim;
        let end = start + head_dim;
        if end > w_o.ncols() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "head {head} out of range for layer {layer} (W_O has {} cols)",
                w_o.ncols()
            )));
        }
        // W_O has shape (hidden_size, n_heads * head_dim); the head slice is columns
        // [start..end]. We want (head_dim, hidden_size) so each row is "this column of
        // pre_o contributes this residual delta". Transpose the slice.
        let slice = w_o.slice(ndarray::s![.., start..end]);
        Ok(slice.t().to_owned().into_pyarray(py))
    }

    /// Zero-ablate the post-layer residual at the listed `ablate_layers`,
    /// then capture last-token residuals at `capture_layers`. Mirrors
    /// lazarus's `ablate_layers` + measurement workflow.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (hidden_size,)` for
    /// each capture layer (post-ablation).
    #[pyo3(signature = (prompt, ablate_layers, capture_layers))]
    pub(super) fn forward_ablate<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        ablate_layers: Vec<usize>,
        capture_layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let residuals = py.detach(|| {
            let mut ablate = ZeroAblateHook::for_layers(ablate_layers);
            self.hooked_residuals(prompt, &capture_layers, &mut ablate)
        })?;
        vectors_dict(py, residuals)
    }

    /// Add `alpha * v` to the last-token row of the post-layer residual at
    /// each (layer, vector, alpha) entry, then capture last-token
    /// residuals at `capture_layers`. Mirrors lazarus's `steer_and_generate`
    /// at the residual-readback level.
    ///
    /// `steers` is a list of `(layer, numpy_vector, alpha)` tuples.
    #[pyo3(signature = (prompt, steers, capture_layers))]
    pub(super) fn forward_steer<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        steers: Vec<(usize, PyReadonlyArray1<f32>, f32)>,
        capture_layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let steers = owned_steers(steers)?;
        let residuals = py.detach(|| {
            let mut steer = steer_hook(steers);
            self.hooked_residuals(prompt, &capture_layers, &mut steer)
        })?;
        vectors_dict(py, residuals)
    }

    /// Activation patching. Run `donor_prompt`, capture post-layer
    /// residuals at the `(layer, position)` coords in `coords`, then run
    /// `recipient_prompt` with those residuals patched in at the same
    /// coords. Returns last-token residuals at `capture_layers` (post-
    /// patch).
    ///
    /// Mirrors lazarus's `patch_activations`. Uses the vindex WalkFfn path
    /// so patches are measured against the same mechanism as inference.
    #[pyo3(signature = (donor_prompt, recipient_prompt, coords, capture_layers))]
    pub(super) fn patch_activations<'py>(
        &self,
        py: Python<'py>,
        donor_prompt: &str,
        recipient_prompt: &str,
        coords: Vec<(usize, usize)>,
        capture_layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let residuals = py.detach(|| {
            let donor_tokens = self.encode(donor_prompt)?;
            let recipient_tokens = self.encode(recipient_prompt)?;
            let walk_ffn = self.walk_ffn();
            let donor =
                capture_donor_state_with_ffn(&self.weights, &donor_tokens, &coords, &walk_ffn);
            let trace = patch_and_trace_with_ffn(
                &self.weights,
                &recipient_tokens,
                &donor,
                &capture_layers,
                &walk_ffn,
            );
            Ok::<_, PyErr>(trace.residuals)
        })?;
        vectors_dict(py, residuals)
    }
}

impl PyWalkModel {
    /// Walk-FFN forward over `prompt` recording the full post-layer residual
    /// at `layers`, with an optional extra hook composed ahead of the
    /// recorder. Pure Rust; callers run it detached.
    fn record_post_layer(
        &self,
        prompt: &str,
        layers: &[usize],
        extra: Option<&mut dyn LayerHook>,
    ) -> PyResult<LayerMatrices> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = self.walk_ffn();
        let mut record = RecordHook::for_layers(layers.iter().copied());
        {
            let mut hooks: Vec<&mut dyn LayerHook> = Vec::with_capacity(2);
            if let Some(extra) = extra {
                hooks.push(extra);
            }
            hooks.push(&mut record);
            let mut composite = CompositeHook::new(hooks);
            let _ = trace_forward_full_hooked(
                &self.weights,
                &token_ids,
                layers,
                false,
                0,
                false,
                &walk_ffn,
                &mut composite,
            );
        }
        Ok(record.post_layer)
    }

    /// Walk-FFN forward over `prompt` with `hook` active, returning the
    /// last-token residual at each of `capture_layers`.
    fn hooked_residuals(
        &self,
        prompt: &str,
        capture_layers: &[usize],
        hook: &mut dyn LayerHook,
    ) -> PyResult<LayerVectors> {
        let token_ids = self.encode(prompt)?;
        let trace = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            capture_layers,
            false,
            0,
            false,
            &self.walk_ffn(),
            hook,
        );
        Ok(trace.residuals)
    }
}

/// Copy `(layer, numpy vector, alpha)` steers into owned Rust values so they
/// can cross into a detached closure.
pub(super) fn owned_steers(
    steers: Vec<(usize, PyReadonlyArray1<f32>, f32)>,
) -> PyResult<Vec<(usize, Array1<f32>, f32)>> {
    steers
        .into_iter()
        .map(|(layer, vec, alpha)| Ok((layer, Array1::from_vec(vec.as_slice()?.to_vec()), alpha)))
        .collect()
}

/// A [`SteerHook`] adding each `alpha * v` at its layer.
pub(super) fn steer_hook(steers: Vec<(usize, Array1<f32>, f32)>) -> SteerHook {
    steers
        .into_iter()
        .fold(SteerHook::new(), |hook, (layer, v, alpha)| {
            hook.add(layer, v, alpha)
        })
}

fn matrices_dict(py: Python<'_>, matrices: LayerMatrices) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (layer, mat) in matrices {
        out.set_item(layer, mat.into_pyarray(py))?;
    }
    Ok(out)
}

fn vectors_dict(py: Python<'_>, vectors: LayerVectors) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (layer, residual) in vectors {
        out.set_item(layer, residual.into_pyarray(py))?;
    }
    Ok(out)
}
