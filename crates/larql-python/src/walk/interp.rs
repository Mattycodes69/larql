//! WalkModel: mechanistic-interp surface (lazarus parity) — captures, ablations, steering, patching.

use larql_inference::forward::{
    capture_donor_state_with_ffn, patch_and_trace_with_ffn, trace_forward_attn_only_capture_pre_o,
    trace_forward_attn_only_with_head_zero, trace_forward_full_hooked, AttnZeroHook, FFNZeroHook,
    RecordHook, SteerHook, ZeroAblateHook,
};
use larql_inference::WalkFfn;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::types::PyDict;
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

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
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let mut hook = RecordHook::for_layers(layers.iter().copied());
        let _ = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut hook,
        );

        let out = PyDict::new(py);
        for (layer, mat) in hook.post_layer.iter() {
            // Last-token row only — matches the convention everywhere else
            // in larql_inference. Full matrix available via
            // `forward_with_capture` if a caller needs every position.
            let last = mat.row(mat.nrows() - 1).to_vec();
            out.set_item(*layer, last.into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let mut hook = RecordHook::for_layers(layers.iter().copied());
        let _ = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut hook,
        );

        let out = PyDict::new(py);
        for (layer, mat) in hook.post_layer.iter() {
            out.set_item(*layer, mat.clone().into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let n_layers = self.weights.num_layers;
        let mut ffn_zero = FFNZeroHook::for_layers(0..n_layers);
        let mut record = RecordHook::for_layers(layers.iter().copied());
        {
            let mut composite = larql_inference::forward::CompositeHook::new(vec![
                &mut ffn_zero as &mut dyn larql_inference::forward::LayerHook,
                &mut record as &mut dyn larql_inference::forward::LayerHook,
            ]);
            let _ = trace_forward_full_hooked(
                &self.weights,
                &token_ids,
                &layers,
                false,
                0,
                false,
                &walk_ffn,
                &mut composite,
            );
        }

        let out = PyDict::new(py);
        for (layer, mat) in record.post_layer.iter() {
            out.set_item(*layer, mat.clone().into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let n_layers = self.weights.num_layers;
        let mut attn_zero = AttnZeroHook::for_layers(0..n_layers);
        let mut record = RecordHook::for_layers(layers.iter().copied());
        {
            let mut composite = larql_inference::forward::CompositeHook::new(vec![
                &mut attn_zero as &mut dyn larql_inference::forward::LayerHook,
                &mut record as &mut dyn larql_inference::forward::LayerHook,
            ]);
            let _ = trace_forward_full_hooked(
                &self.weights,
                &token_ids,
                &layers,
                false,
                0,
                false,
                &walk_ffn,
                &mut composite,
            );
        }

        let out = PyDict::new(py);
        for (layer, mat) in record.post_layer.iter() {
            out.set_item(*layer, mat.clone().into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let head_zero_map: HashMap<usize, Vec<usize>> = head_zeros.into_iter().collect();
        let captures = trace_forward_attn_only_with_head_zero(
            &self.weights,
            &token_ids,
            &layers,
            &head_zero_map,
        );
        let out = PyDict::new(py);
        for (layer, mat) in captures.iter() {
            out.set_item(*layer, mat.clone().into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let captures = trace_forward_attn_only_capture_pre_o(&self.weights, &token_ids, &layers);
        let out = PyDict::new(py);
        for (layer, mat) in captures.iter() {
            out.set_item(*layer, mat.clone().into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let mut ablate = ZeroAblateHook::for_layers(ablate_layers);
        let trace = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &capture_layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut ablate,
        );

        let out = PyDict::new(py);
        for (layer, residual) in trace.residuals {
            out.set_item(layer, residual.into_pyarray(py))?;
        }
        Ok(out)
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
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);

        let mut steer = SteerHook::new();
        for (layer, vec, alpha) in steers {
            let arr = Array1::from_vec(vec.as_slice()?.to_vec());
            steer = steer.add(layer, arr, alpha);
        }
        let trace = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &capture_layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut steer,
        );

        let out = PyDict::new(py);
        for (layer, residual) in trace.residuals {
            out.set_item(layer, residual.into_pyarray(py))?;
        }
        Ok(out)
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
        let donor_tokens = self.encode(donor_prompt)?;
        let recipient_tokens = self.encode(recipient_prompt)?;

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let donor = capture_donor_state_with_ffn(&self.weights, &donor_tokens, &coords, &walk_ffn);
        let trace = patch_and_trace_with_ffn(
            &self.weights,
            &recipient_tokens,
            &donor,
            &capture_layers,
            &walk_ffn,
        );

        let out = PyDict::new(py);
        for (layer, residual) in trace.residuals {
            out.set_item(layer, residual.into_pyarray(py))?;
        }
        Ok(out)
    }
}
