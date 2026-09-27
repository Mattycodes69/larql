//! WalkModel: logit lens, vocab projection and hooked generation.

use larql_inference::forward::{
    embedding_neighbors as li_embedding_neighbors, embedding_row as li_embedding_row,
    embedding_row_scaled as li_embedding_row_scaled, logit_lens_topk,
    project_through_unembed as li_project_through_unembed, track_race as li_track_race,
    track_token as li_track_token, unembedding_row as li_unembedding_row, SteerHook,
    ZeroAblateHook,
};
use larql_inference::WalkFfn;
use larql_kv::generation::generate_cached_hooked;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::types::PyDict;

#[allow(unused_imports)]
use super::*;

#[pymethods]
impl PyWalkModel {
    // ── Logit lens / vocab projection ──────────────────────────────────────

    /// Project `residual` through final norm + lm_head + softcap and
    /// return the top-`k` `(token_id, probability)` pairs.
    #[pyo3(signature = (residual, k=10))]
    pub(super) fn logit_lens(
        &self,
        residual: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        Ok(logit_lens_topk(&self.weights, residual.as_slice()?, k))
    }

    /// Probability of `target_token_id` at the residual.
    pub(super) fn track_token_at(
        &self,
        residual: PyReadonlyArray1<f32>,
        target_token_id: u32,
    ) -> PyResult<f32> {
        Ok(li_track_token(
            &self.weights,
            residual.as_slice()?,
            target_token_id,
        ))
    }

    /// Top-k per layer for a `dict[layer] -> residual` mapping.
    /// Returns `dict[layer] -> List[(token_id, prob)]`.
    #[pyo3(signature = (residuals, k=5))]
    pub(super) fn track_race<'py>(
        &self,
        py: Python<'py>,
        residuals: &Bound<'py, PyDict>,
        k: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut pairs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(residuals.len());
        for (key, val) in residuals.iter() {
            let layer: usize = key.extract()?;
            let arr: PyReadonlyArray1<f32> = val.extract()?;
            pairs.push((layer, arr.as_slice()?.to_vec()));
        }
        let race = li_track_race(&self.weights, &pairs, k);
        let out = PyDict::new(py);
        for (layer, top) in race {
            out.set_item(layer, top)?;
        }
        Ok(out)
    }

    /// Top-`k` vocab tokens by cosine similarity to `query` against `W_E`.
    /// Returns `[(token_id, cosine), ...]` descending.
    #[pyo3(signature = (query, k=10))]
    pub(super) fn embedding_neighbors(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        Ok(li_embedding_neighbors(&self.weights, query.as_slice()?, k))
    }

    /// Raw `lm_head @ vec` projection — top-`k` `(token_id, logit)` pairs.
    /// **No final norm, no softcap, no softmax.** This is the DLA
    /// primitive — apply it to a head's contribution or any direction
    /// you want to read out as a vocabulary distribution without the
    /// model's final-stage normalisation.
    #[pyo3(signature = (vec, k=10))]
    pub(super) fn project_through_unembed(
        &self,
        vec: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        Ok(li_project_through_unembed(
            &self.weights,
            vec.as_slice()?,
            k,
        ))
    }

    /// Embedding row for `token_id`. `scaled=True` (default) returns the
    /// row multiplied by `embed_scale` so it matches what the forward
    /// pass writes into the residual. `scaled=False` returns the raw
    /// matrix row.
    #[pyo3(signature = (token_id, scaled=true))]
    pub(super) fn embedding_for<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
        scaled: bool,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let row = if scaled {
            li_embedding_row_scaled(&self.weights, token_id)
        } else {
            li_embedding_row(&self.weights, token_id)
        };
        Ok(row.map(|r| r.into_pyarray(py)))
    }

    /// Unembedding (`lm_head`) row for `token_id` — the direction whose
    /// dot product with the final residual gives the raw logit for that
    /// token (before any norm/softcap/scaling).
    pub(super) fn unembedding_for<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        Ok(li_unembedding_row(&self.weights, token_id).map(|r| r.into_pyarray(py)))
    }

    /// Multi-token generation with a `LayerHook` active on **every layer
    /// of every step** (prefill + each decode step). Mirrors lazarus's
    /// `steer_and_generate` and `ablate_and_generate` workflows.
    ///
    /// Pass an `ablate_layers` list to zero the post-layer residual at
    /// those layers, and/or a `steers` list of `(layer, vector, alpha)`
    /// triples to add `alpha * v` to the last-token row at those layers.
    /// Both apply on every step. Returns the generated string and the
    /// raw token ids.
    ///
    /// **Backend note**: this routes to the CPU KV-cache path. The
    /// Metal-fast `predict` is hook-free by design (kernel pipeline is
    /// fused). For mech-interp use cases hooks-on-CPU is the right
    /// trade.
    #[pyo3(signature = (prompt, max_new_tokens, ablate_layers=None, steers=None))]
    pub(super) fn generate_with_hooks(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_layers: Option<Vec<usize>>,
        steers: Option<Vec<(usize, PyReadonlyArray1<f32>, f32)>>,
    ) -> PyResult<(String, Vec<u32>)> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);

        // Build the active hook(s). When both ablate + steer are present,
        // wrap them in a CompositeHook; otherwise pass the single hook
        // directly so we don't pay for the extra dispatch.
        let mut ablate = ZeroAblateHook::for_layers(ablate_layers.unwrap_or_default());
        let mut steer = SteerHook::new();
        if let Some(steers) = steers {
            for (layer, vec, alpha) in steers {
                let arr = Array1::from_vec(vec.as_slice()?.to_vec());
                steer = steer.add(layer, arr, alpha);
            }
        }

        let mut composite = larql_inference::forward::CompositeHook::new(vec![
            &mut ablate as &mut dyn larql_inference::forward::LayerHook,
            &mut steer as &mut dyn larql_inference::forward::LayerHook,
        ]);

        let mut generated_text = String::new();
        let ids = generate_cached_hooked(
            &self.weights,
            &self.tokenizer,
            &walk_ffn,
            &token_ids,
            max_new_tokens,
            None,
            None,
            &mut composite,
            |_id, text| generated_text.push_str(text),
        );
        Ok((generated_text, ids))
    }
}
