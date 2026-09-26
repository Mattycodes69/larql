//! Joint-criterion feature selection — tests whether the K=200 walk
//! failure is a *selection* problem (gate-score ranking is wrong) or a
//! *coverage* problem (the top-K-by-anything just isn't enough).
//!
//! The current production walk picks top-K by `|gate_score|`. The
//! actual contribution of feature `i` to the residual at this position
//! is `silu(gate_i) · up_i · down_row_i`. Ranking by `|gate|` alone
//! ignores the magnitudes of the `up` and `down` components.
//!
//! This module exposes:
//! - `down_row_norms(layer)` — lazy per-layer cache of `‖down_row‖`
//!   (computed from the dequantised down cache).
//! - `up_row_norms(layer)`   — same, for `‖up_row‖` (Q4K up).
//! - `joint_gate_knn(layer, residual, top_k, kind)` — get full gate
//!   scores via `gate_scores_batch_backend`, weight by the chosen
//!   joint criterion, take top-K. Returns `(feat_idx, raw_gate_score)`
//!   so the FFN math downstream is unchanged.

use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::helpers::selection_weight_cmp_desc;
use super::shortlist::{criterion_inputs, criterion_weight, rerank_cmp};
use super::WalkFfn;
use crate::vindex::walk_config::FeatureSelector;
use larql_vindex::{FFN_DOWN, FFN_UP};

impl<'a> WalkFfn<'a> {
    /// Public view of `down_row_norms` for probes/examples. Same lazy
    /// cache.
    pub fn down_row_norms_pub(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        self.down_row_norms(layer)
    }

    /// Public view of `up_row_norms`.
    pub fn up_row_norms_pub(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        self.up_row_norms(layer)
    }

    /// Public view of `compute_full_up_scores`.
    pub fn compute_full_up_scores_pub(
        &self,
        layer: usize,
        residual: &Array1<f32>,
    ) -> Option<Vec<f32>> {
        self.compute_full_up_scores(layer, residual)
    }

    /// Lazy per-layer `‖down_row‖`. Triggers
    /// `kquant_ffn_layer(layer, FFN_DOWN)` on first call, then caches
    /// the norms.
    pub(super) fn down_row_norms(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        if let Some(Some(arc)) = self.down_norms_cache.borrow().get(layer) {
            return Some(Arc::clone(arc));
        }
        let down_data = self.index.kquant_ffn_layer(layer, FFN_DOWN)?;
        let num_features = self.index.num_features(layer);
        let hidden = self.weights.hidden_size;
        if down_data.len() < num_features * hidden {
            return None;
        }
        let mut norms = Vec::with_capacity(num_features);
        for feat in 0..num_features {
            let row = &down_data[feat * hidden..(feat + 1) * hidden];
            let sumsq: f32 = row.iter().map(|v| v * v).sum();
            norms.push(sumsq.sqrt());
        }
        let arc = Arc::new(norms);
        let mut cache = self.down_norms_cache.borrow_mut();
        if cache.len() <= layer {
            cache.resize_with(layer + 1, || None);
        }
        cache[layer] = Some(Arc::clone(&arc));
        Some(arc)
    }

    /// Compute all per-feature up scores `⟨up_row, residual⟩` at this
    /// layer for the given residual. Prefers native f32 + BLAS; falls
    /// back to Q4K `kquant_matmul_transb`. Returns a Vec of length
    /// `num_features`.
    pub(super) fn compute_full_up_scores(
        &self,
        layer: usize,
        residual: &Array1<f32>,
    ) -> Option<Vec<f32>> {
        let num_features = self.index.num_features(layer);
        let hidden = self.weights.hidden_size;
        if num_features == 0 || residual.len() != hidden {
            return None;
        }

        // Native f32 path — BLAS / GPU dot.
        if let Some(up_view) = self.index.up_layer_matrix(layer) {
            let x_2d = Array2::from_shape_vec((1, hidden), residual.to_vec()).ok()?;
            let result = larql_compute::dot_proj_gpu(&x_2d, &up_view, self.backend);
            if result.shape() == [1, num_features] {
                return Some(result.row(0).to_vec());
            }
            return None;
        }

        // Q4K path — batched Q4 matmul against the layer's up bytes.
        let x_slice = residual.as_slice()?;
        let y = self
            .index
            .kquant_matmul_transb(layer, 1, x_slice, 1, self.backend)?;
        if y.len() == num_features {
            Some(y)
        } else {
            None
        }
    }

    /// Lazy per-layer `‖up_row‖`. Triggers
    /// `kquant_ffn_layer(layer, FFN_UP)` on first call, then caches
    /// the norms.
    pub(super) fn up_row_norms(&self, layer: usize) -> Option<Arc<Vec<f32>>> {
        if let Some(Some(arc)) = self.up_norms_cache.borrow().get(layer) {
            return Some(Arc::clone(arc));
        }
        let up_data = self.index.kquant_ffn_layer(layer, FFN_UP)?;
        let num_features = self.index.num_features(layer);
        let hidden = self.weights.hidden_size;
        if up_data.len() < num_features * hidden {
            return None;
        }
        let mut norms = Vec::with_capacity(num_features);
        for feat in 0..num_features {
            let row = &up_data[feat * hidden..(feat + 1) * hidden];
            let sumsq: f32 = row.iter().map(|v| v * v).sum();
            norms.push(sumsq.sqrt());
        }
        let arc = Arc::new(norms);
        let mut cache = self.up_norms_cache.borrow_mut();
        if cache.len() <= layer {
            cache.resize_with(layer + 1, || None);
        }
        cache[layer] = Some(Arc::clone(&arc));
        Some(arc)
    }

    /// Top-K features by a joint criterion. Computes full gate scores
    /// once via `gate_scores_batch_backend`, multiplies by per-feature
    /// weights derived from `kind` (formulas single-sourced in
    /// `shortlist::criterion_weight`), takes top-K by `|weighted|` in
    /// the deterministic `rerank_cmp` order, and returns
    /// `(feat_idx, raw_gate_score)` so the FFN math downstream is
    /// unchanged.
    ///
    /// Falls back to the production `gate_walk` path if the joint norms
    /// can't be computed (e.g. no Q4K cache yet), so the walk still
    /// produces output. A **NaN** weighted score, by contrast, panics
    /// (`selection_weight_cmp_desc`) — the same contract as
    /// `top_k_by_abs` on the production gate-KNN path; it must never
    /// silently scramble the selection.
    pub(super) fn joint_gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        kind: FeatureSelector,
    ) -> Vec<(usize, f32)> {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            return Vec::new();
        }
        let hidden = self.weights.hidden_size;

        // Full gate scores in one batched gemv.
        let x = ndarray::Array2::from_shape_vec((1, hidden), residual.to_vec())
            .expect("residual shape (1, hidden)");
        let scores = match self
            .index
            .gate_scores_batch_backend(layer, &x, self.backend)
        {
            Some(s) => s,
            None => {
                // No batched gate-score path — fall back to gate_walk
                // (production behaviour). Random selection also lands
                // here since it doesn't need joint norms either.
                return self.fallback_top_k(layer, residual, top_k, kind);
            }
        };
        let row = scores.row(0);
        if row.len() != num_features {
            return self.fallback_top_k(layer, residual, top_k, kind);
        }

        // Per-feature joint weight — formulas single-sourced in
        // `shortlist::criterion_weight` (shared with the two-stage
        // rerank so the two paths cannot drift). Random has no
        // criterion (`criterion_inputs` returns None) and short-circuits.
        let Some(needs) = criterion_inputs(kind) else {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut idxs: Vec<usize> = (0..num_features).collect();
            idxs.shuffle(&mut rng);
            idxs.truncate(top_k.min(num_features));
            return idxs.into_iter().map(|i| (i, row[i])).collect();
        };
        let down_norms = if needs.down_norm {
            match self.down_row_norms(layer) {
                Some(n) => Some(n),
                None => return self.fallback_top_k(layer, residual, top_k, kind),
            }
        } else {
            None
        };
        let up_norms = if needs.up_norm {
            match self.up_row_norms(layer) {
                Some(n) => Some(n),
                None => return self.fallback_top_k(layer, residual, top_k, kind),
            }
        } else {
            None
        };
        let up_scores = if needs.up_score {
            match self.compute_full_up_scores(layer, residual) {
                Some(s) => Some(s),
                None => return self.fallback_top_k(layer, residual, top_k, kind),
            }
        } else {
            None
        };
        let use_gelu = self.selector_use_gelu();
        let at = |v: &Option<Arc<Vec<f32>>>, i: usize| {
            v.as_ref()
                .map(|v| v.get(i).copied().unwrap_or(0.0))
                .unwrap_or(0.0)
        };
        let mut weighted: Vec<(usize, f32, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, &g)| {
                let u = up_scores
                    .as_ref()
                    .map(|v| v.get(i).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                let w =
                    criterion_weight(kind, use_gelu, g, u, at(&up_norms, i), at(&down_norms, i));
                (i, g, w)
            })
            .collect();

        // Partial sort to the top-K by weighted score, then a full sort
        // of those K into the deterministic rerank order (`rerank_cmp`)
        // so the reported rank order matches the two-stage path. NaN
        // weights panic (`selection_weight_cmp_desc`) — same contract
        // as `top_k_by_abs` on the production gate-KNN path.
        let take = top_k.min(num_features);
        weighted.select_nth_unstable_by(take.saturating_sub(1).min(num_features - 1), |a, b| {
            selection_weight_cmp_desc(a.2, b.2)
        });
        weighted.truncate(take);
        weighted.sort_unstable_by(rerank_cmp);
        weighted.into_iter().map(|(i, g, _)| (i, g)).collect()
    }

    /// Pool-restricted top-K by gate-score: compute full gate scores
    /// via `gate_scores_batch_backend`, restrict to the supplied pool
    /// of feature indices, take top-K within the pool by `|gate|`.
    /// Returns `(feat_idx, raw_gate_score)` so downstream FFN math is
    /// unchanged.
    pub(super) fn pool_restricted_gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        pool: &[usize],
    ) -> Vec<(usize, f32)> {
        if pool.is_empty() {
            return Vec::new();
        }
        let hidden = self.weights.hidden_size;
        let x = ndarray::Array2::from_shape_vec((1, hidden), residual.to_vec())
            .expect("residual shape (1, hidden)");
        let scores = match self
            .index
            .gate_scores_batch_backend(layer, &x, self.backend)
        {
            Some(s) => s,
            None => {
                // No batched gate path — fall back to production hits.
                return self.fallback_top_k(layer, residual, top_k, FeatureSelector::GateOnly);
            }
        };
        let row = scores.row(0);

        let mut weighted: Vec<(usize, f32, f32)> = pool
            .iter()
            .filter_map(|&i| {
                if i < row.len() {
                    let g = row[i];
                    Some((i, g, g.abs()))
                } else {
                    None
                }
            })
            .collect();
        if weighted.is_empty() {
            return Vec::new();
        }
        // NaN weights panic — same contract as `top_k_by_abs`.
        let take = top_k.min(weighted.len());
        let nth = take.saturating_sub(1).min(weighted.len() - 1);
        weighted.select_nth_unstable_by(nth, |a, b| selection_weight_cmp_desc(a.2, b.2));
        weighted.truncate(take);
        weighted.into_iter().map(|(i, g, _)| (i, g)).collect()
    }

    /// Precomputed-route gate scoring — the **cheap routing** path.
    ///
    /// Unlike `pool_restricted_gate_knn` (which calls
    /// `gate_scores_batch_backend` = a full gate projection over *all*
    /// features, then filters to the pool), this computes the gate score
    /// for **only the pool features** via per-row Q4K dots — O(|pool|),
    /// not O(num_features). It models hash routing (Exp 27): the feature
    /// set is decided upstream (token-deterministic), so selection never
    /// touches the full gate matrix.
    ///
    /// Returns `(feature, gate_score)` for each pool feature, in pool
    /// order (no ranking needed — the route *is* the selection). Falls
    /// back to `None` when the layer exposes no Q4K interleaved gate
    /// bytes with a registered `row_dot` (the only storage this fast path
    /// supports; callers then route to `pool_restricted_gate_knn`).
    pub(super) fn local_pool_gate_knn(
        &self,
        layer: usize,
        x_slice: &[f32],
        pool: &[usize],
    ) -> Option<Vec<(usize, f32)>> {
        if pool.is_empty() {
            return Some(Vec::new());
        }
        let hidden = self.weights.hidden_size;
        // Interleaved Q4K layout is [gate, up, down]; gate is slot 0.
        let slices = self.index.interleaved_kquant_layer_data(layer)?;
        let (gate_bytes, gate_tag) = (slices[0].0, slices[0].1);
        let info = larql_vindex::quant::registry::lookup(gate_tag)?;
        let row_dot = info.row_dot?;
        let bytes_per_row = info.bytes_per_row(hidden)?;
        let n_rows = gate_bytes.len() / bytes_per_row;
        Some(
            pool.iter()
                .filter_map(|&feat| {
                    if feat >= n_rows {
                        return None;
                    }
                    let start = feat * bytes_per_row;
                    let end = start + bytes_per_row;
                    let g = row_dot(&gate_bytes[start..end], x_slice).unwrap_or(0.0);
                    Some((feat, g))
                })
                .collect(),
        )
    }

    /// Fallback when the joint-scoring path can't run — falls back to
    /// the production `gate_walk` → `gate_knn_q4` → `gate_knn` chain so
    /// the walk still produces output.
    fn fallback_top_k(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        kind: FeatureSelector,
    ) -> Vec<(usize, f32)> {
        if !matches!(kind, FeatureSelector::GateOnly) {
            // A joint selector (or Random null hypothesis) silently
            // degrading to production GateOnly invalidates the A/B —
            // the results carry the wrong label. Make every such call
            // observable (2026-07-30 review, M10).
            self.trace_path(layer, "selector:fallback");
            self.selector_fallbacks
                .set(self.selector_fallbacks.get() + 1);
        }
        self.production_gate_chain(layer, residual, top_k)
    }
}

#[cfg(test)]
mod tests;
