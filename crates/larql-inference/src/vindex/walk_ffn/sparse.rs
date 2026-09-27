//! Sparse walk path — zero matrix multiplications.
//!
//! The hot path for FFN inference on the LARQL vindex. For each position:
//!
//!   1. `gate_knn` → top-K features (HNSW / batched brute-force / gate-walk)
//!   2. For each feature:
//!      - `up_score  = dot(up_row(feat), x)`         via unified ffn_row_dot
//!      - `activated = silu(gate_score) * up_score`   (GEGLU)
//!      - `out      += activated * down_row(feat)`   via unified ffn_row_scaled_add
//!
//! The "unified" accessors in the `GateIndex` trait dispatch through
//! FP4 → native f32 → Q4K backends in priority order, so this single
//! function is **format-blind** — the same code path serves FP4, Q4K,
//! and native f32 vindexes. Adding a new storage format doesn't touch
//! this file.
//!
//! Four specialisations are layered on top, each in its own module:
//!
//! - **Full-K gemv fast path** (`sparse_gemv.rs`): when K ≥ 80% of
//!   num_features, the per-feature loop is mathematically equivalent to
//!   three dense matmuls — route through BLAS gemm / Q4K direct matmul.
//! - **Route selection** (`sparse_route.rs`): the per-position decision
//!   tree picking WHICH features the walk visits (cell router / pools /
//!   selector dispatch).
//! - **Gather-contiguous Q4K kernel** (`sparse_gather.rs`): known-pool
//!   routes gather gate/up/down bytes contiguous and run fused kernels.
//! - **Parallel Q4K down-cache path** (`sparse_parallel.rs`): for
//!   medium-K on Q4K-only vindexes, cache the dequantised down layer
//!   and parallelise feature chunks over rayon.
//!
//! This file keeps the entry point (`walk_ffn_sparse`), its preflight,
//! and the serial per-feature loop — the canonical correctness
//! baseline; it always works because `ffn_row_*` always has *some*
//! backend.

use ndarray::Array2;

use super::helpers::hits_len_ge_intermediate;
use super::observe::Observe;
use super::thresholds::GATHER_MIN_FEATURES;
use super::WalkFfn;
use crate::ffn::{FfnActivations, SparseActivations};
use crate::vindex::walk_config::FeatureSelector;
use larql_vindex::{FFN_DOWN, FFN_UP};

/// Dispatch-trace / per-position kernel label for the serial
/// per-feature loop — single source for `trace_path` and the runtime
/// trace's observation labels (2026-07-30 review, item 17).
pub(super) const PATH_SPARSE_SERIAL: &str = "sparse:serial";
/// Same, for the gather-contiguous Q4K kernel (`sparse_gather.rs`).
pub(super) const PATH_SPARSE_GATHER: &str = "sparse:gather_q4k";

impl<'a> WalkFfn<'a> {
    /// Sparse walk FFN — see module docs.
    ///
    /// In `Observe::Record` mode the walk emits exactly the `(feature,
    /// activation)` pairs it computes (full-K gemv reports its intrinsic
    /// dense matrix); in `Skip` mode no activation buffer exists at all
    /// — the returned observation is `None`.
    pub(super) fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
        observe: Observe,
    ) -> Option<(Array2<f32>, Option<FfnActivations>)> {
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        // Prefer native f32 mmap (zero-copy). When no native mmap is
        // available we still run — the inner loops dispatch per-row
        // through `ffn_row_dot` / `ffn_row_scaled_add`, which the
        // GateIndex trait routes to FP4 or Q4K or last-resort native
        // as appropriate. The only thing we can't do with neither
        // native f32 mmap, Q4K storage, nor FP4 storage is the serial
        // per-feature loop — those all fail and bail.
        let up_native = self.index.up_layer_matrix(layer);
        let down_native = self.index.down_layer_matrix(layer);
        let row_fallback = up_native.is_none() || down_native.is_none();
        if row_fallback
            && self.index.interleaved_kquant_layer_data(layer).is_none()
            && !self.index.has_fp4_storage()
        {
            return None;
        }

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

        // Hint the kernel to start streaming layer N+1's Q4_K/Q6_K bytes
        // into the page cache while we work on N. No-op when there's no
        // Q4_K mmap, no manifest, or `layer+1` is out of range.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        // Observation record — only in Record mode; the per-feature
        // loops below push exactly what they compute into it. The old
        // dense `seq_len × intermediate` zero-fill is gone.
        let mut obs = observe
            .recording()
            .then(|| SparseActivations::new(seq_len, intermediate));

        let layer_has_overrides = self.index.has_overrides_at(layer);
        let up_bias_for_layer = if !is_gated {
            arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk).cloned())
        } else {
            None
        };
        let activation_floor = self.config.effective_activation_floor();

        // ── Full-K gemv fast path (`sparse_gemv.rs`) ─────────────────────
        // Skipped when a non-default selector is configured, a per-layer
        // pool restriction is set, or a two-stage shortlist is requested:
        // in all three cases gemv would bypass the alternative selection
        // structure, so we force the walk.
        let selector_forces_walk = !matches!(self.config.selector, FeatureSelector::GateOnly)
            || self.config.pool_per_layer.is_some()
            || self.config.cell_router.is_some()
            || self.config.shortlist_m.is_some();
        let k_is_full =
            !selector_forces_walk && hits_len_ge_intermediate(&self.config, layer, intermediate);
        if !layer_has_overrides && is_gated && k_is_full {
            if let Some((gemv_out, act)) =
                self.sparse_full_k_gemv(layer, x, up_native, down_native, use_gelu, intermediate)
            {
                // The gemv computes every feature; its activation matrix
                // is intrinsic, so the observation is honestly Dense.
                return Some((gemv_out, observe.dense(act)));
            }
        }

        // ── Per-position sparse loop ─────────────────────────────────────
        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();
            let x_slice_owned: Vec<f32>;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_slice_owned = x_owned.as_slice().unwrap().to_vec();
                &x_slice_owned
            };

            let top_k = self.top_k_for(layer);

            // ── Gather-contiguous Q4K fast path (`sparse_gather.rs`) ─────
            // For a KNOWN-pool route (precomputed pool or cell-router, no
            // within-pool ranking) the active feature set is decided without
            // gate scores, so we skip the scattered `local_pool_gate_knn` and
            // gather gate+up+down (down from the feature-major sidecar)
            // contiguous, running the fused kernel in one cache-friendly pass.
            // Fixes the ~4× per-row overhead at faithful K; re-gathers every
            // position (the content-addressed pool moves per token). Declines
            // (→ scalar paths) unless gated, no overrides, Q4K up, the down
            // sidecar is loaded, and the route has ≥ GATHER_MIN_FEATURES.
            if is_gated
                && !layer_has_overrides
                && up_native.is_none()
                && !self.config.rank_within_pool
                && self.index.has_down_features_kquant()
            {
                if let Some(feats) = self.gather_route_feats(layer, x_slice, top_k) {
                    if feats.len() >= GATHER_MIN_FEATURES {
                        if let Some(g) =
                            self.gather_q4k_accumulate(layer, &feats, x_slice, use_gelu, hidden)
                        {
                            let mut out_row = out.row_mut(s);
                            out_row.as_slice_mut().unwrap().copy_from_slice(&g.out);
                            if let Some(o) = obs.as_mut() {
                                // The gate/up dots the fused kernels
                                // actually computed ride into the record.
                                for (i, &feat) in feats.iter().enumerate() {
                                    o.record_scored(
                                        s,
                                        feat,
                                        g.acts[i],
                                        Some(g.gate_scores[i]),
                                        Some(g.up_scores[i]),
                                    );
                                }
                                o.set_kernel(s, PATH_SPARSE_GATHER);
                            }
                            self.trace_path(layer, PATH_SPARSE_GATHER);
                            continue;
                        }
                    }
                }
            }

            let t_gate = std::time::Instant::now();
            let hits = self.select_route_hits(layer, &x_owned, x_slice, top_k);
            let gate_knn_ns = t_gate.elapsed().as_nanos() as u64;

            let mut out_row = out.row_mut(s);

            // Parallel Q4K-down-cache path (`sparse_parallel.rs`).
            if self.try_parallel_q4k_down(
                layer,
                &hits,
                x_row,
                x_slice,
                up_native,
                use_gelu,
                down_native.is_none(),
                is_gated,
                layer_has_overrides,
                gate_knn_ns,
                &mut out_row,
                s,
                obs.as_mut(),
            ) {
                continue;
            }

            // Serial per-feature loop — the correctness baseline.
            for (feat, gate_score) in hits {
                // `(activation, gate-position score, up score)` — the
                // scores are recorded alongside the activation so the
                // runtime trace reports the EXECUTED projections
                // (2026-07-30 review, item 17). Non-gated archs have a
                // single projection: its post-bias value is the
                // gate-position score, and there is no up score.
                let (act, gate_obs, up_obs) = if is_gated {
                    let up_ov = if layer_has_overrides {
                        self.index.up_override(layer, feat)
                    } else {
                        None
                    };
                    let up_score = if let Some(up_ov) = up_ov.filter(|o| o.len() == hidden) {
                        ndarray::ArrayView1::from(up_ov).dot(&x_row)
                    } else if let Some(ref up_view) = up_native {
                        up_view.row(feat).dot(&x_row)
                    } else {
                        // Unified dispatch: FP4 → native → Q4K, per GateIndex.
                        self.index.ffn_row_dot(layer, FFN_UP, feat, x_slice)?
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    (activated_gate * up_score, gate_score, Some(up_score))
                } else {
                    let mut v = gate_score;
                    if let Some(ref bias) = up_bias_for_layer {
                        if feat < bias.len() {
                            v += bias[feat];
                        }
                    }
                    let act = if use_gelu {
                        crate::ffn::gelu_tanh(v)
                    } else {
                        v * crate::ffn::sigmoid(v)
                    };
                    (act, v, None)
                };

                if let Some(o) = obs.as_mut() {
                    // Recorded regardless of the floor: the floor gates
                    // the down accumulate, not activation observation.
                    o.record_scored(s, feat, act, Some(gate_obs), up_obs);
                }

                if act.abs() > activation_floor {
                    let down_ov = if layer_has_overrides {
                        self.index.down_override(layer, feat)
                    } else {
                        None
                    };
                    if let Some(override_down) = down_ov.filter(|o| o.len() == hidden) {
                        out_row.scaled_add(act, &ndarray::ArrayView1::from(override_down));
                        continue;
                    }
                    if let Some(ref down_view) = down_native {
                        out_row.scaled_add(act, &down_view.row(feat));
                    } else {
                        let out_slice = out_row.as_slice_mut().unwrap();
                        // Unified dispatch: FP4 → native → Q4K-via-cache, per GateIndex.
                        if !self
                            .index
                            .ffn_row_scaled_add(layer, FFN_DOWN, feat, act, out_slice)
                        {
                            return None;
                        }
                    }
                }
            }

            if let Some(o) = obs.as_mut() {
                o.set_kernel(s, PATH_SPARSE_SERIAL);
            }
        }

        // Down bias
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, PATH_SPARSE_SERIAL);
        Some((out, obs.map(FfnActivations::Sparse)))
    }
}

#[cfg(test)]
mod tests;
