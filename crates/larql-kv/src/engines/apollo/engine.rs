//! ApolloEngine — retrieval-augmented generation via vec_inject.
//!
//! At prefill: routes the prompt through the RoutingIndex, retrieves the
//! most relevant VecInjectEntry records, computes a combined injection delta
//! (scaled token embeddings), then runs the forward pass on the context
//! (window_tokens ++ query_tokens) with the delta injected at `crystal_layer`.
//!
//! At decode: extends the context by one token per step and re-runs the
//! forward pass with the same injection delta. Generation is O(N) per step —
//! there is no K/V cache; accuracy comes from the injection residual.
//!
//! Memory: ~2.8 MB for 176 windows × 3,585 entries on the Apollo 11 corpus,
//! vs ~25.8 GB Standard KV at 370K tokens (~20,000× compression).
//!
//! Simplifications vs the full Python pipeline:
//! - Injection is at the last token position only (Python does per-entry
//!   `position_in_window`).
//! - Routing uses tf-idf-lite on raw token IDs (no stemming/stopwords).
//! - Boundary-residual replay not yet wired (`prefill_to_layer` is a TODO).

use ndarray::{s, Array1, Array2};
use thiserror::Error;

use super::entry::{InjectionConfig, VecInjectEntry};
use super::routing::{RoutingBuildError, RoutingIndex, RoutingQuery};
use super::store::ApolloStore;
use crate::{EngineInfo, RetrievalEngine};
use larql_inference::forward::{embed_tokens_pub, forward_from_layer, forward_raw_logits};
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;

/// (context_tokens, injection_delta, boundary_residual, crystal_layer)
type InjectionPrep = (Vec<u32>, ndarray::Array1<f32>, Option<Vec<f32>>, usize);

// ─── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ApolloError {
    #[error("store not loaded")]
    StoreNotLoaded,
    #[error("routing index not built — call build_routing_index() first")]
    RoutingNotBuilt,
    #[error("invalid window id: {0}")]
    InvalidWindowId(u16),
    #[error("forward pass failed")]
    Forward,
    #[error("no windows matched query (routing returned empty)")]
    NoMatch,
    #[error(transparent)]
    Routing(#[from] RoutingBuildError),
}

// ─── Free helpers ────────────────────────────────────────────────────────────

/// Number of leading query tokens to drop when building the
/// `window_tokens ++ query_tokens` context. Exactly one token is dropped,
/// and only when the query starts with the *known* BOS id; `bos = None`
/// means the BOS id is unknown and nothing is dropped.
fn bos_skip_count(query_ids: &[u32], bos: Option<u32>) -> usize {
    match bos {
        Some(b) if query_ids.first() == Some(&b) => 1,
        _ => 0,
    }
}

/// Fail-closed injection-layer gate, mirroring the markov engines'
/// `check_residual_recompute_preconditions` placement at prefill entry —
/// the earliest point the engine sees both the model (`num_layers`) and
/// the store (`crystal_layer`).
///
/// `forward_layer_range` applies the injection delta only on the iteration
/// where `layer == injection_layer`, so a layer outside the executed range
/// is never reached and the retrieval injection silently no-ops — the worst
/// failure mode for an engine whose only contract is task-level accuracy:
///   - `injection_layer >= num_layers`: unreachable on every path (the
///     default `injection_layer=30` on a <=30-layer model trips this);
///   - compressed path (store has boundary residuals) runs only
///     `crystal_layer..num_layers`, so `injection_layer < crystal_layer`
///     is also unreachable (e.g. `apollo:layer=25` against a `crystal=30`
///     store).
pub(crate) fn check_injection_layer_preconditions(
    engine_name: &str,
    weights: &ModelWeights,
    config: &InjectionConfig,
    store: &ApolloStore,
) -> Result<(), EngineError> {
    if config.injection_layer >= weights.num_layers {
        return Err(EngineError::InvariantViolation {
            what: format!(
                "{engine_name}: injection_layer ({}) >= num_layers ({}) — the forward pass \
                 never reaches the perturbation layer, so retrieval injection would \
                 silently no-op; set injection_layer < num_layers",
                config.injection_layer, weights.num_layers,
            ),
        });
    }
    let crystal = store.manifest.crystal_layer;
    if !store.boundaries.is_empty() && config.injection_layer < crystal {
        return Err(EngineError::InvariantViolation {
            what: format!(
                "{engine_name}: injection_layer ({}) < crystal_layer ({crystal}) — the \
                 compressed forward runs only crystal_layer..num_layers, so retrieval \
                 injection would be silently skipped; set injection_layer >= crystal_layer",
                config.injection_layer,
            ),
        });
    }
    Ok(())
}

// ─── Trace types ─────────────────────────────────────────────────────────────

/// Summary of a single query answered by the engine.
#[derive(Debug, Clone)]
pub struct QueryTrace {
    pub routed_windows: Vec<u16>,
    pub injected_entries: Vec<VecInjectEntry>,
    pub context_tokens: usize,
    pub top1_token_id: u32,
    pub top1_logit: f32,
}

// ─── Engine struct ────────────────────────────────────────────────────────────

pub struct ApolloEngine {
    pub store: Option<ApolloStore>,
    pub routing: RoutingIndex,
    pub config: InjectionConfig,
    /// State maintained between prefill and decode steps.
    pub(super) context_tokens: Vec<u32>,
    pub(super) injection_delta: Option<Array1<f32>>,
    /// Boundary residual for the routed window (output of layer `crystal_layer - 1`).
    /// When `Some`, `prefill` and `decode_step` use `forward_from_layer` instead of
    /// running all 34 layers — ~8.5× faster on Gemma 3 4B (crystal_layer=30 → 4 layers).
    pub(super) boundary_residual: Option<Vec<f32>>,
    pub(super) crystal_layer: usize,
    /// Engine-owned f32 dequant scratch for the Q4K path — `prefill_quant`/
    /// `decode_step_quant` dequantise attn+FFN into it; `prefill`/`decode_step`
    /// resolve `forward_raw_logits`/`forward_from_layer` through a
    /// `WeightsView::with_scratch` over it. Empty on the dense path. Keeps
    /// `weights` immutable (no `weights.tensors` mutation).
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

impl ApolloEngine {
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            store: None,
            routing: RoutingIndex::new(),
            config,
            context_tokens: Vec::new(),
            injection_delta: None,
            boundary_residual: None,
            crystal_layer: 0,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_store(mut self, store: ApolloStore) -> Self {
        self.store = Some(store);
        self
    }

    pub fn build_routing_index(&mut self) -> Result<(), ApolloError> {
        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        self.routing = RoutingIndex::from_store(store)?;
        Ok(())
    }

    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }
    pub fn has_store(&self) -> bool {
        self.store.is_some()
    }
    pub fn store(&self) -> Option<&ApolloStore> {
        self.store.as_ref()
    }
    pub fn routing(&self) -> &RoutingIndex {
        &self.routing
    }

    /// Return the top-k entries most relevant to `query_token_ids`,
    /// scoped to `candidate_windows`. Uses seed + proximity + fact-group +
    /// backfill ranking.
    pub fn retrieve_entries(
        &self,
        query_token_ids: &[u32],
        candidate_windows: &[u16],
    ) -> Result<Vec<VecInjectEntry>, ApolloError> {
        const PROXIMITY_RADIUS: u16 = 10;
        let store = self.store.as_ref().ok_or(ApolloError::StoreNotLoaded)?;
        if query_token_ids.is_empty() {
            return Ok(vec![]);
        }
        let qset: std::collections::HashSet<u32> = query_token_ids.iter().copied().collect();
        let wset: std::collections::HashSet<u16> = candidate_windows.iter().copied().collect();
        let in_candidate = |e: &VecInjectEntry| wset.is_empty() || wset.contains(&e.window_id);
        let entry_key =
            |e: &VecInjectEntry| (e.window_id, e.position_in_window, e.token_id, e.fact_id);

        let seeds: Vec<&VecInjectEntry> = store
            .entries
            .iter()
            .filter(|e| in_candidate(e) && qset.contains(&e.token_id))
            .collect();

        if seeds.is_empty() {
            let mut scored: Vec<(VecInjectEntry, f32)> = store
                .entries
                .iter()
                .filter(|e| in_candidate(e))
                .map(|e| (*e, e.coefficient))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(self.config.top_k);
            return Ok(scored.into_iter().map(|(e, _)| e).collect());
        }

        let seed_facts: std::collections::HashSet<u16> = seeds.iter().map(|e| e.fact_id).collect();
        let seed_positions: std::collections::HashSet<(u16, u16)> = seeds
            .iter()
            .map(|e| (e.window_id, e.position_in_window))
            .collect();

        let mut scored: Vec<(VecInjectEntry, f32)> = Vec::new();
        let mut seen: std::collections::HashSet<(u16, u16, u32, u16)> =
            std::collections::HashSet::new();

        for e in &seeds {
            scored.push((**e, e.coefficient));
            seen.insert(entry_key(e));
        }
        for e in store.entries.iter().filter(|e| in_candidate(e)) {
            let k = entry_key(e);
            if seen.contains(&k) {
                continue;
            }
            let near = seed_positions.iter().any(|(w, p)| {
                *w == e.window_id
                    && (e.position_in_window as i32 - *p as i32).abs() <= PROXIMITY_RADIUS as i32
            });
            if near {
                scored.push((*e, e.coefficient * 1.3));
                seen.insert(k);
            }
        }
        for e in store
            .entries
            .iter()
            .filter(|e| in_candidate(e) && seed_facts.contains(&e.fact_id))
        {
            let k = entry_key(e);
            if !seen.contains(&k) {
                scored.push((*e, e.coefficient * 1.3));
                seen.insert(k);
            }
        }
        if scored.len() < self.config.top_k {
            let mut pool: Vec<&VecInjectEntry> = store
                .entries
                .iter()
                .filter(|e| in_candidate(e) && !seen.contains(&entry_key(e)))
                .collect();
            pool.sort_by(|a, b| {
                b.coefficient
                    .partial_cmp(&a.coefficient)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for e in pool.into_iter().take(self.config.top_k - scored.len()) {
                scored.push((*e, e.coefficient * 0.8));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);
        Ok(scored.into_iter().map(|(e, _)| e).collect())
    }

    /// Build the injection delta, context, and optional boundary residual
    /// for a set of query tokens.
    /// Returns `(context_tokens, injection_delta, boundary_residual, crystal_layer)`.
    pub(super) fn prepare_injection(
        &self,
        weights: &ModelWeights,
        query_ids: &[u32],
    ) -> Option<InjectionPrep> {
        let store = self.store.as_ref()?;
        let q = RoutingQuery {
            token_ids: query_ids.to_vec(),
        };
        let routed = self.routing.resolve(&q, 3);
        let top_window = *routed.first()?;

        let entries = self.retrieve_entries(query_ids, &[top_window]).ok()?;
        let window_tokens = store.window_tokens.get(top_window as usize)?;

        // Context = window_tokens ++ query_tokens. A BOS at the front of the
        // query would land mid-context, so it is dropped — but only when the
        // BOS id is actually known: the explicit engine config wins, else the
        // architecture's structural hook (populated for models whose
        // tokenizer omits BOS and larql prepends it, e.g. Gemma 4). Neither
        // known → nothing is stripped; there is no hardcoded model default.
        let bos = self
            .config
            .bos_token_id
            .or_else(|| weights.arch.bos_token_id());
        let mut context: Vec<u32> = window_tokens.clone();
        context.extend_from_slice(&query_ids[bos_skip_count(query_ids, bos)..]);

        // Injection delta: sum of answer-side entry embeddings.
        let hidden = weights.hidden_size;
        let mut delta = vec![0.0f32; hidden];
        let qset: std::collections::HashSet<u32> = query_ids.iter().copied().collect();
        for e in &entries {
            if qset.contains(&e.token_id) {
                continue;
            }
            let emb = embed_tokens_pub(weights, &[e.token_id]);
            let scale = e.coefficient * self.config.inject_coefficient;
            for (i, v) in emb.row(0).iter().enumerate() {
                delta[i] += v * scale;
            }
        }

        // Boundary residual: if the store has one for this window, the compressed
        // path can skip layers 0..crystal_layer entirely. Injection-layer
        // reachability (vs both crystal_layer and num_layers) is enforced
        // fail-closed by `check_injection_layer_preconditions` at prefill.
        let boundary = store.boundaries.get(top_window as usize).cloned();
        let crystal = store.manifest.crystal_layer;

        Some((context, Array1::from(delta), boundary, crystal))
    }

    /// One-shot query: route → retrieve → inject → forward. Uses the compressed
    /// path (boundary + 4 layers) when the store has boundary residuals.
    pub fn query_greedy(&self, weights: &ModelWeights, query_ids: &[u32]) -> Option<QueryTrace> {
        let (context, delta, boundary, crystal) = self.prepare_injection(weights, query_ids)?;
        let perturb = Some((self.config.injection_layer, delta.view()));
        let raw = if let Some(ref bnd) = boundary {
            // Compressed: skip layers 0..crystal, run only crystal..34 (~4 layers)
            forward_from_layer(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                query_ids,
                bnd,
                crystal,
                perturb,
            )
        } else {
            forward_raw_logits(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &context,
                perturb,
            )
        };
        let (top1_id, top1_logit) = raw
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i as u32, v))?;
        let q = RoutingQuery {
            token_ids: query_ids.to_vec(),
        };
        let routed = self.routing.resolve(&q, 3);
        let entries = self
            .retrieve_entries(query_ids, routed.get(..1).unwrap_or(&[]))
            .unwrap_or_default();
        Some(QueryTrace {
            routed_windows: routed,
            injected_entries: entries,
            context_tokens: context.len(),
            top1_token_id: top1_id,
            top1_logit,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

// ─── RetrievalEngine impl ─────────────────────────────────────────────────────
//
// Apollo implements [`RetrievalEngine`] rather than `KvEngine`. Its
// per-step contract differs: no per-token K/V append (state is the
// retrieval `injection_delta` + `boundary_residual` + token list), no
// `FfnBackend` dispatch (forward goes through `forward_from_layer` /
// `forward_raw_logits` directly), and its `None` returns map to
// `RetrievalMiss` rather than the per-layer backend failures of
// `KvEngine`.
//
// That missing dispatch is why both entry points refuse a hybrid-MoE
// architecture outright (see `APOLLO_HAS_NO_FFN_SEAM`) rather than serving it
// a dense-only forward that would answer for a different model. Construction sites build it as
// [`larql_inference::AnyEngine::Retrieval`] so the harness branches
// once at the top of the autoregressive loop.

/// Why Apollo cannot dispatch experts, for the refusal it owes a MoE model.
///
/// Structural, not an omission: `forward_from_layer` / `forward_raw_logits`
/// live in `larql-compute` *below* the `FfnBackend` seam and construct their
/// own `ViewFfn` over the dense weights, so no caller-supplied backend — and
/// therefore no bound expert route — can reach them. Giving Apollo real
/// dispatch means threading an `FfnBackend` through `forward_layer_range`,
/// which is a change to the forward, not to this engine.
const APOLLO_ENGINE_NAME: &str = "apollo";

const APOLLO_HAS_NO_FFN_SEAM: &str =
    "its forward runs below the FfnBackend seam and builds its own dense FFN";

impl RetrievalEngine for ApolloEngine {
    fn name(&self) -> &str {
        APOLLO_ENGINE_NAME
    }

    fn info(&self) -> EngineInfo {
        let windows = self.store.as_ref().map_or(0, |s| s.window_tokens.len());
        let entries = self.store.as_ref().map_or(0, |s| s.entries.len());
        let store_kb = self.store.as_ref().map_or(0, |s| s.total_bytes()) / 1024;
        let crystal = self.store.as_ref().map_or(0, |s| s.manifest.crystal_layer);
        let has_boundaries = self
            .store
            .as_ref()
            .is_some_and(|s| !s.boundaries.is_empty());
        let path = if has_boundaries {
            format!("compressed(layer={crystal})")
        } else {
            "uncompressed".into()
        };
        EngineInfo {
            name: "apollo".into(),
            description: format!(
                "retrieval+injection [{path}]: {windows} windows, {entries} entries, {store_kb}KB",
            ),
            backend: "cpu".into(),
            config: format!(
                "inject_layer={}, coef={}, top_k={}",
                self.config.injection_layer, self.config.inject_coefficient, self.config.top_k,
            ),
        }
    }

    /// Prefill routes token_ids, retrieves entries, builds the injection delta,
    /// and runs the forward pass.
    ///
    /// **Compressed path** (when store has boundary residuals): runs only
    /// `crystal_layer..num_layers` (~4 layers for Gemma 3 4B), ~8.5× faster.
    ///
    /// **Uncompressed path** (no boundaries): full forward over window+query tokens.
    fn prefill(
        &mut self,
        weights: &ModelWeights,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        crate::engines::refuse_if_moe(APOLLO_ENGINE_NAME, APOLLO_HAS_NO_FFN_SEAM, weights)?;
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| EngineError::RetrievalMiss {
                reason: "apollo store not attached".into(),
            })?;
        // Fail closed on unreachable injection layers before any forward work.
        check_injection_layer_preconditions("apollo", weights, &self.config, store)?;
        if self.routing.is_empty() {
            self.routing =
                RoutingIndex::from_store(store).map_err(|e| EngineError::InvariantViolation {
                    what: format!("apollo: {e}"),
                })?;
        }

        let (context, delta, boundary, crystal) = self
            .prepare_injection(weights, token_ids)
            .ok_or_else(|| EngineError::RetrievalMiss {
                reason: "prepare_injection returned None (no candidates)".into(),
            })?;
        let perturb = Some((self.config.injection_layer, delta.view()));

        let raw = if let Some(ref bnd) = boundary {
            // Compressed: boundary residual acts as position-0; skip layers 0..crystal.
            forward_from_layer(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                token_ids,
                bnd,
                crystal,
                perturb,
            )
        } else {
            forward_raw_logits(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &context,
                perturb,
            )
        };

        // Cache decode state.
        self.context_tokens = if boundary.is_some() {
            token_ids.to_vec() // compressed: just the query
        } else {
            context
        };
        self.injection_delta = Some(delta);
        self.boundary_residual = boundary;
        self.crystal_layer = crystal;

        let last = raw.h_pre_norm.shape()[0] - 1;
        Ok(raw.h_pre_norm.slice(s![last..=last, ..]).to_owned())
    }

    /// Extend by one token. Uses the boundary compressed path when available
    /// (4 layers), otherwise full 34-layer re-forward.
    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        crate::engines::refuse_if_moe(APOLLO_ENGINE_NAME, APOLLO_HAS_NO_FFN_SEAM, weights)?;
        self.context_tokens.push(token_id);
        let delta =
            self.injection_delta
                .as_ref()
                .ok_or_else(|| EngineError::InvariantViolation {
                    what: "decode_step called before prefill (injection_delta missing)".into(),
                })?;
        let perturb = Some((self.config.injection_layer, delta.view()));

        let raw = if let Some(ref bnd) = self.boundary_residual {
            // Compressed: re-run only crystal_layer..num_layers over growing query.
            forward_from_layer(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &self.context_tokens,
                bnd,
                self.crystal_layer,
                perturb,
            )
        } else {
            forward_raw_logits(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &self.context_tokens,
                perturb,
            )
        };

        let last = raw.h_pre_norm.shape()[0] - 1;
        Ok(raw.h_pre_norm.slice(s![last..=last, ..]).to_owned())
    }

    /// Apollo's quant path dequants BOTH attention and FFN tensors —
    /// the forward goes through `forward_from_layer` / `forward_raw_logits`
    /// which expect both. The trait default only dequants attn, so we
    /// override.
    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        for layer in 0..weights.num_layers {
            let _ = larql_inference::vindex::insert_q4k_layer_tensors(
                &mut self.dequant_scratch,
                weights,
                index,
                layer,
            );
        }
        self.prefill(weights, token_ids)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        for layer in 0..weights.num_layers {
            let _ = larql_inference::vindex::insert_q4k_layer_tensors(
                &mut self.dequant_scratch,
                weights,
                index,
                layer,
            );
        }
        self.decode_step(weights, token_id)
    }

    fn memory_bytes(&self) -> usize {
        self.store.as_ref().map_or(0, |s| s.total_bytes())
    }
}

#[cfg(test)]
mod tests;
