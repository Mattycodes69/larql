//! Reusable prefill + decode helpers that orchestrate the per-layer
//! loop via [`KvDispatch`] primitives.
//!
//! These are the engine-facing equivalents of
//! [`crate::forward::kv_prefill_run`] and
//! [`crate::forward::kv_decode_step_run`], rewritten to call
//! `backend.attention_prefill` / `backend.attention_step` per layer
//! instead of the direct `run_attention_*` functions.
//!
//! **Parity:** the helpers below produce bit-identical output to the
//! legacy `kv_prefill_run` / `kv_decode_step_run` when driven against
//! [`super::cpu::CpuKvHandle`] (verified in this file's
//! tests). Engines migrate from the legacy helpers to these helpers
//! in Step 3c of the ComputeBackend redesign.
//!
//! **Three outcomes, never two.** Every helper here returns
//! `Result<Option<T>, BoxRefusal>`, the same split
//! [`FfnBackend::forward_moe_full_layer`](crate::ffn::FfnBackend::forward_moe_full_layer)
//! makes one ring down:
//!
//! ```text
//! Ok(Some(_))   the dispatch produced a complete result
//! Ok(None)      nothing to do, or the backend declined this shape
//! Err(refusal)  a routed operation was required and did not execute
//! ```
//!
//! `Ok(None)` carries exactly what a bare `None` used to: the engine
//! turns it into `EngineError::BackendFailure` and may try another
//! path. `Err` is the channel that did not exist before — an engine
//! receiving it knows the layer is incomplete, so a strict route can
//! refuse the token instead of returning the dense half of a layer
//! whose experts never ran.
//!
//! Hooks are not threaded through these helpers — the existing
//! hooked decode path
//! ([`crate::forward::generate_cached_hooked`]) keeps using the legacy
//! helpers because the trait surface doesn't carry `LayerHook`.
//! That's by design (`compute-backend-redesign.md` §4.2 non-goals).

use larql_execution::BoxRefusal;
use ndarray::Array2;

use super::{EngineBackend, KvHandle};
use crate::async_compute_backend::AsyncComputeBackend;
use crate::ffn::FfnBackend;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::{embed_tokens_pub, run_ffn};

/// What every helper in this module returns: the three outcomes, named once.
///
/// `Ok(Some(t))` executed, `Ok(None)` not applicable, `Err(_)` refused. Spelled
/// as an alias rather than repeated six times so the contract has somewhere to
/// be documented and cannot drift between the sync and async twins.
pub type DispatchOutcome<T> = Result<Option<T>, BoxRefusal>;

/// A completed prefill: the last row of the post-FFN hidden state, plus one
/// K/V handle per layer, in layer order.
pub type PrefilledCache = (Array2<f32>, Vec<KvHandle>);

/// Per-layer FFN + PLE + layer_scalar dispatch for the KV-cached engine
/// path, MoE-aware.
///
/// On hybrid-MoE architectures, try the backend's
/// [`FfnBackend::forward_moe_full_layer`] hook first — it returns the full
/// layer output (dense `h1` + experts `h2` + combine + outer-norm + PLE +
/// layer_scalar; see `moe_ffn_block_cpu`, which applies the last two
/// internally). A remote-MoE backend ([`crate::ffn::RemoteMoeFfn`])
/// implements it to dispatch experts to the shards, giving CPU
/// `--moe-shards` a real KV cache. When the hook declines (`None`) — or
/// the model is dense — fall back to the standard dense FFN followed by
/// `apply_per_layer_embedding` + `apply_layer_scalar`, mirroring the
/// legacy `kv_prefill_run` / `kv_decode_step_run` per-layer sequence
/// exactly (the issue-#98 fix; both are no-ops on non-Gemma-4 archs).
///
/// A refusal propagates. This is what made the strict policy real: the
/// hook's three outcomes stay three all the way out to the engine, so a
/// route that declined a required expert cannot be answered with the
/// dense half of its own layer.
fn ffn_or_moe_layer(
    weights: larql_models::WeightsView,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    ple_input: Option<&Array2<f32>>,
) -> Result<Array2<f32>, BoxRefusal> {
    let phase = larql_compute::phase_timing::start();
    // Pure MoE (GPT-OSS, GraniteMoE, OLMoE) takes this hook exactly as
    // hybrid does — its FFN *is* the expert block. Gating on hybrid alone
    // sent pure-MoE layers to the dense `run_ffn` below, which asks for
    // gate/up/down tensors those checkpoints do not have.
    if weights.arch.is_moe() || weights.arch.is_hybrid_moe() {
        // `?` propagates; `None` falls through. Not applicable means this
        // backend does not serve the layer and the local dispatch below is the
        // correct answer — never a refusal wearing its shape.
        if let Some(h_out) = ffn.forward_moe_full_layer(layer, h_post_attn)? {
            larql_compute::phase_timing::finish(phase, "layer.ffn_block");
            return Ok(h_out);
        }
    }
    let (h_post_ffn, _) = run_ffn(&weights, h_post_attn, layer, ffn, false);
    let mut h_out = apply_per_layer_embedding(&weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(&weights, &mut h_out, layer);
    larql_compute::phase_timing::finish(phase, "layer.ffn_block");
    Ok(h_out)
}

/// Prefill the K/V cache through every layer using `backend`'s
/// [`KvDispatch::attention_prefill`] intent. Returns the last row of
/// the post-FFN hidden state plus per-layer K/V handles.
///
/// `window` is passed through to the backend per layer — backends with
/// windowed-attention shader variants may use it; CPU backends ignore
/// it (the cache simply isn't clipped after prefill on this path —
/// callers that want a clipped prefill should call
/// [`KvDispatch::clip_kv`] per-layer after this returns).
///
/// Three outcomes, per this module's contract: `Ok(Some(_))` prefilled,
/// `Ok(None)` nothing to prefill or the backend declined, `Err(_)` a
/// required routed operation refused.
pub fn kv_prefill_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> DispatchOutcome<PrefilledCache> {
    if prompt_ids.is_empty() {
        return Ok(None);
    }
    let h = embed_tokens_pub(&weights, prompt_ids);
    kv_prefill_from_hidden_via_dispatch(
        backend,
        weights,
        ffn,
        &h,
        Some(prompt_ids),
        window,
        index.map(|v| v as &dyn larql_compute::KvIndex),
    )
}

/// Multi-modal-aware peer of [`kv_prefill_via_dispatch`]. Takes
/// pre-built initial hidden state (e.g. from
/// `larql_compute::forward::embed_plan` on an `EmbeddingPlan` that mixes
/// `Tokens` and `Precomputed` chunks) and drives the rest of prefill
/// unchanged.
///
/// `token_ids` feeds `precompute_per_layer_inputs` — PLE architectures
/// (Gemma 4 E-series) need one token identity per hidden row. Pass
/// `Some` whenever the rows are pure token embeds; `None` (the MM path,
/// where precomputed vision/audio rows have no token ids) skips PLE, so
/// PLE archs must prefill via the token entry point.
///
/// The text-only call `kv_prefill_via_dispatch(prompt_ids)` and
/// `kv_prefill_from_hidden_via_dispatch(embed_tokens_pub(prompt_ids),
/// Some(prompt_ids))` produce bit-identical output by construction — the
/// former is a thin wrapper around the latter. Pinned by tests at the
/// bottom of this module.
pub fn kv_prefill_from_hidden_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    token_ids: Option<&[u32]>,
    window: Option<usize>,
    index: Option<&dyn larql_compute::KvIndex>,
) -> DispatchOutcome<PrefilledCache> {
    if initial_hidden.nrows() == 0 {
        return Ok(None);
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = initial_hidden.clone();
    // Empty for non-PLE archs (and for `token_ids: None`) — then
    // `ple_inputs.get(layer)` yields `None` and PLE is a no-op.
    let ple_inputs = match token_ids {
        Some(ids) => precompute_per_layer_inputs(&weights, initial_hidden, ids),
        None => Vec::new(),
    };

    for layer in 0..num_layers {
        let _t_attn = std::time::Instant::now();
        // A declining backend is not a refusal — it is this dispatch having no
        // answer, which is what `Ok(None)` has always meant to the engine.
        let Some((h_post_attn, mut handle)) =
            backend.attention_prefill(weights, &h, layer, window, index)
        else {
            return Ok(None);
        };
        crate::decode_stages::record_attn(_t_attn.elapsed().as_nanos());
        if let Some(w) = window {
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        h = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn, ple_inputs.get(layer))?;
    }

    Ok(Some((last_row_as_2d(&h), handles)))
}

/// Run one autoregressive decode step using `backend`'s
/// [`KvDispatch::attention_step`] intent per layer.
///
/// `handles` must contain one [`KvHandle`] per layer in `weights`. The
/// caller is responsible for tracking `abs_position` (the absolute
/// token index of the new token — usually `prompt_len + step_idx`).
///
/// `window` is forwarded to the backend's clip step per layer when
/// `Some`. Returns the post-FFN hidden state for the new token
/// (shape `[1, hidden]`).
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    token_id: u32,
    abs_position: usize,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> DispatchOutcome<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_via_dispatch: handles.len() must equal weights.num_layers"
    );
    let h_new = embed_tokens_pub(&weights, &[token_id]);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[token_id]);
    let mut h_step = h_new;

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let _t_attn = std::time::Instant::now();
        let Some(h_post_attn) = backend.attention_step(
            weights,
            &h_step,
            handle,
            layer,
            abs_position,
            index.map(|v| v as &dyn larql_compute::KvIndex),
        ) else {
            return Ok(None);
        };
        crate::decode_stages::record_attn(_t_attn.elapsed().as_nanos());
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        h_step = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn, ple_inputs.get(layer))?;
    }

    Ok(Some(h_step))
}

/// Run one autoregressive decode step whose input is a pre-built hidden
/// row rather than a token id — the decode-time peer of
/// [`kv_prefill_from_hidden_via_dispatch`], closing the seam ADR-0023
/// deferred ("decode is text-out by definition" stopped being true with
/// MOSS-TTS-Realtime, whose step input is a 17-table summed embedding).
///
/// Identical to [`kv_decode_step_via_dispatch`] except the embedding
/// lookup is the caller's, and there are no PLE inputs: per-layer
/// embeddings are token-keyed, so an architecture with
/// `has_per_layer_embeddings` cannot take this path (callers must not
/// route PLE models here).
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_from_hidden_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    hidden_row: &Array2<f32>,
    abs_position: usize,
    window: Option<usize>,
    index: Option<&dyn larql_compute::KvIndex>,
) -> DispatchOutcome<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_from_hidden_via_dispatch: handles.len() must equal weights.num_layers"
    );
    debug_assert_eq!(
        hidden_row.nrows(),
        1,
        "kv_decode_step_from_hidden_via_dispatch: expects exactly one new row"
    );
    let mut h_step = hidden_row.clone();

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let _t_attn = std::time::Instant::now();
        let Some(h_post_attn) =
            backend.attention_step(weights, &h_step, handle, layer, abs_position, index)
        else {
            return Ok(None);
        };
        crate::decode_stages::record_attn(_t_attn.elapsed().as_nanos());
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        h_step = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn, None)?;
    }

    Ok(Some(h_step))
}

// ── Async variants ──────────────────────────────────────────────────
//
// Mirror the sync helpers above but drive the per-layer loop through
// [`AsyncComputeBackend`]. Per `async-compute-backend.md` §11.5 v1: FFN
// stays on host, so the loop reads the post-attention `AttentionHandle`
// per layer before running FFN. The win at A4 (deferred dispatch) comes
// from K/V appends fusing into the *next* layer's attention command
// buffer — `read_hidden` only forces commit on the hidden, not on the
// cache write. v2 (Step A6+) adds `ffn_step_async` for full
// one-commit-per-decode-step shape.
//
// Engines opting in via `with_async_backend` route through these.

/// Async equivalent of [`kv_prefill_via_dispatch`].
///
/// Calls `backend.attention_prefill_async` per layer, reads the hidden
/// to drive FFN on host, then proceeds. Calls `backend.flush()` once at
/// the end so any deferred work clears before returning.
pub fn kv_prefill_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> DispatchOutcome<PrefilledCache> {
    if prompt_ids.is_empty() {
        return Ok(None);
    }
    let h = embed_tokens_pub(&weights, prompt_ids);
    kv_prefill_from_hidden_via_dispatch_async(
        backend,
        weights,
        ffn,
        &h,
        Some(prompt_ids),
        window,
        index.map(|v| v as &dyn larql_compute::KvIndex),
    )
}

/// Async multi-modal-aware peer of [`kv_prefill_via_dispatch_async`].
/// Same shape as [`kv_prefill_from_hidden_via_dispatch`] (including the
/// `token_ids` PLE contract) but routes per-layer attention through
/// `AsyncComputeBackend` and reads each hidden via `read_hidden` before
/// FFN. Flushes once at the end.
///
/// Bit-identity contract: same as the sync peer. Pinned by the parity
/// test at the bottom of this module — sync vs async must agree on
/// CPU paths, MM vs text must agree when text is the input.
pub fn kv_prefill_from_hidden_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    token_ids: Option<&[u32]>,
    window: Option<usize>,
    index: Option<&dyn larql_compute::KvIndex>,
) -> DispatchOutcome<PrefilledCache> {
    if initial_hidden.nrows() == 0 {
        return Ok(None);
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = initial_hidden.clone();
    // Empty for non-PLE archs (and for `token_ids: None`) — then
    // `ple_inputs.get(layer)` yields `None` and PLE is a no-op.
    let ple_inputs = match token_ids {
        Some(ids) => precompute_per_layer_inputs(&weights, initial_hidden, ids),
        None => Vec::new(),
    };

    for layer in 0..num_layers {
        let (h_post_attn_handle, mut handle) =
            backend.attention_prefill_async(weights, &h, layer, window, index);
        if let Some(w) = window {
            // Sync clip — backends with deferred dispatch must flush
            // before clip per spec §11.3.
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        h = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn, ple_inputs.get(layer))?;
    }

    if backend.flush().is_err() {
        return Ok(None);
    }
    Ok(Some((last_row_as_2d(&h), handles)))
}

/// Async equivalent of [`kv_decode_step_via_dispatch`].
///
/// One decode step. Reads the per-layer hidden for FFN dispatch (v1
/// pattern). Flushes at the end of the step so the next call starts
/// from a quiescent backend.
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    token_id: u32,
    abs_position: usize,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> DispatchOutcome<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_via_dispatch_async: handles.len() must equal weights.num_layers"
    );
    let h_new = embed_tokens_pub(&weights, &[token_id]);
    // PLE inputs are per-token — recompute for this single-token decode
    // step, matching the legacy `kv_decode_step_run` recipe exactly.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[token_id]);
    let mut h_step = h_new;

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let h_post_attn_handle = backend.attention_step_async(
            weights,
            &h_step,
            handle,
            layer,
            abs_position,
            index.map(|v| v as &dyn larql_compute::KvIndex),
        );
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        h_step = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn, ple_inputs.get(layer))?;
    }

    if backend.flush().is_err() {
        return Ok(None);
    }
    Ok(Some(h_step))
}

/// Async mirror of [`kv_decode_step_from_hidden_via_dispatch`] — same
/// contract, driven through [`AsyncComputeBackend`]. No PLE inputs, as
/// with the sync variant.
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_from_hidden_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    hidden_row: &Array2<f32>,
    abs_position: usize,
    window: Option<usize>,
    index: Option<&dyn larql_compute::KvIndex>,
) -> DispatchOutcome<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_from_hidden_via_dispatch_async: handles.len() must equal weights.num_layers"
    );
    let mut h_step = hidden_row.clone();

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let h_post_attn_handle =
            backend.attention_step_async(weights, &h_step, handle, layer, abs_position, index);
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        h_step = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn, None)?;
    }

    if backend.flush().is_err() {
        return Ok(None);
    }
    Ok(Some(h_step))
}

fn last_row_as_2d(h: &Array2<f32>) -> Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

#[cfg(test)]
mod tests;
