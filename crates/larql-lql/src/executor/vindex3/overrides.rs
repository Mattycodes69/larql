//! Override composition, residual capture and continuation for VINDEX3.

use crate::error::LqlError;
use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    ContinuationFactory, SelectedContinuation,
};
use larql_vindex::tokenizers::Tokenizer;

#[allow(unused_imports)]
use super::*;

/// One observed pass over the runtime's plan, returning the last
/// position's **normed FFN input** at `layer` — V2's exact install
/// statistic (its walk-FFN trace captures the post-norm vector the
/// gates multiply), so a gate built from it fires on the prompt that
/// produced it (V3-LQL-3B capture: the KNN key and the compose gate
/// direction). The pass is the same canonical traversal INFER runs;
/// the tap is a subscription, never a second executor.
/// The overlay's compose edits as executor operand overrides — `None`
/// while no compose state exists, so callers keep the plain (bit-for-
/// bit identical) execution path.
pub(crate) fn compose_overrides(
    runtime: &V3Runtime,
    overlay: &larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
) -> Result<Option<larql_inference::vindex3::OperandOverrides>, LqlError> {
    if !overlay.has_vector_state() {
        return Ok(None);
    }
    overlay
        .operand_overrides(runtime.plan())
        .map(Some)
        .map_err(|e| LqlError::exec("failed to derive operand overrides", e))
}

pub(crate) fn capture_layer_residual(
    runtime: &V3Runtime,
    tokenizer: &Tokenizer,
    prompt: &str,
    layer: usize,
    overrides: Option<&larql_inference::vindex3::OperandOverrides>,
    bos: Option<u32>,
) -> Result<Vec<f32>, LqlError> {
    let prompt_ids = encode_v3_prompt(tokenizer, prompt, bos)?;
    let mut captured: Option<Vec<f32>> = None;
    let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
        if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event {
            if index == layer {
                captured = trace.ffn_input.last().cloned();
            }
        }
        Ok(())
    };
    // The capture runs over the same effective program INFER runs —
    // V2's contract (its capture forward observes the patch overlay).
    let continuation = v3_continuation(runtime)?;
    match overrides {
        Some(overrides) => {
            runtime.execute_streaming_overlaid(&prompt_ids, overrides, &continuation, &mut sink)
        }
        None => runtime.execute_streaming(&prompt_ids, &continuation, &mut sink),
    }
    .map_err(|e| LqlError::exec("v3 capture pass failed", e))?;
    captured.ok_or_else(|| LqlError::Execution(format!("no residual captured at layer {layer}")))
}

/// LQL's continuation provider — the canonical cache, named by its
/// factory's identity (CONTINUATION-PLUGIN-1, C3) and selected against the
/// runtime's program before anything runs.
pub(crate) fn v3_continuation(runtime: &V3Runtime) -> Result<SelectedContinuation, LqlError> {
    runtime
        .select_continuation(
            &shipped_continuations(),
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
        )
        .map_err(|e| LqlError::exec("v3 continuation selection failed", e))
}

/// HF configs a container carries that declare the BOS token id, most
/// authoritative first. Both arrive with the M2 capability snapshot.
pub(super) const BOS_DECLARING_FILES: [&str; 2] =
    ["generation_config.json", "tokenizer_config.json"];

/// The BOS token id this container **declares**, or `None` when it
/// declares none.
///
/// The V2 surface gets this fact from a reconstructed
/// `ModelArchitecture` (`arch.bos_token_id()`); V3 must not rediscover
/// architecture, so it reads the checkpoint's own declaration, carried
/// into the container by the capability snapshot. For Gemma 4 — the
/// architecture that needs BOS *and* omits it from the tokenizer's
/// `TemplateProcessing.single` template — both sources say 2, which is
/// what `v2_and_v3_prompt_encoders_agree_on_a_bos_requiring_model`
/// pins.
///
/// A container predating the capability snapshot carries neither file
/// and declares nothing; it encodes exactly as it did before.
pub(crate) fn declared_bos_token(container: &std::path::Path) -> Option<u32> {
    for name in BOS_DECLARING_FILES {
        let Ok(text) = std::fs::read_to_string(container.join(name)) else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if let Some(id) = json.get("bos_token_id").and_then(|v| v.as_u64()) {
            return u32::try_from(id).ok();
        }
    }
    None
}

/// Encode a prompt for the V3 program, prepending the container's
/// declared BOS when the tokenizer's own post-processor did not.
///
/// Shares `maybe_prepend_bos` with the V2 path (`encode_prompt`), so
/// the "prepend only if missing" contract has one implementation.
pub(crate) fn encode_v3_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    bos: Option<u32>,
) -> Result<Vec<u32>, LqlError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| LqlError::Execution(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err(LqlError::Execution("prompt tokenises to empty".into()));
    }
    Ok(larql_inference::maybe_prepend_bos(ids, bos))
}

/// Softmax over the logits, then the top-k `(token_id, probability)`
/// pairs, ties keeping the lower id (the greedy sampler's rule).
pub(crate) fn top_k_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut scored: Vec<(u32, f32)> = exps
        .iter()
        .enumerate()
        .map(|(id, &e)| (id as u32, e / sum))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
