//! CPU MoE expert dispatch: the server's wrapper around
//! [`larql_inference::ffn::expert_fold`].
//!
//! The fold itself — hoisted pre-experts norm, per-thread scratch, rayon
//! fold, checked packed-table slicing — lives in larql-inference so any
//! front-end can run a shard. This module resolves the served model's
//! weights, passes the server's env flags in as options, and maps a
//! missing model or weights to a `ServerError`.
//!
//! Both functions keep the fold's contract: they return
//! `(weighted_sum, experts_run)`, and callers MUST compare `experts_run`
//! against [`count_nonzero_weights`] and turn any shortfall into a loud
//! error.

use larql_compute::Q8KActivation;
use larql_inference::ffn::expert_fold::{
    fold_experts_cpu, fold_experts_q8k_prenormed, ExpertFoldOptions,
};

pub use larql_inference::ffn::expert_fold::count_nonzero_weights;

use crate::env_flags;
use crate::error::ServerError;
use crate::state::AppState;

/// CPU expert dispatch over the post-attention residual. See
/// [`fold_experts_cpu`].
pub fn run_experts_cpu_batch(
    state: &AppState,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<(Vec<f32>, usize), ServerError> {
    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let opts = ExpertFoldOptions {
        disable_q4k_direct: env_flags::disable_q4k_direct(),
        timing: env_flags::moe_timing_enabled(),
    };
    Ok(fold_experts_cpu(
        &weights,
        layer,
        h_post_attn,
        expert_ids,
        expert_weights,
        opts,
    ))
}

/// Expert dispatch with a client-side pre-normed, pre-quantised Q8K
/// activation. See [`fold_experts_q8k_prenormed`].
pub fn run_experts_cpu_batch_q8k_prenormed(
    state: &AppState,
    layer: usize,
    q8k: &Q8KActivation,
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> Result<(Vec<f32>, usize), ServerError> {
    let model = state.model_or_err(None)?;
    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    Ok(fold_experts_q8k_prenormed(
        &weights,
        layer,
        q8k,
        expert_ids,
        expert_weights,
    ))
}
