//! CPU expert fold.
//!
//! [`fold_experts_cpu`] hoists `pre_experts_norm` out of the per-expert
//! loop (rms_norm is invariant of expert id), quantises the activation to
//! Q8_K once when the per-layer Q4_K direct kernel is enabled, and folds K
//! expert outputs directly into a per-worker accumulator via rayon. Replaces
//! the historical `expert_ids.par_iter().filter_map(run_expert).collect()`
//! pattern that re-applied pre_norm K times and allocated three Vec<f32>
//! per matmul.
//!
//! **Contract.** Both folds return `(weighted_sum, experts_run)`: the
//! router-weighted sum across the active experts (length = hidden) plus the
//! count of experts that actually contributed. Zero-weight pairs are
//! legitimately skipped and count neither as requested nor as run; an
//! expert whose bytes can't be resolved (unowned or out-of-range
//! `(layer, expert_id)`) is silently absent from the sum, so callers MUST
//! compare `experts_run` against [`count_nonzero_weights`] and turn any
//! shortfall into a loud error — a partial sum is a silently corrupt number
//! (dec-readiness review, "the number is a lie" class). Post-experts norm
//! is the caller's: the fold stops one step short so the same numbers are
//! summable across shards.

use larql_compute::cpu::ops::moe::{
    pre_experts_norm, quantize_h_norm_for_q4k, run_single_expert_into,
    run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::{ExpertMlp, Q8KActivation, QuantFormat};
use rayon::prelude::*;

use super::packed::packed_bf16_expert;
use crate::ModelWeights;

/// Caller-chosen switches for [`fold_experts_cpu`]. A front-end reads its
/// own configuration (env flags, CLI arguments) and passes the answer in.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertFoldOptions {
    /// Keep Q4_K experts on the BLAS-on-cached-f32 path instead of the
    /// direct Q4_K × Q8_K kernel (kernel-debug A/B comparison).
    pub disable_q4k_direct: bool,
    /// Print a per-call stage breakdown to stderr.
    pub timing: bool,
}

thread_local! {
    // Per-rayon-thread scratch. 16 cores on M3 Max → up to 16 instances live
    // for the lifetime of the worker thread; replaces 3 fresh Vec<f32> heap
    // allocations per expert call.
    static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
        const { std::cell::RefCell::new(None) };
}

/// Fold the router-weighted sum of `expert_ids` at `layer` over the
/// post-attention residual `h_post_attn`. See the module docs for the
/// `(weighted_sum, experts_run)` contract.
pub fn fold_experts_cpu(
    weights: &ModelWeights,
    layer: usize,
    h_post_attn: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
    opts: ExpertFoldOptions,
) -> (Vec<f32>, usize) {
    use std::time::Instant;
    let t_start = Instant::now();

    let arch = &*weights.arch;
    let hidden = h_post_attn.len();
    if hidden == 0 || expert_ids.is_empty() {
        return (vec![0.0f32; hidden], 0);
    }
    let inter = arch.moe_intermediate_size();
    let activation = crate::activation_from_arch(arch);
    let per_layer_ffn = weights.has_per_layer_ffn();
    let inter_padded = if per_layer_ffn {
        q4k_padded(inter)
    } else {
        inter
    };
    let t_arch = t_start.elapsed();

    // Hoist pre_experts_norm: same input residual for all K experts; rms_norm
    // is invariant of the expert id, so doing it once per frame saves K-1
    // redundant passes per layer.
    let t_norm_start = Instant::now();
    let pre_norm_slice: &[f32] = arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|key| weights.vectors.get(&key))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let h_norm = pre_experts_norm(
        h_post_attn,
        pre_norm_slice,
        arch.norm_weight_offset(),
        arch.norm_eps(),
    );
    let t_norm = t_norm_start.elapsed();

    let format = if per_layer_ffn {
        QuantFormat::Q4_K
    } else {
        QuantFormat::BF16
    };

    // For Q4_K weights, quantise h_norm to Q8_K once per layer (shared
    // across all K active experts). Enables the SDOT-based direct-Q4K
    // matvec kernel — bypasses the f32 dequant cache entirely. Default-on
    // when format is Q4_K and the activation length is divisible by 256
    // (always true for production hidden sizes).
    let q4k_direct = matches!(format, QuantFormat::Q4_K) && !opts.disable_q4k_direct;
    let h_norm_q8k = if q4k_direct {
        quantize_h_norm_for_q4k(&h_norm)
    } else {
        None
    };

    // Resolve (gate_up, down) bytes for one expert. Pulled out of the
    // rayon closure so the closure body is small and the legacy BF16 path
    // doesn't fight the borrow checker on `weights` / `arch`.
    let resolve_bytes = |eid: usize| -> Option<(&[u8], &[u8])> {
        if per_layer_ffn {
            weights.get_layer_entry_bytes(layer, eid)
        } else {
            let gu_key = arch.packed_experts_gate_up_key(layer)?;
            let dn_key = arch.packed_experts_down_key(layer)?;
            let gu_all = weights.get_packed_bytes(&gu_key)?;
            let dn_all = weights.get_packed_bytes(&dn_key)?;
            packed_bf16_expert(gu_all, dn_all, eid, hidden, inter)
        }
    };

    let mlp = ExpertMlp::gated(activation);
    let t_fold_start = Instant::now();
    let out = weighted_fold(
        expert_ids,
        expert_weights,
        hidden,
        (inter, inter_padded),
        resolve_bytes,
        |scratch, gu_bytes, dn_bytes| match h_norm_q8k.as_ref() {
            Some(q8k) => {
                run_single_expert_q4k_q8k_into(scratch, q8k, gu_bytes, dn_bytes, inter, mlp)
            }
            None => {
                run_single_expert_into(scratch, &h_norm, gu_bytes, dn_bytes, inter, format, mlp)
            }
        },
    );

    if opts.timing {
        let t_par = t_fold_start.elapsed();
        eprintln!(
            "[run_experts_cpu] layer={layer} K={} arch={:.2}ms norm={:.2}ms \
             par_fold={:.2}ms total={:.2}ms",
            expert_ids.len(),
            t_arch.as_secs_f32() * 1000.0,
            t_norm.as_secs_f32() * 1000.0,
            t_par.as_secs_f32() * 1000.0,
            t_start.elapsed().as_secs_f32() * 1000.0,
        );
    }
    out
}

/// Fold with a pre-quantised Q8K activation — skips `pre_experts_norm` and
/// `quantize_h_norm_for_q4k` because the client already did both (4× less
/// upload traffic). Per-layer Q4_K experts only. Same contract as
/// [`fold_experts_cpu`].
pub fn fold_experts_q8k_prenormed(
    weights: &ModelWeights,
    layer: usize,
    q8k: &Q8KActivation,
    expert_ids: &[usize],
    expert_weights: &[f32],
) -> (Vec<f32>, usize) {
    let hidden = q8k.qs.len();
    // An empty activation has nothing to run an expert over. (An empty
    // request needs no guard: the fold over zero ids is already zero.)
    if hidden == 0 {
        return (Vec::new(), 0);
    }
    let arch = &*weights.arch;
    let inter = arch.moe_intermediate_size();
    let mlp = ExpertMlp::gated(crate::activation_from_arch(arch));
    weighted_fold(
        expert_ids,
        expert_weights,
        hidden,
        (inter, q4k_padded(inter)),
        |eid| weights.get_layer_entry_bytes(layer, eid),
        |scratch, gu_bytes, dn_bytes| {
            run_single_expert_q4k_q8k_into(scratch, q8k, gu_bytes, dn_bytes, inter, mlp)
        },
    )
}

/// Number of experts a request actually asked to run: zero-weight pairs are
/// a legitimate no-op (the folds filter them before resolving bytes) and
/// must not be counted when comparing against `experts_run`.
pub fn count_nonzero_weights(expert_weights: &[f32]) -> usize {
    expert_weights.iter().filter(|&&w| w != 0.0).count()
}

/// `inter` rounded up to a whole number of Q4_K blocks.
fn q4k_padded(inter: usize) -> usize {
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    inter.div_ceil(block) * block
}

/// The shared fold: run each non-zero-weight expert whose bytes resolve into
/// a per-worker hidden-sized accumulator, then reduce across workers.
///
/// Replaces collecting K `(Vec<f32>, weight)` partials and summing them
/// serially — that forced an 11 KB allocation per expert per layer
/// (≈2.7 MB/token at 30 MoE layers × top-K=8) and serialised the final
/// accumulation on one thread. Each accumulator also counts the experts it
/// ran, so an unresolvable expert surfaces as `experts_run < requested` at
/// the caller instead of vanishing into a partial sum.
fn weighted_fold<'w, R, E>(
    expert_ids: &[usize],
    expert_weights: &[f32],
    hidden: usize,
    (inter, inter_padded): (usize, usize),
    resolve_bytes: R,
    run_expert: E,
) -> (Vec<f32>, usize)
where
    R: Fn(usize) -> Option<(&'w [u8], &'w [u8])> + Sync,
    E: for<'s> Fn(&'s mut ExpertScratch, &[u8], &[u8]) -> &'s [f32] + Sync,
{
    expert_ids
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || (vec![0.0f32; hidden], 0usize),
            |(mut acc, n_run), (&eid, &w)| {
                let Some((gu_bytes, dn_bytes)) = resolve_bytes(eid) else {
                    return (acc, n_run);
                };
                SCRATCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let scratch = borrow
                        .get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
                    // Resize-on-shape-change: one process can host models
                    // of different shapes (rare, but cheap to handle).
                    if !scratch_fits(scratch, hidden, inter, inter_padded) {
                        *scratch = ExpertScratch::new(hidden, inter, inter_padded);
                    }
                    let h2 = run_expert(scratch, gu_bytes, dn_bytes);
                    for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                        *a += w * v;
                    }
                });
                (acc, n_run + 1)
            },
        )
        .reduce(
            || (vec![0.0f32; hidden], 0usize),
            |(mut a, na), (b, nb)| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                (a, na + nb)
            },
        )
}

/// Whether a thread's cached `ExpertScratch` has every buffer sized for this
/// model. Checking only one buffer lets a model with the same `inter` but a
/// different `hidden` reuse a wrongly sized output buffer.
fn scratch_fits(scratch: &ExpertScratch, hidden: usize, inter: usize, inter_padded: usize) -> bool {
    scratch.gate_out.len() == inter
        && scratch.act.len() == inter_padded
        && scratch.out.len() == hidden
}
