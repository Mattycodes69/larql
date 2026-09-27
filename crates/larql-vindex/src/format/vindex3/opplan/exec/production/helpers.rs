//! Bias, router and aggregation helpers for the production backend.

use super::super::backend::{AttentionCall, ProjectedQkv, RoutedFfnCall};
use super::super::kernels::sigmoid;
use super::super::timing::{timed, OpClass};
use crate::error::VindexError;
use larql_compute::attention::softmax::{softmax_in_place, softmax_in_place_f32};
use larql_compute::ffn::expert_weight::router;
use larql_compute::residual::rms_norm_heads_no_weight_eps;
use larql_compute::MoeGateRule;
use larql_models::config::GateUpLayout;
use larql_models::config::{AttentionSinkSpec, ExpertRoutingPolicy, GateUpBranch, MoeRouterKind};

#[allow(unused_imports)]
use super::*;

/// The output-projection bias, added after `w_o`.
pub(in super::super) fn add_output_bias(call: &AttentionCall<'_>, out: &mut [f32]) {
    if let Some(o) = call.bias.as_ref().and_then(|bias| bias.o) {
        add_bias_in_place(out, o);
    }
}

/// `x[i] += b[i]`; a length mismatch is a geometry bug closure refuses,
/// so it panics rather than pads.
pub(super) fn add_bias_in_place(x: &mut [f32], b: &[f32]) {
    assert_eq!(
        x.len(),
        b.len(),
        "bias length must equal the projection's rows"
    );
    for (x, b) in x.iter_mut().zip(b) {
        *x += b;
    }
}

/// Gate and up: the two branches sharing one fused operand.
pub(in super::super) const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// The vector the router projects: `x` for every family but Gemma 4,
/// whose router reads the raw residual conditioned by a scale-less RMS
/// norm (served `rms_norm_no_weight`), the learned `router.scale` and
/// `hidden^-0.5` — the served `moe_router_input` arithmetic under HF's
/// input choice. Every conditioning operand must be present.
pub(in super::super) fn router_input(call: &RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
    if call.router_kind != MoeRouterKind::TopKRenormScaled {
        // `router_input`, not `x`. Until K3-LATENTMOE-1 these were the
        // same vector for every non-Gemma-4 family, so reading `x` here
        // was indistinguishable from honouring the field — the seam was
        // declared and not carried, and nothing could tell. A latent
        // routed branch hands the experts a projection of the block
        // input and the router the block input itself, and taking `x`
        // here would route on the bottleneck: a different model, and one
        // no shape check can see, since the router matrix would simply
        // be applied to a vector of the wrong width.
        return Ok(call.router_input.unwrap_or(call.x).to_vec());
    }
    let missing = |what: &str| {
        VindexError::Parse(format!(
            "TopKRenormScaled router without its {what}; the plan must carry it"
        ))
    };
    let router_scale = call.router_scale.ok_or_else(|| missing("router scale"))?;
    let eps = call
        .router_norm_eps
        .ok_or_else(|| missing("router norm eps"))?;
    let residual = call.router_input.unwrap_or(call.x);
    // The served scale-less RMS norm, the whole vector as one "head".
    let normed = rms_norm_heads_no_weight_eps(&as_row(residual), 1, call.hidden, eps);
    let mut conditioned: Vec<f32> = normed.iter().copied().collect();
    let root_hidden_inv = (call.hidden as f32).powf(-0.5);
    for (v, s) in conditioned.iter_mut().zip(router_scale) {
        *v *= s * root_hidden_inv;
    }
    Ok(conditioned)
}

/// Route one token through the served selection rule
/// (`larql-compute`'s `router::select`) over the router logits — shared
/// glue, so the production and device backends select identically and
/// exactly as the served path does. Gemma 4 selects with the served
/// renormalised-softmax rule and then applies its per-expert scale to the
/// selected weights (served `moe_route_from_router_input`'s
/// `RenormalizedSoftmax` + `PerExpert` arms).
pub(in super::super) fn select_experts(
    call: &RoutedFfnCall<'_>,
    logits: &mut [f32],
) -> Result<Vec<(usize, f32)>, VindexError> {
    if call.router_kind == MoeRouterKind::TopKRenormScaled {
        let per_expert = call.router_per_expert_scale.ok_or_else(|| {
            VindexError::Parse(
                "TopKRenormScaled router without its per-expert scale; the plan must carry it"
                    .to_string(),
            )
        })?;
        let mut selected = router::select(
            logits,
            call.top_k,
            ExpertRoutingPolicy::NormalisedOverSelected,
        );
        for (e, w) in &mut selected {
            *w *= per_expert[*e];
        }
        return Ok(selected);
    }
    let mut selected = if call.router_kind == MoeRouterKind::Sigmoid {
        sigmoid_select(logits, call.router_bias, call.top_k, call.routing_policy)
    } else {
        if let Some(bias) = call.router_bias {
            for (l, b) in logits.iter_mut().zip(bias) {
                *l += b;
            }
        }
        router::select(logits, call.top_k, call.routing_policy)
    };
    if call.branch_scale != 1.0 {
        for (_, w) in &mut selected {
            *w *= call.branch_scale;
        }
    }
    Ok(selected)
}

/// The reference's renormalisation guard: `weights / (sum + 1e-20)`, so a
/// selection whose scores all underflow divides by something.
pub(super) const SIGMOID_RENORM_EPS: f32 = 1e-20;

/// The sigmoid router (DeepSeek-V3, Kimi, GLM-5.3-Flash): every expert's
/// score is `sigmoid(logit)`, independent of the others; the correction
/// bias moves which experts are SELECTED and never what they WEIGH; the
/// selected raw scores are the weights, renormalised to sum to one under
/// [`ExpertRoutingPolicy::NormalisedOverSelected`] and kept raw otherwise.
/// Ties rank by first index, as `torch.topk` does.
pub(in super::super) fn sigmoid_select(
    logits: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    policy: ExpertRoutingPolicy,
) -> Vec<(usize, f32)> {
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let keys: Vec<f32> = match bias {
        Some(bias) => scores.iter().zip(bias).map(|(s, b)| s + b).collect(),
        None => scores.clone(),
    };
    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    // A stable sort on the key keeps equal keys in index order.
    ranked.sort_by(|&a, &b| {
        keys[b]
            .partial_cmp(&keys[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(top_k.min(logits.len()));
    let mut selected: Vec<(usize, f32)> = ranked.iter().map(|&e| (e, scores[e])).collect();
    if policy == ExpertRoutingPolicy::NormalisedOverSelected && selected.len() > 1 {
        let sum = selected.iter().map(|(_, w)| w).sum::<f32>() + SIGMOID_RENORM_EPS;
        for (_, w) in &mut selected {
            *w /= sum;
        }
    }
    selected
}

/// One selected expert's inner activation from its fused gate/up output
/// (bias already added): rows read through the declared layout, combined
/// by the served gate rule.
pub(in super::super) fn expert_inner(
    call: &RoutedFfnCall<'_>,
    layout: GateUpLayout,
    fused: &[f32],
) -> Vec<f32> {
    let rule = MoeGateRule::from_arch(call.gate_policy, call.activation);
    (0..call.intermediate)
        .map(|i| {
            let g = fused[layout.row(GateUpBranch::Gate, i, call.intermediate)];
            let u = fused[layout.row(GateUpBranch::Up, i, call.intermediate)];
            rule.combine(g, u)
        })
        .collect()
}

/// `x[i] += bias[expert-th row]` for a per-expert bias stored flat.
pub(in super::super) fn add_expert_bias(x: &mut [f32], bias: Option<&[f32]>, expert: usize) {
    if let Some(bias) = bias {
        let rows = x.len();
        add_bias_in_place(x, &bias[expert * rows..(expert + 1) * rows]);
    }
}

/// One query position's scores, softmax and weighted-V aggregation —
/// the production softmax kernel over whatever K/V storage the caller
/// abstracts through `key_of`/`value_of`. Shared by the production and
/// device backends (the device deliberately runs production glue so a
/// divergence is attributable to device matmul arithmetic alone); the
/// gate and output projections stay with each backend's own matmuls.
/// The first source position a query at `position` may attend to — the
/// plan's retention authority ([`HistoryRange`](super::super::kv::HistoryRange)),
/// shared by the kernel and the head tap. The policy is written there,
/// once; this only reads it.
pub(in super::super) fn source_start(call: &AttentionCall<'_>, position: usize) -> usize {
    call.history().required_start(position)
}

pub(in super::super) fn aggregate_heads<'k>(
    call: &AttentionCall<'_>,
    position: usize,
    query: &[f32],
    key_of: impl Fn(usize) -> &'k [f32],
    value_of: impl Fn(usize) -> &'k [f32],
) -> Vec<f32> {
    aggregate_heads_keeping(call, position, query, key_of, value_of, None)
}

/// [`aggregate_heads`] that, when asked, keeps each head's softmax
/// distribution after it has been consumed (V3-HEAD-OBS-1). The
/// arithmetic is identical with or without `keep`: the distribution is
/// moved out after the weighted sum, never recomputed or reordered.
pub(in super::super) fn aggregate_heads_keeping<'k>(
    call: &AttentionCall<'_>,
    position: usize,
    query: &[f32],
    key_of: impl Fn(usize) -> &'k [f32],
    value_of: impl Fn(usize) -> &'k [f32],
    mut keep: Option<&mut Vec<Vec<f32>>>,
) -> Vec<f32> {
    let head_dim = call.head_dim;
    let q_rows = call.num_q_heads * head_dim;
    let group = call.num_q_heads / call.num_kv_heads;
    let start = source_start(call, position);
    let _t = timed(OpClass::AttentionCore);
    let mut concat = vec![0.0f32; q_rows];
    for q_head in 0..call.num_q_heads {
        let kv_head = q_head / group;
        let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let mut scores: Vec<f32> = (start..=position)
            .map(|key_position| {
                let k_slice = &key_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                let dot: f32 = q_slice.iter().zip(k_slice).map(|(a, b)| a * b).sum();
                let scaled = dot * call.score_scale as f32;
                match call.logit_softcapping {
                    Some(cap) => cap * (scaled / cap).tanh(),
                    None => scaled,
                }
            })
            .collect();
        match &call.sinks {
            // The served path's own sink softmax; exhaustive on the judged
            // semantics so a new variant must be implemented before it
            // can execute here.
            Some(sinks) => {
                let AttentionSinkSpec::SoftmaxDenominator = sinks.spec;
                softmax_in_place(&mut scores, Some(sinks.logits[q_head]));
            }
            None => softmax_in_place_f32(&mut scores),
        }
        let head_out = &mut concat[q_head * head_dim..(q_head + 1) * head_dim];
        for (offset, key_position) in (start..=position).enumerate() {
            let v_slice = &value_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
            let weight = scores[offset];
            for (acc, v) in head_out.iter_mut().zip(v_slice) {
                *acc += weight * v;
            }
        }
        if let Some(kept) = keep.as_deref_mut() {
            kept.push(scores);
        }
    }
    concat
}

/// One position's projections, plus the gate half when it came out of
/// the same product.
///
/// Private to this backend: `ProjectedQkv` is the shared seam type and
/// the reference backend deliberately still projects the gate a second
/// time — it is the literal transcription, and the oracle's value is that
/// it does the obvious thing. That the two agree to 4e-7 is what licenses
/// this sharing.
pub(in super::super) struct ProjectedAttention {
    pub(in super::super) qkv: ProjectedQkv,
    /// The gate's values when the plan said they are the other half of
    /// the query projection. `None` for a gate with its own operand, and
    /// for a layer with no gate at all.
    pub(in super::super) gate: Option<Vec<f32>>,
}
