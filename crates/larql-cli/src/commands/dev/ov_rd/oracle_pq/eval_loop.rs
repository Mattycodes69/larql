//! The held-out evaluation: for every eval prompt, the baseline forward,
//! then per `(head, config)` the oracle-PQ forward, the Mode D check and
//! every enabled probe.

use std::collections::HashMap;

use larql_inference::encode_prompt;

use crate::commands::dev::ov_rd::metrics::{
    argmax, kl_logp, log_softmax, max_abs_diff, token_prob, top_k_indices,
};
use crate::commands::dev::ov_rd::oracle_pq_forward::{
    final_logits, forward_q4k_oracle_pq_head, forward_q4k_oracle_pq_mode_d_head,
};
use crate::commands::dev::ov_rd::oracle_pq_reports::OraclePqPointAccumulator;
use crate::commands::dev::ov_rd::reports::OraclePqPromptReport;
use crate::commands::dev::ov_rd::types::{HeadId, PqConfig, PromptRecord};

use super::probe::{
    head_label, lookup, lookup_head, EvalContext, HeadConfigMap, OracleModeD, ProbeResult,
};
use super::registry::ProbeRegistry;
use super::setup::{BaseFit, LoadedModel};

/// Fallback label for a prompt with neither an id nor a stratum.
const DEFAULT_PROMPT_LABEL: &str = "prompt";
/// Stratum of a prompt that declares none.
const UNKNOWN_STRATUM: &str = "unknown";
const TOP5: usize = 5;
const TOP2: usize = 2;

/// Baseline next-token facts of one prompt.
struct Baseline {
    logp: Vec<f64>,
    top1: u32,
    top2: u32,
    top1_prob: f64,
    top2_prob: f64,
}

/// Everything the loop reads besides the model.
pub(super) struct EvalInputs<'a> {
    pub(super) heads: &'a [HeadId],
    pub(super) configs: &'a [PqConfig],
    pub(super) base: &'a BaseFit,
    pub(super) registry: &'a ProbeRegistry,
    pub(super) majority_codes: &'a HeadConfigMap<Vec<usize>>,
    pub(super) mode_d_check: bool,
}

pub(super) fn evaluate_prompts(
    model: &mut LoadedModel,
    inputs: &EvalInputs<'_>,
    eval_prompts: &[PromptRecord],
) -> ProbeResult<HashMap<(HeadId, PqConfig), OraclePqPointAccumulator>> {
    let mut accumulators = HashMap::new();
    for head in inputs.heads {
        for &config in inputs.configs {
            accumulators.insert((*head, config), OraclePqPointAccumulator::new());
        }
    }
    for (prompt_idx, record) in eval_prompts.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or(DEFAULT_PROMPT_LABEL);
        eprintln!("  [{}/{}] {}", prompt_idx + 1, eval_prompts.len(), label);

        let token_ids = encode_prompt(&model.tokenizer, &*model.weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or(UNKNOWN_STRATUM);
        let baseline = baseline(model, &token_ids);
        for head in inputs.heads {
            for &config in inputs.configs {
                let accumulator = accumulators
                    .get_mut(&(*head, config))
                    .expect("oracle PQ accumulator missing");
                evaluate_point(
                    model,
                    inputs,
                    &baseline,
                    PromptInput {
                        token_ids: &token_ids,
                        stratum,
                        label,
                    },
                    *head,
                    config,
                    accumulator,
                )?;
            }
        }
    }
    Ok(accumulators)
}

fn baseline(model: &LoadedModel, token_ids: &[u32]) -> Baseline {
    let hidden = larql_inference::vindex::predict_kquant_hidden(
        &model.weights,
        token_ids,
        &model.index,
        None,
    );
    let logits = final_logits(&model.weights, &hidden);
    let logp = log_softmax(&logits);
    let top1 = argmax(&logits);
    let top2 = top_k_indices(&logits, TOP2).get(1).copied().unwrap_or(top1);
    Baseline {
        top1_prob: token_prob(&logp, top1),
        top2_prob: token_prob(&logp, top2),
        logp,
        top1,
        top2,
    }
}

#[derive(Clone, Copy)]
struct PromptInput<'a> {
    token_ids: &'a [u32],
    stratum: &'a str,
    label: &'a str,
}

/// The Mode D side of one point: `(kl, top1, top1_agree, in_top5,
/// coefficient-vs-Mode-D max logit diff)`, all `None` without
/// `--mode-d-check`.
type ModeDOutcome = (
    Option<f64>,
    Option<u32>,
    Option<bool>,
    Option<bool>,
    Option<f64>,
);

fn evaluate_point(
    model: &mut LoadedModel,
    inputs: &EvalInputs<'_>,
    baseline: &Baseline,
    prompt: PromptInput<'_>,
    head: HeadId,
    config: PqConfig,
    accumulator: &mut OraclePqPointAccumulator,
) -> ProbeResult {
    let base = inputs.base;
    let basis = lookup_head(&base.bases, head, "basis for oracle PQ")?;
    let head_means = lookup_head(&base.means, head, "position means for oracle PQ")?;
    let pca_basis = lookup_head(&base.pca_bases, head, "empirical PCA basis for oracle PQ")?;
    let codebook = base
        .codebooks
        .get(&(head, config))
        .ok_or_else(|| format!("missing PQ codebook for {}", head_label(head)))?;
    let (pq_hidden, metrics, oracle_codes_by_position) = forward_q4k_oracle_pq_head(
        &mut model.weights,
        prompt.token_ids,
        &model.index,
        head,
        basis,
        pca_basis,
        head_means,
        codebook,
        prompt.stratum,
    )?;
    let pq_logits = final_logits(&model.weights, &pq_hidden);
    let pq_logp = log_softmax(&pq_logits);
    let kl = kl_logp(&baseline.logp, &pq_logp);
    let pq_top1 = argmax(&pq_logits);
    let pq_top5 = top_k_indices(&pq_logits, TOP5);
    let pq_top2 = top_k_indices(&pq_logits, TOP2)
        .get(1)
        .copied()
        .unwrap_or(pq_top1);
    let pq_top1_prob = token_prob(&pq_logp, pq_top1);
    let pq_top2_prob = token_prob(&pq_logp, pq_top2);

    let (mode_d_kl, mode_d_top1, mode_d_top1_agree, baseline_top1_in_mode_d_top5, coeff_diff): ModeDOutcome =
        if inputs.mode_d_check {
            let mode_d_table = lookup(&base.mode_d_tables, head, config, "Mode D table for")?;
            let mode_d_hidden = forward_q4k_oracle_pq_mode_d_head(
                &mut model.weights,
                prompt.token_ids,
                &model.index,
                head,
                basis,
                pca_basis,
                head_means,
                codebook,
                mode_d_table,
                prompt.stratum,
            )?;
            let mode_d_logits = final_logits(&model.weights, &mode_d_hidden);
            let mode_d_logp = log_softmax(&mode_d_logits);
            let mode_d_top1 = argmax(&mode_d_logits);
            let mode_d_top5 = top_k_indices(&mode_d_logits, TOP5);
            (
                Some(kl_logp(&baseline.logp, &mode_d_logp)),
                Some(mode_d_top1),
                Some(baseline.top1 == mode_d_top1),
                Some(mode_d_top5.contains(&baseline.top1)),
                Some(max_abs_diff(&pq_logits, &mode_d_logits)),
            )
        } else {
            (None, None, None, None, None)
        };

    let mut ctx = EvalContext {
        weights: &mut model.weights,
        index: &model.index,
        token_ids: prompt.token_ids,
        stratum: prompt.stratum,
        label: prompt.label,
        head,
        config,
        baseline_logp: &baseline.logp,
        baseline_top1: baseline.top1,
        oracle_codes_by_position: &oracle_codes_by_position,
        oracle_mode_d: OracleModeD {
            kl: mode_d_kl.unwrap_or(kl),
            top1_agree: mode_d_top1_agree.unwrap_or(false),
            baseline_top1_in_top5: baseline_top1_in_mode_d_top5.unwrap_or(false),
        },
        mode_d_tables: &base.mode_d_tables,
        majority_codes: inputs.majority_codes,
        accumulator,
    };
    for probe in &inputs.registry.probes {
        if probe.enabled() {
            probe.evaluate(&mut ctx)?;
        }
    }

    ctx.accumulator.add(OraclePqPromptReport {
        id: prompt.label.to_string(),
        stratum: prompt.stratum.to_string(),
        kl,
        delta_cross_entropy_bits: kl / std::f64::consts::LN_2,
        baseline_top1: baseline.top1,
        pq_top1,
        top1_agree: baseline.top1 == pq_top1,
        baseline_top1_in_pq_top5: pq_top5.contains(&baseline.top1),
        baseline_top1_prob: baseline.top1_prob,
        baseline_top2: baseline.top2,
        baseline_top2_prob: baseline.top2_prob,
        baseline_top1_margin: baseline.top1_prob - baseline.top2_prob,
        pq_top1_prob,
        pq_prob_of_baseline_top1: token_prob(&pq_logp, baseline.top1),
        pq_top1_margin: pq_top1_prob - pq_top2_prob,
        mode_d_kl,
        mode_d_top1,
        mode_d_top1_agree,
        baseline_top1_in_mode_d_top5,
        coeff_mode_d_max_abs_logit_diff: coeff_diff,
        pre_wo_l2: metrics.pre_wo_l2,
        wo_visible_l2: metrics.wo_visible_l2,
    });
    Ok(())
}
