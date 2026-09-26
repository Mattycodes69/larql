//! Quotient-class validation, class metrics and replacement evaluation.

use super::super::metrics::{
    argmax, bool_rate, kl_logp, log_softmax, mean, percentile, top_k_indices,
};
use super::super::oracle_pq_forward::{final_logits, forward_q4k_predicted_address_mode_d_head};
use super::super::program::Program;
use super::super::types::HeadId;
use larql_vindex::VectorIndex;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[allow(unused_imports)]
use super::*;

pub(super) fn validate_quotient_classes(
    program: &Program,
    observed_classes: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let declared = program
        .terminal_classes
        .iter()
        .map(|tc| tc.representative_code)
        .collect::<BTreeSet<_>>();
    if declared.is_empty() {
        return Err(
            "program declares no terminal_classes; probe target would be raw PQ codes".into(),
        );
    }
    let leaked = observed_classes
        .iter()
        .copied()
        .filter(|code| !declared.contains(code))
        .collect::<Vec<_>>();
    if !leaked.is_empty() {
        return Err(format!(
            "program leaves raw PQ codes outside the behavioral quotient: {:?}; use a program that canonicalizes every observed code into terminal_classes",
            leaked
        )
        .into());
    }
    Ok(())
}

pub(super) fn class_metrics(truth: &[usize], pred: &[usize], classes: &[usize]) -> ClassMetrics {
    let class_to_idx = classes
        .iter()
        .enumerate()
        .map(|(idx, &code)| (code, idx))
        .collect::<HashMap<_, _>>();
    let mut confusion = vec![vec![0usize; classes.len()]; classes.len()];
    let mut correct = 0usize;
    for (&t, &p) in truth.iter().zip(pred.iter()) {
        if t == p {
            correct += 1;
        }
        if let (Some(&ti), Some(&pi)) = (class_to_idx.get(&t), class_to_idx.get(&p)) {
            confusion[ti][pi] += 1;
        }
    }
    let mut per_class_f1 = BTreeMap::new();
    let mut f1s = Vec::new();
    for (idx, &class_code) in classes.iter().enumerate() {
        let tp = confusion[idx][idx] as f64;
        let row_sum = confusion[idx].iter().sum::<usize>() as f64;
        let col_sum = confusion.iter().map(|row| row[idx]).sum::<usize>() as f64;
        let precision = if col_sum > 0.0 { tp / col_sum } else { 0.0 };
        let recall = if row_sum > 0.0 { tp / row_sum } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        per_class_f1.insert(class_code, f1);
        f1s.push(f1);
    }
    ClassMetrics {
        accuracy: if truth.is_empty() {
            0.0
        } else {
            correct as f64 / truth.len() as f64
        },
        macro_f1: mean(&f1s),
        per_class_f1,
        confusion,
    }
}

pub(super) fn evaluate_replacement<F>(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    head: HeadId,
    group: usize,
    mode_d_table: &super::super::pq::ModeDTable,
    prompts: &[ProbePrompt],
    mut predicted_classes: F,
) -> Result<ReplacementMetrics, Box<dyn std::error::Error>>
where
    F: FnMut(&ProbePrompt, &BTreeMap<ProbeSource, Vec<Vec<f32>>>) -> Vec<usize>,
{
    let mut prompt_kls = Vec::new();
    let mut top1_agree = Vec::new();
    let mut top5_keep = Vec::new();
    for prompt in prompts {
        let predicted = predicted_classes(prompt, &prompt.features);
        let mut remapped_codes = prompt.oracle_codes.clone();
        for (codes, &class_code) in remapped_codes.iter_mut().zip(predicted.iter()) {
            if group < codes.len() {
                codes[group] = class_code;
            }
        }
        let h = forward_q4k_predicted_address_mode_d_head(
            weights,
            &prompt.token_ids,
            index,
            head,
            mode_d_table,
            &remapped_codes,
            &prompt.stratum,
        )?;
        let logits = final_logits(weights, &h);
        let logp = log_softmax(&logits);
        let top1 = argmax(&logits);
        let top5 = top_k_indices(&logits, 5);
        prompt_kls.push(kl_logp(&prompt.baseline_logp, &logp));
        top1_agree.push(top1 == prompt.baseline_top1);
        top5_keep.push(top5.contains(&prompt.baseline_top1));
    }
    Ok(ReplacementMetrics {
        mean_kl: mean(&prompt_kls),
        p95_kl: percentile(prompt_kls.clone(), 0.95),
        max_kl: prompt_kls.iter().copied().fold(0.0_f64, f64::max),
        top1_agreement: bool_rate(top1_agree.into_iter()),
        top5_retention: bool_rate(top5_keep.into_iter()),
    })
}
