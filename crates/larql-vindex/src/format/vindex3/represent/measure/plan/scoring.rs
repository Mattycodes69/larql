//! **Scoring the bank**: the null arm, then every sample through both arms,
//! in batches an arm may score concurrently.
//!
//! Concurrency changes wall time only. Each sample is scored exactly as it
//! would be alone ([`TeacherForcedArm::score_batch`]'s contract), per-position
//! numbers are computed from those logits alone, and everything is assembled
//! in bank order. When several positions fail, the refusal is the first in
//! bank order, as a sequential run would report it.

use rayon::prelude::*;

use super::arm::TeacherForcedArm;
use super::metrics::{self, MetricError, PositionMetrics};
use super::sketch::SketchSpec;
use super::{
    execution, first_difference, inadmissible, PlanExecutionFailure, PlanInadmissible, PlanRefusal,
};
use crate::format::vindex3::represent::token_bank::TokenBank;

/// Samples handed to an arm at once. It bounds the logits held in memory
/// (two arms × this many samples × positions × vocabulary) and gives an arm
/// enough independent work to fill the machine.
pub const SCORE_BATCH: usize = 8;

fn score_batch(
    arm: &mut dyn TeacherForcedArm,
    batch: &[Vec<u32>],
) -> Result<Vec<Vec<Vec<f32>>>, PlanRefusal> {
    let name = arm.describe().arm.clone();
    let scored = arm
        .score_batch(batch)
        .map_err(|detail| execution(PlanExecutionFailure::StepRefused { arm: name, detail }))?;
    if scored.len() != batch.len() {
        return Err(execution(PlanExecutionFailure::StepRefused {
            arm: arm.describe().arm.clone(),
            detail: format!("scored {} of {} samples", scored.len(), batch.len()),
        }));
    }
    Ok(scored)
}

/// The null arm: the reference scores each of its samples twice, and the two
/// must agree in every bit. Returns the first scoring of each, for reuse.
pub(super) fn null_arm(
    bank: &TokenBank,
    reference: &mut dyn TeacherForcedArm,
    samples: usize,
    bank_refusal: impl Fn(super::TokenBankError) -> PlanRefusal,
) -> Result<Vec<Vec<Vec<f32>>>, PlanRefusal> {
    let mut cache = Vec::with_capacity(samples);
    for sample in 0..samples {
        let ids = bank.read(sample).map_err(&bank_refusal)?;
        let mut twice = score_batch(reference, &[ids.clone(), ids])?.into_iter();
        let (first, second) = (twice.next(), twice.next());
        let (Some(first), Some(second)) = (first, second) else {
            unreachable!("score_batch returned one result per sample, checked above")
        };
        if let Some(position) = first_difference(&first, &second) {
            return Err(inadmissible(PlanInadmissible::NullArmNotZero {
                sample,
                position,
            }));
        }
        cache.push(first);
    }
    Ok(cache)
}

/// One position's numbers and sketch, or the refusal it earns.
type Scored = Result<(PositionMetrics, Option<Vec<f32>>), PlanRefusal>;

/// Every position of one sample, computed in parallel and returned in order.
fn score_positions(
    sample: usize,
    category: &str,
    ids: &[u32],
    reference: &[Vec<f32>],
    candidate: &[Vec<f32>],
    sketch: Option<&SketchSpec>,
) -> Vec<Scored> {
    reference
        .par_iter()
        .zip(candidate)
        .enumerate()
        .map(|(position, (r, c))| {
            let next = ids.get(position + 1).copied();
            let scored = metrics::position_metrics(r, c, next).map_err(|e| match e {
                MetricError::NonFinite { arm } => execution(PlanExecutionFailure::StepRefused {
                    arm: arm.into(),
                    detail: format!("sample {sample} position {position}: a logit is not finite"),
                }),
                other => inadmissible(PlanInadmissible::PositionCountMismatch {
                    sample,
                    detail: format!("position {position}: {other:?}"),
                }),
            })?;
            let row = PositionMetrics {
                sample,
                position,
                category: category.to_string(),
                kl: scored.kl,
                top1_agree: scored.top1_agree,
                top5_overlap: scored.top5_overlap,
                delta_nll: scored.delta_nll,
                reference_margin: scored.reference_margin,
                reference_entropy: scored.reference_entropy,
                max_abs_delta: scored.max_abs_delta,
                mean_abs_delta: scored.mean_abs_delta,
            };
            Ok((row, sketch.map(|spec| spec.project(r, c))))
        })
        .collect()
}

/// What the measurement produced, in bank order.
pub(super) struct Measured {
    pub positions: Vec<PositionMetrics>,
    pub sketches: Vec<Vec<f32>>,
    pub samples_read: usize,
}

/// Every sample through both arms, `batch` samples at a time.
/// `reference_cache` holds the null arm's scorings of the first samples,
/// which are reused rather than re-scored.
#[allow(clippy::too_many_arguments)]
pub(super) fn measure_samples(
    bank: &TokenBank,
    sequences: usize,
    batch_size: usize,
    reference: &mut dyn TeacherForcedArm,
    candidate: &mut dyn TeacherForcedArm,
    mut reference_cache: Vec<Vec<Vec<f32>>>,
    sketch: Option<&SketchSpec>,
    bank_refusal: impl Fn(super::TokenBankError) -> PlanRefusal,
) -> Result<Measured, PlanRefusal> {
    let mut out = Measured {
        positions: Vec::new(),
        sketches: Vec::new(),
        samples_read: 0,
    };
    let cached = reference_cache.len();
    let batch_size = batch_size.max(1);
    for first in (0..sequences).step_by(batch_size) {
        let samples: Vec<usize> = (first..sequences.min(first + batch_size)).collect();
        let mut batch = Vec::with_capacity(samples.len());
        for &sample in &samples {
            batch.push(bank.read(sample).map_err(&bank_refusal)?);
            out.samples_read += 1;
        }
        let uncached: Vec<Vec<u32>> = samples
            .iter()
            .zip(&batch)
            .filter(|(&sample, _)| sample >= cached)
            .map(|(_, ids)| ids.clone())
            .collect();
        let mut fresh = score_batch(reference, &uncached)?.into_iter();
        let reference_logits: Vec<Vec<Vec<f32>>> = samples
            .iter()
            .map(|&sample| match reference_cache.get_mut(sample) {
                Some(cached) => std::mem::take(cached),
                None => fresh
                    .next()
                    .expect("one fresh scoring per uncached sample, checked by score_batch"),
            })
            .collect();
        let candidate_logits = score_batch(candidate, &batch)?;
        for (((&sample, ids), r), c) in samples
            .iter()
            .zip(&batch)
            .zip(&reference_logits)
            .zip(&candidate_logits)
        {
            if r.len() != ids.len() || c.len() != ids.len() {
                return Err(inadmissible(PlanInadmissible::PositionCountMismatch {
                    sample,
                    detail: format!(
                        "{} ids, reference scored {}, candidate scored {}",
                        ids.len(),
                        r.len(),
                        c.len()
                    ),
                }));
            }
            let category = &bank.manifest().samples[sample].category;
            for scored in score_positions(sample, category, ids, r, c, sketch) {
                let (row, projected) = scored?;
                out.positions.push(row);
                out.sketches.extend(projected);
            }
        }
    }
    Ok(out)
}
