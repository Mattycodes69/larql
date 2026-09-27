//! MLA operands and matrix-shape helpers.

use super::super::super::{ComponentOpPlan, MlaOp, MlaQueryProjection, OperandRef};
use super::super::accounting::{
    declared_resident_for, expectations, BlockGeometry, Bound, Expectation, ResidencyBudget,
    ResourceLedger,
};
use super::super::backend::{PlanBackend, WeightFormat};
use super::super::operands::OperandSource;
use super::super::realization::{
    realization_residency, RealizationId, RealizationRecord, SelectionReason,
};
use super::super::weights::{load_weight, LoadedWeight};
use crate::error::VindexError;

#[allow(unused_imports)]
use super::*;

impl MlaOperands {
    pub(in super::super) fn bound<'a>(&'a self, op: &'a MlaOp) -> Vec<Bound<'a>> {
        let mut bound = match (&self.query, &op.query) {
            (MlaQueryOperands::Direct { q_proj }, MlaQueryProjection::Direct { q_proj: r }) => {
                vec![Bound::one(r, q_proj)]
            }
            (
                MlaQueryOperands::LowRank {
                    q_a_proj, q_b_proj, ..
                },
                MlaQueryProjection::LowRank {
                    q_a_proj: ra,
                    q_b_proj: rb,
                    ..
                },
            ) => vec![Bound::one(ra, q_a_proj), Bound::one(rb, q_b_proj)],
            _ => unreachable!("query operands loaded for a form the op does not declare"),
        };
        bound.extend([
            Bound::one(&op.kv_a_proj, &self.kv_a_proj),
            Bound::one(&op.kv_b_proj, &self.kv_b_proj),
            Bound::one(&op.out_proj, &self.o_proj),
        ]);
        if let (Some(r), Some(w)) = (&op.output_gate, &self.output_gate) {
            bound.push(Bound::one(r, w));
        }
        bound
    }

    pub(super) fn load(
        op: &super::super::super::MlaOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let kv_a_norm_eps = op.kv_a_norm_eps.ok_or_else(|| {
            VindexError::Parse(
                "this MLA layer carries no epsilon for its latent norm (`kv_a_layernorm`), \
                 which is NOT the layer's `rms_norm_eps` on any judged checkpoint; refusing \
                 to substitute one"
                    .to_string(),
            )
        })?;
        let query = match &op.query {
            MlaQueryProjection::Direct { q_proj } => MlaQueryOperands::Direct {
                q_proj: matrix(q_proj)?,
            },
            MlaQueryProjection::LowRank {
                q_a_proj,
                q_a_norm,
                q_b_proj,
                q_a_norm_eps,
            } => MlaQueryOperands::LowRank {
                q_a_proj: matrix(q_a_proj)?,
                q_a_norm: store.load(q_a_norm)?,
                q_b_proj: matrix(q_b_proj)?,
                // Its OWN epsilon. Refused rather than borrowed from the
                // latent norm above, whose value it happens to equal on
                // every checkpoint judged so far — a shared cause, one
                // class default used twice, and not a shared authority.
                q_a_norm_eps: q_a_norm_eps.ok_or_else(|| {
                    VindexError::Parse(
                        "this MLA layer factorises its query but carries no epsilon for \
                         `q_a_layernorm`, which is NOT the layer's `rms_norm_eps` and is not \
                         the latent norm's either; refusing to substitute one"
                            .to_string(),
                    )
                })?,
            },
        };
        Ok(Self {
            op: op.clone(),
            query,
            kv_a_proj: matrix(&op.kv_a_proj)?,
            kv_b_proj: matrix(&op.kv_b_proj)?,
            o_proj: matrix(&op.out_proj)?,
            kv_a_norm: store.load(&op.kv_a_norm)?,
            kv_a_norm_eps,
            output_gate: op.output_gate.as_ref().map(matrix).transpose()?,
        })
    }

    /// Every matrix, for residency accounting — including whichever
    /// query form's projections this layer loaded.
    pub(in super::super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut matrices = match &self.query {
            MlaQueryOperands::Direct { q_proj } => vec![q_proj],
            MlaQueryOperands::LowRank {
                q_a_proj, q_b_proj, ..
            } => vec![q_a_proj, q_b_proj],
        };
        matrices.extend([&self.kv_a_proj, &self.kv_b_proj, &self.o_proj]);
        matrices.extend(self.output_gate.as_ref());
        matrices
    }

    /// The f32 operands that are not matrix traffic: the latent norm, and
    /// the query latent's norm when the query is factorised.
    pub(in super::super) fn glue_bytes(&self) -> usize {
        let query_norm = match &self.query {
            MlaQueryOperands::Direct { .. } => 0,
            MlaQueryOperands::LowRank { q_a_norm, .. } => std::mem::size_of_val(&q_a_norm[..]),
        };
        std::mem::size_of_val(&self.kv_a_norm[..]) + query_norm
    }

    pub(in super::super) fn weights(
        &self,
    ) -> Result<super::super::mla::MlaWeights<'_>, VindexError> {
        Ok(super::super::mla::MlaWeights {
            query: match (&self.query, &self.op.query) {
                (MlaQueryOperands::Direct { q_proj }, MlaQueryProjection::Direct { q_proj: r }) => {
                    super::super::mla::MlaQueryWeights::Direct {
                        q_proj: matrix_rows(q_proj, r)?,
                    }
                }
                (
                    MlaQueryOperands::LowRank {
                        q_a_proj,
                        q_a_norm,
                        q_b_proj,
                        q_a_norm_eps,
                    },
                    MlaQueryProjection::LowRank {
                        q_a_proj: ra,
                        q_b_proj: rb,
                        ..
                    },
                ) => super::super::mla::MlaQueryWeights::LowRank {
                    q_a_proj: matrix_rows(q_a_proj, ra)?,
                    q_a_norm,
                    q_b_proj: matrix_rows(q_b_proj, rb)?,
                    q_a_norm_eps: *q_a_norm_eps,
                },
                _ => unreachable!("query operands loaded for a form the op does not declare"),
            },
            kv_a_proj: matrix_rows(&self.kv_a_proj, &self.op.kv_a_proj)?,
            kv_b_proj: matrix_rows(&self.kv_b_proj, &self.op.kv_b_proj)?,
            o_proj: matrix_rows(&self.o_proj, &self.op.out_proj)?,
            kv_a_norm: &self.kv_a_norm,
            kv_a_norm_eps: self.kv_a_norm_eps,
            output_gate: match (&self.output_gate, &self.op.output_gate) {
                (Some(w), Some(r)) => Some(matrix_rows(w, r)?),
                (None, None) => None,
                _ => unreachable!("MLA gate operand loaded for an op that does not declare it"),
            },
        })
    }
}

/// A resident matrix as row ranges, cut to the geometry the op declares.
///
/// The geometry comes from the OP and never from the slice length: a
/// resident slab is page-padded, so `len / in_dim` can exceed the number
/// of rows the matrix has.
pub(super) fn matrix_rows<'a>(
    w: &'a LoadedWeight,
    r: &OperandRef,
) -> Result<super::super::cpu::WeightRows<'a>, VindexError> {
    let (out_dim, in_dim) = two_dims(r)?;
    w.slice().rows(out_dim, in_dim)
}

/// A matrix operand's `[out, in]` geometry.
///
/// Fails closed on anything else: a projection is two-dimensional, and a
/// caller that inferred `out_dim` from a slice length instead would read
/// page padding as extra rows.
pub(super) fn two_dims(r: &OperandRef) -> Result<(usize, usize), VindexError> {
    match r.shape.as_slice() {
        [out_dim, in_dim] => Ok((*out_dim, *in_dim)),
        other => Err(VindexError::Parse(format!(
            "operand `{}` has shape {other:?}; a dense projection is `[out, in]`",
            r.tensor
        ))),
    }
}

/// Resolves the load format for ONE matrix operand.
///
/// A function rather than a value because the question is now per matrix
/// and not per class: a layer hands its q/k/v/o — or its five delta
/// projections — to the same resolver and can get different answers, which
/// is what lets a `48 x 5120` gate stay f32 inside a stack whose `10240 x
/// 5120` projections do not.
pub(in super::super) type FormatFor<'a> =
    &'a dyn Fn(&OperandRef) -> Result<WeightFormat, VindexError>;

/// Resolve, admit and select a realization for every planned operand
/// `slice` of `plan` will load — BEFORE any byte is read.
///
/// The backend is handed the planned operation and the registry's facts
/// for the stored dtype, never the bytes and never a label to match on.
/// Every refusal is collected, so the caller sees the whole problem and
/// not its first symptom. An operand the container does not hold is
/// skipped here: the loader refuses it by name, as it always has.
pub fn select_realizations<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    backend: &B,
    slice: &ExecutionSlice,
) -> Result<Vec<RealizationRecord>, VindexError> {
    Ok(select_records(plan, store, backend, slice)?
        .into_iter()
        .map(|(record, _)| record)
        .collect())
}

/// [`select_realizations`], then held against `budget`: while the plan's
/// PHYSICAL working set or per-token touch exceeds it, the record with
/// the largest resident saving among its own candidates is re-selected
/// to the cheaper-resident one the backend had considered, with reason
/// [`SelectionReason::BudgetPolicy`]; when no candidate can bring the
/// plan inside, the whole preparation is refused BEFORE any payload byte
/// with the irreducible deficit and the alternatives that were tried.
/// Under [`ResidencyBudget::UNBOUNDED`] this is exactly
/// [`select_realizations`].
pub fn select_realizations_within<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    backend: &B,
    slice: &ExecutionSlice,
    budget: &ResidencyBudget,
) -> Result<Vec<RealizationRecord>, VindexError> {
    let mut selected = select_records(plan, store, backend, slice)?;
    // The access policy is the budget's, not the candidate's: every mapped
    // pin executes under the one the caller declared.
    for (record, _) in &mut selected {
        let under_policy = record
            .selection
            .realization
            .with_access(budget.expert_access);
        record.repin(under_policy);
    }
    // The floor gates SELECTION, so it is asked about what was selected
    // — here, before any budget pressure — and not only about the
    // shallower extents the loop below considers moving to.
    super::super::fidelity_carriage::enforce_floor(
        selected.iter().map(|(record, _)| record),
        budget.fidelity,
    )?;
    let geometry = BlockGeometry::executor();
    let stored_len = |op: &OperandRef| store.stored_len(op);
    let mut switches: Vec<String> = Vec::new();
    loop {
        let records: Vec<RealizationRecord> = selected.iter().map(|(r, _)| r.clone()).collect();
        let priced = expectations(&records, stored_len, geometry);
        let ledger = ResourceLedger::aggregate(&priced);
        let deficit = budget.deficit(&ledger);
        if deficit.is_zero() {
            // No second floor check here, deliberately. The initial
            // selection was enforced above, and the ONLY assignment to
            // `extent.selected` after it takes its extent from
            // `shallowest_saving`, which already filters candidates by
            // `fidelity.admits`. Reselection is therefore enforced at the
            // point of choice, and a re-check on the way out could never
            // fire — a refusal that cannot happen is not a guarantee, it
            // is decoration, and this plane exists to remove those.
            return Ok(records);
        }
        // Preparation overruns are answered by reading LESS of an
        // artifact, which is an extent decision and nothing else: a
        // realization change moves what is held, never what is opened.
        // Only extents the plan's fidelity floor admits are considered, so
        // a shallower selection is a quality decision the caller made and
        // not one the budget made for them.
        if deficit.prepare > 0 {
            if let Some((i, extent, saving)) = shallowest_saving(&selected, budget) {
                let (record, _) = &mut selected[i];
                switches.push(format!(
                    "`{}` extent depth {} → {} (opens {:.2} GB less)",
                    record.planned.operand.tensor,
                    record.extent.selected.depth,
                    extent.depth,
                    saving as f64 / 1e9
                ));
                record.extent.selected = extent;
                record.selection.reason = SelectionReason::BudgetPolicy;
                continue;
            }
            if deficit.physical == 0 && deficit.touch_per_token == 0 {
                return Err(VindexError::Parse(budget_refusal(
                    budget, &ledger, &deficit, &priced, &switches,
                )));
            }
        }
        // The best switch: the largest resident saving any record can make
        // by moving to another of ITS OWN candidates.
        let mut best: Option<(usize, RealizationId, u64)> = None;
        for (i, (record, facts)) in selected.iter().enumerate() {
            let current = record.selection.realization;
            let now = declared_resident_for(
                &record.planned,
                current,
                record.selection.residency,
                geometry,
            );
            for candidate in &record.selection.candidates {
                if *candidate == current {
                    continue;
                }
                let then = declared_resident_for(
                    &record.planned,
                    *candidate,
                    realization_residency(facts, *candidate),
                    geometry,
                );
                if then < now {
                    let saving = now - then;
                    if best.is_none_or(|(_, _, s)| saving > s) {
                        best = Some((i, *candidate, saving));
                    }
                }
            }
        }
        let Some((i, candidate, saving)) = best else {
            return Err(VindexError::Parse(budget_refusal(
                budget, &ledger, &deficit, &priced, &switches,
            )));
        };
        let (record, facts) = &mut selected[i];
        switches.push(format!(
            "`{}` {} → {} (saves {:.2} GB resident)",
            record.planned.operand.tensor,
            record.selection.realization.name(),
            candidate.name(),
            saving as f64 / 1e9
        ));
        record.selection.residency = realization_residency(facts, candidate);
        record.repin(candidate);
        record.selection.reason = SelectionReason::BudgetPolicy;
    }
}

/// The refusal a budget produces: what the plan demands, what the budget
/// allows, the irreducible deficit, the largest committed operands that
/// have nowhere cheaper to go, and every alternative already taken.
pub(super) fn budget_refusal(
    budget: &ResidencyBudget,
    ledger: &ResourceLedger,
    deficit: &super::super::accounting::BudgetDeficit,
    priced: &[Expectation],
    switches: &[String],
) -> String {
    const GB: f64 = 1e9;
    const LARGEST: usize = 5;
    let mut out = format!(
        "the plan cannot be prepared within the residency budget before any payload byte: \
         physical working set {:.2} GB (resident {:.2} + staging peak {:.2} + page-in per \
         token {:.2}) against {}; touch per token {:.2} GB against {}; irreducible deficit: \
         {:.2} GB physical, {:.2} GB per token",
        ledger.physical_working_set() as f64 / GB,
        ledger.resident as f64 / GB,
        ledger.transient_peak as f64 / GB,
        ledger.page_in_per_token as f64 / GB,
        budget
            .physical_bytes
            .map(|b| format!("{:.2} GB", b as f64 / GB))
            .unwrap_or_else(|| "no physical limit".to_string()),
        ledger.touch_per_token as f64 / GB,
        budget
            .throughput
            .map(|t| format!("{:.2} GB per token", t.bytes_per_token() as f64 / GB))
            .unwrap_or_else(|| "no throughput limit".to_string()),
        deficit.physical as f64 / GB,
        deficit.touch_per_token as f64 / GB,
    );
    // The preparation dimension names the floor with it: a refusal that
    // says only "too many bytes" hides that a shallower extent existed and
    // the plan's own quality requirement ruled it out.
    if deficit.prepare > 0 {
        out.push_str(&format!(
            "; preparation opens {:.2} GB against {}, a deficit of {:.2} GB, under a fidelity \
             requirement of {}",
            ledger.read_to_prepare as f64 / GB,
            budget
                .prepare_bytes
                .map(|b| format!("{:.2} GB", b as f64 / GB))
                .unwrap_or_else(|| "no preparation limit".to_string()),
            deficit.prepare as f64 / GB,
            budget.fidelity.describe(),
        ));
    }
    let mut largest: Vec<&Expectation> = priced
        .iter()
        .filter(|e| e.resources().resident > 0)
        .collect();
    largest.sort_by_key(|e| std::cmp::Reverse(e.resources().resident));
    out.push_str("; largest committed operands with no cheaper candidate: ");
    out.push_str(
        &largest
            .iter()
            .take(LARGEST)
            .map(|e| {
                format!(
                    "`{}` {} {:.2} GB",
                    e.operand.tensor,
                    e.realization.name(),
                    e.resources().resident as f64 / GB
                )
            })
            .collect::<Vec<_>>()
            .join(", "),
    );
    if switches.is_empty() {
        out.push_str("; no alternative was cheaper");
    } else {
        out.push_str(&format!(
            "; alternatives already taken ({}): {}",
            switches.len(),
            switches.join(", ")
        ));
    }
    out
}
