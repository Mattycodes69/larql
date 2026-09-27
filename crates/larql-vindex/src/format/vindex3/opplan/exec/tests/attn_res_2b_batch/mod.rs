//! **K3-ATTNRES-1 2b — the batch traversal carries ONE residual history
//! per position.**
//!
//! Frozen in `docs/arch-conformance/forecasts/k3-attnres-1-traverse.json`
//! against the oracle commit `ec7da08d`. 2a proved the decode traversal
//! against that oracle; this module proves the batch traversal carries a
//! DISTINCT history per position, agrees with the decode traversal to
//! the bit, and is caught by three positional controls. It lifts
//! nothing: the public loader still refuses, and a resume point carrying
//! a history plane is still refused by name.
//!
//! # The invariant, stated once
//!
//! **Batching may vectorise the branch computation. It may not merge,
//! share, reorder or reinterpret residual history.** Every position
//! snapshots at the same layers — the schedule is a property of depth,
//! not of the token — so all three histories always hold the same COUNT.
//! What separates them is their CONTENTS. A traversal that built one
//! shared history and handed it to every row would produce the right
//! counts, the right candidate counts, a plausible distribution at every
//! site, and a wrong model.
//!
//! # What is foreign here and what is not
//!
//! ```text
//! FOREIGN     the SCHEDULE and the probability BAND, both
//!             read from the oracle's export — which layers reduce, over
//!             how many candidates, where the boundary events fall, and
//!             the non-saturated interval the whole rung is only
//!             measurable inside.
//!
//! NOT FOREIGN A7 compares the batch traversal against the DECODE
//!             traversal. That is a self-consistency check, and it is
//!             worth having only because the decode arm is itself
//!             anchored against the oracle in 2a. Its provenance is
//!             borrowed, and this module says so rather than presenting
//!             batch/decode agreement as an external result.
//! ```
//!
//! The per-site ARITHMETIC is not re-compared against the oracle here:
//! a batch run from tokens enters on the substrate's own embeddings
//! rather than the oracle's, so its states are not the oracle's states.
//! 2a already made that comparison at the only place it is meaningful —
//! the reduction, replayed from the oracle's own recorded entering
//! state. What 2b adds instead is that every batch site reduction is
//! required to equal `attention_residual::reduce` called directly on the
//! state the witness recorded for that position, so the batch path
//! cannot have grown a second, batched implementation of the reduction.
//!
//! # The two traversals emit in different orders, and A7 accounts for it
//!
//! Decode is position-major: a whole stack for position 0, then for
//! position 1. Batch is layer-major: layer 0 for every position, then
//! layer 1. The global event sequences therefore CANNOT be equal, and a
//! test that compared them directly would be asserting an interleaving
//! rather than a topology. A7 compares the per-position SUBSEQUENCE,
//! which preserves every ordering claim the topology actually makes —
//! the order of sites and boundary events within one position's stack.
//!
//! # Why three positions, and the control that proves it was necessary
//!
//! `a_single_position_batch_hides_every_positional_defect` runs the same
//! witness at batch size one and shows all three controls are INVISIBLE
//! there — the swap and the offset need two rows to act on, and
//! broadcasting position 0's state onto position 0 is the identity. A
//! witness built on a one-position fixture would report this transition
//! green while proving nothing about it. The separation the three
//! positions actually achieve is measured first, and every later
//! assertion in the file is void if that measurement fails.

use super::super::attention_residual::{self, History};
use super::super::decode::DecodeSession;
use super::super::hyper_connection::Mutation;
use super::super::kv::RowKvState;
use super::super::observe::{
    AttnResBoundaryRecord, AttnResSiteRecord, HcSite, StepEvent, StepObserver,
};
use super::super::prepared::PreparedOperands;
use super::super::reference::ReferenceBackend;
use super::super::{
    execute_prepared_streaming_mutated, FinalOutput, FinalState, Plane, PlaneEvent, ResumePoint,
};
use super::attn_res_substrate::{
    prepare, substrate, Oracle, Substrate, HIDDEN, LAYERS, MAX_PROB, MIN_PROB, NORM_EPS, POSITIONS,
    TOLERANCE,
};

/// Three distinct tokens, deliberately neither ascending nor adjacent,
/// so no position can be confused with its index and no defect that
/// reverses the order is hidden by a monotone fixture.
pub(super) const TOKENS: [u32; POSITIONS] = [1, 4, 2];

/// The margin the separation precondition requires between any two
/// positions' state at any site.
///
/// A fixed multiple of the comparison tolerance rather than a number
/// read off this fixture: the question it answers is "could a positional
/// defect hide inside the noise floor", and the noise floor is
/// [`TOLERANCE`]. 100x is the margin that makes the answer unambiguous
/// without pinning a coincidence of these particular weights.
const SEPARATION: f32 = TOLERANCE * 100.0;

/// One site at one position, in the shape BOTH traversals emit.
///
/// Every field is compared by A7, and the derived ones are derived the
/// same way on both paths — `prefix_after` from the history's prefix
/// falling back to the branch output, exactly as the decode traversal
/// does when a boundary reset leaves the branch output standing alone.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SiteRow {
    pub(super) layer: usize,
    pub(super) site: HcSite,
    pub(super) position: usize,
    pub(super) candidate_count: usize,
    pub(super) snapshot_count_before: usize,
    pub(super) probs: Vec<f32>,
    pub(super) mixed: Vec<f32>,
    pub(super) prefix_before: Vec<f32>,
    pub(super) prefix_after: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundaryRow {
    pub(super) layer: usize,
    pub(super) position: usize,
    pub(super) snapshots_before: usize,
    pub(super) snapshots_after: usize,
    pub(super) value: Vec<f32>,
    pub(super) entering_prefix: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Event {
    Site(SiteRow),
    Boundary(BoundaryRow),
}

impl Event {
    fn position(&self) -> usize {
        match self {
            Self::Site(s) => s.position,
            Self::Boundary(b) => b.position,
        }
    }

    fn layer(&self) -> usize {
        match self {
            Self::Site(s) => s.layer,
            Self::Boundary(b) => b.layer,
        }
    }
}

#[derive(Default, Clone)]
pub(super) struct Witness {
    pub(super) events: Vec<Event>,
}

impl Witness {
    /// One position's events, in emission order. The unit A7 compares:
    /// the two traversals interleave positions differently and agree
    /// within one.
    pub(super) fn at(&self, position: usize) -> Vec<Event> {
        self.events
            .iter()
            .filter(|e| e.position() == position)
            .cloned()
            .collect()
    }

    /// Every site record, excluding the exit — which the batch path
    /// reduces for the last position only, and which the exit assertion compares as an
    /// OUTPUT rather than as a record.
    pub(super) fn sites(&self) -> Vec<&SiteRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Site(s) if s.layer < LAYERS => Some(s),
                _ => None,
            })
            .collect()
    }

    pub(super) fn boundaries(&self) -> Vec<&BoundaryRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Boundary(b) => Some(b),
                Event::Site(_) => None,
            })
            .collect()
    }

    pub(super) fn site(&self, layer: usize, site: HcSite, position: usize) -> Option<&SiteRow> {
        self.sites()
            .into_iter()
            .find(|s| s.layer == layer && s.site == site && s.position == position)
    }

    /// The snapshot values this position had accumulated when the site
    /// at `layer` was entered, oldest first — reconstructed from the
    /// witness's OWN boundary rows.
    ///
    /// The boundary event of a layer falls BETWEEN its two sites, so the
    /// attention site of layer L reads the events of layers strictly
    /// before L and the FFN site reads L's own as well. The shared-
    /// reduction assertion rebuilds the
    /// reduction from this and is what proves the executor's recorded
    /// `snapshot_count_before` describes a real set.
    pub(super) fn snapshots_for(
        &self,
        layer: usize,
        site: HcSite,
        position: usize,
    ) -> Vec<Vec<f32>> {
        self.boundaries()
            .into_iter()
            .filter(|b| b.position == position)
            .filter(|b| match site {
                HcSite::Attention => b.layer < layer,
                HcSite::Ffn => b.layer <= layer,
            })
            .map(|b| b.value.clone())
            .collect()
    }
}

impl StepObserver for Witness {
    fn event(&mut self, _event: StepEvent) {}

    fn attention_residual_site(&mut self, r: AttnResSiteRecord<'_>) {
        self.events.push(Event::Site(SiteRow {
            layer: r.layer,
            site: r.site,
            position: r.position,
            candidate_count: r.candidate_count,
            snapshot_count_before: r.snapshot_count_before,
            probs: r.probs.to_vec(),
            mixed: r.mixed_vector.to_vec(),
            prefix_before: r.prefix_before.to_vec(),
            prefix_after: r.prefix_after.to_vec(),
        }));
    }

    fn attention_residual_boundary(&mut self, r: AttnResBoundaryRecord<'_>) {
        self.events.push(Event::Boundary(BoundaryRow {
            layer: r.layer,
            position: r.position,
            snapshots_before: r.snapshots_before,
            snapshots_after: r.snapshots_after,
            value: r.value.to_vec(),
            entering_prefix: r.entering_prefix.to_vec(),
        }));
    }
}

pub(super) struct Run {
    pub(super) witness: Witness,
    /// The last position's state as the run left it.
    ///
    /// The two paths tap this at DIFFERENT points, and the difference is
    /// not incidental: `StepRun::exit` is the vector the exit reduction
    /// produced, BEFORE the final norm, while a batch `FinalState` is
    /// what the component hands on, AFTER it. the exit assertion applies the norm to the
    /// decode tap rather than comparing the two raw — a test that
    /// compared them directly would be comparing a pre-norm vector to a
    /// post-norm one and calling the difference a defect.
    pub(super) exit: Vec<f32>,
    pub(super) logits: Option<Vec<f32>>,
}

/// Sequential decode over `tokens`, one step per position. The session
/// carries the KV across steps, so position p attends over 0..=p exactly
/// as the batch traversal does; the residual history is built fresh per
/// step, because it is a property of one forward pass.
pub(super) fn decode(sub: &Substrate, tokens: &[u32], mutation: Mutation) -> Run {
    let (_store, ops) = prepare(sub);
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let mut exit = Vec::new();
    let mut logits = None;
    for &token in tokens {
        let step = session
            .step_mutated(token, &mut witness, mutation)
            .expect("the decode step runs the topology");
        exit = step.exit.expect("a whole-stack image reduces at the exit");
        logits = step.logits;
    }
    Run {
        witness,
        exit,
        logits,
    }
}

/// One batch traversal over `tokens`, collected into the same record
/// shape the decode observer produces.
pub(super) fn batch(sub: &Substrate, tokens: &[u32], mutation: Mutation) -> Run {
    let (_store, ops) = prepare(sub);
    let backend = ReferenceBackend::new();
    let mut witness = Witness::default();
    let out = collect(sub, &ops, tokens, None, mutation, &backend, &mut witness)
        .expect("the batch traversal runs the topology");
    let exit = match out.exit {
        FinalState::Hidden(h) => h,
        other => panic!("a whole-stack batch image must exit on a hidden state, got {other:?}"),
    };
    Run {
        witness,
        exit,
        logits: out.logits,
    }
}

/// The plane-event sink, shared by every batch run in this file.
fn collect(
    sub: &Substrate,
    ops: &PreparedOperands,
    tokens: &[u32],
    resume: Option<ResumePoint>,
    mutation: Mutation,
    backend: &ReferenceBackend,
    witness: &mut Witness,
) -> Result<FinalOutput, crate::error::VindexError> {
    let out = execute_prepared_streaming_mutated(
        &sub.plan,
        ops,
        tokens,
        backend,
        resume,
        &mut |event| {
            match event {
                PlaneEvent::AttentionResidualSite(plane) => {
                    for position in 0..plane.reductions.len() {
                        let reduction = &plane.reductions[position];
                        // The same fallback the decode traversal uses: a
                        // boundary reset leaves the branch output as the
                        // whole prefix.
                        let prefix_after = plane.histories_out[position]
                            .prefix()
                            .unwrap_or(&plane.branch_outputs[position])
                            .to_vec();
                        witness.events.push(Event::Site(SiteRow {
                            layer: plane.layer,
                            site: plane.site,
                            position,
                            candidate_count: reduction.probs.len(),
                            snapshot_count_before: plane.snapshot_counts_before[position],
                            probs: reduction.probs.clone(),
                            mixed: reduction.mixed.clone(),
                            prefix_before: plane.prefixes_before[position].clone(),
                            prefix_after,
                        }));
                    }
                }
                PlaneEvent::AttentionResidualBoundary(plane) => {
                    for position in 0..plane.values.len() {
                        witness.events.push(Event::Boundary(BoundaryRow {
                            layer: plane.layer,
                            position,
                            snapshots_before: plane.snapshots_before,
                            snapshots_after: plane.snapshots_after,
                            value: plane.values[position].clone(),
                            entering_prefix: plane.entering_prefixes[position].clone(),
                        }));
                    }
                }
                PlaneEvent::Embedded(_) | PlaneEvent::Layer { .. } => {}
                PlaneEvent::HyperConnectionSite(_) => {
                    panic!("an attention-residual plan emitted a hyper-connection site")
                }
            }
            Ok(())
        },
        mutation,
        None,
    )?;
    Ok(out)
}

/// Every unordered pair of positions.
fn pairs() -> Vec<(usize, usize)> {
    (0..POSITIONS)
        .flat_map(|left| ((left + 1)..POSITIONS).map(move |right| (left, right)))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// A7's comparison, returning the first disagreement rather than
/// panicking, so the control tests can require it to FAIL and say where.
pub(super) fn assert_a7(batched: &Witness, stepped: &Witness) -> Result<(), String> {
    for position in 0..POSITIONS {
        let left = batched.at(position);
        let right: Vec<Event> = stepped
            .at(position)
            .into_iter()
            .filter(|e| e.layer() < LAYERS)
            .collect();
        if left.len() != right.len() {
            return Err(format!(
                "position {position}: {} batch events against {} decode events",
                left.len(),
                right.len()
            ));
        }
        for (index, (a, b)) in left.iter().zip(&right).enumerate() {
            if a != b {
                return Err(format!(
                    "position {position} event {index}: batch {a:?} != decode {b:?}"
                ));
            }
        }
    }
    Ok(())
}

mod the_precondition_every_later_assertion_r;
mod the_reduction_is_the_shared_one;
