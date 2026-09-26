//! **Versioned acceptance criteria, and the evidence they judge.**
//!
//! `quality_proven` must never be a boolean somebody sets. It has to
//! mean *passed a named gate*, because acceptance criteria change: they
//! get tightened when a representation ships, loosened for a draft
//! model, and compared across models. A record saying "quality: ok" with
//! no gate behind it is unfalsifiable a month later, and worse, is
//! indistinguishable from one that passed a much weaker bar.
//!
//! So the evidence carries the gate it was judged by, and the verdict is
//! **derived, never stored** — [`QualityEvidence`] holds only the
//! criteria and the measurements, so it is not possible to construct a
//! passing verdict over a failing bank.
//!
//! ## Routing evidence is kept apart from logit evidence
//!
//! Both live in one bank, but they answer different questions, and for a
//! MoE representation the difference is the whole diagnosis:
//!
//! - **logits moved, routing stable** — an arithmetic effect. The same
//!   experts ran and their outputs shifted. Usually smooth in the
//!   representation's error, and usually the thing more bits fix.
//! - **routing moved** — a decision changed. A different expert ran, so
//!   the downstream effect is not a small perturbation of the same
//!   function, and more bits *on the experts* may not be the response at
//!   all; the router or its inputs may be what needs protecting.
//!
//! Collapsing them into one "quality" number would make those two look
//! identical and send the precision-map search in the wrong direction.

use serde::{Deserialize, Serialize};

/// Acceptance criteria, identified so a claim can name what it passed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGate {
    /// e.g. `kimi-logit-v1`. Changing a threshold means a NEW id.
    pub id: String,
    /// Fewest positions the bank must cover for its tail statistics to
    /// mean anything. A p99 over nineteen positions is not a p99.
    pub positions_min: u64,
    pub kl_p99_max: f64,
    /// Raw discrete-change counts. `None` means **this gate does not
    /// judge on counts** — not that it permits any number of them.
    ///
    /// v1 and v2 set all three. v3 sets none, because the counts were
    /// measured and found unable to separate a coin flip from an
    /// overturned decision: at layer 26, six argmax flips gave up a
    /// median of 0.1 % of probability, and 232 top-10 changes moved a
    /// median of 0.33 % of top-10 mass one rank. The counts remain in
    /// the evidence as diagnostics; they stop being authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_flip_max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_change_max: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_flip_max: Option<u64>,
    /// **Consequence limits.** The most probability any single argmax
    /// flip may give up, the top-k mass any position may displace at
    /// p99, and how much of the routed mixture a layer may move.
    ///
    /// These are what make v3 stricter where it matters: one large
    /// mixture replacement fails instantly, while a thousand
    /// microscopic eighth-versus-ninth expert swaps pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_mass_displaced_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_mass_displaced_p99_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_mixture_mass_p99_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_mixture_mass_max: Option<f64>,
    /// Least baseline mass the bank's truncation must have covered at
    /// its WORST position, for the KL above to be judged at all.
    ///
    /// `None` — v1's shape — means the gate does not ask, and a bank
    /// with no coverage record is judged on its other criteria.
    /// Introducing this on an existing gate id would silently re-date
    /// every claim that ever cited it, so it arrives with a new id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covered_mass_min: Option<f64>,
    /// Whether this gate requires the bank's numbers to have been
    /// measured over the model's OWN activations.
    ///
    /// `None` means the gate does not judge the substrate — appropriate
    /// for a mechanism gate. `Some(true)` means a synthetic or unstated
    /// substrate fails, which is what any gate carrying a MAGNITUDE must
    /// set. There is deliberately no `Some(false)`-means-forbidden case:
    /// a gate never demands a synthetic substrate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_model_activations: Option<bool>,
}

/// What the bank measured about the output distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogitEvidence {
    pub kl_p50: f64,
    pub kl_p95: f64,
    pub kl_p99: f64,
    pub max_logit_delta: f64,
    /// Positions where the argmax changed.
    pub top1_flips: u64,
    /// Positions where the top-10 SET or its order changed.
    pub top10_changes: u64,
}

/// What the bank measured about routing — deliberately separate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingEvidence {
    /// Selected-expert-id changes, summed over positions and layers.
    pub route_flips: u64,
    /// Positions where ANY layer routed differently. A handful of flips
    /// concentrated in one position is a different fact from the same
    /// count spread across all of them.
    pub positions_with_route_change: u64,
    /// Layers that ever routed differently, so a fix can be scoped by
    /// depth instead of applied to the whole role.
    pub layers_with_route_change: u64,
    /// **How CLOSE the routing decisions that changed actually were.**
    ///
    /// The selection-score gap, in the BASELINE arm, between the last
    /// selected expert and the best unselected one, at each layer whose
    /// route changed — so a small value means the perturbation crossed
    /// a near-tie and a large one means it overturned a decision the
    /// router was confident about.
    ///
    /// Counting route changes cannot tell those apart, and they are not
    /// the same behavioural event. `None` when nothing changed, or when
    /// the bank was built without score evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_margin: Option<Distribution>,
    /// **How much MIXTURE MASS the changed routes moved**, as a
    /// fraction of the layer's routed combine weight.
    ///
    /// A swap of the 8th expert for the 9th at nearly equal weight
    /// moves almost nothing; replacing a heavily-weighted expert moves
    /// a lot. The count is identical in both cases, which is why the
    /// mass is carried separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_weight_mass_moved: Option<Distribution>,
    /// The SHALLOWEST layer that ever routed differently, if any.
    ///
    /// The diagnostic that separates two failure modes a count cannot:
    /// a perturbed layer changing its OWN routing (the candidate's
    /// experts were selected differently) versus a perturbed layer
    /// leaving its own routing intact and moving a LATER router's input
    /// — a cascade. Measured on Kimi layer 1 at Q6_K, the layer-1
    /// expert union was identical while twenty-five later layers moved,
    /// which is the second and needs a different response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_layer_with_route_change: Option<u64>,
}

/// A measured quantity's shape, not just its extremes.
///
/// Carried whole because "the worst case was large" and "most cases
/// were large" call for different responses, and a single number
/// cannot distinguish them — the question these exist to answer is
/// whether a criterion is failing on a few real events or on a mass of
/// marginal ones.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub count: u64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Distribution {
    /// Nearest-rank percentiles over `values`, which this sorts.
    ///
    /// Nearest-rank for the same reason the KL percentiles are: a
    /// reported value should be one some observation actually produced.
    pub fn of(values: &mut [f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let at = |p: f64| {
            let rank = (p * values.len() as f64).ceil().max(1.0) as usize;
            values[rank.min(values.len()) - 1]
        };
        Some(Self {
            count: values.len() as u64,
            min: values[0],
            p50: at(0.50),
            p95: at(0.95),
            p99: at(0.99),
            max: values[values.len() - 1],
        })
    }
}

pub use super::statistic::{Better, Statistic};

mod evidence;
pub use evidence::*;

/// One bank of measurements over a fixed token sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityBank {
    pub positions: u64,
    pub logits: LogitEvidence,
    pub routing: RoutingEvidence,
    /// Smallest baseline probability mass any position's truncation
    /// covered — how much of the distribution the KL above can SEE.
    ///
    /// Recorded because a KL over a truncation that covered a third of
    /// the mass is a different measurement from one that covered all of
    /// it, and nothing else in the bank distinguishes them. Measured on
    /// the real bank: a teacher-forced sequence's FIRST position has no
    /// context and is near-flat over 163,840 ids, so a short truncation
    /// sees almost none of it — top-128 covered 0.307, top-2048 0.729.
    ///
    /// `None` for a bank whose builder did not record it. Judged only
    /// by gates that ask for it — see [`kimi_logit_v2`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_covered_mass: Option<f64>,
    /// **Where the activations behind every number above came from.**
    ///
    /// Optional so older records deserialize, but absence is refused by
    /// any gate that asks — see [`Criterion::ActivationAuthority`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activations: Option<super::activation::ActivationSource>,
    /// **How close the top-10 orderings that changed actually were.**
    ///
    /// The baseline's rank-10-minus-rank-11 logit gap at each position
    /// whose top-10 changed. The top-10 criterion counts any change
    /// including a reordering, and a reordering of two near-tied
    /// candidates is not the same event as a genuine preference
    /// change — this is the evidence that tells them apart.
    ///
    /// EVIDENCE, not a criterion: no gate reads it yet, deliberately,
    /// because a threshold set before the distribution is known is a
    /// guess wearing a version number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_margin: Option<Distribution>,
    /// The CANDIDATE's gap at the same two ids. Read against
    /// [`Self::top10_margin`]: if the boundary was near-tied and the
    /// candidate's gap is comparably small, the ordering flipped
    /// because the two were indistinguishable, not because the model's
    /// preference changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_candidate_margin: Option<Distribution>,
    /// Probability mass displaced across the top-k, 0 = none, 1 =
    /// disjoint. The top-k analogue of
    /// [`RoutingEvidence::route_weight_mass_moved`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_mass_displaced: Option<Distribution>,
    /// Furthest rank move of any id. A swap of ranks 10 and 11 is 1; a
    /// candidate arriving from rank 400 is 390.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top10_rank_displacement: Option<Distribution>,
    /// **How much margin the argmax flips actually crossed.** The
    /// strictest criterion in the contract, and the last one to get
    /// severity evidence: a coin-flip between near-equal candidates and
    /// an overturned confident choice are both "one flip" without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_margin: Option<Distribution>,
    /// The candidate's gap over the same pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_candidate_margin: Option<Distribution>,
    /// Probability the baseline gave up by switching winner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_mass_displaced: Option<Distribution>,
}

/// Which criterion a bank failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Criterion {
    Positions,
    KlP99,
    Top1Flips,
    Top10Changes,
    RouteFlips,
    /// An argmax flip gave up more probability than a near-tie would.
    Top1Displacement,
    /// A top-k reordering displaced more mass than a near-tie would.
    TopKDisplacement,
    /// A routing change moved more of the routed mixture than a
    /// near-boundary swap would.
    RouteDisplacement,
    /// The bank's KL was blind to too much of the distribution, or did
    /// not record how much it saw. A gate that asks for coverage and
    /// gets `None` must fail rather than assume the truncation was
    /// wide enough.
    CoveredMass,
    /// The bank did not say where its activations came from, or said
    /// they were synthetic. A gate that asks for model activations and
    /// gets either must fail: a synthetic substrate may prove mechanism
    /// and fire controls, never a magnitude, and an unstated one is
    /// indistinguishable from a synthetic one. See
    /// [`super::activation`].
    ActivationAuthority,
    /// plan-v1: KL p99, nats over the full vocabulary.
    PlanKlP99,
    /// plan-v1: mean KL, nats over the full vocabulary.
    PlanKlMean,
    /// plan-v1: share of positions whose argmax moved.
    PlanTop1Disagreement,
    /// plan-v1: mean ΔNLL of the actual next token.
    PlanDeltaNllMean,
}

impl Criterion {
    pub fn name(self) -> &'static str {
        match self {
            Criterion::Positions => "positions",
            Criterion::KlP99 => "kl_p99",
            Criterion::Top1Flips => "top1_flips",
            Criterion::Top10Changes => "top10_changes",
            Criterion::RouteFlips => "route_flips",
            Criterion::Top1Displacement => "top1_mass_displaced",
            Criterion::TopKDisplacement => "top10_mass_displaced",
            Criterion::RouteDisplacement => "route_mixture_mass",
            Criterion::CoveredMass => "min_covered_mass",
            Criterion::ActivationAuthority => "activations",
            Criterion::PlanKlP99 => "plan_kl_p99",
            Criterion::PlanKlMean => "plan_kl_mean",
            Criterion::PlanTop1Disagreement => "plan_top1_disagreement",
            Criterion::PlanDeltaNllMean => "plan_delta_nll_mean",
        }
    }
}

/// The outcome of judging a bank by a gate.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityVerdict {
    pub gate_id: String,
    /// Empty means passed. Each entry names the criterion and what it
    /// saw against what it required.
    pub failures: Vec<(Criterion, String)>,
}

impl QualityVerdict {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn describe(&self) -> String {
        if self.passed() {
            return format!("passed {}", self.gate_id);
        }
        let f: Vec<String> = self
            .failures
            .iter()
            .map(|(c, d)| format!("{}: {d}", c.name()))
            .collect();
        format!("failed {} — {}", self.gate_id, f.join("; "))
    }
}

impl QualityGate {
    /// Whether the absence of a displacement distribution means
    /// "nothing changed" rather than "nothing was measured".
    ///
    /// A bank with zero route flips legitimately has no routing
    /// displacement to report, and must not fail for it. A bank that
    /// changed things and recorded no severity is a different matter.
    fn nothing_changed(bank: &QualityBank, criterion: Criterion) -> bool {
        match criterion {
            Criterion::Top1Displacement => bank.logits.top1_flips == 0,
            Criterion::TopKDisplacement => bank.logits.top10_changes == 0,
            Criterion::RouteDisplacement => bank.routing.route_flips == 0,
            _ => false,
        }
    }

    /// Judge a bank. Every criterion is checked, not short-circuited, so
    /// a report says everything that is wrong rather than the first
    /// thing.
    pub fn evaluate(&self, bank: &QualityBank) -> QualityVerdict {
        let mut failures = Vec::new();
        if bank.positions < self.positions_min {
            failures.push((
                Criterion::Positions,
                format!("{} < {} required", bank.positions, self.positions_min),
            ));
        }
        if bank.logits.kl_p99 > self.kl_p99_max {
            failures.push((
                Criterion::KlP99,
                format!("{:.3e} > {:.3e}", bank.logits.kl_p99, self.kl_p99_max),
            ));
        }
        for (limit, got, criterion) in [
            (
                self.top1_flip_max,
                bank.logits.top1_flips,
                Criterion::Top1Flips,
            ),
            (
                self.top10_change_max,
                bank.logits.top10_changes,
                Criterion::Top10Changes,
            ),
            (
                self.route_flip_max,
                bank.routing.route_flips,
                Criterion::RouteFlips,
            ),
        ] {
            if let Some(limit) = limit {
                if got > limit {
                    failures.push((criterion, format!("{got} > {limit}")));
                }
            }
        }
        // Consequence. A criterion the gate asks for and the bank did
        // not record FAILS: an unmeasured consequence is not a small
        // one, and defaulting it to zero is the unfalsifiable claim
        // this module exists to prevent.
        for (limit, observed, criterion, what) in [
            (
                self.top1_mass_displaced_max,
                bank.top1_mass_displaced.map(|d| d.max),
                Criterion::Top1Displacement,
                "top-1 probability given up",
            ),
            (
                self.top10_mass_displaced_p99_max,
                bank.top10_mass_displaced.map(|d| d.p99),
                Criterion::TopKDisplacement,
                "top-10 mass displaced at p99",
            ),
            (
                self.route_mixture_mass_p99_max,
                bank.routing.route_weight_mass_moved.map(|d| d.p99),
                Criterion::RouteDisplacement,
                "routed mixture moved at p99",
            ),
            (
                self.route_mixture_mass_max,
                bank.routing.route_weight_mass_moved.map(|d| d.max),
                Criterion::RouteDisplacement,
                "routed mixture moved at max",
            ),
        ] {
            let Some(limit) = limit else { continue };
            match observed {
                // Nothing changed at all, so nothing was displaced.
                None if Self::nothing_changed(bank, criterion) => {}
                None => failures.push((
                    criterion,
                    format!("{what} was not recorded, and this gate requires <= {limit:.4}"),
                )),
                Some(got) if got <= limit => {}
                Some(got) => failures.push((criterion, format!("{what} {got:.4} > {limit:.4}"))),
            }
        }
        if self.require_model_activations == Some(true) {
            if let Some(refusal) =
                super::activation::refuse_unless_model_execution(bank.activations.as_ref())
            {
                failures.push((Criterion::ActivationAuthority, refusal.to_string()));
            }
        }
        if let Some(required) = self.covered_mass_min {
            match bank.min_covered_mass {
                Some(got) if got >= required => {}
                Some(got) => failures.push((
                    Criterion::CoveredMass,
                    format!("{got:.4} < {required:.4} required"),
                )),
                None => failures.push((
                    Criterion::CoveredMass,
                    format!(
                        "not recorded, and this gate requires at least {required:.4} — a KL                          of unknown coverage is not evidence"
                    ),
                )),
            }
        }
        QualityVerdict {
            gate_id: self.id.clone(),
            failures,
        }
    }
}

#[cfg(test)]
#[path = "quality_tests.rs"]
mod tests;
