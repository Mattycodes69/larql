//! Quality evidence, and the implemented gates by id.

use crate::error::VindexError;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

/// A bank together with the gate it is judged by.
///
/// The verdict is NOT a field. Storing it would allow a record whose
/// stored verdict and stored numbers disagree, which is exactly the
/// unfalsifiable claim this module exists to prevent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityEvidence {
    pub gate: QualityGate,
    pub bank: QualityBank,
}

impl QualityEvidence {
    pub fn verdict(&self) -> QualityVerdict {
        self.gate.evaluate(&self.bank)
    }

    /// **The promotion report: what decided, and what merely happened.**
    ///
    /// The two are separated on purpose. A reader who sees "211 routing
    /// decisions changed" with no context assumes the worst; a reader
    /// who sees that number under DIAGNOSTICS, beside a measured
    /// consequence under AUTHORITY, can tell the divergence was found,
    /// weighed and judged. Hiding the counts would be worse than
    /// either — they are reported in full, they simply do not decide.
    pub fn report(&self) -> String {
        let verdict = self.verdict();
        let failed = |c: Criterion| verdict.failures.iter().any(|(k, _)| *k == c);
        let mut out = format!("QUALITY_GATE: {}\n\nAUTHORITY:\n", self.gate.id);
        // Each row states the MEASURED authority statistic against its
        // bound — the statistic the criterion is actually judged on,
        // never a neighbouring percentile. A bare PASS invites a later
        // reader to reconstruct the comparison from whatever number is
        // nearest to hand, and a p95 cannot establish a p99 bound.
        let against = |measured: Option<f64>, bound: Option<f64>, dir: &str| match (measured, bound)
        {
            (Some(m), Some(b)) => format!("{m:.3e} vs {dir} {b:.3e}"),
            (None, Some(b)) => format!("no changes vs {dir} {b:.3e}"),
            _ => String::new(),
        };
        let route = self.bank.routing.route_weight_mass_moved.as_ref();
        let route_detail = [
            self.gate
                .route_mixture_mass_p99_max
                .map(|b| format!("p99 {}", against(route.map(|d| d.p99), Some(b), "<="))),
            self.gate
                .route_mixture_mass_max
                .map(|b| format!("max {}", against(route.map(|d| d.max), Some(b), "<="))),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("; ");
        let count_detail = format!(
            "top1 {} top10 {} route {}",
            against(
                Some(self.bank.logits.top1_flips as f64),
                self.gate.top1_flip_max.map(|b| b as f64),
                "<=",
            ),
            against(
                Some(self.bank.logits.top10_changes as f64),
                self.gate.top10_change_max.map(|b| b as f64),
                "<=",
            ),
            against(
                Some(self.bank.routing.route_flips as f64),
                self.gate.route_flip_max.map(|b| b as f64),
                "<=",
            ),
        );
        let asked: [(&str, bool, Criterion, String); 7] = [
            (
                "positions",
                true,
                Criterion::Positions,
                format!("{} vs >= {}", self.bank.positions, self.gate.positions_min),
            ),
            (
                "kl_p99",
                true,
                Criterion::KlP99,
                format!(
                    "{:.3e} vs <= {:.3e}",
                    self.bank.logits.kl_p99, self.gate.kl_p99_max
                ),
            ),
            (
                "covered_mass",
                self.gate.covered_mass_min.is_some(),
                Criterion::CoveredMass,
                against(self.bank.min_covered_mass, self.gate.covered_mass_min, ">="),
            ),
            (
                "top1_mass_displaced",
                self.gate.top1_mass_displaced_max.is_some(),
                Criterion::Top1Displacement,
                format!(
                    "max {}",
                    against(
                        self.bank.top1_mass_displaced.as_ref().map(|d| d.max),
                        self.gate.top1_mass_displaced_max,
                        "<=",
                    )
                ),
            ),
            (
                "top10_mass_displaced",
                self.gate.top10_mass_displaced_p99_max.is_some(),
                Criterion::TopKDisplacement,
                format!(
                    "p99 {}",
                    against(
                        self.bank.top10_mass_displaced.as_ref().map(|d| d.p99),
                        self.gate.top10_mass_displaced_p99_max,
                        "<=",
                    )
                ),
            ),
            (
                "route_mass",
                self.gate.route_mixture_mass_p99_max.is_some()
                    || self.gate.route_mixture_mass_max.is_some(),
                Criterion::RouteDisplacement,
                route_detail,
            ),
            (
                "discrete_counts",
                self.gate.top1_flip_max.is_some(),
                Criterion::Top1Flips,
                count_detail,
            ),
        ];
        for (name, asked_for, criterion, detail) in asked {
            if !asked_for {
                continue;
            }
            out.push_str(&format!(
                "  {name:<22} {}   {detail}\n",
                if failed(criterion) { "FAIL" } else { "PASS" }
            ));
        }
        out.push_str(&format!(
            "\nDIAGNOSTICS (recorded, not authoritative):\n  \
             top1 flips             {}\n  top10 changes          {}\n  \
             route flips            {}\n  positions              {}\n",
            self.bank.logits.top1_flips,
            self.bank.logits.top10_changes,
            self.bank.routing.route_flips,
            self.bank.positions,
        ));
        if let Some(l) = self.bank.routing.first_layer_with_route_change {
            out.push_str(&format!("  first changed layer    {l}\n"));
        }
        out
    }

    /// The gate id this evidence passed, if it passed.
    pub fn proven_by(&self) -> Option<&str> {
        self.verdict().passed().then_some(self.gate.id.as_str())
    }

    /// Whether the logits moved while routing stayed put — the two
    /// mechanisms a MoE precision decision has to tell apart.
    pub fn is_arithmetic_only(&self) -> bool {
        self.bank.routing.route_flips == 0
    }
}

/// **`kimi-logit-v1` — the first acceptance contract for Kimi Linear.**
///
/// Named, and therefore FROZEN. If these thresholds turn out too strict
/// or too lax, the answer is `kimi-logit-v2`; editing this function
/// would silently re-date every claim that ever cited v1, which is the
/// one thing a versioned gate exists to prevent.
///
/// The values are a stated PRIOR, not a calibration — no candidate has
/// been through a bank yet, and guessing thresholds from a distribution
/// nobody has seen is how a bad contract gets frozen. They are written
/// as fractions of an 8192-position bank so the reasoning is inspectable:
///
/// | criterion | value | fraction | why |
/// |---|---|---|---|
/// | `positions_min` | 4096 | half the bank | a p99 needs a tail; 19 positions has none |
/// | `kl_p99_max` | 1e-3 nats | — | at the 99th percentile, well under where sampled text visibly diverges |
/// | `top1_flip_max` | 8 | 0.1 % | greedy decoding changes at all here, so it is the strictest of the four |
/// | `top10_change_max` | 82 | 1 % | reordering inside the top-10 is real but rarely decisive |
/// | `route_flip_max` | 82 | 1 % | a routing change is a decision change, held to the same 1 % |
///
/// The bar this must clear before it is used on anything: a NULL arm
/// (BF16 against itself) has to pass it with every count at zero. A gate
/// its own reference cannot satisfy is measuring the harness.
/// **Every gate this build implements, by name.**
///
/// A record names its gate; this resolves it. The alternative — a
/// caller choosing the gate and the record merely describing one — is
/// how `q2a_teacher_forced` came to evaluate `kimi-logit-v3` while the
/// optimiser record declared `kimi-logit-balanced-v1`: two authorities,
/// and the stored one was not the one that judged.
///
/// **No fallback.** An unknown id is refused, never resolved to
/// whichever gate this file happens to consider current. A gate decides
/// what is admissible; silently substituting one re-answers every
/// verdict drawn under the name that was asked for.
pub fn gate_by_id(id: &str) -> Result<QualityGate, VindexError> {
    let gate = match id {
        "kimi-logit-v1" => kimi_logit_v1(),
        "kimi-logit-v2" => kimi_logit_v2(),
        "kimi-logit-v3" => kimi_logit_v3(),
        "kimi-logit-balanced-v1" => kimi_logit_balanced_v1(),
        other => {
            return Err(VindexError::Parse(format!(
                "no gate named `{other}` is implemented by this build — a run judged under \
                 another gate would carry a verdict nobody asked for"
            )))
        }
    };
    debug_assert_eq!(
        gate.id, id,
        "a gate must answer to the name it is looked up by"
    );
    Ok(gate)
}

/// The gates [`gate_by_id`] resolves, so a test can assert the registry
/// and the constructors do not drift apart.
pub const IMPLEMENTED_GATES: [&str; 4] = [
    "kimi-logit-v1",
    "kimi-logit-v2",
    "kimi-logit-v3",
    "kimi-logit-balanced-v1",
];

pub fn kimi_logit_v1() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v1".into(),
        positions_min: 4096,
        kl_p99_max: 1e-3,
        top1_flip_max: Some(8),
        top10_change_max: Some(82),
        route_flip_max: Some(82),
        // v1 does not ask about coverage. See `kimi_logit_v2`.
        covered_mass_min: None,
        // Nor about the substrate. These gates predate
        // ACTIVATION-AUTHORITY-1; their banks were real, but the record
        // does not SAY so, and a gate is not permitted to assume it.
        // A new gate id is required to start judging it — changing a
        // threshold under an existing id is what the id exists to
        // prevent.
        require_model_activations: None,
        // Nor about consequence. See `kimi_logit_v3`.
        top1_mass_displaced_max: None,
        top10_mass_displaced_p99_max: None,
        route_mixture_mass_p99_max: None,
        route_mixture_mass_max: None,
    }
}

/// **`kimi-logit-v2` — v1 plus a bank-validity criterion.**
///
/// Same five thresholds, unchanged and deliberately so: this is not a
/// re-tuning, and a candidate's numbers mean exactly what they meant
/// under v1. What changes is that the KL must be a KL *of something* —
/// the bank has to say how much of the baseline distribution its
/// truncation covered, and that has to be most of it.
///
/// A new id rather than a field added to v1, because a criterion
/// changes what "passed this gate" MEANS. Every claim that ever cited
/// v1 was judged without a coverage requirement, and editing v1 in
/// place would silently re-date all of them as though they had met one.
///
/// `covered_mass_min` is **0.60**, and it is a floor on the WORST
/// position rather than the mean. Its justification is the measurement
/// that motivated it: on Kimi's teacher-forced bank the minimum is
/// driven by each sequence's FIRST position, which has no context and
/// is near-flat over 163,840 ids — top-128 covered 0.307 there and
/// top-2048 covered 0.729, while the p99 barely moved (8.425e-2 →
/// 7.984e-2). So 0.60 rejects the truncation that was demonstrably too
/// narrow, admits the one that was demonstrably wide enough, and does
/// not pretend a context-free position can be made sharp.
pub fn kimi_logit_v2() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v2".into(),
        covered_mass_min: Some(0.60),
        require_model_activations: None,
        ..kimi_logit_v1()
    }
}

/// **`kimi-logit-v3` — judged by CONSEQUENCE, not by discrete-change
/// counts.**
///
/// v1 and v2 asked whether a discrete boundary was crossed. Measurement
/// showed that question cannot separate the two events it most needs
/// to, at every level of the contract:
///
/// | count | what it actually was, measured at layer 26 |
/// |---|---|
/// | 6 argmax flips | median 0.1 % of probability given up, max 0.49 % |
/// | 232 top-10 changes | median 0.33 % of top-10 mass, one rank |
/// | 0 route flips | — |
///
/// while at layer 1 a single routing change could replace 36 % of the
/// routed mixture. Counting scores those identically.
///
/// So v3 keeps the counts as DIAGNOSTICS and drops them as authority,
/// and adds limits on how much actually moved. That is not a loosening:
/// it fails instantly on one large mixture replacement that v1 would
/// have accepted inside its 82-flip allowance, while no longer failing
/// on a thousand microscopic eighth-versus-ninth expert swaps.
///
/// **The thresholds are read off the measured distributions**, which is
/// the whole point of not writing this gate earlier:
///
/// | criterion | value | why that number |
/// |---|---|---|
/// | `kl_p99_max` | 1e-3 | unchanged from v1; layers 25-26 sit under it, 24 just over |
/// | `covered_mass_min` | 0.60 | v2's, unchanged |
/// | `top1_mass_displaced_max` | 0.05 | every late-band flip gave up <= 0.026; a flip surrendering a twentieth of the model's probability is a preference change, not a tie |
/// | `top10_mass_displaced_p99_max` | 0.10 | observed p99 0.022-0.042 across the band; max seen 0.105 |
/// | `route_mixture_mass_p99_max` | 0.15 | observed p99 0.094-0.105 in the late band |
/// | `route_mixture_mass_max` | 0.25 | late band caps at 0.163; layer 1 reaches 0.361 |
///
/// A criterion this gate asks for and the bank did not record FAILS,
/// unless nothing changed at all — an unmeasured consequence is not a
/// small one.
pub fn kimi_logit_v3() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v3".into(),
        top1_flip_max: None,
        top10_change_max: None,
        route_flip_max: None,
        top1_mass_displaced_max: Some(0.05),
        top10_mass_displaced_p99_max: Some(0.10),
        route_mixture_mass_p99_max: Some(0.15),
        route_mixture_mass_max: Some(0.25),
        ..kimi_logit_v2()
    }
}

/// **`kimi-logit-balanced-v1` — noticeable but bounded movement,
/// calibrated from a measured consequence ladder on TWO banks.**
///
/// Not a relaxation ratio over v3: every limit is drawn from the
/// empirical gap between the last candidate whose changes stayed
/// small and local and the first whose consequences changed character,
/// measured at 8,192 positions on the selection bank AND the held-out
/// bank (zero window overlap, never used to choose topology):
///
/// | anchor @ 8192 | kl p99 (sel/held) | worst top-1 give-up | verdict here |
/// |---|---|---|---|
/// | strict map (experts 24-26 + KDA 24,25, all Q8_0) | 5.99e-4 / — | 0.020 | PASS |
/// | wide map (+ KDA 21,22) | 9.65e-4 / 1.23e-3 | 0.055 / 0.028 | PASS |
/// | flagship (experts 20-26 + KDA 20-25 plateau) | 2.38e-3 / 2.60e-3 | 0.094 / 0.058 | PASS |
/// | B3 (experts 16-26) | 4.74e-3 / — | 0.181, 63 severe overturns | **FAIL** |
///
/// The ladder's ORDERING reproduced on the held-out bank with
/// magnitudes shifting ±30-50%, so each limit carries bank-to-bank
/// margin above the flagship's worst bank rather than sitting on one
/// measurement:
///
/// | criterion | limit | why |
/// |---|---|---|
/// | `kl_p99_max` | 3.5e-3 | flagship worst bank 2.60e-3 × drift margin; refuses B3's 4.74e-3 |
/// | `top1_mass_displaced_max` | 0.12 | flagship worst bank 0.094 × margin; refuses B3's 0.181 — the character change IS this dimension |
/// | `top10_mass_displaced_p99_max` | 0.12 | flagship 0.076 both banks + margin |
/// | route limits | unchanged from v3 | measured NON-discriminating in the corridor (p99 0.126-0.134 from strict to B3) — no evidence justifies loosening |
/// | `covered_mass_min` | 0.55 | the held-out bank's flattest position covers 0.577 at top-2048 — a property of that bank, not of any candidate; 0.60 would make the held-out bank unusable while 0.55 still refuses a blind instrument |
///
/// KL is bounded but is deliberately NOT the definition — B3 taught
/// that a diagnostic-benign map can hide dozens of confident overturns
/// that only authority scale reveals, and the wide map taught that a
/// single overturn's severity (0.055) can exceed strict's whole budget
/// while everything else stays local. The contract boundary is where
/// high-consequence top-1 overturns become materially larger and more
/// frequent, and it was chosen on the held-out bank: the selection
/// bank chose the corridor; the held-out bank chose the contract.
pub fn kimi_logit_balanced_v1() -> QualityGate {
    QualityGate {
        id: "kimi-logit-balanced-v1".into(),
        kl_p99_max: 3.5e-3,
        covered_mass_min: Some(0.55),
        require_model_activations: None,
        top1_mass_displaced_max: Some(0.12),
        top10_mass_displaced_p99_max: Some(0.12),
        ..kimi_logit_v3()
    }
}
