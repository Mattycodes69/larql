//! Priced records and Pareto-world fixtures.

use super::super::super::byte_ledger::{ByteLedger, ScopeBytes};
use super::super::super::compiler::read_source_identity;
use super::super::super::decision::SearchCandidate;
use super::super::super::diagnostic::{DiagnosticPolicy, DiagnosticVector};
use super::super::super::execution_cost::{ExecutionCostModel, ExecutionCostObservation};
use super::super::super::map::{Exception, PrecisionMap};
use super::super::super::measurement::{EvidenceScale, TailSupportPolicy};
use super::super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::super::participation::ParticipationDeclaration;
use super::super::super::policy::Role;
use super::super::super::promotion::PromotionCandidate;
use super::super::super::quality::{kimi_logit_balanced_v1, QualityGate};
use super::super::super::search_evidence::SearchCalibrationRegistry;
use super::super::accounting::read_source_storage;
use super::super::resolved::{layout_admission, PACK_LAYOUT_ADMISSION};
use std::collections::{BTreeMap, BTreeSet};

#[allow(unused_imports)]
use super::*;

impl PricedRecord {
    pub fn new(container: &std::path::Path) -> Self {
        Self {
            container: container.to_path_buf(),
            applied: BTreeSet::new(),
            layout: PACK_LAYOUT_ADMISSION.to_string(),
            accounting: "logical-bytes/v1".to_string(),
            shape: vec![64, 64],
            accounting_from: None,
            distinct_edits: false,
            protocol: None,
            intent: None,
            gate: kimi_logit_balanced_v1(),
        }
    }

    /// The applied set the record's next question is asked FROM.
    pub fn applied(mut self, applied: BTreeSet<String>) -> Self {
        self.applied = applied;
        self
    }

    /// The layout policy the record declares. Whatever is named here is
    /// what state resolution and price-table construction must both
    /// resolve to; nothing may fall back to this build's favourite.
    pub fn layout(mut self, declared: &str) -> Self {
        self.layout = declared.to_string();
        self
    }

    /// The physical accounting procedure the record declares.
    pub fn accounting(mut self, declared: &str) -> Self {
        self.accounting = declared.to_string();
        self
    }

    /// The surface's tensor shape — it decides whether the pack layout
    /// can hold the tensor, which is what makes the declared layout
    /// policy observable.
    pub fn shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    /// Read the accounting facts from ANOTHER container.
    pub fn accounting_from(mut self, other: &std::path::Path) -> Self {
        self.accounting_from = Some(other.to_path_buf());
        self
    }

    /// Give the two edits different resolutions, so the policy has more
    /// than one opportunity to order rather than two routes to one.
    pub fn distinct_edits(mut self) -> Self {
        self.distinct_edits = true;
        self
    }

    /// Carry the declarations the standing intent's digests stand for,
    /// AND search under them — the consistent case, and what a real
    /// record must always be.
    pub fn with_protocol(mut self, protocol: MeasurementProtocol) -> Self {
        self.intent = Some(protocol.intent(EvidenceScale::Authority));
        self.protocol = Some(protocol);
        self
    }

    /// Carry declarations while searching under the FIXTURE's intent —
    /// the inconsistent case, which a record must be refused for. It has
    /// no legitimate use outside a test that asserts the refusal, which
    /// is why it is a separate call rather than an argument.
    pub fn with_protocol_only(mut self, protocol: MeasurementProtocol) -> Self {
        self.protocol = Some(protocol);
        self
    }

    /// The behavioural contract the record's conclusions are drawn
    /// against. Supplied so a test can carry a gate whose id this build
    /// implements and whose thresholds have moved — the case a run must
    /// refuse rather than judge under whichever definition it compiled.
    pub fn gate(mut self, gate: QualityGate) -> Self {
        self.gate = gate;
        self
    }

    pub fn build(self) -> SearchSnapshot {
        let model = read_source_identity(&self.container).expect("identity");
        let facts_root = self
            .accounting_from
            .unwrap_or_else(|| self.container.clone());
        let facts_model = read_source_identity(&facts_root).expect("identity");
        let accounting = read_source_storage(&facts_root, &facts_model).expect("storage facts");
        let priced = read_source_storage(&self.container, &model).expect("storage facts");

        // The container's own tensors, at an NVFP4-admissible shape.
        let surface = TensorSurface::new(priced.tensors().map(|(id, _)| {
            SurfaceTensor::new(
                &id.object,
                &id.tensor,
                Role::DecoderLinear,
                self.shape.clone(),
            )
        }))
        .expect("one entry per stored tensor");

        // The role is in the map's domain and a blanket exception
        // protects it, so the base state presents source bytes
        // throughout. Each edit lifts that protection, which is the
        // direction the objective wants.
        let compile_everything = || Exception {
            projection: None,
            layers: None,
            encoding: Some(DTYPE_NVFP4.into()),
        };
        let base_map = PrecisionMap {
            name: "protect-everything".into(),
            encoding: DTYPE_NVFP4.into(),
            roles: vec!["decoder-linear".into()],
            exceptions: vec![Exception {
                projection: None,
                layers: None,
                encoding: None,
            }],
        };
        // Two edits that resolve identically: one physical state reached
        // by two routes, which is the case `MeasurementOpportunity`
        // exists for — two realizations, ONE experiment.
        let second = match self.distinct_edits {
            // Compiles only the q projections, so it resolves to a
            // different physical state and the policy has two
            // opportunities to order rather than two routes to one.
            true => Exception {
                projection: Some("q_proj".into()),
                layers: None,
                encoding: Some(DTYPE_NVFP4.into()),
            },
            false => compile_everything(),
        };
        let vocabulary = ActionVocabulary::new([
            MapEdit::new("compile-all", compile_everything()),
            MapEdit::new("compile-all-by-another-name", second),
        ])
        .expect("distinct names");

        // The root: the base map resolved, priced by the same procedure
        // the record declares. A writer may compute; the record stores
        // only the resulting node.
        let layout = layout_admission(PACK_LAYOUT_ADMISSION).expect("this build implements it");
        let root_state = RepresentationState::resolve(&model, &surface, &base_map, layout);
        let root_bytes = priced
            .bind(&model, &surface)
            .ok()
            .and_then(|bound| {
                SurfaceFootprint::new(
                    &bound,
                    &surface,
                    layout,
                    &PackCompiledBytes,
                    &[DTYPE_NVFP4.to_string()],
                )
                .ok()
                .map(|f| f.logical_bytes(&root_state))
            })
            .unwrap_or_else(|| {
                // A shape the pack cannot hold prices as source
                // throughout, which is what the base map presents anyway.
                LogicalBytes::new(priced.tensors().map(|(_, f)| f.logical_bytes.get()).sum())
            });
        let root = ResolvedState::new(root_state.clone(), root_bytes);

        SearchSnapshot::new(
            SearchSpace {
                surface,
                base_map,
                vocabulary,
                applied: self.applied,
            },
            SearchConfig {
                objective: Objective::MinimiseLogicalBytes,
                gate: Some(self.gate.into()),
                tail_support: TailSupportPolicy::route_cal_1(),
                calibrations: SearchCalibrationRegistry::default(),
                diagnostic_policy: DiagnosticPolicy::bs2_kimi_v1(),
                semantics: SearchSemantics::new(
                    "exchange-1-out-1-in/v1",
                    "ruling-1-three-prunes/v1",
                    "search-evidence-ladder/v1",
                    "kimi-balanced-v1-authority-only/v1",
                    "physical-prize-first/v1",
                    &self.accounting,
                    &self.layout,
                ),
                ranking: RankingSemantics::new(RankingRule::PhysicalPrizeFirst),
                standing_intent: self.intent.unwrap_or_else(standing_intent),
                protocol: self.protocol,
            },
            SearchFacts {
                graph: RepresentationStateGraph::new(
                    TransitionPolicy::StrictlyImprovingPhysical,
                    root,
                ),
                measurements: MeasurementRegistry::default(),
                byte_ledgers: BTreeMap::new(),
                execution_cost: ExecutionCostModel::new(Vec::new()),
                accounting: Some(accounting),
            },
        )
    }
}

// ------------------------------------------------- REPRESENT-PARETO-1

/// One ledger for a PARETO-1 world: every scope the decoder reads per
/// token, changed or not, so `fraction_removed` has a denominator.
pub fn pareto_ledger(name: &str, q: u64, k: u64) -> ByteLedger {
    let scope = |scope: &str, family: &str, baseline, candidate| ScopeBytes {
        scope: scope.into(),
        family: family.into(),
        baseline_bytes: baseline,
        candidate_bytes: candidate,
    };
    ByteLedger {
        model: "kimi-linear-48b".into(),
        baseline_representation: "BF16".into(),
        candidate_representation: name.into(),
        scopes: vec![
            scope("q", "attention", 16_384, q),
            scope("k", "attention", 16_384, k),
        ],
    }
}

/// One measured execution observation, shaped like a real one. A
/// FIXTURE: no GPU timing exists for these maps.
///
/// Its beta is the only slope this model has for this checkpoint —
/// `predict` keys on `model_identity` and then on nearest byte
/// fraction, never on codec or realization. PARETO-1 holds physical
/// facts constant across candidates in every world but C2 precisely so
/// its claim does not rest on that slope being right.
///
/// ```text
/// beta = ((10.0 - 8.0)/10.0) / ((32768 - 24576)/32768) = 0.2/0.25 = 0.8
/// ```
pub fn pareto_cost_model() -> ExecutionCostModel {
    ExecutionCostModel::new(vec![ExecutionCostObservation {
        id: "pareto-1-fixture-001".into(),
        machine: "fixture".into(),
        device: "fixture-gpu".into(),
        backend: "metal".into(),
        compiler_commit: "0000000".into(),
        model_identity: "kimi-linear-48b".into(),
        baseline_representation: "BF16".into(),
        candidate_representation: "fixture".into(),
        families_changed: vec!["attention".into()],
        scopes_changed: 1,
        baseline_bytes_per_token: 32_768,
        candidate_bytes_per_token: 24_576,
        baseline_gpu_ms_per_token: 10.0,
        candidate_gpu_ms_per_token: 8.0,
        fixed_overhead_ms: 1.0,
        benchmark_protocol: "fixture".into(),
        evidence: vec![],
    }])
}

/// The frozen accepted quality vectors, as `(kl_p99, route_flips)`.
/// Lower is better on both. Held in the same magnitude band as the P1
/// instantiation so no world changes the classification regime.
///
/// ONE definition, used by both the direct worlds and the OPT-6
/// integration, so the two layers can be asserted to run the SAME
/// experiment rather than two experiments that resemble each other.
pub const PARETO_BETTER: (f64, u64) = (3.4000e-3, 1200);
/// See [`PARETO_BETTER`].
pub const PARETO_WORSE: (f64, u64) = (3.9000e-3, 1600);
/// See [`PARETO_BETTER`]. Used where the two candidates must be
/// indistinguishable on quality.
pub const PARETO_MIDDLE: (f64, u64) = (3.6500e-3, 1400);
/// The PARENT baseline both layers measure against. It sits BELOW the
/// candidates on kl, so every move consumes kl headroom and classifies
/// `Priced` rather than `Unpriced`. A baseline worse than its children
/// makes every move free, which is a different experiment.
pub const PARETO_PARENT: (f64, u64) = (3.3532e-3, 1427);

/// **One PARETO-1 world — everything a world varies, and nothing else.**
///
/// The two candidates, their identities, the graph and its actions, the
/// parent reading, the cost model and ROUTE-CAL-1's registry are FIXED
/// across every world. A world differs only in the accepted
/// `(kl_p99, route_flips)` of each candidate and — in exactly one world
/// — the candidate byte ledgers that price them.
///
/// **The registry is why this fixture exists.** `config()` carries
/// `TailSupportPolicy::route_cal_1()` and
/// `SearchCalibrationRegistry::default()` — ROUTE-CAL-1's policy
/// without ROUTE-CAL-1's calibrations. `DiagnosticReading::evidence`
/// asks at a hardcoded `EvidenceScale::Diagnostic`, where an
/// unregistered statistic is `Unusable` with no `is_priceable`
/// fallback, so under the default registry NOTHING orders and every
/// comparison is vacuous. Carrying the real registry is what leaves
/// `KlP99` and `RouteFlipRate` — and only those two — able to rank.
pub struct ParetoWorld {
    pub(super) a_quality: (f64, u64),
    pub(super) b_quality: (f64, u64),
    pub(super) a_bytes: (u64, u64),
    pub(super) b_bytes: (u64, u64),
    pub(super) positions: u64,
}

impl ParetoWorld {
    /// Candidate A's identity, exactly as `promotion_candidates` labels
    /// it from `edge.action().label`.
    pub const A: &'static str = "\u{2212}M26 +K24";
    /// Candidate B's identity.
    pub const B: &'static str = "\u{2212}K25 +H";

    /// **Physical facts INERT across the pair.** Both ledgers remove
    /// 8,192 of 32,768 bytes, so `fraction_removed` is 0.25 on each,
    /// both predict `gpu_ms_saved = 2.0`, and stage 5 of
    /// `decide_promotion` cannot separate the candidates. Only the
    /// accepted quality values can.
    pub fn inert(a_quality: (f64, u64), b_quality: (f64, u64)) -> Self {
        Self {
            a_quality,
            b_quality,
            a_bytes: (8_192, 16_384),
            b_bytes: (16_384, 8_192),
            positions: 8192,
        }
    }

    /// **B removes half as much.** `fraction_removed` is 0.125 against
    /// A's 0.25, so the predicted gains are 1.0 against 2.0.
    ///
    /// The ONLY world where stage 5 is allowed to decide, and the only
    /// place this asymmetry may appear. Everywhere else a physical
    /// difference would make a quality result unattributable.
    pub fn physically_separated(a_quality: (f64, u64), b_quality: (f64, u64)) -> Self {
        Self {
            a_quality,
            b_quality,
            a_bytes: (8_192, 16_384),
            b_bytes: (16_384, 12_288),
            positions: 8192,
        }
    }

    /// Measure every bank at this corpus depth. Below the p99 support
    /// floor the candidates stay orderable but lose pricing, which is
    /// the whole point of the measurement ladder.
    pub fn at_depth(mut self, positions: u64) -> Self {
        self.positions = positions;
        self
    }

    pub fn snapshot(&self) -> SearchSnapshot {
        let mut graph =
            RepresentationStateGraph::new(TransitionPolicy::StrictlyImprovingPhysical, p());
        for (child, action, who) in [
            (
                t1(),
                Action::new(Self::A).removing(["M26"]).adding(["K24"]),
                "pareto-1/A",
            ),
            (
                s2(),
                Action::new(Self::B).removing(["K25"]).adding(["H"]),
                "pareto-1/B",
            ),
        ] {
            graph
                .apply(p().physical_id(), action, child, Provenance::new(who))
                .expect("both children are physically lighter than the parent");
        }

        let mut measurements = MeasurementRegistry::new();
        for (s, (kl, flips)) in [
            (p(), PARETO_PARENT),
            (t1(), self.a_quality),
            (s2(), self.b_quality),
        ] {
            let mut bank = authority_reading(kl, flips);
            bank.positions = self.positions;
            measurements
                .record_fixture(key_for(&s, EvidenceScale::Authority), bank)
                .expect("record");
        }

        let mut config = config();
        config.calibrations = SearchCalibrationRegistry::route_cal_1();

        let mut facts = facts(graph, measurements);
        facts.byte_ledgers = BTreeMap::from([
            (
                p().physical_id().clone(),
                pareto_ledger("P", 16_384, 16_384),
            ),
            (
                t1().physical_id().clone(),
                pareto_ledger("T1", self.a_bytes.0, self.a_bytes.1),
            ),
            (
                s2().physical_id().clone(),
                pareto_ledger("S2", self.b_bytes.0, self.b_bytes.1),
            ),
        ]);
        facts.execution_cost = pareto_cost_model();

        SearchSnapshot::new(space(), config, facts)
    }
}

/// **The P1 instantiation, pinned.** Its numbers are the ones P1 scored
/// and must keep producing: both candidates `Priced`, tier 2,
/// `gpu_ms_saved` exactly 2.0, equal across the pair.
pub fn pareto_p1_snapshot() -> SearchSnapshot {
    ParetoWorld::inert((3.6480e-3, 1570), (4.0563e-3, 1309)).snapshot()
}

/// The shared assessment and the policies the comparator reads.
///
/// One assessment, cloned into every synthetic candidate. FRONTIER
/// rungs require identical `MoveClass` and tier across a set — mixed
/// classes make stage 1 the thing being measured — so sharing it makes
/// that structural rather than asserted-and-hoped.
/// The template at a chosen measurement depth. Below the p99 support
/// floor the assessment classifies `Unscorable`; at or above it,
/// `Priced`. That difference is what an escalation buys, and what the
/// spend ladder is made of.
pub fn pareto_candidate_template_at(
    positions: u64,
) -> (
    PromotionCandidate,
    SearchCalibrationRegistry,
    TailSupportPolicy,
    DiagnosticPolicy,
) {
    let snap = ParetoWorld::inert(PARETO_BETTER, PARETO_WORSE)
        .at_depth(positions)
        .snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");
    let config = snap.config();
    (
        candidates[0].promotion.clone(),
        config.calibrations.clone(),
        config.tail_support.clone(),
        config.diagnostic_policy.clone(),
    )
}

pub fn pareto_candidate_template() -> (
    PromotionCandidate,
    SearchCalibrationRegistry,
    TailSupportPolicy,
    DiagnosticPolicy,
) {
    let snap = pareto_p1_snapshot();
    let candidates = snap
        .promotion_candidates(EvidenceScale::Authority)
        .expect("the cost model covers this model");
    let config = snap.config();
    (
        candidates[0].promotion.clone(),
        config.calibrations.clone(),
        config.tail_support.clone(),
        config.diagnostic_policy.clone(),
    )
}

/// One synthetic candidate carrying a chosen `(kl_p99, route_flips)`.
/// Everything except the diagnostic vector comes from the template.
pub fn pareto_candidate(
    id: &str,
    promotion: &PromotionCandidate,
    policy: &DiagnosticPolicy,
    kl: f64,
    route_flips: u64,
) -> SearchCandidate {
    SearchCandidate {
        id: id.to_string(),
        promotion: promotion.clone(),
        diagnostic: DiagnosticVector::of(policy, &authority_reading(kl, route_flips)),
        participation: ParticipationDeclaration::all_affected(),
    }
}
