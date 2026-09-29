//! Gates 2–9 over a real container: the glimmer fixture's draft and
//! target decoder stacks share tensor names, so one `(projection, layer)`
//! group spans both objects, exactly as the map grammar addresses them.

use std::collections::BTreeMap;

use super::super::super::compiler::read_source_identity;
use super::super::super::map::Exception;
use super::super::super::nvfp4_pack::DTYPE_NVFP4;
use super::super::super::policy::{layer_of, Role};
use super::super::accounting::read_source_storage;
use super::super::action_space::{ActionVocabulary, MapEdit};
use super::super::footprint::PackCompiledBytes;
use super::super::resolved::{PackLayoutAdmission, PACK_LAYOUT_ADMISSION};
use super::super::surface::SurfaceTensor;
use super::super::tests::container;
use super::*;

/// The group whose shape no NVFP4 pack admits (k = 8 is not a multiple
/// of the 16-wide group).
const REFUSED: (&str, u32) = ("q_proj", 2);

struct Fixture {
    _dir: tempfile::TempDir,
    model: SourceIdentity,
    surface: TensorSurface,
    footprint: SurfaceFootprint,
    base: PrecisionMap,
}

fn is_linear(tensor: &str) -> bool {
    tensor.ends_with("_proj.weight") || tensor == "fc.weight"
}

/// A `[numel / 64, 64]` shape carrying the stored bytes (BF16), so
/// larger tensors cost more; the refused group gets k = 8.
fn shape(tensor: &str, bf16_bytes: u64) -> Vec<usize> {
    let numel = (bf16_bytes / 2) as usize;
    let refused = projection_of(tensor) == Some(REFUSED.0) && layer_of(tensor) == Some(REFUSED.1);
    if refused {
        vec![numel / 8, 8]
    } else {
        vec![numel / 64, 64]
    }
}

fn fixture() -> Fixture {
    let dir = container::glimmer();
    let model = read_source_identity(dir.path()).expect("identity");
    let facts = read_source_storage(dir.path(), &model).expect("facts");
    let surface = TensorSurface::new(facts.tensors().map(|(id, fact)| {
        let role = if is_linear(&id.tensor) {
            Role::DecoderLinear
        } else {
            Role::Norm
        };
        SurfaceTensor::new(
            &id.object,
            &id.tensor,
            role,
            shape(&id.tensor, fact.logical_bytes.get()),
        )
    }))
    .expect("one entry per tensor");
    let bound = facts.bind(&model, &surface).expect("bound");
    let footprint = SurfaceFootprint::new(
        &bound,
        &surface,
        &PackLayoutAdmission,
        &PackCompiledBytes,
        &[DTYPE_NVFP4.to_string()],
    )
    .expect("priced");
    Fixture {
        _dir: dir,
        model,
        surface,
        footprint,
        base: PrecisionMap {
            name: "base".into(),
            encoding: DTYPE_NVFP4.into(),
            roles: vec!["decoder-linear".into()],
            exceptions: vec![],
        },
    }
}

impl Fixture {
    fn inputs(&self) -> ProposalInputs<'_> {
        ProposalInputs {
            model: &self.model,
            surface: &self.surface,
            base: &self.base,
            layout: &PackLayoutAdmission,
            layout_id: PACK_LAYOUT_ADMISSION,
            grouping: Grouping::PerProjectionLayer,
            footprint: &self.footprint,
            ceiling: None,
            pins: BTreeMap::new(),
            cuts: vec![],
            prior: None,
        }
    }

    fn propose(&self, inputs: ProposalInputs<'_>, k: usize) -> Proposals {
        let problem = ProposalProblem::new(inputs).expect("problem");
        BranchAndBound {
            node_limit: u64::MAX,
        }
        .propose(&problem, k)
        .expect("proposals")
    }
}

fn states(p: &Proposals) -> Vec<RepresentationStateId> {
    p.proposals.iter().map(|p| p.state.id().clone()).collect()
}

#[test]
fn every_proposal_presents_the_bytes_the_footprint_prices() {
    let f = fixture();
    let out = f.propose(f.inputs(), 8);
    assert_eq!(out.proposals.len(), 8);
    assert_eq!(out.outcome, SearchOutcome::Complete);
    for (i, p) in out.proposals.iter().enumerate() {
        let priced = f
            .footprint
            .try_logical_bytes(&p.state)
            .expect("same surface");
        assert_eq!(p.record.bytes, priced.get(), "rank {}", i + 1);
        assert_eq!(p.record.rank, i + 1);
        assert_eq!(&p.record.state, p.state.id());
        let resolved =
            RepresentationState::resolve(&f.model, &f.surface, &p.map, &PackLayoutAdmission);
        assert_eq!(
            resolved.id(),
            p.state.id(),
            "the record names the map's state"
        );
    }
    assert!(out
        .proposals
        .windows(2)
        .all(|w| w[0].record.bytes <= w[1].record.bytes));
    // Compiling every eligible group is cheapest; the refused group is
    // held at source because compiling it is structurally meaningless.
    assert_eq!(
        out.proposals[0].record.protected,
        vec![GroupKey::new(REFUSED.0, REFUSED.1)]
    );
    assert_eq!(
        out.lower_bound.map(LogicalBytes::get),
        Some(out.proposals[0].record.bytes)
    );
    // A group spans objects: draft and target both hold `0.mlp.down_proj`.
    let problem = ProposalProblem::new(f.inputs()).unwrap();
    assert!(problem
        .groups()
        .any(|g| g == &GroupKey::new("down_proj", 0)));
    assert!(problem.fixed().contains(&(
        "draft.feature_projector".to_string(),
        "fc.weight".to_string()
    )));
}

#[test]
fn every_proposal_passes_the_map_check_and_a_dead_rule_is_refused() {
    let f = fixture();
    let out = f.propose(f.inputs(), 8);
    for p in &out.proposals {
        p.map
            .check_against(
                f.surface
                    .entries()
                    .iter()
                    .map(|t| (t.role, t.tensor.as_str())),
            )
            .expect("a synthesised map decides every exception");
    }
    let problem = ProposalProblem::new(f.inputs()).unwrap();
    let err = problem
        .synthesize(&[GroupKey::new("q_proj", 99)], "dead".into())
        .unwrap_err()
        .to_string();
    assert!(err.contains("check refuses"), "{err}");
}

#[test]
fn an_exact_no_good_removes_that_state_and_no_other() {
    let f = fixture();
    let before = f.propose(f.inputs(), 4);
    let cut = before.proposals[0].state.id().clone();
    let mut inputs = f.inputs();
    inputs.cuts = vec![Cut::ExactNoGood { state: cut.clone() }];
    let after = f.propose(inputs, 3);
    assert_eq!(states(&after), states(&before)[1..].to_vec());
    assert!(!states(&after).contains(&cut));
    assert!(after.proposals[0]
        .record
        .cuts
        .contains(&Cut::ExactNoGood { state: cut }.id()));
    // The states that differ from the cut state in one group survive:
    // the cut is not widened into a claim about any one choice.
    let cut_protected = &before.proposals[0].record.protected;
    for p in &after.proposals {
        let differ = p
            .record
            .protected
            .iter()
            .filter(|g| !cut_protected.contains(g))
            .count()
            + cut_protected
                .iter()
                .filter(|g| !p.record.protected.contains(g))
                .count();
        assert!(differ >= 1);
    }
    assert_eq!(
        after.proposals[0].record.protected.len(),
        cut_protected.len() + 1
    );
}

#[test]
fn a_structural_cut_never_appears_and_is_recorded() {
    let f = fixture();
    let group = GroupKey::new("o_proj", 1);
    let mut inputs = f.inputs();
    inputs.cuts = vec![Cut::Structural {
        group: group.clone(),
        choice: GroupChoice::Compile,
        reason: "no kernel".into(),
    }];
    let out = f.propose(inputs, 5);
    for p in &out.proposals {
        assert!(p.record.protected.contains(&group), "Compile was cut");
        assert!(p
            .record
            .cuts
            .contains(&"structural:o_proj@1:compile:no kernel".to_string()));
        assert!(p
            .record
            .cuts
            .iter()
            .any(|c| c.starts_with("structural:q_proj@2:compile:layout refuses")));
    }
}

#[test]
fn a_truncated_search_says_so_and_identical_inputs_give_identical_records() {
    let f = fixture();
    let complete = f.propose(f.inputs(), 3);
    let problem = ProposalProblem::new(f.inputs()).unwrap();
    let truncated = BranchAndBound { node_limit: 5 }
        .propose(&problem, 3)
        .unwrap();
    assert!(matches!(truncated.outcome, SearchOutcome::NodeLimit { .. }));
    let bound = truncated.lower_bound.expect("bounded");
    assert!(bound.get() <= complete.proposals[0].record.bytes);

    let again = f.propose(f.inputs(), 3);
    let records = |p: &Proposals| {
        p.proposals
            .iter()
            .map(|p| serde_json::to_string(&p.record).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(records(&complete), records(&again));
    assert_eq!(
        complete.proposals[0].record.solver_revision,
        SOLVER_REVISION
    );
}

struct Prefers {
    projection: &'static str,
    evidence: SearchEvidence,
}

impl OrderingPrior for Prefers {
    fn evidence(&self) -> SearchEvidence {
        self.evidence.clone()
    }
    fn identity(&self) -> String {
        format!("prefers-protecting-{}", self.projection)
    }
    fn score(&self, group: &GroupKey, choice: GroupChoice) -> f64 {
        if group.tensor().map(|(p, _)| p) == Some(self.projection) && choice == GroupChoice::Source
        {
            -1.0
        } else {
            0.0
        }
    }
}

#[test]
fn a_prior_reorders_only_equal_byte_states_and_unusable_is_refused() {
    let f = fixture();
    let plain = f.propose(f.inputs(), 12);
    let prior = Prefers {
        projection: "k_proj",
        evidence: SearchEvidence::OrderingProxy {
            calibration: "test".into(),
        },
    };
    let mut inputs = f.inputs();
    inputs.prior = Some(&prior);
    let ordered = f.propose(inputs, 12);
    let bytes = |p: &Proposals| {
        p.proposals
            .iter()
            .map(|p| p.record.bytes)
            .collect::<Vec<_>>()
    };
    assert_eq!(bytes(&plain), bytes(&ordered), "the byte order never moves");
    assert_ne!(
        states(&plain),
        states(&ordered),
        "a tie was broken differently"
    );
    // Within one byte level the preferred protection comes first; without
    // the prior the canonical assignment order puts `v_proj` first.
    let first_tie = |p: &Proposals| {
        let i = p
            .proposals
            .windows(2)
            .position(|w| w[0].record.bytes == w[1].record.bytes)
            .expect("k_proj and v_proj are the same size, so ties exist");
        p.proposals[i].record.protected.clone()
    };
    assert!(first_tie(&plain)
        .iter()
        .any(|g| g.tensor().map(|(p, _)| p) == Some("v_proj")));
    assert!(first_tie(&ordered)
        .iter()
        .any(|g| g.tensor().map(|(p, _)| p) == Some("k_proj")));
    assert_eq!(
        ordered.proposals[0]
            .record
            .prior
            .as_ref()
            .map(|p| p.identity.as_str()),
        Some("prefers-protecting-k_proj")
    );

    let unusable = Prefers {
        projection: "v_proj",
        evidence: SearchEvidence::Unusable,
    };
    let mut inputs = f.inputs();
    inputs.prior = Some(&unusable);
    assert!(ProposalProblem::new(inputs).is_err());
}

#[test]
fn malformed_problems_are_refused() {
    let f = fixture();
    let mut with_exception = f.base.clone();
    with_exception
        .exceptions
        .push(super::super::super::map::Exception {
            projection: Some("q_proj".into()),
            layers: None,
            encoding: None,
        });
    let mut inputs = f.inputs();
    inputs.base = &with_exception;
    assert!(ProposalProblem::new(inputs).is_err());

    let mut inputs = f.inputs();
    inputs.pins = BTreeMap::from([(GroupKey::new("q_proj", 99), GroupChoice::Source)]);
    assert!(ProposalProblem::new(inputs).is_err());

    let mut inputs = f.inputs();
    inputs.cuts = vec![Cut::Structural {
        group: GroupKey::new("nope", 0),
        choice: GroupChoice::Source,
        reason: "x".into(),
    }];
    assert!(ProposalProblem::new(inputs).is_err());

    // A ceiling below the cheapest state leaves nothing feasible.
    let mut inputs = f.inputs();
    inputs.ceiling = Some(LogicalBytes::new(1));
    let out = f.propose(inputs, 3);
    assert!(out.proposals.is_empty());
    assert_eq!(out.lower_bound, None);
}

#[test]
fn a_proposals_protections_compile_to_the_same_state() {
    let f = fixture();
    let mut inputs = f.inputs();
    inputs.pins = BTreeMap::from([
        (GroupKey::new("up_proj", 0), GroupChoice::Source),
        (GroupKey::new("up_proj", 1), GroupChoice::Source),
        (GroupKey::new("up_proj", 3), GroupChoice::Source),
    ]);
    let out = f.propose(inputs, 4);
    for p in &out.proposals {
        // What `compile_representation` would be handed: the base map's
        // encoding and roles with these protections, as `from_policy`
        // writes them.
        let compiled = PrecisionMap {
            exceptions: p.protections.as_exceptions(),
            ..f.base.clone()
        };
        let state =
            RepresentationState::resolve(&f.model, &f.surface, &compiled, &PackLayoutAdmission);
        assert_eq!(state.id(), p.state.id());
    }
    // Consecutive layers of one projection merge into one rule.
    let top = &out.proposals[0].map.exceptions;
    assert!(top
        .iter()
        .any(|e| e.projection.as_deref() == Some("up_proj") && e.layers == Some((0, 1))));
    assert!(top
        .iter()
        .any(|e| e.projection.as_deref() == Some("up_proj") && e.layers == Some((3, 3))));
}

/// One declared group per projection, over every layer it has: a partition
/// of the eligible tensors derived from the fixture's own surface.
fn per_projection_vocabulary(f: &Fixture) -> ActionVocabulary {
    let mut spans: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for key in group_keys(&f.surface, &f.base) {
        let (projection, layer) = key.tensor().expect("1a keys are per tensor");
        let span = spans
            .entry(projection.to_string())
            .or_insert((layer, layer));
        *span = (span.0.min(layer), span.1.max(layer));
    }
    ActionVocabulary::new(spans.into_iter().map(|(projection, (lo, hi))| {
        MapEdit::new(
            format!("all-{projection}"),
            Exception {
                projection: Some(projection),
                layers: Some((lo, hi)),
                encoding: None,
            },
        )
    }))
    .unwrap()
}

/// Every state a declared problem can reach, priced, cheapest first: the
/// exhaustive answer the solver's K best must reproduce.
fn exhaustive_bytes(f: &Fixture, problem: &ProposalProblem<'_>) -> Vec<u64> {
    let groups: Vec<GroupKey> = problem.groups().cloned().collect();
    let mut priced: BTreeMap<RepresentationStateId, u64> = BTreeMap::new();
    for mask in 0u32..(1 << groups.len()) {
        let protected: Vec<GroupKey> = groups
            .iter()
            .enumerate()
            .filter(|(i, _)| mask >> i & 1 == 1)
            .map(|(_, g)| g.clone())
            .collect();
        let (map, _) = problem.synthesize(&protected, "x".into()).unwrap();
        let state = problem.state_of(&map);
        let bytes = f.footprint.try_logical_bytes(&state).unwrap().get();
        priced.insert(state.id().clone(), bytes);
    }
    let mut bytes: Vec<u64> = priced.into_values().collect();
    bytes.sort();
    bytes
}

#[test]
fn a_declared_vocabulary_searches_its_edits_and_meets_every_1a_gate() {
    let f = fixture();
    let vocabulary = per_projection_vocabulary(&f);
    let mut inputs = f.inputs();
    inputs.grouping = Grouping::Declared(&vocabulary);
    let problem = ProposalProblem::new(inputs).expect("a partition");
    let names: Vec<String> = problem.groups().map(GroupKey::describe).collect();
    let declared: Vec<String> = vocabulary.names().map(str::to_string).collect();
    assert_eq!(names, declared, "one variable per declared edit");
    let expected = exhaustive_bytes(&f, &problem);
    let k = expected.len().min(8);
    let out = BranchAndBound {
        node_limit: u64::MAX,
    }
    .propose(&problem, k)
    .unwrap();
    // Gate 1: the K best equal exhaustive enumeration over the declared space.
    let found: Vec<u64> = out.proposals.iter().map(|p| p.record.bytes).collect();
    assert_eq!(found, expected[..k]);
    for p in &out.proposals {
        // Gate 2: the bytes the footprint prices.
        let priced = f.footprint.try_logical_bytes(&p.state).unwrap().get();
        assert_eq!(p.record.bytes, priced);
        // Gate 3: the map check.
        p.map
            .check_against(
                f.surface
                    .entries()
                    .iter()
                    .map(|t| (t.role, t.tensor.as_str())),
            )
            .expect("a synthesised map decides every exception");
        // Gate 4: what compile_representation is handed resolves to the
        // proposed state, and so does the vocabulary's own map for the
        // applied set a measurement request is built from.
        let compiled = PrecisionMap {
            exceptions: p.protections.as_exceptions(),
            ..f.base.clone()
        };
        let state =
            RepresentationState::resolve(&f.model, &f.surface, &compiled, &PackLayoutAdmission);
        assert_eq!(state.id(), p.state.id());
        let applied = p.record.protected.iter().map(GroupKey::describe).collect();
        let from_vocabulary = vocabulary.map_for(&f.base, &applied).unwrap();
        let state = RepresentationState::resolve(
            &f.model,
            &f.surface,
            &from_vocabulary,
            &PackLayoutAdmission,
        );
        assert_eq!(state.id(), p.state.id());
    }
    // The declared rules are part of the problem's identity.
    assert_ne!(problem.id(), ProposalProblem::new(f.inputs()).unwrap().id());
}

#[test]
fn a_vocabulary_that_is_not_a_partition_of_source_rules_is_refused() {
    let f = fixture();
    let good = per_projection_vocabulary(&f);
    let refuse = |vocabulary: &ActionVocabulary| {
        let mut inputs = f.inputs();
        inputs.grouping = Grouping::Declared(vocabulary);
        let from_problem = ProposalProblem::new(inputs)
            .err()
            .expect("refused")
            .to_string();
        let from_check = check_declared(&f.surface, &f.base, vocabulary)
            .unwrap_err()
            .to_string();
        assert_eq!(
            from_problem, from_check,
            "one definition of a valid vocabulary"
        );
        from_problem
    };
    let with = |extra: MapEdit| {
        ActionVocabulary::new(good.edits().iter().cloned().chain([extra])).unwrap()
    };
    let rule = |projection: &str, layers, encoding: Option<&str>| Exception {
        projection: Some(projection.into()),
        layers,
        encoding: encoding.map(str::to_string),
    };
    let first = good.edits()[0].exceptions()[0].clone();
    let projection = first.projection.clone().unwrap();
    let layer = first.layers.unwrap().0;

    let overlap = with(MapEdit::new(
        "again",
        rule(&projection, Some((layer, layer)), None),
    ));
    assert!(refuse(&overlap).contains("must not overlap"));

    let gap = ActionVocabulary::new(good.edits()[1..].iter().cloned()).unwrap();
    assert!(refuse(&gap).contains("must cover every eligible tensor"));

    let dead = with(MapEdit::new(
        "nothing",
        rule("absent_proj", Some((0, 1)), None),
    ));
    assert!(refuse(&dead).contains("dead rule"));

    for bad in [
        rule("absent_proj", None, None),
        rule("absent_proj", Some((3, 1)), None),
        rule("absent_proj", Some((0, 1)), Some(DTYPE_NVFP4)),
        Exception {
            projection: None,
            layers: Some((0, 1)),
            encoding: None,
        },
    ] {
        assert!(refuse(&with(MapEdit::new("bad", bad))).contains("a group rule names"));
    }
    check_declared(&f.surface, &f.base, &good).expect("the partition is valid");
}

#[test]
fn a_per_tensor_key_keeps_the_form_records_were_written_in() {
    let key = GroupKey::new("q_proj", 3);
    let written = serde_json::to_value(&key).unwrap();
    assert_eq!(
        written,
        serde_json::json!({"projection": "q_proj", "layer": 3})
    );
    assert_eq!(serde_json::from_value::<GroupKey>(written).unwrap(), key);
    let declared = GroupKey::declared("attn-qkv-q1");
    let written = serde_json::to_value(&declared).unwrap();
    assert_eq!(written, serde_json::json!({"name": "attn-qkv-q1"}));
    assert_eq!(
        serde_json::from_value::<GroupKey>(written).unwrap(),
        declared
    );
    assert_eq!(declared.tensor(), None);
    assert_eq!(key.describe(), "q_proj@3");
}
