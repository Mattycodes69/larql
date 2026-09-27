//! Graph-based fixture: real encoded/compiled bytes, no K3 or manifest-only authority.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use super::super::{
    actuate::{executor::Observed, prepare::PreparedExperiment},
    compile_representation,
    compiler::read_source_identity,
    map::{Exception, PrecisionMap},
    measure::outcome::VerifiedFacts,
    measurement::EvidenceScale,
    policy::classify_in,
    state::{
        self, fixtures,
        snapshot::{SearchSnapshot, SearchSpace},
        ActionVocabulary, LogicalBytes, MapEdit, PackLayoutAdmission, RepresentationState,
        RepresentationStateGraph, ResolvedState, SurfaceTensor, TensorSurface, TransitionPolicy,
    },
    RepresentSpec,
};
use super::artifact::MeasurementArtifact;
use super::state_evidence::{ArtifactStateEvidence, EstablishedState};
use super::*;
use crate::format::vindex3::{
    encode::segment::read_segment_header,
    fixtures::{dense_f32_model, encode_fixture_container},
    index::Vindex3Index,
};

pub(super) struct Fixture {
    pub dir: tempfile::TempDir,
    pub source: PathBuf,
    pub candidate: PathBuf,
    pub corpus: PathBuf,
    pub snapshot: SearchSnapshot,
    pub prepared: PreparedExperiment,
    /// The corpus depth this fixture was built at. Every observation it
    /// seals reports it, and the bank and `VerifiedFacts` declare it.
    pub positions: u64,
}

impl Fixture {
    /// The default fixture: 8 positions, as every caller before
    /// PARETO-1 expects. Unchanged, and deliberately so — LOOP-1 is a
    /// closed milestone and its fixture must stay byte-for-byte
    /// behaviourally identical.
    pub fn new() -> Self {
        Self::with_positions(8)
    }

    /// A fixture at a chosen corpus depth.
    ///
    /// Depth is not a free parameter. `TailSupportPolicy::route_cal_1`
    /// needs `5.0 / (1 - 0.99)` = 500 observations to support a p99, so
    /// below 500 positions every p99 criterion is unpriceable at
    /// authority scale and a positive-gain move classifies
    /// `Unscorable`. PARETO-1's priced witness runs at exactly 500 and
    /// its negative control at 499.
    pub fn with_positions(positions: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("checkpoint");
        std::fs::create_dir(&checkpoint).unwrap();
        let source = dir.path().join("source");
        encode_fixture_container(dense_f32_model, &checkpoint, &source, "target");
        let index: Vindex3Index =
            serde_json::from_slice(&std::fs::read(source.join("index.json")).unwrap()).unwrap();
        let mut entries = BTreeMap::new();
        for rep in index.representations.values() {
            let (header, _) = read_segment_header(&source.join(&rep.segment)).unwrap();
            for tensor in header.tensors {
                let role = classify_in(true, &rep.object, &tensor.name, &tensor.shape);
                entries.insert(
                    (rep.object.clone(), tensor.name.clone()),
                    SurfaceTensor::new(&rep.object, &tensor.name, role, tensor.shape),
                );
            }
        }
        let surface = TensorSurface::new(entries.into_values()).unwrap();
        let mut spec = RepresentSpec::nvfp4();
        spec.protect = spec.protect.layers(1, 1);
        let mut base_map =
            PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);
        base_map.exceptions.push(Exception {
            projection: None,
            layers: None,
            encoding: None,
        });
        let vocabulary = ActionVocabulary::new([
            MapEdit::new(
                "compile-all",
                Exception {
                    projection: None,
                    layers: Some((0, 0)),
                    encoding: Some(spec.encoding.clone()),
                },
            ),
            MapEdit::new(
                "compile-q",
                Exception {
                    projection: Some("q_proj".into()),
                    layers: Some((0, 0)),
                    encoding: Some(spec.encoding.clone()),
                },
            ),
        ])
        .unwrap();
        let corpus = dir.path().join("corpus");
        std::fs::create_dir(&corpus).unwrap();
        let rows = vec![0u8; positions as usize * 64 * 4];
        std::fs::write(corpus.join("seq_0.f32"), &rows).unwrap();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "sequences":1,"positions":positions,"hidden":64,
            "token_ids":[vec![0u32; positions as usize]],"regime":"teacher-forced",
            "payload_authority":"teacher-forced-bank-payload/v1",
            "payloads":{"seq_0.f32":{"len":rows.len(),"sha256":super::super::compile::hash_bytes(&rows)}}
        })).unwrap();
        std::fs::write(corpus.join("manifest.json"), &manifest).unwrap();
        let mut protocol = fixtures::protocol();
        protocol.bank = state::EvidenceBank::new(
            "kimi-teacher-forced/v1",
            super::super::compile::hash_bytes(&manifest),
            ["seq-000"],
            positions as u32,
        );
        let seed = fixtures::PricedRecord::new(&source)
            .with_protocol(protocol)
            .build();
        let model = read_source_identity(&source).unwrap();
        let root = RepresentationState::resolve(&model, &surface, &base_map, &PackLayoutAdmission);
        let mut facts = seed.facts().clone();
        let source_bytes = facts
            .accounting
            .as_ref()
            .unwrap()
            .tensors()
            .map(|(_, fact)| fact.logical_bytes.get())
            .sum();
        facts.graph = RepresentationStateGraph::new(
            TransitionPolicy::StrictlyImprovingPhysical,
            ResolvedState::new(root, LogicalBytes::new(source_bytes)),
        );
        let snapshot = SearchSnapshot::new(
            SearchSpace {
                surface,
                base_map,
                vocabulary,
                applied: BTreeSet::new(),
            },
            seed.config().clone(),
            facts,
        );
        let prepared = PreparedExperiment::of(&snapshot);
        assert!(prepared.is_ready(), "{prepared:?}");
        let candidate = dir.path().join("candidate");
        drop(compile_representation(&source, &candidate, &spec).unwrap());
        let actual = ArtifactStateEvidence::establish(&candidate).unwrap();
        assert_eq!(
            actual.established(),
            prepared.request().unwrap().key().state()
        );
        let compiled: Vec<_> = actual
            .state()
            .decisions()
            .decisions()
            .iter()
            .filter(|d| matches!(d.encoding, state::ResolvedEncoding::Compiled(_)))
            .collect();
        assert_eq!(compiled.len(), 7);
        assert!(compiled.iter().all(|d| d.tensor.starts_with("0.")));
        Self {
            dir,
            source,
            candidate,
            corpus,
            snapshot,
            prepared,
            positions,
        }
    }
    pub fn evidence(&self) -> EstablishedState {
        ArtifactStateEvidence::establish(&self.candidate).unwrap()
    }
    pub fn observed(&self, kl: f64) -> Observed {
        self.observed_with(kl, 0)
    }
    /// The same observation with the ROUTE FLIP count settable.
    ///
    /// `observed` hardcoded zero, which left `Statistic::RouteFlipRate`
    /// constant across every candidate and therefore unable to order
    /// anything — one of the two statistics ROUTE-CAL-1 registers as an
    /// ordering proxy. A caller that needs two independent orderable
    /// dimensions needs this one.
    pub fn observed_with(&self, kl: f64, route_flips: u64) -> Observed {
        let mut observation = fixtures::authority_reading(kl, route_flips);
        observation.positions = self.positions;
        observation.routing.route_weight_mass_moved = None;
        observation.top10_mass_displaced = None;
        observation.top1_mass_displaced = None;
        Observed {
            key: self.prepared.request().unwrap().key().clone(),
            observation: observation.into(),
            verified: VerifiedFacts {
                compiled_layers: vec![0],
                compiled_projections: [
                    "down_proj",
                    "gate_proj",
                    "up_proj",
                    "k_proj",
                    "o_proj",
                    "q_proj",
                    "v_proj",
                ]
                .map(str::to_owned)
                .to_vec(),
                attribution_checked_layers: vec![0],
                seal_checked_operands: 7,
                invariant_neighbour_layer: Some(1),
                positions: self.positions,
                gate_evaluated: self.snapshot.gate().unwrap().id().to_string(),
            }
            .into(),
            execution_note: "tiny deterministic observation fixture".into(),
        }
    }
    pub fn artifact(&self, kl: f64) -> MeasurementArtifact {
        MeasurementArtifact::from_execution(&self.prepared, &self.observed(kl), &self.evidence())
            .unwrap()
    }
    pub fn sources(&self) -> IngestionSources<'_> {
        IngestionSources {
            container: &self.source,
            candidate: &self.candidate,
            corpus: &self.corpus,
        }
    }
}

fn refused_without_change(
    f: &Fixture,
    snapshot: &mut SearchSnapshot,
    artifact: &MeasurementArtifact,
) -> IngestionRefusal {
    let before = snapshot.clone();
    let bytes = serde_json::to_vec(snapshot).unwrap();
    let next = snapshot.next_experiment().unwrap();
    let frontier = snapshot.frontier();
    let promotions = format!(
        "{:?}",
        snapshot.promotion_candidates(EvidenceScale::Authority)
    );
    let refusal = ingest(snapshot, &f.prepared, artifact, &f.sources()).unwrap_err();
    assert_eq!(*snapshot, before);
    assert_eq!(serde_json::to_vec(snapshot).unwrap(), bytes);
    assert_eq!(snapshot.next_experiment().unwrap(), next);
    assert_eq!(snapshot.frontier(), frontier);
    assert_eq!(
        format!(
            "{:?}",
            snapshot.promotion_candidates(EvidenceScale::Authority)
        ),
        promotions
    );
    refusal
}

mod tests_basics;
mod tests_basics_2;
