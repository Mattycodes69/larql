use super::*;

#[test]
fn every_ingestion_refusal_preserves_its_actionable_authority() {
    use super::super::super::candidate_authority::CandidateAuthorityRefusal as C;
    use super::super::artifact::ArtifactRefusal as A;
    fn must_say(r: &IngestionRefusal) -> Vec<String> {
        match r {
            IngestionRefusal::Artifact(a) => match a {
                A::SealBroken { expected, observed }
                | A::SubstitutedExperiment { expected, observed } => {
                    vec![expected.clone(), observed.clone()]
                }
                A::Unsealable { detail } | A::NotAuthorised { detail } => vec![detail.clone()],
            },
            IngestionRefusal::Candidate(c) => match c {
                C::Invalid(detail) => vec![detail.clone()],
                C::Missing => vec!["no completed representation authority".into()],
                C::RequestedState { expected, observed } => {
                    vec![expected.to_string(), observed.to_string()]
                }
                C::Version {
                    field,
                    expected,
                    observed,
                } => vec![field.clone(), expected.clone(), observed.clone()],
                C::Binding {
                    what,
                    expected,
                    observed,
                } => vec![what.clone(), expected.clone(), observed.clone()],
                C::StoredState { stored, recomputed } => {
                    vec![stored.to_string(), recomputed.to_string()]
                }
            },
            IngestionRefusal::Authority {
                what,
                expected,
                observed,
            } => vec![what.clone(), expected.clone(), observed.clone()],
            IngestionRefusal::Malformed(detail) => vec![detail.clone()],
            IngestionRefusal::IncompleteRun { missing, observed } => {
                let mut required = missing.clone();
                required.push(format!("{observed:?}"));
                required
            }
            IngestionRefusal::Conflict(c) => vec![
                format!("{:?}", c.key),
                format!("{:?}", c.expected),
                format!("{:?}", c.observed),
            ],
        }
    }
    let f = Fixture::new();
    let x = f.evidence().established().clone();
    let y = RepresentationState::resolve(
        &read_source_identity(&f.source).unwrap(),
        &f.snapshot.space().surface,
        &f.snapshot.space().base_map,
        &PackLayoutAdmission,
    )
    .id()
    .clone();
    let mut cases = vec![
        IngestionRefusal::Authority {
            what: "bank".into(),
            expected: "expected-bank".into(),
            observed: "observed-bank".into(),
        },
        IngestionRefusal::Malformed("missing input rows".into()),
        IngestionRefusal::IncompleteRun {
            missing: vec!["compiled layers".into(), "seal/read witness".into()],
            observed: Box::new(VerifiedFacts::default().into()),
        },
        IngestionRefusal::Conflict(Box::new(state::key::MeasurementConflict {
            key: f.observed(0.01).key,
            expected: f.observed(0.01).observation,
            observed: f.observed(0.02).observation,
        })),
    ];
    cases.extend(
        [
            A::SealBroken {
                expected: "sealed".into(),
                observed: "changed".into(),
            },
            A::SubstitutedExperiment {
                expected: "requested".into(),
                observed: "restated".into(),
            },
            A::Unsealable {
                detail: "not serializable".into(),
            },
            A::NotAuthorised {
                detail: "no ready experiment".into(),
            },
        ]
        .into_iter()
        .map(IngestionRefusal::Artifact),
    );
    cases.extend(
        [
            C::Invalid("unreadable candidate".into()),
            C::Missing,
            C::RequestedState {
                expected: x.clone(),
                observed: y.clone(),
            },
            C::Version {
                field: "schema".into(),
                expected: "v1".into(),
                observed: "v99".into(),
            },
            C::Binding {
                what: "payload".into(),
                expected: "sealed-payload".into(),
                observed: "actual-payload".into(),
            },
            C::StoredState {
                stored: y,
                recomputed: x,
            },
        ]
        .into_iter()
        .map(IngestionRefusal::Candidate),
    );
    for r in cases {
        for required in must_say(&r) {
            assert!(r.to_string().contains(&required), "{r} omitted {required}");
        }
    }
}

#[test]
fn changed_baseline_segment_refuses_even_when_candidate_and_source_declaration_stay_valid() {
    let f = Fixture::new();
    let artifact = f.artifact(0.01);
    let before = read_source_identity(&f.source).unwrap();
    let entry = &before.semantic.representations[0];
    let path = f.source.join(&entry.segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    // Break the source/candidate hardlink before replacing baseline bytes.
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    assert_eq!(read_source_identity(&f.source).unwrap(), before);
    assert_eq!(f.evidence().established(), artifact.key().state());
    let refusal = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
    let shown = refusal.to_string();
    assert!(shown.contains("source container payloads"), "{shown}");
    assert!(shown.contains(&entry.segment_sha256));
    assert!(shown.contains(&super::super::super::compile::hash_bytes(&bytes)));
}

/// MEASURE-PLAN-2 W2 at ingestion: a plan-v1 reading sealed into an
/// artifact for a Kimi-gated record is refused before anything reads its
/// values, and the record is untouched.
#[test]
fn a_plan_reading_is_refused_by_a_kimi_gated_record() {
    use super::super::super::measure::plan::metrics::Aggregate;
    use super::super::super::reading::{Observation, PlanObservation, PlanProcedure};
    let f = Fixture::new();
    let all = Aggregate {
        positions: 8,
        kl_mean: 1e-4,
        kl_p50: 1e-4,
        kl_p99: 2e-4,
        kl_max: 3e-4,
        top1_agreement: 1.0,
        top5_overlap_mean: 1.0,
        delta_nll_mean: Some(0.0),
        max_abs_delta_mean: 0.1,
        max_abs_delta_p99: 0.2,
    };
    let mut observed = f.observed(0.001);
    observed.observation = Observation::Plan(PlanObservation {
        procedure: PlanProcedure,
        sequences: 1,
        positions: 8,
        all,
        by_category: vec![],
        by_margin_band: vec![],
    });
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
    let mut snapshot = f.snapshot.clone();
    let refusal = refused_without_change(&f, &mut snapshot, &artifact);
    assert!(
        matches!(&refusal, IngestionRefusal::Authority { what, .. } if what == "reading kind of the procedure"),
        "{refusal}"
    );
}

/// Integrity refusals the record must name: a bank manifest that cannot be
/// read, and a source container whose declared seals are intact but whose
/// payload bytes are not what they seal. Each refuses without changing a
/// fact, and names the authority that failed.
#[test]
fn an_unreadable_bank_manifest_or_corrupted_source_payload_is_refused_by_name() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let mut snapshot = f.snapshot.clone();
    std::fs::remove_file(
        f.corpus
            .join(super::super::super::actuate::artifacts::BANK_MANIFEST),
    )
    .unwrap();
    let r = refused_without_change(&f, &mut snapshot, &a);
    assert!(
        matches!(&r, IngestionRefusal::Authority { what, .. } if what == "bank manifest"),
        "{r:?}"
    );

    let f = Fixture::new();
    let a = f.artifact(0.01);
    let mut snapshot = f.snapshot.clone();
    let index: Vindex3Index =
        serde_json::from_slice(&std::fs::read(f.source.join("index.json")).unwrap()).unwrap();
    let segment = f
        .source
        .join(&index.representations.values().next().unwrap().segment);
    let mut bytes = std::fs::read(&segment).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    std::fs::write(&segment, bytes).unwrap();
    let r = refused_without_change(&f, &mut snapshot, &a);
    assert!(
        matches!(&r, IngestionRefusal::Authority { what, .. } if what == "source container payloads"),
        "{r:?}"
    );
}

/// A run can report a complete, sealed set of facts that still disagree
/// with the record: positions measured other than the bank's, or a gate
/// evaluated other than the record's. The seal proves the report is the
/// executor's; ingestion must still refuse facts that are not this
/// record's, naming which.
#[test]
fn sealed_facts_that_disagree_with_the_record_are_refused_by_name() {
    for (what, edit) in [("reported measured positions", 0u8), ("reported gate", 1u8)] {
        let f = Fixture::new();
        let mut observed = f.observed(0.01);
        let facts = observed
            .verified
            .as_kimi_mut()
            .expect("the fixture's run reports Kimi-procedure facts");
        if edit == 0 {
            facts.positions += 1;
        } else {
            facts.gate_evaluated = "a-gate-this-record-never-named/v1".into();
        }
        assert!(facts.complete(), "{what}: the report is otherwise complete");
        let artifact =
            MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
        artifact.verify_seal().unwrap();
        let r = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
        assert!(
            matches!(&r, IngestionRefusal::Authority { what: named, .. } if named == what),
            "{what}: {r:?}"
        );
    }
}
