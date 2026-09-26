use super::*;

#[test]
fn freshly_sealed_incomplete_run_with_correct_positions_and_gate_refuses() {
    let f = Fixture::new();
    let mut observed = f.observed(0.01);
    observed.verified = VerifiedFacts {
        positions: 8,
        gate_evaluated: f.snapshot.gate().unwrap().id().to_string(),
        ..Default::default()
    }
    .into();
    assert!(!observed.verified.as_kimi().unwrap().complete());
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
    artifact.verify_seal().unwrap();
    let refusal = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
    let shown = refusal.to_string();
    for obligation in [
        "compiled layers",
        "compiled projections",
        "attribution",
        "seal/read",
        "invariant neighbour",
    ] {
        assert!(shown.contains(obligation), "{shown} omitted {obligation}");
    }
}

#[test]
fn each_missing_run_validity_obligation_refuses_transactionally() {
    let f = Fixture::new();
    for obligation in [
        "compiled layers",
        "compiled projections",
        "attribution checks covering compiled layers",
        "seal/read witness",
        "invariant neighbour",
        "non-zero positions",
        "gate",
        "partial attribution",
        "all",
    ] {
        let mut observed = f.observed(0.01);
        assert!(observed.verified.as_kimi().unwrap().complete());
        let report = observed.verified.as_kimi_mut().unwrap();
        match obligation {
            "compiled layers" => report.compiled_layers.clear(),
            "compiled projections" => report.compiled_projections.clear(),
            "attribution checks covering compiled layers" => {
                report.attribution_checked_layers.clear()
            }
            "seal/read witness" => report.seal_checked_operands = 0,
            "invariant neighbour" => report.invariant_neighbour_layer = None,
            "non-zero positions" => report.positions = 0,
            "gate" => report.gate_evaluated.clear(),
            "partial attribution" => report.compiled_layers.push(1),
            "all" => *report = VerifiedFacts::default(),
            _ => unreachable!(),
        }
        assert!(!report.complete());
        let artifact =
            MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
        artifact.verify_seal().unwrap();
        let refusal = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
        let IngestionRefusal::IncompleteRun {
            missing,
            observed: actual,
        } = &refusal
        else {
            panic!("{obligation}: {refusal}");
        };
        assert_eq!(actual.as_ref(), &observed.verified);
        if obligation == "all" {
            assert_eq!(missing.len(), 7);
        } else {
            let expected = if obligation == "partial attribution" {
                "attribution checks covering compiled layers"
            } else {
                obligation
            };
            assert!(missing.iter().any(|m| m == expected), "{refusal}");
        }
        for absent in missing {
            assert!(refusal.to_string().contains(absent), "{refusal}");
        }
        assert!(refusal
            .to_string()
            .contains(&format!("{:?}", observed.verified)));
    }
}

#[test]
fn accepted_once_and_identical_duplicate_is_transactionally_idempotent() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    assert!(a.verified().as_kimi().unwrap().complete());
    let mut snapshot = f.snapshot.clone();
    let first = ingest(&mut snapshot, &f.prepared, &a, &f.sources()).unwrap();
    assert!(first.recorded);
    assert_eq!(snapshot.measurements().len(), 1);
    assert_eq!(first.accepted.candidate().established(), a.key().state());
    assert_eq!(first.accepted.observation(), a.observation());
    assert_eq!(first.accepted.artifact_seal(), a.seal());
    assert_eq!(
        first.accepted.source_semantic_digest(),
        a.source_semantic_digest()
    );
    assert_eq!(
        first.accepted.bank_manifest_sha256(),
        f.snapshot.protocol().unwrap().bank.manifest_sha256
    );
    let before = snapshot.clone();
    assert!(
        !ingest(&mut snapshot, &f.prepared, &a, &f.sources())
            .unwrap()
            .recorded
    );
    assert_eq!(snapshot, before);
    let mut expected = f.snapshot.facts().clone();
    expected.measurements = snapshot.measurements().clone();
    assert_eq!(snapshot.facts(), &expected);
}

#[test]
fn valid_y_is_established_then_refused_for_requested_x_without_any_scientific_change() {
    let f = Fixture::new();
    let mut source_spec = RepresentSpec::nvfp4();
    source_spec.protect = super::super::super::policy::Protections::default().projection("q_proj");
    let y = f.dir.path().join("valid-y");
    drop(compile_representation(&f.source, &y, &source_spec).unwrap());
    let evidence = ArtifactStateEvidence::establish(&y).unwrap();
    let requested = f.prepared.request().unwrap().key().state().clone();
    assert_ne!(evidence.established(), &requested);
    let a = MeasurementArtifact::from_execution(&f.prepared, &f.observed(0.01), &evidence).unwrap();
    a.verify_seal().unwrap();
    assert_eq!(a.key().state(), &requested);
    let mut snapshot = f.snapshot.clone();
    let wrong = Fixture { candidate: y, ..f };
    let r = refused_without_change(&wrong, &mut snapshot, &a);
    assert!(
        matches!(&r, IngestionRefusal::Authority {what,..} if what == "candidate representation state")
    );
    assert!(r.to_string().contains(requested.as_str()));
    assert!(r.to_string().contains(evidence.established().as_str()));
}

#[test]
fn conflicting_duplicate_preserves_both_observations_and_all_facts() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let b = f.artifact(0.02);
    let mut snapshot = f.snapshot.clone();
    ingest(&mut snapshot, &f.prepared, &a, &f.sources()).unwrap();
    let r = refused_without_change(&f, &mut snapshot, &b);
    let IngestionRefusal::Conflict(c) = &r else {
        panic!("{r:?}")
    };
    assert_eq!(&c.expected, a.observation());
    assert_eq!(&c.observed, b.observation());
    assert_eq!(&c.key, a.key());
    let shown = r.to_string();
    assert!(shown.contains("0.01") && shown.contains("0.02"));
    assert!(
        shown.contains(a.key().bank().as_str()) && shown.contains(a.key().instrument().as_str())
    );
}

#[test]
fn tampered_artifact_and_malformed_observation_refuse_before_writes() {
    let f = Fixture::new();
    let mut snapshot = f.snapshot.clone();
    let a = f.artifact(0.01);
    let mut doc = serde_json::to_value(&a).unwrap();
    doc["execution_note"] = serde_json::json!("changed");
    let tampered = serde_json::from_value(doc).unwrap();
    assert!(matches!(
        refused_without_change(&f, &mut snapshot, &tampered),
        IngestionRefusal::Artifact(_)
    ));
    for kind in ["positions", "nonfinite", "count", "empty-report"] {
        let mut observed = f.observed(0.01);
        match kind {
            "positions" => observed.observation.as_kimi_mut().unwrap().positions = 0,
            "nonfinite" => observed.observation.as_kimi_mut().unwrap().logits.kl_p99 = f64::NAN,
            "count" => {
                observed
                    .observation
                    .as_kimi_mut()
                    .unwrap()
                    .logits
                    .top1_flips = 9
            }
            _ => observed.verified = VerifiedFacts::default().into(),
        }
        let a = MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence()).unwrap();
        refused_without_change(&f, &mut snapshot, &a);
    }
}

#[test]
fn protocol_source_bank_and_candidate_mutations_refuse_transactionally() {
    for kind in [
        "protocol",
        "gate",
        "source",
        "bank",
        "candidate",
        "missing-candidate",
    ] {
        let f = Fixture::new();
        let a = f.artifact(0.01);
        let mut snapshot = f.snapshot.clone();
        match kind {
            "protocol" | "gate" => {
                let mut config = snapshot.config().clone();
                if kind == "protocol" {
                    config
                        .protocol
                        .as_mut()
                        .unwrap()
                        .bank
                        .samples
                        .push("other".into());
                } else {
                    match config.gate.as_mut().unwrap() {
                        super::super::super::reading::Gate::Kimi(gate) => gate.positions_min += 1,
                        super::super::super::reading::Gate::Plan(gate) => gate.positions_min += 1,
                    }
                }
                snapshot =
                    SearchSnapshot::new(snapshot.space().clone(), config, snapshot.facts().clone());
            }
            "source" => {
                let p = f.source.join("index.json");
                let mut d: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                d["model"] = serde_json::json!("different");
                std::fs::write(p, serde_json::to_vec(&d).unwrap()).unwrap();
            }
            "bank" => std::fs::write(f.corpus.join("manifest.json"), b"changed").unwrap(),
            "candidate" => std::fs::write(f.candidate.join("index.json"), b"changed").unwrap(),
            _ => std::fs::remove_file(f.candidate.join("candidate.json")).unwrap(),
        }
        let before = snapshot.clone();
        // A malformed protocol can itself make next_experiment unavailable;
        // compare both results instead of requiring a successful selection.
        let next = format!("{:?}", snapshot.next_experiment());
        assert!(
            ingest(&mut snapshot, &f.prepared, &a, &f.sources()).is_err(),
            "{kind}"
        );
        assert_eq!(snapshot, before, "{kind}");
        assert_eq!(format!("{:?}", snapshot.next_experiment()), next);
    }
}

#[test]
fn a_valid_sealed_artifact_for_another_authorisation_names_both_keys() {
    use super::super::super::actuate::{prepare::Ready, request::MeasurementRequest};
    let f = Fixture::new();
    let expected = f.prepared.request().unwrap().key();
    let other_key = state::MeasurementKey::new(
        expected.state(),
        expected.bank(),
        EvidenceScale::Diagnostic,
        expected.instrument(),
    );
    let request = MeasurementRequest::of(
        &f.snapshot,
        &other_key,
        &BTreeSet::from(["compile-all".into()]),
    )
    .unwrap();
    let other = PreparedExperiment::Ready(Box::new(Ready {
        request,
        physical_delta: 0,
        routes: 1,
        considered: 1,
    }));
    let observed = Observed {
        key: other_key.clone(),
        ..f.observed(0.01)
    };
    let a = MeasurementArtifact::from_execution(&other, &observed, &f.evidence()).unwrap();
    a.verify_seal().unwrap();
    let r = refused_without_change(&f, &mut f.snapshot.clone(), &a);
    assert!(matches!(&r,IngestionRefusal::Authority{what,..} if what=="measurement key"));
    assert!(r.to_string().contains(expected.as_str()));
    assert!(r.to_string().contains(other_key.as_str()));
}

#[test]
fn caller_verdict_fields_are_rejected_on_artifact_deserialization() {
    let f = Fixture::new();
    for field in ["verdict", "rank", "promotion", "admissible"] {
        let mut doc = serde_json::to_value(f.artifact(0.01)).unwrap();
        doc[field] = serde_json::json!("PASS");
        assert!(
            serde_json::from_value::<MeasurementArtifact>(doc).is_err(),
            "{field}"
        );
    }
}

#[test]
fn unreadable_authorities_and_invalid_distributions_cannot_change_facts() {
    for kind in ["source", "bank", "distribution", "schema", "no-protocol"] {
        let f = Fixture::new();
        let mut a = f.artifact(0.01);
        let mut snapshot = f.snapshot.clone();
        match kind {
            "source" => std::fs::remove_file(f.source.join("index.json")).unwrap(),
            "bank" => std::fs::remove_file(f.corpus.join("manifest.json")).unwrap(),
            "distribution" => {
                let mut observed = f.observed(0.01);
                observed.observation.as_kimi_mut().unwrap().top1_margin =
                    Some(super::super::super::quality::Distribution {
                        count: 0,
                        min: 0.0,
                        p50: 1.0,
                        p95: 0.5,
                        p99: 2.0,
                        max: 3.0,
                    });
                a = MeasurementArtifact::from_execution(&f.prepared, &observed, &f.evidence())
                    .unwrap();
            }
            "schema" => {
                let mut doc = serde_json::to_value(&snapshot).unwrap();
                doc["schema"] = serde_json::json!("unknown/v9");
                snapshot = serde_json::from_value(doc).unwrap();
            }
            _ => {
                let mut config = snapshot.config().clone();
                config.protocol = None;
                snapshot =
                    SearchSnapshot::new(snapshot.space().clone(), config, snapshot.facts().clone());
            }
        }
        let before = snapshot.clone();
        assert!(
            ingest(&mut snapshot, &f.prepared, &a, &f.sources()).is_err(),
            "{kind}"
        );
        assert_eq!(snapshot, before);
    }
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let mut snapshot = f.snapshot.clone();
    assert!(ingest(
        &mut snapshot,
        &PreparedExperiment::Exhausted,
        &a,
        &f.sources()
    )
    .is_err());
    assert_eq!(snapshot, f.snapshot);
}

#[test]
fn persisted_artifact_round_trip_accepts_and_missing_fields_refuse() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let bytes = serde_json::to_vec(&a).unwrap();
    let mut snapshot = f.snapshot.clone();
    assert!(
        ingest_bytes(&mut snapshot, &f.prepared, &bytes, &f.sources())
            .unwrap()
            .recorded
    );
    for field in [
        "key",
        "observation",
        "candidate_authority_digest",
        "verified",
        "seal",
    ] {
        let mut doc = serde_json::to_value(&a).unwrap();
        doc.as_object_mut().unwrap().remove(field);
        let before = snapshot.clone();
        assert!(matches!(
            ingest_bytes(
                &mut snapshot,
                &f.prepared,
                &serde_json::to_vec(&doc).unwrap(),
                &f.sources()
            ),
            Err(IngestionRefusal::Malformed(_))
        ));
        assert_eq!(snapshot, before);
    }
}

#[test]
fn self_consistent_but_empty_or_unimplemented_protocols_are_refused() {
    for kind in ["empty-bank", "unknown-procedure"] {
        let f = Fixture::new();
        let mut config = f.snapshot.config().clone();
        let protocol = config.protocol.as_mut().unwrap();
        if kind == "empty-bank" {
            protocol.bank.samples.clear();
        } else {
            protocol.procedure = "unknown/v1".into();
        }
        config.standing_intent = protocol.intent(EvidenceScale::Authority);
        let mut snapshot = SearchSnapshot::new(
            f.snapshot.space().clone(),
            config,
            f.snapshot.facts().clone(),
        );
        let prepared = PreparedExperiment::of(&snapshot);
        assert!(prepared.is_ready());
        let observed = Observed {
            key: prepared.request().unwrap().key().clone(),
            ..f.observed(0.01)
        };
        let artifact =
            MeasurementArtifact::from_execution(&prepared, &observed, &f.evidence()).unwrap();
        let before = snapshot.clone();
        let refusal = ingest(&mut snapshot, &prepared, &artifact, &f.sources()).unwrap_err();
        assert!(matches!(refusal, IngestionRefusal::Malformed(_)));
        assert_eq!(snapshot, before);
    }
}

#[test]
fn changed_bank_rows_with_unchanged_manifest_and_key_refuse() {
    let f = Fixture::new();
    let a = f.artifact(0.01);
    let path = f.corpus.join("seq_0.f32");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(path, bytes).unwrap();
    let mut snapshot = f.snapshot.clone();
    let refusal = refused_without_change(&f, &mut snapshot, &a);
    assert!(refusal.to_string().contains("bank payload"), "{refusal}");
}

#[test]
fn bank_authority_requires_seals_dimensions_and_the_executed_sample_prefix() {
    for kind in [
        "legacy",
        "version",
        "regime",
        "positions",
        "hidden",
        "inventory",
        "overflow",
        "seal",
        "length",
        "unreadable",
        "schema",
        "sample-order",
    ] {
        let mut f = Fixture::new();
        let path = f.corpus.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        match kind {
            "legacy" => {
                manifest
                    .as_object_mut()
                    .unwrap()
                    .remove("payload_authority");
            }
            "version" => manifest["payload_authority"] = serde_json::json!("unknown/v1"),
            "regime" => manifest["regime"] = serde_json::json!("free-running"),
            "positions" => manifest["positions"] = serde_json::json!(7),
            "hidden" => manifest["hidden"] = serde_json::json!(0),
            "inventory" => manifest["token_ids"] = serde_json::json!([]),
            "overflow" => manifest["hidden"] = serde_json::json!(u64::MAX),
            "seal" => manifest["payloads"] = serde_json::json!({}),
            "length" => manifest["payloads"]["seq_0.f32"]["len"] = serde_json::json!(1),
            "unreadable" => std::fs::remove_file(f.corpus.join("seq_0.f32")).unwrap(),
            _ => {}
        }
        let bytes = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(path, &bytes).unwrap();
        let mut config = f.snapshot.config().clone();
        let protocol = config.protocol.as_mut().unwrap();
        protocol.bank.manifest_sha256 = super::super::super::compile::hash_bytes(&bytes);
        if kind == "schema" {
            protocol.bank.schema = "unknown/v9".into();
        }
        if kind == "sample-order" {
            protocol.bank.samples = vec!["seq-001".into()];
        }
        config.standing_intent = protocol.intent(EvidenceScale::Authority);
        f.snapshot = SearchSnapshot::new(
            f.snapshot.space().clone(),
            config,
            f.snapshot.facts().clone(),
        );
        f.prepared = PreparedExperiment::of(&f.snapshot);
        assert!(f.prepared.is_ready());
        let a = f.artifact(0.01);
        let mut snapshot = f.snapshot.clone();
        let refusal = refused_without_change(&f, &mut snapshot, &a);
        assert!(refusal.to_string().contains("bank"), "{kind}: {refusal}");
    }
}

#[test]
fn bank_relocation_preserves_authority_but_reordering_does_not() {
    let mut f = Fixture::new();
    let path = f.corpus.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["sequences"] = serde_json::json!(2);
    manifest["token_ids"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!([1, 1, 1, 1, 1, 1, 1, 1]));
    manifest["payloads"]["seq_1.f32"] = manifest["payloads"]["seq_0.f32"].clone();
    std::fs::copy(f.corpus.join("seq_0.f32"), f.corpus.join("seq_1.f32")).unwrap();
    let bytes = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let mut config = f.snapshot.config().clone();
    let protocol = config.protocol.as_mut().unwrap();
    protocol.bank.manifest_sha256 = super::super::super::compile::hash_bytes(&bytes);
    protocol.bank.samples = vec!["seq-000".into(), "seq-001".into()];
    config.standing_intent = protocol.intent(EvidenceScale::Authority);
    f.snapshot = SearchSnapshot::new(
        f.snapshot.space().clone(),
        config,
        f.snapshot.facts().clone(),
    );
    f.prepared = PreparedExperiment::of(&f.snapshot);
    let observed = |fixture: &Fixture| {
        let mut o = fixture.observed(0.01);
        o.observation.as_kimi_mut().unwrap().positions = 16;
        o.verified.as_kimi_mut().unwrap().positions = 16;
        o
    };
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed(&f), &f.evidence()).unwrap();
    let relocated = f.dir.path().join("moved-corpus");
    std::fs::rename(&f.corpus, &relocated).unwrap();
    f.corpus = relocated;
    assert!(
        ingest(
            &mut f.snapshot.clone(),
            &f.prepared,
            &artifact,
            &f.sources()
        )
        .unwrap()
        .recorded
    );
    let mut config = f.snapshot.config().clone();
    let protocol = config.protocol.as_mut().unwrap();
    protocol.bank.samples.reverse();
    config.standing_intent = protocol.intent(EvidenceScale::Authority);
    f.snapshot = SearchSnapshot::new(
        f.snapshot.space().clone(),
        config,
        f.snapshot.facts().clone(),
    );
    f.prepared = PreparedExperiment::of(&f.snapshot);
    let artifact =
        MeasurementArtifact::from_execution(&f.prepared, &observed(&f), &f.evidence()).unwrap();
    let r = refused_without_change(&f, &mut f.snapshot.clone(), &artifact);
    assert!(r.to_string().contains("bank sample order"));
}
