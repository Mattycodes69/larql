use super::*;
use crate::engines::semantic_promotion::scoring::{score_trajectory, PositionScore};

fn key(operator: &str) -> CapabilityKey {
    CapabilityKey {
        model: ModelFingerprint::new("gemma3-4b-q4k#abc"),
        runtime: RuntimeFingerprint::new("semantic-promotion(standard)/cpu"),
        prompt_protocol: PromptProtocolId::new("terse-number-v1"),
        placement_protocol: PlacementProtocolId::new("boundary"),
        record_digest: [0u8; 32],
        record_schema_version: 1,
        materialisation_kind: MaterialisationKind::TokenSequence,
        materialisation_version: 1,
        materialisation_digest: [0u8; 32],
        operation: OperationSignature::new(operator, ["operand"], "integer", 1),
    }
}

fn metrics(kls: &[f32], payload: u32) -> QualificationMetrics {
    let trace: Vec<_> = kls
        .iter()
        .map(|&kl_bits| PositionScore {
            kl_bits,
            top1_equal: kl_bits < 1.0,
        })
        .collect();
    score_trajectory(&trace, payload, DEFAULT_TOLERANCE_BITS).unwrap()
}

fn certificate(operator: &str, object: ScoredObject, kls: &[f32]) -> QualificationCertificate {
    QualificationCertificate {
        id: QualificationId::from_counter(1),
        key: key(operator),
        record: RecordId::from_counter(1),
        scored_object: object,
        metrics: metrics(kls, object.payload_tokens()),
        tolerance_bits: DEFAULT_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption: ConsumptionMode::OperandForOperation,
        general_state_evidence: None,
    }
}

/// EXP-25 `transform_inc` / canonical numbers.
const INC_CANONICAL: &[f32] = &[0.0008, 0.0003, 0.0001, 0.001, 0.1472, 0.0044];

#[test]
fn a_certificate_must_clear_its_own_tolerance() {
    let bad = certificate(
        "increment",
        ScoredObject::ReachableTrajectory { payload_tokens: 4 },
        INC_CANONICAL,
    );
    assert!(!bad.is_self_consistent());
    assert!(matches!(
        bad.check_applies(
            &key("increment"),
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        ),
        Err(PromotionError::InvariantViolation(_))
    ));
}

#[test]
fn the_same_numbers_are_self_consistent_as_a_payload_claim() {
    let ok = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    assert!(ok.is_self_consistent());
    assert!(ok
        .check_applies(
            &key("increment"),
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        )
        .is_ok());
}

#[test]
fn a_certificate_for_another_operation_does_not_apply() {
    let cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    assert_eq!(
        cert.check_applies(
            &key("reverse_digits"),
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        )
        .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

#[test]
fn a_certificate_for_another_record_does_not_apply() {
    let cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    assert_eq!(
        cert.check_applies(
            &key("increment"),
            RecordId::from_counter(2),
            &RegimeObservation::at_boundary()
        )
        .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

#[test]
fn a_changed_materialisation_encoding_invalidates_the_certificate() {
    let cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    let mut live = key("increment");
    live.materialisation_version = 2;
    assert_eq!(
        cert.check_applies(
            &live,
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        )
        .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

#[test]
fn shadow_comparison_promotes_only_with_passing_controls() {
    let base = ShadowComparison {
        record: RecordId::from_counter(1),
        key: key("increment"),
        scored_object: ScoredObject::PayloadOnly { payload_tokens: 4 },
        metrics: metrics(INC_CANONICAL, 4),
        tolerance_bits: DEFAULT_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption: ConsumptionMode::OperandForOperation,
        controls_ok: false,
    };
    assert_eq!(
        base.clone()
            .into_certificate(QualificationId::from_counter(1))
            .unwrap_err(),
        PromotionError::OperationNotQualified
    );

    let passing = ShadowComparison {
        controls_ok: true,
        ..base
    };
    let cert = passing
        .into_certificate(QualificationId::from_counter(1))
        .unwrap();
    assert!(cert.is_self_consistent());
}

#[test]
fn shadow_comparison_never_establishes_general_state_evidence() {
    let shadow = ShadowComparison {
        record: RecordId::from_counter(1),
        key: key("increment"),
        scored_object: ScoredObject::PayloadOnly { payload_tokens: 4 },
        metrics: metrics(INC_CANONICAL, 4),
        tolerance_bits: DEFAULT_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption: ConsumptionMode::OperandForOperation,
        controls_ok: true,
    };
    let cert = shadow
        .into_certificate(QualificationId::from_counter(9))
        .unwrap();
    assert!(cert.general_state_evidence.is_none());
}

#[test]
fn shadow_comparison_refuses_when_the_metrics_fail() {
    let shadow = ShadowComparison {
        record: RecordId::from_counter(1),
        key: key("increment"),
        scored_object: ScoredObject::ReachableTrajectory { payload_tokens: 4 },
        metrics: metrics(INC_CANONICAL, 4),
        tolerance_bits: DEFAULT_TOLERANCE_BITS,
        established_over: RegimeEvidence::distance_invariant(),
        consumption: ConsumptionMode::OperandForOperation,
        controls_ok: true,
    };
    assert_eq!(
        shadow
            .into_certificate(QualificationId::from_counter(1))
            .unwrap_err(),
        PromotionError::OperationNotQualified
    );
}

/// A certificate earned at one distance is not evidence at another.
///
/// EXP-25CF made this concrete: a record that discharged correctly at
/// 64K was re-applied as an operand at 4K, in both histories. Without
/// the band, the 64K certificate would have silently licensed the 4K
/// promotion.
#[test]
fn a_certificate_does_not_apply_outside_the_distance_it_was_measured_at() {
    let mut cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    // A band, because this arm is testing containment. Building one
    // from two measured points would be the interpolation error
    // `RegimeEvidence::Point` exists to prevent — see the point-evidence
    // test below.
    cert.established_over = RegimeEvidence::ValidatedBand(RegimeBounds {
        source_distance: DistanceBand::between(32_768, 131_072),
        record_distance: DistanceBand::unbounded(),
    });
    let live = key("increment");
    let record = RecordId::from_counter(1);

    let at_64k = RegimeObservation {
        source_distance_tokens: 49_000,
        record_distance_tokens: 0,
    };
    assert!(cert.check_applies(&live, record, &at_64k).is_ok());

    let at_4k = RegimeObservation {
        source_distance_tokens: 2_800,
        record_distance_tokens: 0,
    };
    assert_eq!(
        cert.check_applies(&live, record, &at_4k).unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

/// **Point evidence licenses only the point it was measured at.**
///
/// EXP-25CF has canonical payload passing at both 4K and ~49K while
/// derived *consumption inverts* between them. Two passing points can
/// therefore bracket a behavioural boundary, and a certificate that
/// silently spanned them would authorise the regime where the
/// behaviour has already flipped. A band has to be earned by a sweep.
#[test]
fn point_evidence_does_not_interpolate_between_measurements() {
    let mut cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    cert.established_over = RegimeEvidence::at(49_000, 0);
    let live = key("increment");
    let record = RecordId::from_counter(1);

    let measured = RegimeObservation {
        source_distance_tokens: 49_000,
        record_distance_tokens: 0,
    };
    assert!(cert.check_applies(&live, record, &measured).is_ok());
    assert!(cert.established_over.is_point());

    // Anywhere else — including *between* two measured points — is a
    // mismatch, not an approximation.
    for elsewhere in [32_768u64, 40_000, 60_000, 4_096] {
        let observed = RegimeObservation {
            source_distance_tokens: elsewhere,
            record_distance_tokens: 0,
        };
        assert_eq!(
            cert.check_applies(&live, record, &observed).unwrap_err(),
            PromotionError::QualificationMismatch,
            "point evidence must not cover {elsewhere}"
        );
    }
}

/// A validated band does interpolate — that is what earning one means.
#[test]
fn a_validated_band_covers_its_interior() {
    let mut cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    cert.established_over = RegimeEvidence::ValidatedBand(RegimeBounds {
        source_distance: DistanceBand::between(4_096, 65_536),
        record_distance: DistanceBand::unbounded(),
    });
    assert!(!cert.established_over.is_point());
    let live = key("increment");
    for inside in [4_096u64, 32_768, 65_536] {
        let observed = RegimeObservation {
            source_distance_tokens: inside,
            record_distance_tokens: 0,
        };
        assert!(cert
            .check_applies(&live, RecordId::from_counter(1), &observed)
            .is_ok());
    }
}

#[test]
fn distance_bands_are_inclusive_and_orderless() {
    let b = DistanceBand::between(131_072, 32_768);
    assert_eq!(b.min_tokens, 32_768);
    assert!(b.contains(32_768) && b.contains(131_072) && b.contains(60_000));
    assert!(!b.contains(32_767) && !b.contains(131_073));
    assert!(DistanceBand::at(7).contains(7));
    assert!(!DistanceBand::at(7).contains(8));
    assert!(DistanceBand::unbounded().contains(u64::MAX));
}

/// Consumption is an independent axis: the same numbers can back a
/// certificate for either mode, so it is never inferred from them.
#[test]
fn consumption_mode_is_not_derived_from_the_metrics() {
    let operand = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    let mut answer = operand.clone();
    answer.consumption = ConsumptionMode::AnswerWithoutReapplication;
    assert_eq!(operand.metrics, answer.metrics);
    assert_ne!(operand.consumption, answer.consumption);
    // Both remain self-consistent — the metrics say nothing about it.
    assert!(operand.is_self_consistent() && answer.is_self_consistent());
}

/// **A certificate is keyed on the exact record, not a value class.**
///
/// EXP-25CF's frontier: at 8K the true-value claim read as an operand
/// while the *corrupt* claim read as an answer, same geometry; and
/// across the sweep the corrupt arm's mode oscillated where the true
/// arm's was monotone. So two records of the same shape, same schema,
/// same operation can be consumed differently. Only the content
/// distinguishes them.
#[test]
fn certificates_do_not_transfer_between_records_of_the_same_shape() {
    let cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    let mut live = key("increment");
    // Same schema, same operation, same everything — different content.
    live.record_digest = [7u8; 32];
    assert_eq!(
        cert.check_applies(
            &live,
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        )
        .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

/// **Same assertion, different rendering, no transfer.**
///
/// `record_digest` groups renderings of one authority; it does not
/// license capability between them. Behaviour follows the stimulus,
/// so the materialisation digest has to match too.
#[test]
fn a_certificate_does_not_transfer_between_renderings_of_one_assertion() {
    let cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    let mut live = key("increment");
    // Same semantic record — same record_digest — worded differently.
    assert_eq!(live.record_digest, cert.key.record_digest);
    live.materialisation_digest = [3u8; 32];
    assert_eq!(
        cert.check_applies(
            &live,
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        )
        .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

/// A changed placement protocol invalidates the certificate, like any
/// other key field.
#[test]
fn a_changed_placement_protocol_invalidates_the_certificate() {
    let cert = certificate(
        "increment",
        ScoredObject::PayloadOnly { payload_tokens: 4 },
        INC_CANONICAL,
    );
    let mut live = key("increment");
    live.placement_protocol = PlacementProtocolId::new("record-far-from-question");
    assert_eq!(
        cert.check_applies(
            &live,
            RecordId::from_counter(1),
            &RegimeObservation::at_boundary()
        )
        .unwrap_err(),
        PromotionError::QualificationMismatch
    );
}

#[test]
fn only_the_permissive_policy_accepts_shadow_evidence() {
    assert!(!QualificationPolicy::ExactCertificate.accepts_shadow());
    assert!(QualificationPolicy::AllowSessionShadow.accepts_shadow());
}

#[test]
fn fingerprint_constructors_round_trip() {
    assert_eq!(ModelFingerprint::new("m").as_str(), "m");
    assert_eq!(RuntimeFingerprint::new("r").as_str(), "r");
    assert_eq!(PromptProtocolId::new("p").as_str(), "p");
}
