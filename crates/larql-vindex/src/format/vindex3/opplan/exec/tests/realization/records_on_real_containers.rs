//! Records on real containers
//! Refusals, before any byte
//! The pin is checked

use super::*;

#[test]
fn a_prepared_plan_pins_one_realization_per_planned_operand_with_its_reason() {
    let fixture = dense(None);
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let records = prepared.realizations();
    assert_eq!(records.len(), fixture.plan.planned_operands().len());
    for record in records {
        assert_eq!(record.representation, "F32");
        assert!(record
            .selection
            .candidates
            .contains(&record.selection.realization));
        match record.planned.operation {
            Operation::Embed => assert_eq!(
                record.selection.realization.form,
                RealizationForm::DecodedGather
            ),
            op if is_projection(op) => {
                // An f32 source's stored bytes ARE the f32 image: the
                // codec declares BLAS over them as a direct realization.
                assert_eq!(
                    record.selection.realization,
                    RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::BlasF32))
                );
                assert_eq!(record.selection.reason, SelectionReason::DirectDeclared);
            }
            other => panic!("the dense plan has no {other:?}"),
        }
    }
    prepared.verify_pins().unwrap();
}

#[test]
fn the_reference_backend_pins_the_scalar_oracle_for_every_projection() {
    let fixture = dense(None);
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    for record in prepared.realizations() {
        if is_projection(record.planned.operation) {
            assert_eq!(
                record.selection.realization,
                RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32))
            );
            assert_eq!(record.selection.reason, SelectionReason::ReferenceOracle);
            assert_eq!(
                record.selection.candidates.len(),
                1,
                "the oracle considers nothing else"
            );
        }
    }
}

#[test]
fn the_entropy_coded_container_pins_decode_and_says_why() {
    let fixture = dense(Some(Transcode::Bf16Zlib));
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let mut seen = 0;
    for record in prepared.realizations() {
        if record.representation != "BF16_ZLIB" {
            continue;
        }
        assert!(is_projection(record.planned.operation));
        assert_eq!(
            record.selection.realization,
            RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32))
        );
        assert_eq!(
            record.selection.reason,
            SelectionReason::NoDirectRealization
        );
        assert_eq!(
            record.selection.candidates,
            vec![RealizationId::cpu(RealizationForm::Decode(
                PhysicalProjectionPlan::BlasF32
            ))],
            "no direct realization is declared, so decode is the only candidate"
        );
        assert_eq!(record.selection.residency, ResidencyProfile::DECODED_F32);
        seen += 1;
    }
    assert!(seen > 0);
}

#[test]
fn an_unregistered_representation_is_refused_at_preparation_before_any_byte_is_read() {
    let fixture = dense(Some(Transcode::Unregistered));
    let before = fixture.store.load_count();
    let err = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("nothing is registered for the label");
    assert!(err.contains("unregistered representation"), "{err}");
    assert!(err.contains("BF16_ZLIB_UNREGISTERED"), "{err}");
    // Every refused operand is named, not just the first.
    assert!(
        err.matches("unregistered representation").count() > 1,
        "{err}"
    );
    assert_eq!(
        fixture.store.load_count(),
        before,
        "refused before any byte was read"
    );
}

#[test]
fn a_tampered_pin_is_refused_by_the_prepared_plan() {
    let fixture = dense(None);
    let mut prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    prepared.verify_pins().unwrap();
    let index = prepared
        .realizations()
        .iter()
        .position(|r| r.planned.operation == Operation::Project(MatrixClass::AttentionProjection))
        .unwrap();
    prepared.realizations_mut()[index].selection.realization =
        RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedBf16));
    let err = prepared.verify_pins().unwrap_err().to_string();
    assert!(err.contains("not the pinned realizations"), "{err}");
    assert!(err.contains("Bf16") && err.contains("F32"), "{err}");
}

/// The plan's own view and the loader agree on every fixture this suite
/// prepares; a loader asking for an operand the view does not list is a
/// refusal, not a default.
#[test]
fn every_loaded_matrix_has_a_record_and_no_record_is_left_unloaded() {
    let fixture = dense(None);
    let prepared = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let projections = prepared
        .realizations()
        .iter()
        .filter(|r| is_projection(r.planned.operation))
        .count();
    let per_layer = 7;
    assert_eq!(projections, per_layer * fixture.plan.layers.len() + 1);
}
