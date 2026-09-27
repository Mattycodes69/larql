//! Presentation over the records

use super::*;

#[test]
fn the_summary_renders_the_structured_records_and_adds_nothing() {
    let f = fixture(dense_f32_model, Some(Transcode::Bf16Zlib));
    let ops = prepared(&f);
    let text = render_selection_summary(ops.realizations());
    assert!(text.starts_with("realizations:\n"), "{text}");
    for record in ops.realizations() {
        assert!(text.contains(&record.representation), "{text}");
        assert!(
            text.contains(&record.selection.realization.name()),
            "{text}"
        );
        assert!(text.contains(record.selection.reason.name()), "{text}");
    }
    let lines = text.lines().count() - 1;
    let distinct: std::collections::BTreeSet<_> = ops
        .realizations()
        .iter()
        .map(|r| {
            (
                r.representation.clone(),
                r.selection.realization.name(),
                r.selection.reason.name(),
            )
        })
        .collect();
    assert_eq!(
        lines,
        distinct.len(),
        "one line per (representation, realization, reason)"
    );
}

/// The observation side refuses a pairing that contradicts itself,
/// and the reconciliation refuses a stray object and a missing one.
#[test]
fn reconciliation_refuses_strays_omissions_and_disagreeing_forms() {
    let f = fixture(dense_f32_model, None);
    let op = an_ffn_projection(&f.plan);
    let f32 = load_weight((&f.store).into(), &op, WeightFormat::F32).unwrap();
    // A different resident form for the same operand: the executor's own
    // re-quantisation, which any f32 source can take.
    let q8 = load_weight((&f.store).into(), &op, WeightFormat::Q8).unwrap();
    let rec = record(
        &op,
        RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)),
        ResidencyProfile::DECODED_F32,
    );
    let expected = expectations(
        std::slice::from_ref(&rec),
        |o| f.store.stored_len(o),
        BlockGeometry::executor(),
    );
    let observed = |w| {
        vec![Bound::one(&op, w)
            .observed(rec.planned.operation, Some(0))
            .unwrap()]
    };
    reconcile(&expected, &observed(&f32)).unwrap();
    let err = reconcile(&expected, &observed(&q8))
        .unwrap_err()
        .to_string();
    assert!(err.contains("pinned F32 but Q8 is resident"), "{err}");
    let err = reconcile(&expected, &[]).unwrap_err().to_string();
    assert!(err.contains("nothing is resident for it"), "{err}");
    let err = reconcile(&[], &observed(&f32)).unwrap_err().to_string();
    assert!(err.contains("nothing was pinned for it"), "{err}");
    let mixed = Bound {
        operand: &op,
        weights: vec![&f32, &q8],
    }
    .observed(rec.planned.operation, Some(0))
    .unwrap_err()
    .to_string();
    assert!(
        mixed.contains("disagree on their representation"),
        "{mixed}"
    );
    let _: Observed = observed(&f32).remove(0);
}
