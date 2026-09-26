use super::*;

#[test]
fn calibration_image_digest_refuses_conflicting_empty_and_non_finite_prefixes() {
    let (_dir, plan, store) = fixture(false);
    let mut prefix = plan.clone();
    prefix.layers.truncate(1);
    prefix.final_norm = None;
    prefix.output = None;
    let sealed = identity::image_digest(&prefix, (&store).into()).unwrap();
    assert!(valid_digest(&sealed));
    assert_eq!(
        sealed,
        identity::image_digest(&prefix, (&store).into()).unwrap(),
        "the same prefix over the same values seals identically"
    );

    // One (object, tensor) named twice with two geometries cannot be sealed.
    let mut conflicting = prefix.clone();
    let attention = conflicting.layers[0].attention.softmax_mut().unwrap();
    attention.q.object = attention.k.object.clone();
    attention.q.tensor = attention.k.tensor.clone();
    attention.q.shape = attention.k.shape.iter().rev().copied().collect();
    assert_ne!(
        attention.q.shape, attention.k.shape,
        "fixture K is not square"
    );
    let error = identity::image_digest(&conflicting, (&store).into()).unwrap_err();
    assert!(
        error.to_string().contains("conflicting operand geometries"),
        "{error}"
    );

    let mut empty = prefix.clone();
    empty.layers.clear();
    empty.embedding = None;
    let error = identity::image_digest(&empty, (&store).into()).unwrap_err();
    assert!(
        error.to_string().contains("prefix contains no operands"),
        "{error}"
    );

    // A candidate whose effective values are non-finite is refused, never sealed.
    let q = &prefix.layers[0].attention.softmax().unwrap().q;
    let mut edits = OperandOverrides::new();
    edits.push(
        q,
        OperandEdit::Row {
            index: 0,
            values: vec![f32::NAN; q.shape[1]],
        },
    );
    let error = PreparedCalibration::prepare(
        &plan,
        OperandSource::overlaid(&store, &edits),
        0,
        Projection::Query,
    )
    .err()
    .expect("a non-finite candidate prefix must refuse");
    assert!(
        error
            .to_string()
            .contains("wrong shape or non-finite values"),
        "{error}"
    );
}

/// Finite weights can still drive the executor's activations to overflow. The
/// tap must surface that as a refusal instead of accumulating a poisoned sum.
#[test]
fn calibration_capture_refuses_non_finite_site_inputs() {
    let (_dir, plan, store) = fixture(false);
    let ffn = plan.layers[0].ffn.as_ref().unwrap().dense().unwrap();
    // Finite, so the prefix seals; large enough that the un-normalized
    // gate * up product entering layer 0's down projection overflows.
    const OVERFLOWING_WEIGHT: f32 = f32::MAX / 2.;
    let mut edits = OperandOverrides::new();
    for op in [ffn.gate.as_ref().unwrap(), &ffn.up] {
        for index in 0..op.shape[0] {
            edits.push(
                op,
                OperandEdit::Row {
                    index,
                    values: vec![OVERFLOWING_WEIGHT; op.shape[1]],
                },
            );
        }
    }
    let prepared = PreparedCalibration::prepare(
        &plan,
        OperandSource::overlaid(&store, &edits),
        0,
        Projection::Down,
    )
    .unwrap();
    let error = prepared
        .capture(&bank(), StatisticKind::DenseGram, &row(&plan))
        .err()
        .expect("an overflowing site input must refuse");
    assert!(
        error
            .to_string()
            .contains("wrong width or non-finite values"),
        "{error}"
    );
}

#[test]
fn calibration_capture_propagates_the_selected_providers_refusal() {
    let (_dir, plan, store) = fixture(false);
    let prepared = PreparedCalibration::prepare(&plan, &store, 1, Projection::Query).unwrap();
    let mut registry = ContinuationRegistry::new();
    registry.register(Box::new(RefusingRow)).unwrap();
    let refusing = registry
        .select(
            &RefusingRow.identity(),
            &ContinuationConfig::empty(),
            &plan_continuation_geometry(&plan).unwrap(),
        )
        .unwrap();
    let error = prepared
        .capture(&bank(), StatisticKind::DenseGram, &refusing)
        .err()
        .expect("a provider refusing the plan's geometry must stop capture");
    assert!(error.to_string().contains("refusing-row"), "{error}");
}
