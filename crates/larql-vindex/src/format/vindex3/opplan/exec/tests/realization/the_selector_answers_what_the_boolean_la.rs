//! The selector answers what the boolean ladder answered

use super::*;

#[test]
fn the_selector_lands_where_the_boolean_ladder_landed_for_every_label_class_size_and_arm() {
    let labels = [
        "BF16",
        "F16",
        "F32",
        "Q4_K",
        "Q6_K",
        "Q8_0",
        "NVFP4",
        "MXFP4",
        "BF16_ZLIB",
    ];
    let mut compared = 0;
    for label in labels {
        let facts = RepresentationFacts::resolve(label);
        assert!(facts.registered.is_some(), "{label} is registered");
        let stored_bf16 = label == "BF16";
        let stored_nvfp4 = label == "NVFP4";
        let stored_kquant = matches!(label, "Q4_K" | "Q6_K" | "Q8_0");
        for (operation, class) in [
            (
                Operation::Project(MatrixClass::AttentionProjection),
                MatrixClass::AttentionProjection,
            ),
            (
                Operation::Project(MatrixClass::FfnProjection),
                MatrixClass::FfnProjection,
            ),
            (Operation::OutputHead, MatrixClass::OutputHead),
        ] {
            for elements in [SMALL, LARGE] {
                for arm in [KQuantExecution::Direct, KQuantExecution::Widen] {
                    let selected = select_cpu(&synthetic(operation, elements), &facts, arm)
                        .unwrap_or_else(|e| panic!("{label} {operation:?} {elements}: {e}"));
                    let expected = old_ladder(
                        class,
                        elements,
                        stored_bf16,
                        stored_nvfp4,
                        stored_kquant,
                        arm,
                    );
                    assert_eq!(
                        selected.realization.format(),
                        expected,
                        "{label} {operation:?} {elements} {arm:?}: {:?}",
                        selected.realization
                    );
                    // The pin is one of the candidates, always.
                    assert!(selected.candidates.contains(&selected.realization));
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, labels.len() * 3 * 2 * 2);
}

/// Direct and decode are DISTINCT realizations of the same operand, told
/// apart by the record and not by the bytes alone.
#[test]
fn direct_and_decode_are_distinct_realizations_of_one_stored_kquant() {
    let facts = RepresentationFacts::resolve("Q6_K");
    let operand = synthetic(Operation::Project(MatrixClass::FfnProjection), LARGE);
    let direct = select_cpu(&operand, &facts, KQuantExecution::Direct).unwrap();
    let decode = select_cpu(&operand, &facts, KQuantExecution::Widen).unwrap();
    assert_eq!(
        direct.realization.form,
        RealizationForm::Direct(PhysicalProjectionPlan::FusedKQuant)
    );
    assert_eq!(
        decode.realization.form,
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)
    );
    assert_ne!(
        direct.residency, decode.residency,
        "different declared costs"
    );
    assert_eq!(decode.residency, ResidencyProfile::DECODED_F32);
    assert_eq!(
        direct.candidates, decode.candidates,
        "one candidate set, two orderings"
    );
}
