//! The prepared plan's pairing and provider arms

use super::*;

#[test]
fn a_hybrid_plan_pairs_dense_projections_one_each_and_banks_over_their_experts() {
    let dir = tempfile::tempdir().unwrap();
    miniature_gemma4(dir.path(), None);
    let container = gemma4_encoded(dir.path());
    let outcome = gemma4_closure(container.path());
    let plan = outcome.plan.expect("the Gemma-4 miniature plans");
    // Layer 3 binds one tensor as both its key and value projection: one
    // stored operand, two operation instances in one layer. The view lists
    // both, the loader binds both, the pairing matches both, and the
    // stored footprint counts the object once.
    let mut counts = std::collections::BTreeMap::new();
    for p in plan.planned_operands() {
        *counts
            .entry((p.operand.tensor.clone(), p.layer))
            .or_insert(0usize) += 1;
    }
    // Two shared objects: the aliased projection within layer 3, and the
    // head tied to the embedding table (two operations, no layer).
    let repeated: Vec<_> = counts.iter().filter(|(_, n)| **n > 1).collect();
    assert_eq!(repeated.len(), 2, "{repeated:?}");
    assert!(repeated.iter().all(|(_, n)| **n == 2), "{repeated:?}");
    assert!(repeated
        .iter()
        .any(|((t, l), _)| t.ends_with("k_proj.weight") && l.is_some()));
    assert!(
        repeated.iter().any(|((_, l), _)| l.is_none()),
        "the tied head"
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let ops = PreparedOperands::load(
        &plan,
        &store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let observed = ops.bound(&plan).unwrap();
    let hybrid_layers = plan
        .layers
        .iter()
        .filter(|l| {
            matches!(
                l.ffn,
                Some(crate::format::vindex3::opplan::LayerFfn::Hybrid(_))
            )
        })
        .count();
    assert!(hybrid_layers > 0, "the miniature has hybrid layers");
    let dense: Vec<_> = observed
        .iter()
        .filter(|o| o.operation == Operation::Project(MatrixClass::FfnProjection))
        .collect();
    let banks: Vec<_> = observed
        .iter()
        .filter(|o| o.operation == Operation::ExpertBankSlice)
        .collect();
    assert!(
        dense.len() >= 2 * hybrid_layers,
        "gate/up and down per hybrid dense branch"
    );
    assert_eq!(banks.len(), 2 * hybrid_layers, "two banks per hybrid layer");
    assert!(banks
        .iter()
        .all(|b| b.allocations == 0 && b.format == WeightFormat::F32));
    let done = ops.reconcile(&plan, (&store).into()).unwrap();
    assert_eq!(done.matched, ops.realizations().len());
    let expected = ops.expectations((&store).into(), BlockGeometry::executor());
    assert_eq!(
        stored_footprint(&expected).operands,
        expected.len() - 2,
        "the aliased projection and the tied table are one stored operand each under two instances"
    );
    assert!(execution_touch(&expected) > stored_footprint(&expected).bytes);
}

#[test]
fn verify_pins_refuses_a_tampered_ffn_pin_and_a_tampered_head_pin() {
    let f = fixture(dense_f32_model, None);
    for (operation, site) in [
        (Operation::Project(MatrixClass::FfnProjection), "ffn"),
        (Operation::OutputHead, "output head"),
    ] {
        let mut ops = prepared(&f);
        let index = ops
            .realizations()
            .iter()
            .position(|r| r.planned.operation == operation)
            .unwrap();
        ops.realizations_mut()[index].selection.realization =
            RealizationId::cpu(RealizationForm::Requantise(PhysicalProjectionPlan::FusedQ8));
        let err = ops.verify_pins().unwrap_err().to_string();
        assert!(
            err.contains(site) && err.contains("not the pinned realizations"),
            "{err}"
        );
    }
}

#[test]
fn a_provider_whose_identity_changed_invalidates_the_preparation() {
    let f = fixture(dense_f32_model, None);
    let ops = prepared(&f);
    let changed = CodecRegistry::new()
        .register(Box::new(ProviderStub { label: "F32" }))
        .unwrap();
    let err = ops.ensure_providers_in(&changed).unwrap_err().to_string();
    assert!(
        err.contains("`F32`") && err.contains("F32 r1") && err.contains("stub-F32 r7"),
        "{err}"
    );
    let providers = ops.providers();
    assert_eq!(
        providers.len(),
        1,
        "one stored label on the dense fixture: {providers:?}"
    );
    assert_eq!(providers[0].0, "F32");
    assert!(providers[0].1.is_some());
}

#[test]
fn bound_refuses_a_plan_that_is_a_different_program_from_the_prepared_one() {
    let dense = fixture(dense_f32_model, None);
    let ops = prepared(&dense);
    let mut headless = dense.plan.clone();
    headless.layers.clear();
    let err = ops.bound(&headless).unwrap_err().to_string();
    assert!(err.contains("prepared but not in the plan"), "{err}");
    let lllf = fixture(
        crate::format::vindex3::fixtures::hybrid_lllf_f32_model,
        None,
    );
    let err = ops.bound(&lllf.plan).unwrap_err().to_string();
    assert!(err.contains("different programs"), "{err}");
}
