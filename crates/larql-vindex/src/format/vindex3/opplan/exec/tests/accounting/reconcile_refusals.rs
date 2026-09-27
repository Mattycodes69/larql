//! The reconciliation's per-form rules, held on hand-built pairs so each
//! refusal is reached by exactly the disagreement it names: a mapping is
//! held to its address space, an owned object never carries a mapping, an
//! allocation's padding is bounded by the page, and a pinned plan that
//! never ran is refused.

use super::super::super::accounting::{Expectation, Reconciliation};
use super::super::super::cpu::ledger::ProjectionLedger;
use super::super::super::realization::MappedAccess;
use super::*;

/// Bytes the hand-built expectation declares.
const DECLARED: u64 = 4096;
/// A mapping's pages resident at observation time — reported, never
/// reconciled.
const RESIDENT_PAGES: u64 = 1024;
const LAYER: Option<usize> = Some(0);
const OPERATION: Operation = Operation::Project(MatrixClass::FfnProjection);

fn operand() -> OperandRef {
    OperandRef {
        object: "target.decoder_stack".to_string(),
        tensor: "0.mlp.up_proj.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![16, 64],
    }
}

fn expectation(form: RealizationForm) -> Expectation {
    let op = operand();
    Expectation {
        logical_elements: op.shape.iter().product(),
        operand: op,
        operation: OPERATION,
        layer: LAYER,
        realization: RealizationId::cpu(form),
        stored_bytes: DECLARED,
        declared_resident: DECLARED,
        staging: 0,
        read_to_prepare: DECLARED,
        verified_bytes: 0,
        dependencies: Vec::new(),
    }
}

fn observed(resident_bytes: u64, mapped_bytes: u64, allocations: usize) -> Observed {
    Observed {
        operand: operand(),
        operation: OPERATION,
        layer: LAYER,
        format: WeightFormat::F32,
        resident_bytes,
        mapped_bytes,
        allocations,
    }
}

fn mapped() -> RealizationForm {
    RealizationForm::MappedStored {
        format: WeightFormat::F32,
        access: MappedAccess::Demand,
    }
}

fn decoded() -> RealizationForm {
    RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)
}

fn refusal(form: RealizationForm, seen: Observed) -> String {
    reconcile(&[expectation(form)], &[seen])
        .unwrap_err()
        .to_string()
}

#[test]
fn a_mapping_reconciles_its_address_space_and_reports_its_resident_pages() {
    let account = reconcile(
        &[expectation(mapped())],
        &[observed(RESIDENT_PAGES, DECLARED, 0)],
    )
    .unwrap();
    assert_eq!(
        account,
        Reconciliation {
            matched: 1,
            mapped: DECLARED,
            mapped_resident: RESIDENT_PAGES,
            ..Reconciliation::default()
        }
    );
}

#[test]
fn a_mapping_of_the_wrong_extent_is_refused() {
    let err = refusal(mapped(), observed(0, DECLARED / 2, 0));
    assert!(err.contains("mapped bytes"), "{err}");
    assert!(
        err.contains("the declaration and the loader disagree"),
        "{err}"
    );
}

#[test]
fn an_owned_pin_bound_to_a_mapping_is_refused() {
    let err = refusal(decoded(), observed(DECLARED, DECLARED, 1));
    assert!(err.contains("a mapping of"), "{err}");
}

#[test]
fn padding_is_bounded_by_one_page_per_allocation() {
    let page = DEVICE_PAGE_ALIGN as u64;
    let within = reconcile(
        &[expectation(decoded())],
        &[observed(DECLARED + page - 1, 0, 1)],
    )
    .unwrap();
    assert_eq!(within.padding, page - 1);
    let err = refusal(decoded(), observed(DECLARED + page, 0, 1));
    assert!(err.contains("resident in 1 allocation(s)"), "{err}");
    let err = refusal(decoded(), observed(DECLARED - 1, 0, 1));
    assert!(
        err.contains("the declaration and the loader disagree"),
        "{err}"
    );
}

#[test]
fn a_pinned_plan_that_never_ran_is_refused() {
    let op = operand();
    let pins = [record(
        &op,
        RealizationId::cpu(decoded()),
        ResidencyProfile::DECODED_F32,
    )];
    let idle = ProjectionLedger::new();
    let err = ledger_correspondence(&pins, &idle).unwrap_err().to_string();
    assert!(err.contains("it never ran"), "{err}");
    // Nothing pinned and nothing run is an empty, agreeing account.
    assert!(ledger_correspondence(&[], &idle).unwrap().is_empty());
}
