//! The receipt binds the evidence files by digest, and `verify_record`
//! refuses a record whose evidence was replaced after the run.

use super::*;
use crate::format::vindex3::represent::measure::plan::record::verify_record;
use crate::format::vindex3::represent::measure::plan::sketch::{
    SketchSpec, SKETCH_FILE, SKETCH_GENERATOR,
};

const DIM: usize = 16;
const SEED: u64 = 5;

/// A finished run with a sketch, and its receipt.
fn sketched_run() -> (Fixture, PlanReceipt) {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let request = PlanMeasureRequest {
        sketch: Some(SketchSpec::new(DIM, SEED).unwrap()),
        ..request(&f)
    };
    let receipt = run(&request, &mut r, &mut c).expect("admissible");
    (f, receipt)
}

fn seal_broken_on(result: Result<PlanReceipt, PlanRefusal>) -> String {
    match result {
        Err(PlanRefusal::Inadmissible(PlanInadmissible::SealMismatch { what, .. })) => what,
        other => panic!("expected a seal mismatch, got {other:?}"),
    }
}

fn flip_last_byte(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn the_receipt_binds_the_sketch_spec_and_both_evidence_files() {
    let (f, receipt) = sketched_run();
    let sketch = receipt
        .sketch
        .as_ref()
        .expect("the receipt binds the sketch");
    assert_eq!(sketch.generator, SKETCH_GENERATOR);
    assert_eq!(sketch.spec, SketchSpec::new(DIM, SEED).unwrap());
    assert_eq!(sketch.positions as u64, receipt.facts.positions);
    assert_eq!(receipt.positions_sha256.len(), 64);
    assert_eq!(verify_record(&f.output).expect("untouched"), receipt);
}

#[test]
fn a_replaced_positions_file_breaks_the_seal() {
    let (f, _) = sketched_run();
    flip_last_byte(&f.output.join(POSITIONS_FILE));
    assert_eq!(seal_broken_on(verify_record(&f.output)), POSITIONS_FILE);
}

#[test]
fn a_replaced_sketch_breaks_the_seal() {
    let (f, _) = sketched_run();
    flip_last_byte(&f.output.join(SKETCH_FILE));
    assert_eq!(seal_broken_on(verify_record(&f.output)), SKETCH_FILE);
}

#[test]
fn a_run_without_a_sketch_binds_its_positions_and_nothing_else() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let receipt = run(&request(&f), &mut r, &mut c).expect("admissible");
    assert!(receipt.sketch.is_none());
    let written = std::fs::read_to_string(f.output.join(RECEIPT_FILE)).unwrap();
    assert!(!written.contains("\"sketch\""), "no sketch key is written");
    assert_eq!(verify_record(&f.output).expect("untouched"), receipt);
}

fn refused(result: Result<PlanReceipt, PlanRefusal>) -> String {
    match result {
        Err(PlanRefusal::Execution(PlanExecutionFailure::RequestRefused { detail })) => detail,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_receipt_from_before_evidence_digests_cannot_be_verified() {
    let (f, _) = sketched_run();
    let path = f.output.join(RECEIPT_FILE);
    let mut written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let receipt = written["receipt"].as_object_mut().unwrap();
    receipt.remove("positions_sha256");
    receipt.remove("sketch");
    std::fs::write(&path, serde_json::to_vec(&written).unwrap()).unwrap();
    assert!(refused(verify_record(&f.output)).contains("predates"));
}

#[test]
fn an_inadmissible_or_missing_receipt_has_no_evidence_to_verify() {
    let tmp = tempfile::tempdir().unwrap();
    match verify_record(tmp.path()) {
        Err(PlanRefusal::Execution(PlanExecutionFailure::ArtifactUnreadable { detail })) => {
            assert!(detail.contains(RECEIPT_FILE), "{detail}")
        }
        other => panic!("expected an unreadable receipt, got {other:?}"),
    }
    std::fs::write(tmp.path().join(RECEIPT_FILE), b"not json").unwrap();
    assert!(matches!(
        verify_record(tmp.path()),
        Err(PlanRefusal::Execution(
            PlanExecutionFailure::ArtifactUnreadable { .. }
        ))
    ));
    std::fs::write(
        tmp.path().join(RECEIPT_FILE),
        br#"{"admissible": false, "refusal": "anything"}"#,
    )
    .unwrap();
    assert!(refused(verify_record(tmp.path())).contains("not admissible"));
    std::fs::write(
        tmp.path().join(RECEIPT_FILE),
        br#"{"admissible": true, "receipt": {"procedure": 1}}"#,
    )
    .unwrap();
    assert!(matches!(
        verify_record(tmp.path()),
        Err(PlanRefusal::Execution(
            PlanExecutionFailure::ArtifactUnreadable { .. }
        ))
    ));
}
