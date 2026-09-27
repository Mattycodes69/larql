//! `plan_resolved` — the one plan-by-source sequence every front door
//! shares: the verdict names each artifact by the spec the caller typed,
//! and a spec list that does not pair one-to-one with the resolved
//! artifacts is refused rather than zipped short.

use super::super::plan_resolved;
use crate::format::vindex3::artifact::resolve;
use crate::format::vindex3::fixtures::dense_f32_model;

#[test]
fn the_verdict_names_each_artifact_by_the_spec_the_caller_gave() {
    let checkpoint = tempfile::tempdir().unwrap();
    dense_f32_model(checkpoint.path());
    let spec = checkpoint.path().to_path_buf();
    let resolved = resolve(&spec).unwrap();
    let plan = plan_resolved(std::slice::from_ref(&spec), vec![resolved]).unwrap();
    assert_eq!(plan.artifacts.len(), 1);
    let source = &plan.artifacts[0].source;
    assert_eq!(source.path, spec.display().to_string());
    assert_eq!(source.revision, None, "a local checkpoint has no commit");
}

#[test]
fn a_spec_list_that_does_not_pair_with_the_artifacts_is_refused() {
    let checkpoint = tempfile::tempdir().unwrap();
    dense_f32_model(checkpoint.path());
    let resolved = resolve(checkpoint.path()).unwrap();
    let err = match plan_resolved(&[], vec![resolved]) {
        Ok(_) => panic!("an unpaired artifact must refuse"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("0 spec(s) but 1 resolved artifact(s)"),
        "{err}"
    );
}
