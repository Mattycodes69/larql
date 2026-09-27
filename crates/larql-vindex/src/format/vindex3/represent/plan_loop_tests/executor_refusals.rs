//! The plan-v1 executor forwards an arm it cannot build as a refusal to
//! instruct, naming the procedure, rather than running a partial pair.

use super::*;
use crate::format::vindex3::represent::actuate::executor::ExecutionRefusal;

/// A component the source container does not hold.
const ABSENT_COMPONENT: &str = "no-such-component";

#[test]
fn an_arm_that_cannot_be_built_is_not_instructable() {
    let f = fixture(None);
    let candidate = f.workdir.join("uniform");
    compile_representation(&f.source, &candidate, &f.spec).unwrap();
    let established = ArtifactStateEvidence::establish(&candidate).unwrap();
    let key = f
        .snapshot
        .standing_intent()
        .key_for(established.established());
    let request = MeasurementRequest::of(&f.snapshot, &key, &BTreeSet::new()).unwrap();
    let locator = DeclaredArtifacts::new()
        .container_at(&f.source)
        .corpus_at(&f.bank)
        .overlay_at(key.state(), &candidate);
    let mut executor = executor(&f);
    executor.component = ABSENT_COMPONENT.into();
    let refusal = executor.execute(&request, &locator).err();
    let Some(ExecutionRefusal::NotInstructable { procedure, detail }) = refusal else {
        panic!("an absent component must refuse to instruct, got {refusal:?}");
    };
    assert_eq!(procedure, executor.procedure());
    assert!(!detail.is_empty());
}
