//! The executor's defence-in-depth refusals and seldom-taken arms, each
//! reached directly: the traversal's own pre-flight refuses these states
//! first, so only a direct call proves the inner refusal still holds if
//! that pre-flight is ever narrowed.

mod attention_operands;
mod batch_boundary;
mod layer_refusals;
mod session_surface;
mod site_refusals;

use std::path::{Path, PathBuf};

use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

const COMPONENT: &str = "target";

/// A checkpoint encoded, planned and prepared whole-stack on the
/// reference backend.
pub(super) struct Prepared {
    _dirs: (tempfile::TempDir, tempfile::TempDir),
    pub(super) plan: ComponentOpPlan,
    pub(super) ops: PreparedOperands,
}

pub(super) fn prepared(write: fn(&Path)) -> Prepared {
    let checkpoint = tempfile::tempdir().unwrap();
    write(checkpoint.path());
    let out = tempfile::tempdir().unwrap();
    let container: PathBuf = out.path().join("fixture.vindex3");
    encode_checkpoint(checkpoint.path(), &container).expect("the fixture encodes");
    let inspection = inspect_container(&container, false).unwrap();
    let plan = plan_component_ops(&inspection, &container, COMPONENT)
        .unwrap()
        .plan
        .expect("the fixture closes");
    let store = OperandStore::open(&container, &inspection).unwrap();
    let ops =
        PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full).unwrap();
    Prepared {
        _dirs: (checkpoint, out),
        plan,
        ops,
    }
}
