//! A decode session reports the image it runs — pins and allocations —
//! and refuses to advance a layer-range image through token ids, which
//! it has no embedding table to look up.

use super::super::super::decode::DecodeSession;
use super::super::super::kv::RowKvState;
use super::super::decode::fixture;
use super::*;
use crate::format::vindex3::fixtures::G_TOKENS;

/// The first layer a tail image executes; every layer before it is the
/// caller's.
const TAIL_START: usize = 1;

#[test]
fn a_session_reports_the_pins_and_allocations_of_its_image() {
    let (_dir, plan, store) = fixture();
    let ops =
        PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full).unwrap();
    let mut kv = RowKvState::default();
    let session = DecodeSession::over_prepared(&plan, &ops, &ReferenceBackend, &mut kv).unwrap();
    assert_eq!(session.realizations(), ops.realizations());
    assert!(!session.realizations().is_empty());
    assert_eq!(
        format!("{:?}", session.allocation_census()),
        format!("{:?}", ops.allocation_census())
    );
}

#[test]
fn a_layer_range_image_refuses_token_ids() {
    let (_dir, plan, store) = fixture();
    let tail = PreparedOperands::load(
        &plan,
        &store,
        &ReferenceBackend,
        ExecutionSlice::LayerRange {
            start: TAIL_START,
            end: plan.layers.len(),
        },
    )
    .unwrap();
    let mut kv = RowKvState::default();
    let mut session =
        DecodeSession::over_prepared(&plan, &tail, &ReferenceBackend, &mut kv).unwrap();
    let err = match session.step_many(&G_TOKENS) {
        Ok(_) => panic!("a layer-range image has no embedding table"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("carries no embedding table"), "{err}");
    assert_eq!(session.position(), 0, "refused before any state moved");
}
