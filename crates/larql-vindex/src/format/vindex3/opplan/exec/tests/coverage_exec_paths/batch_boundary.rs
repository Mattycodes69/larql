//! The batch block-boundary event snapshots only a history plane, at
//! exactly the phase the run's control selects, and records what that
//! control says a snapshot holds.

use super::super::super::attention_residual::{BoundaryPhase, History};
use super::super::super::batch_site::batch_boundary_event;
use super::super::super::hyper_connection::Mutation;
use super::super::super::trace::Plane;
use super::super::super::PlaneEvent;

const HIDDEN: usize = 3;
const LAYER: usize = 2;
const PREFIX: f32 = 1.0;
const ENTERING: f32 = 2.0;
const MIXED: f32 = 3.0;

fn histories() -> Plane {
    Plane::Histories(vec![History::new(vec![PREFIX; HIDDEN])])
}

/// Runs one boundary event and returns the plane and the number of
/// events the sink saw.
fn boundary(mut plane: Plane, phase: BoundaryPhase, mutation: Mutation) -> (Plane, usize) {
    let mut events = 0usize;
    let mut sink = |_: PlaneEvent| {
        events += 1;
        Ok(())
    };
    batch_boundary_event(
        &mut plane,
        phase,
        &[vec![ENTERING; HIDDEN]],
        &[vec![MIXED; HIDDEN]],
        LAYER,
        mutation,
        &mut sink,
    )
    .unwrap();
    (plane, events)
}

fn only_snapshot(plane: &Plane) -> Vec<f32> {
    let Plane::Histories(histories) = plane else {
        panic!("a history plane stays one");
    };
    assert_eq!(histories[0].snapshot_count(), 1);
    histories[0].snapshots()[0].clone()
}

#[test]
fn a_row_plane_has_no_boundary_to_mark() {
    let rows = Plane::Rows(vec![vec![PREFIX; HIDDEN]]);
    let (plane, events) = boundary(
        rows.clone(),
        BoundaryPhase::AfterAttentionReduce,
        Mutation::None,
    );
    assert_eq!(plane, rows);
    assert_eq!(events, 0);
}

#[test]
fn the_snapshot_moves_to_the_phase_its_control_names() {
    let (plane, events) = boundary(
        histories(),
        BoundaryPhase::BeforeAttentionReduce,
        Mutation::AttnResSiteOverNewSnapshots,
    );
    assert_eq!(events, 1);
    assert_eq!(only_snapshot(&plane), vec![ENTERING; HIDDEN]);
}

#[test]
fn a_snapshot_after_the_attention_branch_holds_the_prefix() {
    let (plane, _) = boundary(
        histories(),
        BoundaryPhase::AfterAttentionBranch,
        Mutation::AttnResSnapshotAfterAttention,
    );
    assert_eq!(only_snapshot(&plane), vec![PREFIX; HIDDEN]);
}

#[test]
fn the_mixed_vector_control_snapshots_the_mixed_vector() {
    let (plane, _) = boundary(
        histories(),
        BoundaryPhase::AfterAttentionReduce,
        Mutation::AttnResSnapshotIsMixedVector,
    );
    assert_eq!(only_snapshot(&plane), vec![MIXED; HIDDEN]);
}
