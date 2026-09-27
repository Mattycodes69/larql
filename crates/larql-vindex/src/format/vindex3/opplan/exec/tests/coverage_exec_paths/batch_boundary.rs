//! The batch block-boundary event snapshots only a history plane, at
//! exactly the phase the run's control selects, and records what that
//! control says a snapshot holds.

use super::super::super::attention_residual::{BoundaryPhase, History};
use super::super::super::batch_site::batch_boundary_event;
use super::super::super::hyper_connection::Mutation;
use super::super::super::observe::{CarrierForm, CarrierTransition};
use super::super::super::trace::Plane;
use super::super::super::PlaneEvent;

const HIDDEN: usize = 3;
const LAYER: usize = 2;
/// A nonzero provider base: the boundary's transitions are named at
/// absolute positions `BASE + row` (RESIDUAL-BUS-2 A2).
const BASE: usize = 5;
const PREFIX: f32 = 1.0;
const ENTERING: f32 = 2.0;
const MIXED: f32 = 3.0;

fn histories() -> Plane {
    Plane::Histories(vec![History::new(vec![PREFIX; HIDDEN])])
}

/// What the sink saw: boundary records, and the carrier transitions
/// (RESIDUAL-BUS-1) named beside them, as `(position, transition)`.
#[derive(Default)]
struct Seen {
    boundaries: usize,
    transitions: Vec<(usize, CarrierTransition)>,
}

/// Runs one boundary event and returns the plane and what the sink saw.
fn boundary(mut plane: Plane, phase: BoundaryPhase, mutation: Mutation) -> (Plane, Seen) {
    let mut seen = Seen::default();
    let mut sink = |event: PlaneEvent| {
        match event {
            PlaneEvent::AttentionResidualBoundary(_) => seen.boundaries += 1,
            PlaneEvent::Transition {
                address,
                transition,
            } => {
                assert_eq!(address.layer, LAYER);
                assert_eq!(address.form, CarrierForm::History);
                seen.transitions.push((address.position, transition));
            }
            other => panic!("a boundary event emitted {other:?}"),
        }
        Ok(())
    };
    batch_boundary_event(
        &mut plane,
        phase,
        &[vec![ENTERING; HIDDEN]],
        &[vec![MIXED; HIDDEN]],
        LAYER,
        BASE,
        mutation,
        &mut sink,
    )
    .unwrap();
    (plane, seen)
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
    let (plane, seen) = boundary(
        rows.clone(),
        BoundaryPhase::AfterAttentionReduce,
        Mutation::None,
    );
    assert_eq!(plane, rows);
    assert_eq!(seen.boundaries, 0);
    assert!(seen.transitions.is_empty());
}

#[test]
fn the_snapshot_moves_to_the_phase_its_control_names() {
    let (plane, seen) = boundary(
        histories(),
        BoundaryPhase::BeforeAttentionReduce,
        Mutation::AttnResSiteOverNewSnapshots,
    );
    assert_eq!(seen.boundaries, 1);
    // The snapshot is named; the reset is not, because it belongs to the
    // reference's phase, not the one the control moved the snapshot to.
    assert_eq!(
        seen.transitions,
        vec![(BASE, CarrierTransition::HistorySnapshot)]
    );
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
