//! V3-STREAM-1 witnesses (`docs/v3-stream-1-run-record.md`): S1 lossless,
//! S2 replay identity, S3 sequencing, S4 bounded and counted live loss,
//! S5 observation parity through the recorder, S6 receipt integrity, S7
//! provenance on the record; S8 on a real container behind
//! `LARQL_V3_CONTAINER`.

use std::sync::mpsc::sync_channel;

use larql_vindex::format::vindex3::fixtures::{miniature_glimmer, G_HIDDEN, G_LAYERS, G_TOKENS};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
use larql_vindex::format::vindex3::opplan::exec::observe::{
    NoopObserver, RecordingObserver, StepObserver,
};
use larql_vindex::format::vindex3::opplan::exec::observe_stats::{FixedBasis, StatsObserver};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::provenance::{ExecutionProvenance, RunProvenance};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use crate::vindex3::record::{
    hash_lines, EventKind, RecordError, RecordedEvent, RunIdentity, RunRecord, RunRecorder, Site,
    RECORD_SCHEMA,
};

const COMPONENT: &str = "target";
const DIMS: usize = 3;
const SEED: u64 = 11;
/// F3: a tap this small on the golden run must drop most of it.
const SMALL_TAP: usize = 8;
const LARGE_TAP: usize = 1_000;

struct Fixture {
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn fixture() -> Fixture {
    let container = super::container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _container: container,
        plan: outcome.plan.unwrap(),
        store,
    }
}

fn stats() -> StatsObserver {
    StatsObserver::new(FixedBasis::seeded(G_HIDDEN, DIMS, SEED).unwrap(), None)
}

fn identity() -> RunIdentity {
    RunIdentity::new("run-test", "mini-glimmer", COMPONENT, &G_TOKENS)
}

/// Step every token with `observer`, returning the per-position logits.
fn run<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    observer: &mut dyn StepObserver,
) -> Vec<Vec<f32>> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    G_TOKENS
        .iter()
        .map(|&t| session.step_observed(t, observer).unwrap().logits.unwrap())
        .collect()
}

/// Part-by-part equality that names the first differing event instead
/// of dumping two records.
fn assert_record_eq(read: &RunRecord, written: &RunRecord) {
    assert_eq!(read.identity, written.identity, "identity");
    assert_eq!(read.provenance, written.provenance, "provenance");
    assert_eq!(
        read.provenance_fingerprint, written.provenance_fingerprint,
        "fingerprint"
    );
    assert_eq!(read.receipt, written.receipt, "receipt");
    assert_eq!(read.events.len(), written.events.len(), "event count");
    for (index, (a, b)) in read.events.iter().zip(&written.events).enumerate() {
        assert_eq!(a, b, "event {index} differs after the round trip");
    }
    assert_eq!(read, written);
}

/// F1: per position 1 embedded + 1 entering + per layer 2 sites × (stats,
/// write, boundary) + 1 logits.
fn forecast_events_per_position() -> usize {
    1 + 1 + G_LAYERS * 2 * 3 + 1
}

mod record_basics;
mod v3_lens_1_on_the_record;
