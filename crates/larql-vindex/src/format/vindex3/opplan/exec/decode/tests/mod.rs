//! The decode traversal's site entry and exit refuse a carrier that
//! disagrees with the layer's sites or the declared topology — states
//! preparation refuses first, pinned here so the step's own refusals
//! hold if preparation is ever narrowed — and a boundary event on a
//! carrier that keeps no history is a no-op.

use super::super::observe::NoopObserver;
use super::super::reference::ReferenceBackend;
use super::*;

const HIDDEN: usize = 4;
const STREAMS: usize = 2;
const NORM_EPS: f64 = 1e-6;
const LAYER: usize = 0;
const VALUE: f32 = 0.5;

fn row() -> Vec<f32> {
    vec![VALUE; HIDDEN]
}

fn history() -> Carrier {
    Carrier::History(attention_residual::History::new(row()))
}

fn context() -> SiteContext<'static> {
    SiteContext {
        layer: LAYER,
        site: HcSite::Attention,
        position: 0,
        mutation: Mutation::None,
        intervention: None,
        layer_scale: None,
    }
}

fn topology() -> HyperConnection {
    HyperConnection {
        streams: STREAMS,
        sinkhorn_iters: 1,
        sinkhorn_eps: NORM_EPS,
    }
}

fn refusal<T>(result: Result<T, VindexError>) -> String {
    match result {
        Ok(_) => panic!("a disagreeing carrier must refuse"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_single_stream_carrier_under_a_declared_topology_is_refused() {
    let err = refusal(enter_site(
        &Carrier::Single(row()),
        None,
        Some(topology()),
        NORM_EPS,
        Mutation::None,
    ));
    assert!(err.contains("declared topology disagree"), "{err}");
}

#[test]
fn a_history_carrier_without_attention_residual_sites_is_refused() {
    let err = refusal(enter(
        &history(),
        None,
        None,
        HcSite::Attention,
        None,
        NORM_EPS,
        LAYER,
        Mutation::None,
    ));
    assert!(err.contains("no attention-residual sites"), "{err}");
}

#[test]
fn a_boundary_event_on_a_single_stream_carrier_changes_nothing() {
    let mut carrier = Carrier::Single(row());
    boundary_event(
        &mut carrier,
        BoundaryPhase::AfterAttentionReduce,
        &row(),
        &row(),
        context(),
        &mut NoopObserver,
    );
    assert!(matches!(carrier, Carrier::Single(h) if h == row()));
}

#[test]
fn a_history_carrier_leaving_without_its_entry_is_refused() {
    let mut carrier = history();
    let err = refusal(leave_site(
        &ReferenceBackend,
        &mut carrier,
        row(),
        None,
        None,
        context(),
        &mut NoopObserver,
    ));
    assert!(err.contains("left a site with no site entry"), "{err}");
}

#[test]
fn a_single_stream_carrier_handed_a_reduction_is_refused() {
    let reduction = SiteReduction {
        split: hyper_connection::SinkhornSplit {
            pre: vec![VALUE; STREAMS],
            post: vec![VALUE; STREAMS],
            comb: vec![VALUE; STREAMS * STREAMS],
        },
        reduced: row(),
    };
    let mut carrier = Carrier::Single(row());
    let err = refusal(leave_site(
        &ReferenceBackend,
        &mut carrier,
        row(),
        Some(reduction),
        None,
        context(),
        &mut NoopObserver,
    ));
    assert!(err.contains("received a site reduction"), "{err}");
}
