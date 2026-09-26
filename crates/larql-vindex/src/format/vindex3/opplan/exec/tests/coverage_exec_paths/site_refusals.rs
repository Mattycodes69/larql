//! A batch site whose carrier, operands and declared topology disagree is
//! an executor bug, and the site refuses it by name rather than running
//! the wrong residual programme.

use larql_models::config::HyperConnection;

use super::super::super::attention_residual::History;
use super::super::super::batch_site::{
    enter_batch_site, leave_batch_site, BatchSite, BatchSiteContext,
};
use super::super::super::hyper_connection::Mutation;
use super::super::super::observe::HcSite;
use super::super::super::trace::Plane;
use super::super::super::PlaneEvent;
use super::*;

const HIDDEN: usize = 4;
const POSITIONS: usize = 2;
const NORM_EPS: f64 = 1e-6;
const LAYER: usize = 0;

fn rows() -> Vec<Vec<f32>> {
    vec![vec![1.0; HIDDEN]; POSITIONS]
}

fn bare_site() -> BatchSite<'static> {
    BatchSite {
        layer: LAYER,
        which: HcSite::Attention,
        hyper_connection: None,
        attention_residual: None,
    }
}

fn entry_refusal(plane: &Plane, topology: Option<HyperConnection>) -> String {
    match enter_batch_site(plane, bare_site(), topology, NORM_EPS, Mutation::None) {
        Ok(_) => panic!("a disagreeing site must refuse"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_history_plane_without_attention_residual_sites_is_refused() {
    let plane = Plane::Histories(
        (0..POSITIONS)
            .map(|_| History::new(vec![0.0; HIDDEN]))
            .collect(),
    );
    let err = entry_refusal(&plane, None);
    assert!(err.contains("residual-history plane"), "{err}");
}

#[test]
fn a_row_plane_under_a_declared_hyper_connection_is_refused() {
    let topology = HyperConnection {
        streams: POSITIONS,
        sinkhorn_iters: 1,
        sinkhorn_eps: NORM_EPS,
    };
    let err = entry_refusal(&Plane::Rows(rows()), Some(topology));
    assert!(err.contains("declared topology disagree"), "{err}");
}

#[test]
fn a_row_plane_handed_site_reductions_on_the_way_out_is_refused() {
    let mut plane = Plane::Rows(rows());
    let mut sink = |_: PlaneEvent| Ok(());
    let err = leave_batch_site(
        &ReferenceBackend,
        &mut plane,
        rows(),
        Some(Vec::new()),
        None,
        BatchSiteContext {
            layer: LAYER,
            site: HcSite::Attention,
            mutation: Mutation::None,
        },
        &mut sink,
    )
    .expect_err("a row plane has no reductions to apply")
    .to_string();
    assert!(
        err.contains("a row plane received site reductions"),
        "{err}"
    );
}
