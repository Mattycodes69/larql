//! Rendering for `larql k3-ledger` — I/O, excluded from coverage.
//!
//! Every quoted ratio names its convention (R0). Anything resting on an
//! unmeasured input says so in the output rather than in a footnote.

use super::geometry::K3Geometry;

mod bank_scan;
mod census;
mod serving;
mod tables;
pub use bank_scan::*;
pub use census::*;
pub use serving::*;
pub use tables::*;

type R = Result<(), Box<dyn std::error::Error>>;

fn header(geom: &K3Geometry) {
    println!(
        "K3 geometry: {} layers ({} KDA + {} MLA), {}-of-{} ({:.2}% activation), \
         expert {:.2} MB",
        geom.n_layers,
        geom.n_kda_layers,
        geom.n_mla_layers,
        geom.top_k,
        geom.n_experts,
        100.0 * geom.activation_fraction(),
        geom.expert_bytes as f64 / 1e6,
    );
    println!(
        "activated {:.2} B params (dense {:.2} B + routed {:.2} B); vision {}",
        geom.activated_params() as f64 / 1e9,
        geom.dense_params() as f64 / 1e9,
        geom.routed_activated_params() as f64 / 1e9,
        if geom.vision_included {
            "included"
        } else {
            "EXCLUDED"
        },
    );
    if geom.branches_are_equal() {
        println!(
            "branches w1/w2/w3 equal to the byte -> dropping one caps at {:.2}x (retention {:.3})",
            1.0 / geom.up_fold_retention(),
            geom.up_fold_retention(),
        );
    }
}
