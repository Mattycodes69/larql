//! Whether an addressed carrier may cross a process boundary
//! (RESIDUAL-BUS-2, A3).
//!
//! Addressable is not portable. Every transition has an address
//! ([`super::address::CarrierAddress`]); only some carrier forms have a
//! representation another process can read back. This is the one place
//! that says which, so the boundaries that hand a carrier to another
//! process (the layer-prefix RPC, the CLI's plane files) cannot disagree.
//!
//! In-process resumption is a different question and is not decided here:
//! a bundle `ResumePoint` inside one process is supported, because nothing
//! leaves the process. Continuation state (KV rows, recurrent and latent
//! state) has no export path at all, so it cannot reach a boundary to be
//! refused; a portable form for it, or for bundles and histories, is its
//! own later rung.

use larql_models::config::ResidualTopology;

use super::observe::CarrierForm;
use crate::error::VindexError;

/// The carrier form a component's residual takes.
pub fn carrier_form_of(topology: ResidualTopology) -> CarrierForm {
    match topology {
        ResidualTopology::SingleStream => CarrierForm::Single,
        ResidualTopology::HyperConnection(_) => CarrierForm::Bundle,
        ResidualTopology::AttentionResidual { .. } => CarrierForm::History,
    }
}

/// Refuse a carrier form that has no representation another process can
/// read back.
pub fn ensure_portable(form: CarrierForm) -> Result<(), VindexError> {
    let why = match form {
        CarrierForm::Single => return Ok(()),
        CarrierForm::Bundle => "a hyper-connected bundle has no serialised form",
        CarrierForm::History => {
            "an attention-residual history (prefix plus snapshots) has no serialised form"
        }
    };
    Err(VindexError::Parse(format!(
        "this carrier cannot cross a process boundary: {why}. It is addressable, but not \
         portable; a portable {form:?} carrier is its own rung (RESIDUAL-BUS-2 A3)"
    )))
}
