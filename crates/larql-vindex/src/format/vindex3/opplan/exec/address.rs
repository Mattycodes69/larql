//! Where a carrier transition belongs (RESIDUAL-BUS-2, A1).
//!
//! BUS-1 named every carrier change. A name is not a place: a transition
//! that crosses a process boundary must say which position, which layer,
//! which site and which carrier form it belongs to, or a receiver cannot
//! tell it from one that belongs elsewhere. [`CarrierAddress`] is that
//! place. It is the same value in the batch and the decode traversal.
//!
//! `position` is the ABSOLUTE continuation position: a batch over a
//! provider that already holds state numbers from the provider's base,
//! exactly as decode does (A2). `layer` is plan-absolute. `form` is part
//! of the address because the same `(position, layer, site)` means a
//! different thing on rows, bundles and histories; whether a form may
//! cross a process is a separate question ([`super::portability`]).

use super::observe::{CarrierForm, CarrierTransition, SublayerSite};

/// The place one carrier transition belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CarrierAddress {
    /// The absolute continuation position.
    pub position: usize,
    /// The plan-absolute layer.
    pub layer: usize,
    /// The sublayer site, for a transition that belongs to one. `Enter`,
    /// `Scale` and the boundary events belong to the layer, not a site.
    pub site: Option<SublayerSite>,
    pub form: CarrierForm,
}

impl CarrierAddress {
    /// The address of `transition` at `position` and `layer` on a carrier
    /// of `form`. The site is read from the transition, so the two cannot
    /// disagree.
    pub fn of(
        position: usize,
        layer: usize,
        form: CarrierForm,
        transition: &CarrierTransition,
    ) -> Self {
        Self {
            position,
            layer,
            site: transition.site(),
            form,
        }
    }
}

impl CarrierTransition {
    /// The sublayer site this transition belongs to, if any.
    pub fn site(&self) -> Option<SublayerSite> {
        match *self {
            Self::Add { site }
            | Self::HcUpdate { site }
            | Self::HistoryWrite { site, .. }
            | Self::Intervene { site, .. } => Some(site),
            Self::Enter | Self::Scale | Self::HistorySnapshot | Self::HistoryReset => None,
        }
    }
}
