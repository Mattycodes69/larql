//! Routes transport provider-call records into the VINDEX3 profile capture.
//!
//! Transports (e.g. `larql-router`'s HTTP FFN/expert clients) record through
//! `larql_router_protocol::provider_calls`, which cannot see `larql-vindex`.
//! This adapter forwards those records to
//! `larql_vindex::…::exec::profile`, the thread-local capture the executor
//! owns, so they land in exactly the capture they did when transports called
//! the profiler directly. Coordinator preparation installs it.

use larql_router_protocol::provider_calls::{self, ProviderCallObserver};
use larql_vindex::format::vindex3::opplan::exec::profile;

struct VindexProfile;

impl ProviderCallObserver for VindexProfile {
    fn enabled(&self) -> bool {
        profile::enabled()
    }
    fn record(&self, call: serde_json::Value) {
        profile::record_provider_call(call);
    }
}

static VINDEX_PROFILE: VindexProfile = VindexProfile;

/// Install the VINDEX3 profile as the process's provider-call observer.
/// Idempotent; returns `true` when the installed observer is this one and
/// `false` when some other observer claimed the slot first (records then go
/// there — diagnostics only, execution is unaffected).
pub fn install() -> bool {
    static INSTALLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *INSTALLED.get_or_init(|| provider_calls::install_observer(&VINDEX_PROFILE))
}
