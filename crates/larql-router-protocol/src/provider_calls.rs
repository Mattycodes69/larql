//! Provider-call diagnostics seam.
//!
//! A transport (e.g. `larql-router`'s HTTP FFN/expert clients) wants to
//! attach per-call timing records to whatever profile capture is active on
//! the calling thread. The capture itself lives in `larql-vindex`'s VINDEX3
//! executor, which this crate must not depend on. So the transport talks to
//! this process-wide observer instead, and the crate that owns the capture
//! installs an adapter once (`larql-inference` does so when it prepares a
//! coordinator). Records then land in the same capture as before.
//!
//! With no observer installed, [`enabled`] is `false` and
//! [`record_provider_call`] is a no-op: diagnostics are optional and never
//! change execution.

use std::sync::OnceLock;

/// Receives provider-call records from transports.
pub trait ProviderCallObserver: Send + Sync {
    /// `true` when a capture on the calling thread wants records. Transports
    /// check this before paying for timing and JSON assembly.
    fn enabled(&self) -> bool;
    /// Attach `call` to the calling thread's capture, if any.
    fn record(&self, call: serde_json::Value);
}

static OBSERVER: OnceLock<&'static dyn ProviderCallObserver> = OnceLock::new();

/// Install the process-wide observer. Returns `true` if this call installed
/// it, `false` if an observer was already installed (the first one stays).
/// Installing is idempotent for a caller that always passes the same one.
pub fn install_observer(observer: &'static dyn ProviderCallObserver) -> bool {
    OBSERVER.set(observer).is_ok()
}

/// `true` once any observer is installed.
pub fn observer_installed() -> bool {
    OBSERVER.get().is_some()
}

/// `true` when an observer is installed and wants records on this thread.
pub fn enabled() -> bool {
    OBSERVER.get().is_some_and(|o| o.enabled())
}

/// Forward `call` to the installed observer; a no-op when none is.
pub fn record_provider_call(call: serde_json::Value) {
    if let Some(observer) = OBSERVER.get() {
        observer.record(call);
    }
}
