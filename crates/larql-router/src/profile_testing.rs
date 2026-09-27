//! Test-only provider-call capture.
//!
//! The router's transports record diagnostics through
//! `larql_router_protocol::provider_calls`; in a real coordinator process
//! `larql-inference` installs an observer that forwards them to the VINDEX3
//! profile. The router's own tests have no inference crate, so they install
//! this thread-local capture instead and read back exactly what a transport
//! recorded.

use std::cell::RefCell;

use larql_router_protocol::provider_calls::{self, ProviderCallObserver};

thread_local! {
    static CALLS: RefCell<Option<Vec<serde_json::Value>>> = const { RefCell::new(None) };
}

struct ThreadCapture;

impl ProviderCallObserver for ThreadCapture {
    fn enabled(&self) -> bool {
        CALLS.with(|c| c.borrow().is_some())
    }
    fn record(&self, call: serde_json::Value) {
        CALLS.with(|c| {
            if let Some(calls) = c.borrow_mut().as_mut() {
                calls.push(call);
            }
        });
    }
}

static THREAD_CAPTURE: ThreadCapture = ThreadCapture;

/// An active capture on the current thread. Dropping it disables capture.
pub(crate) struct Capture(());

impl Capture {
    pub(crate) fn start() -> Self {
        provider_calls::install_observer(&THREAD_CAPTURE);
        CALLS.with(|c| *c.borrow_mut() = Some(Vec::new()));
        Self(())
    }
    pub(crate) fn finish_provider_calls(self) -> Vec<serde_json::Value> {
        CALLS.with(|c| c.borrow_mut().take().unwrap_or_default())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        CALLS.with(|c| c.borrow_mut().take());
    }
}
