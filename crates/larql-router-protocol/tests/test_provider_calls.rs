//! The provider-call observer seam. The observer slot is process-wide and
//! can be installed once, so this binary checks the whole lifecycle — before
//! install, install, repeat install — in one ordered test.

use std::cell::RefCell;

use larql_router_protocol::provider_calls::{
    enabled, install_observer, observer_installed, record_provider_call, ProviderCallObserver,
};

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

struct Other;

impl ProviderCallObserver for Other {
    fn enabled(&self) -> bool {
        true
    }
    fn record(&self, _call: serde_json::Value) {
        panic!("a second observer must never replace the first");
    }
}

static CAPTURE: ThreadCapture = ThreadCapture;
static OTHER: Other = Other;

#[test]
fn observer_lifecycle() {
    // No observer: diagnostics are off and recording is a harmless no-op.
    assert!(!observer_installed());
    assert!(!enabled());
    record_provider_call(serde_json::json!({"dropped": true}));

    assert!(install_observer(&CAPTURE), "first install wins");
    assert!(observer_installed());
    assert!(!install_observer(&OTHER), "the slot is taken");
    assert!(!install_observer(&CAPTURE), "repeat install is a no-op");

    // Installed but no capture on this thread: still disabled.
    assert!(!enabled());
    record_provider_call(serde_json::json!({"no_capture": true}));

    CALLS.with(|c| *c.borrow_mut() = Some(Vec::new()));
    assert!(enabled());
    record_provider_call(serde_json::json!({"kind": "ffn", "layer": 3}));
    let calls = CALLS.with(|c| c.borrow_mut().take()).unwrap();
    assert_eq!(calls, vec![serde_json::json!({"kind": "ffn", "layer": 3})]);

    // The capture is per thread: another thread sees it disabled.
    CALLS.with(|c| *c.borrow_mut() = Some(Vec::new()));
    assert!(!std::thread::spawn(enabled).join().unwrap());
}
