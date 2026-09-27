//! Transport provider-call records reach the VINDEX3 profile capture through
//! the protocol crate's observer seam once inference installs its adapter —
//! the path the router's HTTP transports use now that they no longer call
//! the profiler directly.

use larql_inference::vindex3::dense_ffn::profile::Capture;
use larql_inference::vindex3::provider_observer;
use larql_router_protocol::provider_calls;

#[test]
fn transport_records_land_in_the_active_profile_capture() {
    assert!(provider_observer::install(), "inference owns the slot");
    assert!(provider_observer::install(), "install is idempotent");

    // No capture on this thread: transports see diagnostics disabled.
    assert!(!provider_calls::enabled());

    let capture = Capture::start().expect("fresh thread");
    assert!(provider_calls::enabled());
    provider_calls::record_provider_call(serde_json::json!({"wire": "binary-f32-v1"}));
    let calls = capture.finish_provider_calls();
    assert_eq!(calls, vec![serde_json::json!({"wire": "binary-f32-v1"})]);
    assert!(!provider_calls::enabled(), "finishing ends the capture");
}
