//! two-scoreboard timing decomposition (dec-funnel §3 DEC-1A)

use super::*;

#[test]
fn derive_transmit_us_subtracts_serve_from_client_window() {
    // client_ms covers send→receive only, so transmit = client − serve.
    let (t, clamped) = derive_transmit_us(5.0, Some(3200.0));
    assert_eq!(t, Some(5000.0 - 3200.0));
    assert!(!clamped);
}

#[test]
fn derive_transmit_us_none_without_serve() {
    assert_eq!(derive_transmit_us(5.0, None), (None, false));
}

#[test]
fn derive_transmit_us_clamps_negative_to_zero_and_flags() {
    // serve ≥ measured window (clock anomaly): clamp at 0, flag it.
    let (t, clamped) = derive_transmit_us(1.0, Some(1500.0));
    assert_eq!(t, Some(0.0));
    assert!(clamped);
    // Exactly equal is 0 but NOT an anomaly.
    let (t, clamped) = derive_transmit_us(1.0, Some(1000.0));
    assert_eq!(t, Some(0.0));
    assert!(!clamped);
}

#[test]
fn summarize_two_scoreboard_fields_from_samples() {
    let point = SweepPoint {
        batch: 2,
        wire: WireSpec::Plain(WireArm::F32),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfn,
    };
    let stats = SweepPointStats {
        step_ms: vec![10.0, 20.0],
        samples: vec![
            sample(0, 4.0, Some(3.0)),  // serve 3000us, transmit 1000us
            sample(1, 6.0, Some(5.0)),  // serve 5000us, transmit 1000us
            sample(0, 8.0, Some(6.0)),  // serve 6000us, transmit 2000us
            sample(1, 12.0, Some(9.0)), // serve 9000us, transmit 3000us
        ],
    };
    let s = summarize(&point, &stats, None, None);
    // queue_ms structurally 0 at driver concurrency 1.
    assert_eq!(s.queue_ms, 0.0);
    assert_eq!(s.encode_us_p50, 40.0);
    assert_eq!(s.client_decode_us_p50, 25.0);
    // Nearest-rank percentiles over serve [3000, 5000, 6000, 9000]us:
    // p50 idx 2 → 6000, p99 idx 3 → 9000.
    assert_eq!(s.serve_us_p50, Some(6000.0));
    assert_eq!(s.serve_us_p99, Some(9000.0));
    // transmit [1000, 1000, 2000, 3000]us: p50 idx 2 → 2000.
    assert_eq!(s.transmit_us_p50, Some(2000.0));
    assert_eq!(s.transmit_us_p99, Some(3000.0));
    assert_eq!(s.transmit_us_clamped, 0);
    // serve_us is the µs twin of the legacy server_ms scoreboard.
    assert_eq!(s.server_ms_p50, Some(6.0));
}

#[test]
fn summarize_counts_clamped_transmit_samples() {
    let point = SweepPoint {
        batch: 1,
        wire: WireSpec::Plain(WireArm::F32),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfn,
    };
    let stats = SweepPointStats {
        step_ms: vec![1.0],
        // serve (2500us) exceeds the 1ms client window → clamped.
        samples: vec![sample(0, 1.0, Some(2.5)), sample(0, 4.0, Some(2.0))],
    };
    let s = summarize(&point, &stats, None, None);
    assert_eq!(s.transmit_us_clamped, 1);
    // Clamped sample contributes 0 (not a negative) to the transmit
    // series [0, 2000]us — nearest-rank p50/p99 both land on idx 1.
    assert_eq!(s.transmit_us_p50, Some(2000.0));
    assert_eq!(s.transmit_us_p99, Some(2000.0));
}

#[test]
fn summarize_omits_serve_and_transmit_without_timing_data() {
    // Pre-extension server on a trailer endpoint: serve_us all None →
    // serve/transmit keys absent from the serialised summary, while
    // encode/decode/queue (client-measured) are always present.
    let point = SweepPoint {
        batch: 1,
        wire: WireSpec::Plain(WireArm::Q8k),
        dispatch: DispatchMode::Streaming,
        endpoint: Endpoint::WalkFfnQ8k,
    };
    let stats = SweepPointStats {
        step_ms: vec![5.0],
        samples: vec![sample(0, 5.0, None)],
    };
    let s = summarize(&point, &stats, None, None);
    assert_eq!(s.serve_us_p50, None);
    assert_eq!(s.transmit_us_p50, None);
    assert_eq!(s.transmit_us_clamped, 0);
    let json = serde_json::to_value(&s).unwrap();
    assert!(json.get("serve_us_p50").is_none(), "omitted when None");
    assert!(json.get("transmit_us_p50").is_none());
    assert!(json.get("queue_ms").is_some());
    assert!(json.get("encode_us_p50").is_some());
    assert!(json.get("client_decode_us_p50").is_some());
}
