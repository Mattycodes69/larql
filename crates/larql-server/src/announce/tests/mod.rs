//! Unit tests for the grid announce message builders.
use super::*;

fn config() -> AnnounceConfig {
    AnnounceConfig {
        join_url: "http://router:50052".into(),
        model_id: "gemma-test".into(),
        layer_start: 3,
        layer_end: 7,
        listen_url: "http://server:8080".into(),
        ram_bytes: 42,
        grid_key: Some("secret".into()),
        vindex_hash: "abc123".into(),
        shard_sha256: "ab".repeat(32),
        serves_openai: false,
        latency_tracker: Arc::new(LayerLatencyTracker::new()),
        requests_in_flight: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        available_after_drain: None,
        quic_cert_fingerprint: None,
    }
}

#[test]
fn vindex_identity_hash_is_stable_and_hex() {
    let a = vindex_identity_hash("model-a", 30);
    let b = vindex_identity_hash("model-a", 30);
    let c = vindex_identity_hash("model-a", 31);
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.len(), 16);
    assert!(a.chars().all(|ch| ch.is_ascii_hexdigit()));
}

#[test]
fn grid_bearer_value_formats_authorization() {
    let val = grid_bearer_value(Some("secret")).unwrap().unwrap();
    assert_eq!(val.to_str().unwrap(), "Bearer secret");
    assert!(grid_bearer_value(None).unwrap().is_none());
}

#[test]
fn announce_message_copies_config_fields() {
    let cfg = config();
    let msg = announce_message(&cfg);
    let Some(ServerPayload::Announce(announce)) = msg.payload else {
        panic!("expected announce payload");
    };
    assert_eq!(announce.model_id, "gemma-test");
    assert_eq!(announce.layer_start, 3);
    assert_eq!(announce.layer_end, 7);
    assert_eq!(announce.ram_bytes, 42);
    assert_eq!(announce.listen_url, "http://server:8080");
    assert_eq!(announce.vindex_hash, "abc123");
    assert_eq!(announce.shard_sha256, "ab".repeat(32));
}

#[test]
fn heartbeat_message_uses_zeroed_metrics() {
    let tracker = LayerLatencyTracker::new();
    let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut last = 0u64;
    let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
    let Some(ServerPayload::Heartbeat(heartbeat)) = msg.payload else {
        panic!("expected heartbeat payload");
    };
    assert_eq!(heartbeat.cpu_pct, 0.0);
    assert_eq!(heartbeat.ram_used, 0);
    assert_eq!(heartbeat.requests_in_flight, 0);
    assert!(heartbeat.layer_stats.is_empty());
    assert_eq!(heartbeat.req_per_sec, 0.0);
}

#[test]
fn heartbeat_includes_layer_stats_after_recording() {
    let tracker = LayerLatencyTracker::new();
    tracker.record(5, 3.0);
    tracker.record(5, 5.0);
    let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut last = 0u64;
    let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
    let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
        panic!("expected heartbeat");
    };
    assert_eq!(hb.layer_stats.len(), 1);
    assert_eq!(hb.layer_stats[0].layer, 5);
    assert!(hb.layer_stats[0].avg_ms > 0.0);
}

#[test]
fn heartbeat_computes_req_per_sec_from_counter_delta() {
    let tracker = LayerLatencyTracker::new();
    let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut last = 0u64;
    // 50 requests over a 10 s interval = 5 req/s.
    total.store(50, std::sync::atomic::Ordering::Relaxed);
    let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
    let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
        panic!("expected heartbeat");
    };
    assert!(
        (hb.req_per_sec - 5.0).abs() < 0.001,
        "got {}",
        hb.req_per_sec
    );
    assert_eq!(last, 50, "last sample should advance");

    // Second sample: another 30 requests in the same window → 3 req/s.
    total.store(80, std::sync::atomic::Ordering::Relaxed);
    let msg2 = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
    let Some(ServerPayload::Heartbeat(hb2)) = msg2.payload else {
        panic!("expected heartbeat");
    };
    assert!(
        (hb2.req_per_sec - 3.0).abs() < 0.001,
        "got {}",
        hb2.req_per_sec
    );
}

#[test]
fn heartbeat_rate_clamps_to_zero_on_counter_reset() {
    // saturating_sub guards against a counter going backwards. The
    // counter is monotonic in production; this just prevents an
    // underflow spike if a deployer pulls the rug.
    let tracker = LayerLatencyTracker::new();
    let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut last = 100u64;
    let msg = heartbeat_message(&tracker, &rif, &total, &mut last, Duration::from_secs(10));
    let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
        panic!("expected heartbeat");
    };
    assert_eq!(hb.req_per_sec, 0.0);
    assert_eq!(last, 0, "last sample should track the reset counter");
}

#[test]
fn build_available_after_drain_returns_none_without_ram() {
    assert!(
        build_available_after_drain(None, "http://srv", None, None, UnverifiedShards::Refuse)
            .is_none()
    );
}

#[test]
fn build_available_after_drain_uses_default_store_path() {
    let cfg = build_available_after_drain(
        Some(8 * 1024 * 1024 * 1024),
        "http://srv",
        None,
        None,
        UnverifiedShards::Refuse,
    )
    .expect("ram set should produce a config");
    assert_eq!(cfg.ram_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(cfg.listen_url, "http://srv");
    assert_eq!(cfg.store_path, "/tmp/larql-shards");
    assert!(cfg.join_url.is_empty(), "filled per-router by bootstrap");
    assert!(cfg.grid_key.is_none());
    assert_eq!(cfg.unverified_shards, UnverifiedShards::Refuse);
}

#[test]
fn build_available_after_drain_passes_through_overrides() {
    let cfg = build_available_after_drain(
        Some(1),
        "http://srv",
        Some("/mnt/shards"),
        Some("secret"),
        UnverifiedShards::Allow,
    )
    .unwrap();
    assert_eq!(cfg.store_path, "/mnt/shards");
    assert_eq!(cfg.unverified_shards, UnverifiedShards::Allow);
    assert_eq!(cfg.grid_key.as_deref(), Some("secret"));
}

#[test]
fn dropping_message_marks_reassigned() {
    let msg = dropping_message("model".into(), 1, 2);
    let Some(ServerPayload::Dropping(dropping)) = msg.payload else {
        panic!("expected dropping payload");
    };
    assert_eq!(dropping.model_id, "model");
    assert_eq!(dropping.layer_start, 1);
    assert_eq!(dropping.layer_end, 2);
    assert_eq!(dropping.reason, "reassigned");
}
