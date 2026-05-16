//! Route-lookup hot path and its comparator.
//!
//! Owns the `route()` / `route_all()` entry points, the
//! `rebuild_route_table()` cold-path indexer, and the three-tier
//! `compare_servers_for_route()` comparator. The state-mutation
//! callers (register / deregister / update_heartbeat) live in the
//! parent module — they invoke `rebuild_route_table()` whenever
//! topology changes.

use std::collections::HashMap;

use super::{GridState, ServerEntry};

/// Routing comparator used by [`GridState::route`]. Three-tier:
///
///   1. **GT3 per-layer latency** — when both replicas have a value
///      for `layer`, the one with lower `avg_ms` wins. Replicas with
///      a value beat replicas without (NaN-safe).
///   2. **Active-probe RTT** — when neither side has GT3 data, use
///      `rtt_ms` as a wire-only tie-breaker. Replicas with a probe
///      result beat unprobed ones.
///   3. **Requests in flight** — last resort. Always defined.
///
/// Hoisted out of `route()` so the cascade is directly testable
/// without standing up a full `GridState`. NaN-tolerant — partial
/// orderings collapse to `Equal` rather than panicking.
fn compare_servers_for_route(a: &ServerEntry, b: &ServerEntry, layer: u32) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let lat_a = a.layer_latencies.get(&layer).map(|(avg, _)| *avg);
    let lat_b = b.layer_latencies.get(&layer).map(|(avg, _)| *avg);
    match (lat_a, lat_b) {
        (Some(la), Some(lb)) => la.partial_cmp(&lb).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => match (a.rtt_ms, b.rtt_ms) {
            (Some(ra), Some(rb)) => ra.partial_cmp(&rb).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.requests_in_flight.cmp(&b.requests_in_flight),
        },
    }
}

impl GridState {
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .min_by(|a, b| compare_servers_for_route(a, b, layer))
                .map(|s| s.listen_url.clone())
        })
    }

    /// Resolve all layers in one call — one lock acquisition covers the whole batch.
    /// Returns Ok(layer → url) or Err(first layer with no owning shard).
    #[allow(dead_code)]
    pub fn route_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.route(model_id, layer as u32) {
                Some(url) => {
                    out.insert(layer, url);
                }
                None => return Err(layer),
            }
        }
        Ok(out)
    }

    /// Rebuild layer→servers index. Called only on join/leave (cold path).
    pub(super) fn rebuild_route_table(&mut self) {
        let mut rt: HashMap<(String, u32), Vec<String>> = HashMap::new();
        let mut any: HashMap<u32, Vec<String>> = HashMap::new();
        for entry in self.servers.values() {
            for layer in entry.layer_start..=entry.layer_end {
                rt.entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push(entry.server_id.clone());
                any.entry(layer).or_default().push(entry.server_id.clone());
            }
        }
        self.route_table = rt;
        self.any_model_table = any;
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::entry;
    use super::*;

    #[test]
    fn route_uses_inclusive_layer_ranges() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        assert_eq!(state.route(Some("model-a"), 0).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 3).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 5).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 6), None);
    }

    #[test]
    fn route_without_model_uses_any_model_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));

        assert_eq!(state.route(None, 1).as_deref(), Some("http://a"));
        assert_eq!(state.route(None, 2), None);
    }

    #[test]
    fn route_prefers_least_loaded_replica() {
        let mut state = GridState::default();
        let mut busy = entry("busy", "http://busy", "model-a", 0, 4);
        busy.requests_in_flight = 12;
        let mut idle = entry("idle", "http://idle", "model-a", 0, 4);
        idle.requests_in_flight = 1;

        state.register(busy);
        state.register(idle);

        assert_eq!(
            state.route(Some("model-a"), 3).as_deref(),
            Some("http://idle")
        );
    }

    #[test]
    fn route_all_returns_first_uncovered_layer() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        assert_eq!(state.route_all(Some("model-a"), &[0, 1, 2, 3]), Err(2));
    }

    #[test]
    fn route_prefers_lower_layer_latency_over_inflight() {
        // slow has fewer requests_in_flight but higher per-layer latency.
        // fast has more requests but lower layer latency.
        // Router should route to fast.
        let mut state = GridState::default();
        let mut slow = entry("slow", "http://slow", "model-a", 0, 4);
        slow.requests_in_flight = 2;
        slow.layer_latencies.insert(2, (50.0, 80.0)); // 50 ms avg

        let mut fast = entry("fast", "http://fast", "model-a", 0, 4);
        fast.requests_in_flight = 8;
        fast.layer_latencies.insert(2, (5.0, 9.0)); // 5 ms avg

        state.register(slow);
        state.register(fast);

        assert_eq!(
            state.route(Some("model-a"), 2).as_deref(),
            Some("http://fast")
        );
    }

    #[test]
    fn compare_uses_gt3_latency_when_both_replicas_have_it() {
        let mut fast = entry("fast", "http://fast", "m", 0, 4);
        fast.layer_latencies.insert(2, (5.0, 10.0));
        fast.rtt_ms = Some(100.0); // worse RTT but better latency wins
        let mut slow = entry("slow", "http://slow", "m", 0, 4);
        slow.layer_latencies.insert(2, (50.0, 80.0));
        slow.rtt_ms = Some(1.0);
        assert_eq!(
            compare_servers_for_route(&fast, &slow, 2),
            std::cmp::Ordering::Less,
            "GT3 latency must beat RTT when both replicas have layer stats"
        );
    }

    #[test]
    fn compare_falls_through_to_rtt_when_no_gt3() {
        // Neither replica has GT3 data at layer 2; pick the lower RTT.
        let mut close = entry("close", "http://close", "m", 0, 4);
        close.rtt_ms = Some(1.5);
        close.requests_in_flight = 9; // higher load but lower RTT wins
        let mut far = entry("far", "http://far", "m", 0, 4);
        far.rtt_ms = Some(30.0);
        far.requests_in_flight = 1;
        assert_eq!(
            compare_servers_for_route(&close, &far, 2),
            std::cmp::Ordering::Less,
        );
    }

    #[test]
    fn compare_prefers_replica_with_rtt_data_over_unprobed() {
        let mut probed = entry("probed", "http://p", "m", 0, 4);
        probed.rtt_ms = Some(10.0);
        let unprobed = entry("unprobed", "http://u", "m", 0, 4); // rtt_ms = None
        assert_eq!(
            compare_servers_for_route(&probed, &unprobed, 2),
            std::cmp::Ordering::Less,
        );
        assert_eq!(
            compare_servers_for_route(&unprobed, &probed, 2),
            std::cmp::Ordering::Greater,
        );
    }

    #[test]
    fn compare_falls_through_to_requests_in_flight_when_no_latency_no_rtt() {
        let mut idle = entry("idle", "http://idle", "m", 0, 4);
        idle.requests_in_flight = 1;
        let mut busy = entry("busy", "http://busy", "m", 0, 4);
        busy.requests_in_flight = 8;
        // Neither has layer_latencies nor rtt_ms — fallback runs.
        assert_eq!(
            compare_servers_for_route(&idle, &busy, 2),
            std::cmp::Ordering::Less,
        );
    }
}
