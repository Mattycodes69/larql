//! Replication, gap-fill, and Mode B assignment dispatch.
//!
//! Owns the rebalancer-facing surface: under/over-replication
//! detection, effective-target-with-hot-shard-bump bookkeeping,
//! and the `AssignMsg` dispatch into the available pool. The
//! hot-shard signal itself (elevation set + req/sec scan) lives in
//! the parent module; this file only consumes the elevation bit via
//! [`GridState::effective_target_for`].

use std::collections::HashMap;

use larql_router_protocol::{AssignMsg, RouterMessage, RouterPayload};

use super::{GridState, ServerEntry};

impl GridState {
    pub fn find_origin_for(
        &self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> Option<(String, String)> {
        self.servers
            .values()
            .find(|e| {
                e.model_id == model_id && e.layer_start <= layer_start && e.layer_end >= layer_end
            })
            .map(|e| (e.listen_url.clone(), e.vindex_hash.clone()))
    }

    /// Find the first available server that has at least `min_ram_bytes` of
    /// RAM, resolve a serving origin, send it an `AssignMsg`, and move it out
    /// of the available pool.
    ///
    /// Returns `true` if an assignment was sent. Returns `false` either when no
    /// available server has enough RAM, or when no live replica is left to
    /// serve as origin for the gap.
    pub fn try_assign_gap(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        min_ram_bytes: u64,
    ) -> bool {
        let Some((origin_url, shard_hash)) = self.find_origin_for(model_id, layer_start, layer_end)
        else {
            tracing::warn!(
                model_id = %model_id,
                layers = %format!("{layer_start}-{layer_end}"),
                "Grid: cannot fill gap — no live replica to serve as origin"
            );
            return false;
        };
        self.try_assign_gap_with_origin(
            model_id,
            layer_start,
            layer_end,
            &origin_url,
            &shard_hash,
            min_ram_bytes,
        )
    }

    /// Lower-level assign that takes an explicit origin. Used by tests and by
    /// deployments that supply an external (non-grid) origin store.
    pub fn try_assign_gap_with_origin(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        origin_url: &str,
        shard_hash: &str,
        min_ram_bytes: u64,
    ) -> bool {
        // Find a suitable available server.
        let server_id = self
            .available_servers
            .iter()
            .find(|(_, e)| e.ram_bytes >= min_ram_bytes)
            .map(|(id, _)| id.clone());

        let Some(server_id) = server_id else {
            return false;
        };

        let entry = self.available_servers.remove(&server_id).unwrap();
        let msg = RouterMessage {
            payload: Some(RouterPayload::Assign(AssignMsg {
                model_id: model_id.to_owned(),
                layer_start,
                layer_end,
                origin_url: origin_url.to_owned(),
                shard_hash: shard_hash.to_owned(),
            })),
        };
        if entry.sender.try_send(Ok(msg)).is_ok() {
            tracing::info!(
                server_id = %server_id,
                model_id = %model_id,
                layers = %format!("{layer_start}-{layer_end}"),
                origin_url = %origin_url,
                "Grid: Mode B assignment sent"
            );
            true
        } else {
            tracing::warn!(server_id = %server_id, "Grid: Mode B assignment send failed (peer disconnected)");
            false
        }
    }

    /// Phase 4: configure how many replicas the router maintains per shard
    /// range. Setter so the value can come from CLI in main.rs.
    pub fn set_target_replicas(&mut self, n: u32) {
        // 0 would mean "no servers"; clamp to ≥1.
        self.target_replicas = n.max(1);
    }

    /// Current target_replicas value (read-only).
    pub fn target_replicas(&self) -> u32 {
        self.target_replicas
    }

    /// Effective replication target for a specific shard range.
    /// Equal to `target_replicas`, plus 1 when the range is currently
    /// marked elevated by the hot-shard tick.
    pub fn effective_target_for(&self, model_id: &str, layer_start: u32, layer_end: u32) -> u32 {
        let bump = if self
            .elevated_ranges
            .contains(&(model_id.to_owned(), layer_start, layer_end))
        {
            1
        } else {
            0
        };
        self.target_replicas + bump
    }

    /// Phase 4: ranges whose live replica count exceeds the effective
    /// target. Hot ranges have effective target = target + 1, so the
    /// over-replication tick won't strip a freshly-pulled hot spare;
    /// once the hot signal clears, the elevated bump goes away and the
    /// surplus replica is dropped on the next tick.
    pub fn over_replicated_ranges(&self) -> Vec<(String, u32, u32, u32)> {
        let mut counts: HashMap<(String, u32, u32), u32> = HashMap::new();
        for e in self.servers.values() {
            *counts
                .entry((e.model_id.clone(), e.layer_start, e.layer_end))
                .or_default() += 1;
        }
        let mut out = Vec::new();
        for ((model_id, start, end), count) in counts {
            let effective = self.effective_target_for(&model_id, start, end);
            if count > effective {
                out.push((model_id, start, end, count - effective));
            }
        }
        out.sort();
        out
    }

    /// Phase 4: among servers covering `(model_id, layer_start, layer_end)`,
    /// return the one with the lowest `requests_in_flight`. Used by the
    /// over-replication path to pick which replica to drop.
    pub fn least_loaded_in_range(
        &self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> Option<&ServerEntry> {
        self.servers
            .values()
            .filter(|e| {
                e.model_id == model_id && e.layer_start == layer_start && e.layer_end == layer_end
            })
            .min_by_key(|e| e.requests_in_flight)
    }

    /// Phase 4: ranges whose live replica count is below the effective
    /// target (= `target_replicas` plus the hot-shard bump). Skips ranges
    /// that have zero servers — those are handled by `coverage_gaps()` /
    /// `try_fill_all_gaps()` because they need a different
    /// origin-resolution story (no live replica → no origin).
    pub fn under_replicated_ranges(&self) -> Vec<(String, u32, u32, u32)> {
        // Group by (model_id, layer_start, layer_end) → count of servers.
        let mut counts: HashMap<(String, u32, u32), u32> = HashMap::new();
        for e in self.servers.values() {
            *counts
                .entry((e.model_id.clone(), e.layer_start, e.layer_end))
                .or_default() += 1;
        }
        let mut out = Vec::new();
        for ((model_id, start, end), count) in counts {
            let effective = self.effective_target_for(&model_id, start, end);
            if count > 0 && count < effective {
                out.push((model_id, start, end, effective - count));
            }
        }
        out.sort();
        out
    }

    /// Phase 4: walk under-replicated ranges and dispatch one `AssignMsg`
    /// per range to bring counts closer to `target_replicas`. Returns the
    /// number of assignments sent.
    ///
    /// At most one assignment per range per call — a newly-assigned replica
    /// won't register as serving until `ReadyMsg` arrives, so issuing more
    /// than one assignment per range here would over-replicate. Callers run
    /// this periodically (rebalancer) or after Ready/Available events.
    pub fn try_replicate_from_available(&mut self) -> usize {
        let ranges = self.under_replicated_ranges();
        let mut sent = 0;
        for (model_id, start, end, _deficit) in ranges {
            if self.try_assign_gap(&model_id, start, end, 0) {
                sent += 1;
            }
        }
        sent
    }

    /// ADR-0004 Phase 5: send an `AssignMsg` to a specific available
    /// server, identified by `server_id`. Used by the admin `assign_range`
    /// RPC when the operator wants a deterministic destination instead of
    /// "any spare with enough RAM".
    ///
    /// Returns `Ok(())` on dispatch, `Err(msg)` when the server isn't in
    /// the available pool or its outbound channel rejected the message.
    pub fn send_assign_to_named_available(
        &mut self,
        server_id: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        origin_url: &str,
        shard_hash: &str,
    ) -> Result<(), String> {
        let entry = self
            .available_servers
            .remove(server_id)
            .ok_or_else(|| format!("server_id {server_id:?} is not in the available pool"))?;
        let msg = RouterMessage {
            payload: Some(RouterPayload::Assign(AssignMsg {
                model_id: model_id.to_owned(),
                layer_start,
                layer_end,
                origin_url: origin_url.to_owned(),
                shard_hash: shard_hash.to_owned(),
            })),
        };
        if let Err(e) = entry.sender.try_send(Ok(msg)) {
            // Put the entry back so a follow-up call can retry.
            self.available_servers.insert(server_id.to_string(), entry);
            return Err(format!("send to {server_id:?} failed: {e}"));
        }
        tracing::info!(
            server_id,
            model_id,
            layers = %format!("{layer_start}-{layer_end}"),
            origin_url,
            "Grid: admin-targeted AssignMsg sent"
        );
        Ok(())
    }

    /// Scan current coverage gaps and try to fill each one from the available
    /// pool. Returns the number of assignments sent.
    pub fn try_fill_all_gaps(&mut self) -> usize {
        let gaps = self.coverage_gaps();
        let mut sent = 0;
        for (model_id, layer_start, layer_end) in gaps {
            // RAM estimate: we don't have a true upper bound from the gap
            // alone, so fall back to a permissive 0 (any available server is
            // acceptable). Deployments that need RAM-aware placement should
            // call try_assign_gap_with_origin directly with a real estimate.
            if self.try_assign_gap(&model_id, layer_start, layer_end, 0) {
                sent += 1;
            }
        }
        sent
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::entry;
    use super::*;

    #[test]
    fn send_assign_to_named_available_dispatches_to_specific_server() {
        let mut state = GridState::default();
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        state.register_available("target".into(), tx, 1, 0, "/".into());

        state
            .send_assign_to_named_available(
                "target",
                "test-model",
                10,
                14,
                "http://origin:8090",
                "deadbeef",
            )
            .expect("send must succeed");
        let msg = rx
            .try_recv()
            .expect("AssignMsg should have been queued")
            .expect("ok payload");
        let Some(RouterPayload::Assign(a)) = msg.payload else {
            panic!("expected Assign, got {msg:?}");
        };
        assert_eq!(a.model_id, "test-model");
        assert_eq!(a.layer_start, 10);
        assert_eq!(a.layer_end, 14);
        assert_eq!(a.origin_url, "http://origin:8090");
        assert_eq!(a.shard_hash, "deadbeef");
        // Entry consumed.
        assert!(!state.has_available_servers());
    }

    #[test]
    fn send_assign_to_named_available_unknown_id_errors() {
        let mut state = GridState::default();
        let err = state
            .send_assign_to_named_available("no-such", "test-model", 0, 4, "http://origin", "h")
            .unwrap_err();
        assert!(err.contains("not in the available pool"));
    }

    #[test]
    fn send_assign_to_named_available_failed_send_re_inserts_entry() {
        let mut state = GridState::default();
        // Drop the receiver so the send fails.
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        state.register_available("target".into(), tx, 1, 0, "/".into());

        let err = state
            .send_assign_to_named_available("target", "m", 0, 4, "http://origin", "h")
            .unwrap_err();
        assert!(err.contains("failed"));
        // Entry must still be in the pool for a follow-up retry.
        assert!(state.has_available_servers());
    }

    #[test]
    fn find_origin_for_returns_listen_url_and_hash_of_replica() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a:8080", "model-a", 0, 5);
        a.vindex_hash = "deadbeef".into();
        state.register(a);

        let origin = state.find_origin_for("model-a", 0, 5);
        assert_eq!(origin, Some(("http://a:8080".into(), "deadbeef".into())));

        // Wrong model: no origin.
        assert!(state.find_origin_for("other", 0, 5).is_none());
        // Range outside coverage: no origin.
        assert!(state.find_origin_for("model-a", 6, 9).is_none());
    }

    #[test]
    fn try_assign_gap_resolves_origin_from_live_replica() {
        let mut state = GridState::default();
        // Two replicas of layers 0-5 — one will be the origin for a third
        // available server that fills a fresh assignment.
        let mut a = entry("a", "http://a:8080", "model-a", 0, 5);
        a.vindex_hash = "abc".into();
        state.register(a);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        state.register_available("spare".into(), tx, 16 * 1024 * 1024 * 1024, 0, "/".into());

        // Pretend layers 6-10 became a gap and we need to fill it. There's no
        // live replica for that range, so the assignment should be refused.
        assert!(!state.try_assign_gap("model-a", 6, 10, 0));

        // Now ask to fill an existing range — must find http://a:8080 as origin.
        assert!(state.try_assign_gap("model-a", 0, 5, 0));
        let sent = rx.try_recv().expect("AssignMsg should be queued");
        let Ok(RouterMessage {
            payload: Some(RouterPayload::Assign(assign)),
        }) = sent
        else {
            panic!("expected Assign payload, got: {sent:?}");
        };
        assert_eq!(assign.origin_url, "http://a:8080");
        assert_eq!(assign.shard_hash, "abc");
        assert_eq!(assign.layer_start, 0);
        assert_eq!(assign.layer_end, 5);
    }

    #[test]
    fn set_target_replicas_clamps_to_at_least_one() {
        let mut state = GridState::default();
        assert_eq!(state.target_replicas(), 1);
        state.set_target_replicas(0);
        assert_eq!(state.target_replicas(), 1, "0 must clamp to 1");
        state.set_target_replicas(3);
        assert_eq!(state.target_replicas(), 3);
    }

    #[test]
    fn under_replicated_ranges_reports_deficit_per_range() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // Range 0-4: only one server → deficit 1.
        state.register(entry("a", "http://a", "model-x", 0, 4));
        // Range 5-9: two servers → at target.
        state.register(entry("b", "http://b", "model-x", 5, 9));
        state.register(entry("c", "http://c", "model-x", 5, 9));

        let ranges = state.under_replicated_ranges();
        assert_eq!(ranges, vec![("model-x".to_string(), 0, 4, 1)]);
    }

    #[test]
    fn over_replicated_ranges_reports_surplus() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // 3 replicas of 0-4 — surplus 1.
        state.register(entry("a", "http://a", "model-x", 0, 4));
        state.register(entry("b", "http://b", "model-x", 0, 4));
        state.register(entry("c", "http://c", "model-x", 0, 4));
        // 1 replica of 5-9 — under target, not over.
        state.register(entry("d", "http://d", "model-x", 5, 9));

        let over = state.over_replicated_ranges();
        assert_eq!(over, vec![("model-x".to_string(), 0, 4, 1)]);
    }

    #[test]
    fn least_loaded_in_range_picks_lowest_inflight() {
        let mut state = GridState::default();
        let mut a = entry("a", "http://a", "model-x", 0, 4);
        a.requests_in_flight = 5;
        let mut b = entry("b", "http://b", "model-x", 0, 4);
        b.requests_in_flight = 1;
        let mut c = entry("c", "http://c", "model-x", 0, 4);
        c.requests_in_flight = 9;
        state.register(a);
        state.register(b);
        state.register(c);

        let pick = state.least_loaded_in_range("model-x", 0, 4).unwrap();
        assert_eq!(pick.server_id, "b");

        // Wrong range yields None.
        assert!(state.least_loaded_in_range("model-x", 10, 14).is_none());
    }

    #[test]
    fn under_replicated_ranges_ignores_zero_coverage() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // No server for layers 0-4 — that's a *gap*, handled separately.
        // Provide some other coverage to keep the test realistic.
        state.register(entry("a", "http://a", "model-y", 10, 14));
        // model-y[10-14] has 1/2 → under-replicated.
        let ranges = state.under_replicated_ranges();
        assert_eq!(ranges, vec![("model-y".to_string(), 10, 14, 1)]);
    }

    #[test]
    fn try_replicate_from_available_dispatches_one_per_range() {
        let mut state = GridState::default();
        state.set_target_replicas(2);
        // One server covering 0-4 — under-replicated by 1.
        let mut a = entry("a", "http://a", "model-x", 0, 4);
        a.vindex_hash = "ha".into();
        state.register(a);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        state.register_available("spare".into(), tx, 1, 0, "/".into());

        let sent = state.try_replicate_from_available();
        assert_eq!(sent, 1);
        let msg = rx
            .try_recv()
            .expect("AssignMsg should have been delivered")
            .expect("ok payload");
        let Some(RouterPayload::Assign(a)) = msg.payload else {
            panic!("expected Assign payload");
        };
        assert_eq!(a.model_id, "model-x");
        assert_eq!(a.layer_start, 0);
        assert_eq!(a.layer_end, 4);
        assert_eq!(a.origin_url, "http://a");
        assert_eq!(a.shard_hash, "ha");

        // No more spares → second call assigns nothing.
        let again = state.try_replicate_from_available();
        assert_eq!(again, 0);
    }

    #[test]
    fn try_fill_all_gaps_scans_coverage_and_fills() {
        let mut state = GridState::default();
        // Two shards with a gap at layer 2.
        let mut a = entry("a", "http://a:8080", "model-a", 0, 1);
        a.vindex_hash = "ha".into();
        let mut b = entry("b", "http://b:8080", "model-a", 3, 4);
        b.vindex_hash = "hb".into();
        state.register(a);
        state.register(b);
        // No live replica covers layer 2 alone, so coverage_gaps reports it
        // but find_origin_for returns None — try_fill_all_gaps should send 0.
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        state.register_available("spare".into(), tx, 1, 0, "/".into());
        assert_eq!(state.try_fill_all_gaps(), 0);
    }
}
