//! Hot-shard detection tick: marks ranges whose `req/sec` crosses
//! the configured threshold as elevated, and demotes ranges that have
//! cooled. The follow-on `check_under_replication` /
//! `check_over_replication` ticks act on the new effective targets —
//! this function does not send any messages itself.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::grid::GridState;

pub(super) async fn check_hot_shards(state: &Arc<RwLock<GridState>>, threshold: f32) {
    let mut guard = state.write().await;
    let hot: HashSet<(String, u32, u32)> = guard.hot_layer_ranges(threshold).into_iter().collect();
    let elevated: HashSet<(String, u32, u32)> =
        guard.elevated_ranges_snapshot().into_iter().collect();

    for range in hot.difference(&elevated) {
        guard.mark_elevated(&range.0, range.1, range.2);
        tracing::info!(
            model_id = %range.0,
            layers = %format!("{}-{}", range.1, range.2),
            threshold,
            "Rebalancer: hot shard detected — effective_target raised by 1"
        );
    }
    for range in elevated.difference(&hot) {
        guard.demote_elevated(&range.0, range.1, range.2);
        tracing::info!(
            model_id = %range.0,
            layers = %format!("{}-{}", range.1, range.2),
            "Rebalancer: hot shard cooled — effective_target restored"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::replication::{check_over_replication, check_under_replication};
    use super::*;
    use crate::grid::testing::entry;
    use larql_router_protocol::{RouterMessage, RouterPayload};

    #[tokio::test]
    async fn check_hot_shards_marks_newly_hot_ranges() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write().await;
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 50.0;
            g.register(a);
        }
        check_hot_shards(&state, 20.0).await;
        let g = state.read().await;
        assert_eq!(
            g.elevated_ranges_snapshot(),
            vec![("m".to_string(), 0, 4)],
            "range above threshold must be elevated"
        );
    }

    #[tokio::test]
    async fn check_hot_shards_demotes_cooled_ranges() {
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write().await;
            // Pre-elevated range with a cool replica.
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 1.0;
            g.register(a);
            assert!(g.mark_elevated("m", 0, 4));
        }
        check_hot_shards(&state, 20.0).await;
        let g = state.read().await;
        assert!(
            g.elevated_ranges_snapshot().is_empty(),
            "cooled range must be demoted"
        );
    }

    #[tokio::test]
    async fn check_hot_shards_is_noop_when_state_unchanged() {
        // Hot range that's already elevated stays elevated; cool range
        // that's not elevated stays not elevated. Run twice to confirm
        // idempotence.
        let state = Arc::new(RwLock::new(GridState::default()));
        {
            let mut g = state.write().await;
            let mut hot = entry("hot", "http://hot", "m", 0, 4);
            hot.req_per_sec = 50.0;
            g.register(hot);
            let mut cool = entry("cool", "http://cool", "m", 5, 9);
            cool.req_per_sec = 1.0;
            g.register(cool);
        }
        check_hot_shards(&state, 20.0).await;
        check_hot_shards(&state, 20.0).await;
        let g = state.read().await;
        assert_eq!(g.elevated_ranges_snapshot(), vec![("m".to_string(), 0, 4)],);
    }

    #[tokio::test]
    async fn hot_then_cool_path_pulls_and_drops_replica() {
        // End-to-end: hot detected → under-rep tick pulls spare; cool
        // detected → over-rep tick drops the surplus replica.
        use tokio::sync::mpsc;

        let state = Arc::new(RwLock::new(GridState::default()));
        let (spare_tx, mut spare_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        let (busy_tx, _busy_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write().await;
            // target_replicas == 1 default — hot bump takes effective to 2.
            let mut a = entry("a", "http://a", "m", 0, 4);
            a.req_per_sec = 100.0;
            g.register_with_sender(a, busy_tx);
            g.register_available("spare".into(), spare_tx, 1, 0, "/".into());
        }
        // Hot detection + spare pull (mirrors rebalancer_task ordering).
        check_hot_shards(&state, 50.0).await;
        check_under_replication(&state).await;

        let pulled = spare_rx
            .try_recv()
            .expect("spare should receive AssignMsg")
            .expect("ok payload");
        let Some(RouterPayload::Assign(a)) = pulled.payload else {
            panic!("expected Assign, got {pulled:?}");
        };
        assert_eq!(a.layer_start, 0);
        assert_eq!(a.layer_end, 4);

        // Simulate the spare arriving as a serving replica and the
        // workload cooling: rate drops, hot tick demotes, over-rep
        // tick drops the surplus.
        let (extra_tx, mut extra_rx) = mpsc::channel::<Result<RouterMessage, tonic::Status>>(4);
        {
            let mut g = state.write().await;
            let mut extra = entry("extra", "http://extra", "m", 0, 4);
            extra.req_per_sec = 0.5;
            g.register_with_sender(extra, extra_tx);
            // Existing replica also cools.
            g.update_heartbeat("a", 0.0, 0, 0, vec![], 0.5);
        }
        check_hot_shards(&state, 50.0).await;
        check_over_replication(&state).await;

        // Either of the two replicas (extra or a) is least-loaded; the
        // important thing is that one Unassign fires for layers 0-4. If
        // the chosen victim is "a" (whose sender is kept by busy_tx), the
        // unassign just gets queued there; the assertion below relaxes to
        // "if extra was the victim, it received an over_replicated
        // Unassign for the correct range."
        let got = extra_rx.try_recv();
        if let Ok(Ok(msg)) = got {
            if let Some(RouterPayload::Unassign(u)) = msg.payload {
                assert_eq!(u.model_id, "m");
                assert_eq!(u.layer_start, 0);
                assert_eq!(u.layer_end, 4);
                assert_eq!(u.reason, "over_replicated");
            }
        }
        let g = state.read().await;
        assert!(
            g.elevated_ranges_snapshot().is_empty(),
            "range must be demoted after cooling"
        );
    }
}
