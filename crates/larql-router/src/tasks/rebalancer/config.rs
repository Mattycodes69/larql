//! [`RebalancerConfig`] — knobs for the rebalancer background task.
//!
//! Lives in its own file so that `main.rs` can construct/import it
//! without dragging in the rest of the rebalancer module. The
//! defaults mirror the CLI defaults (`--rebalance-interval 30`,
//! `--rebalance-threshold 2.0`, etc.).

use std::time::Duration;

#[derive(Clone)]
pub struct RebalancerConfig {
    /// How often to run the imbalance check.
    pub check_interval: Duration,
    /// Trigger rebalancing when max(avg_ms) / min(avg_ms) exceeds this ratio
    /// across replicas covering the same layer for at least `sustained_window`.
    pub imbalance_threshold: f32,
    /// Sustained imbalance window before action is taken.
    pub sustained_window: Duration,
    /// Servers that haven't sent a heartbeat within this window are evicted
    /// even if the gRPC stream is still alive. Defensive against deadlocked
    /// servers that keep TCP open but stop sending heartbeats. Default 25 s
    /// = 2.5 × the 10 s heartbeat interval.
    pub stale_heartbeat_timeout: Duration,
    /// Hot-shard request-rate threshold (req/s, max across replicas).
    /// `None` disables the check. When set, a shard whose per-replica
    /// req_per_sec exceeds this value is treated as effectively
    /// under-replicated (target + 1) until the rate subsides.
    pub hot_shard_rps_threshold: Option<f32>,
}

impl Default for RebalancerConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            imbalance_threshold: 2.0,
            sustained_window: Duration::from_secs(60),
            stale_heartbeat_timeout: Duration::from_secs(25),
            hot_shard_rps_threshold: None,
        }
    }
}

impl RebalancerConfig {
    pub fn from_cli(interval_secs: u64, threshold: f32) -> Self {
        Self {
            check_interval: Duration::from_secs(interval_secs),
            imbalance_threshold: threshold,
            sustained_window: Duration::from_secs(interval_secs * 2),
            stale_heartbeat_timeout: Duration::from_secs(25),
            hot_shard_rps_threshold: None,
        }
    }

    /// Builder-style setter for the hot-shard threshold so callers
    /// constructed via `default()` / `from_cli()` can add the threshold
    /// without restating every field.
    pub fn with_hot_shard_threshold(mut self, threshold: Option<f32>) -> Self {
        // Treat ≤0 as "disabled" — saves a magic check in the rebalancer.
        self.hot_shard_rps_threshold = threshold.filter(|t| *t > 0.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebalancer_config_defaults() {
        let cfg = RebalancerConfig::default();
        assert_eq!(cfg.check_interval, Duration::from_secs(30));
        assert_eq!(cfg.imbalance_threshold, 2.0);
        assert_eq!(cfg.stale_heartbeat_timeout, Duration::from_secs(25));
    }

    #[test]
    fn from_cli_derives_sustained_window_from_interval() {
        let cfg = RebalancerConfig::from_cli(15, 2.5);
        assert_eq!(cfg.check_interval, Duration::from_secs(15));
        assert_eq!(cfg.imbalance_threshold, 2.5);
        assert_eq!(cfg.sustained_window, Duration::from_secs(30));
        assert_eq!(cfg.stale_heartbeat_timeout, Duration::from_secs(25));
    }

    #[test]
    fn with_hot_shard_threshold_filters_non_positive() {
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(Some(10.0));
        assert_eq!(cfg.hot_shard_rps_threshold, Some(10.0));

        // 0 and negative values disable the check (treated as None).
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(Some(0.0));
        assert_eq!(cfg.hot_shard_rps_threshold, None);
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(Some(-5.0));
        assert_eq!(cfg.hot_shard_rps_threshold, None);
        let cfg = RebalancerConfig::default().with_hot_shard_threshold(None);
        assert_eq!(cfg.hot_shard_rps_threshold, None);
    }
}
