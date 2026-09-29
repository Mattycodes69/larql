//! **Every plan-v1 gate this build implements, by name.**
//!
//! A plan gate's limits are a measured anchor's own values, registered in
//! the commit that banks the anchor's report. Named, and therefore FROZEN:
//! a gate that turns out wrong is replaced by a new version, never edited,
//! because every verdict drawn under an id cites that id's limits.

use super::reading::{PlanGate, PlanProcedure};
use super::state::instrument::MetricSemantics;

/// MEASURE-PLAN-3's gate (`docs/measure-plan-3.md`): no worse than the
/// `late10-ffn` hand map ({`ffn`×Q4} held, uniform NVFP4 elsewhere).
pub const GRANITE_41_3B_PLAN_V1_VS_LATE10_FFN_V1: &str = "granite-4.1-3b-plan-v1-vs-late10-ffn/v1";

/// The plan gates [`plan_gate_by_id`] resolves, so a test can assert the
/// registry and the constructors do not drift apart.
pub const IMPLEMENTED_PLAN_GATES: [&str; 1] = [GRANITE_41_3B_PLAN_V1_VS_LATE10_FFN_V1];

/// The anchor's plan-v1 reading: `bench/measure-plan-3/anchor/`, report
/// sha256 `e9c947135a63bc2ee0655d735bca7d881d28b013354f018af7c2eac34f3d8dc3`,
/// measured 2026-09-28 at commit `31f8e3b5` with the frozen arms
/// (`production` on the source's canonical bytes against `production-nvfp4`
/// on the anchor pack), all 69 Q-BANK-1 sequences, null arm bit-identical.
/// The values are the receipt's `summary.all`, exactly.
mod anchor {
    pub const POSITIONS: u64 = 1667;
    pub const KL_P99: f64 = 0.882304764943293;
    pub const KL_MEAN: f64 = 0.08731425030319556;
    pub const TOP1_AGREEMENT: f64 = 0.8428314337132573;
    pub const DELTA_NLL_MEAN: f64 = -0.011485826309444516;
}

/// **`granite-4.1-3b-plan-v1-vs-late10-ffn/v1`.** A candidate is admitted
/// when it is no worse than the anchor on every criterion. Each limit is
/// the anchor's value, so the anchor admits itself with every margin at
/// zero. Top-1 disagreement is derived exactly as the reading derives it
/// (`1 − agreement`), so the comparison is between identical expressions.
fn granite_41_3b_plan_v1_vs_late10_ffn_v1() -> PlanGate {
    PlanGate {
        procedure: PlanProcedure,
        id: GRANITE_41_3B_PLAN_V1_VS_LATE10_FFN_V1.into(),
        semantics: MetricSemantics::PLAN_V1,
        positions_min: anchor::POSITIONS,
        kl_p99_max: anchor::KL_P99,
        kl_mean_max: Some(anchor::KL_MEAN),
        top1_disagreement_max: Some(1.0 - anchor::TOP1_AGREEMENT),
        delta_nll_mean_max: Some(anchor::DELTA_NLL_MEAN),
    }
}

/// A plan gate by id, or `None` when this build implements none by that
/// name. No fallback: an unknown id is never resolved to a nearby gate.
pub fn plan_gate_by_id(id: &str) -> Option<PlanGate> {
    let gate = match id {
        GRANITE_41_3B_PLAN_V1_VS_LATE10_FFN_V1 => granite_41_3b_plan_v1_vs_late10_ffn_v1(),
        _ => return None,
    };
    debug_assert_eq!(gate.id, id, "a gate answers to the name it is looked up by");
    Some(gate)
}

#[cfg(test)]
mod tests;
