//! The registry resolves what it lists, and the MEASURE-PLAN-3 gate admits
//! its own banked anchor through the same path any reading takes.

use super::*;
use crate::format::vindex3::represent::constraint::ConstraintVector;
use crate::format::vindex3::represent::measure::plan::metrics::Summary;
use crate::format::vindex3::represent::reading::{
    gate_of_any_kind, Gate, Observation, PlanObservation,
};

/// Q-BANK-1's sequence count, the anchor's `--sequences`.
const ANCHOR_SEQUENCES: usize = 69;

/// The banked anchor receipt's summary, read from the repository.
fn anchor_summary() -> Summary {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/measure-plan-3/anchor/receipt.json"
    );
    let written: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("the anchor is banked")).unwrap();
    serde_json::from_value(written["receipt"]["summary"].clone()).unwrap()
}

fn judged(summary: &Summary) -> ConstraintVector {
    let gate = gate_of_any_kind(GRANITE_41_3B_PLAN_V1_VS_LATE10_FFN_V1).unwrap();
    let reading = Observation::Plan(PlanObservation::from_summary(summary, ANCHOR_SEQUENCES));
    ConstraintVector::judge(&gate, &reading).unwrap()
}

#[test]
fn every_listed_plan_gate_resolves_to_a_plan_gate_of_its_own_name() {
    for id in IMPLEMENTED_PLAN_GATES {
        match gate_of_any_kind(id).unwrap() {
            Gate::Plan(gate) => assert_eq!(gate.id, id),
            Gate::Kimi(_) => panic!("{id} resolved to a Kimi gate"),
        }
    }
    assert!(plan_gate_by_id("granite-4.1-3b-plan-v1-vs-late10-ffn/v0").is_none());
}

#[test]
fn the_gate_carries_the_banked_anchor_exactly() {
    let all = anchor_summary().all;
    let Gate::Plan(gate) = gate_of_any_kind(GRANITE_41_3B_PLAN_V1_VS_LATE10_FFN_V1).unwrap() else {
        unreachable!("resolved above as a plan gate")
    };
    assert_eq!(gate.positions_min, all.positions as u64);
    assert_eq!(gate.kl_p99_max, all.kl_p99);
    assert_eq!(gate.kl_mean_max, Some(all.kl_mean));
    assert_eq!(gate.top1_disagreement_max, Some(1.0 - all.top1_agreement));
    assert_eq!(gate.delta_nll_mean_max, all.delta_nll_mean);
    assert_eq!(gate.semantics, MetricSemantics::PLAN_V1);
}

/// MEASURE-PLAN-3 forecast 2's precondition: a gate built from the anchor's
/// values admits the anchor, with every ceiling met at equality.
#[test]
fn the_anchor_admits_itself_at_zero_margin() {
    let standing = judged(&anchor_summary());
    assert!(
        standing.margins.iter().all(|m| m.satisfied()),
        "{standing:?}"
    );
    for margin in standing.spendable() {
        assert_eq!(margin.observed, Some(margin.limit), "{margin:?}");
    }
}

/// Anything worse than the anchor on one criterion is refused on it.
#[test]
fn a_reading_worse_on_any_one_criterion_is_refused() {
    let anchor = anchor_summary();
    let worse: [fn(&mut Summary); 5] = [
        |s| s.all.kl_p99 *= 1.0 + f64::EPSILON * 4.0,
        |s| s.all.kl_mean *= 1.0 + f64::EPSILON * 4.0,
        |s| s.all.top1_agreement -= 1e-12,
        |s| s.all.delta_nll_mean = s.all.delta_nll_mean.map(|d| d + 1e-12),
        |s| s.all.positions -= 1,
    ];
    for (i, make_worse) in worse.iter().enumerate() {
        let mut summary = anchor.clone();
        make_worse(&mut summary);
        let standing = judged(&summary);
        let failed = standing.margins.iter().filter(|m| !m.satisfied()).count();
        assert_eq!(failed, 1, "perturbation {i}: {standing:?}");
    }
}
