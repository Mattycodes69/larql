//! The VINDEX3-boundary authority gate: parser consumption is not
//! representation authority.
//!
//! Every test here pairs a positive and a negative arm on the *same* key,
//! because the instrument's claim is discriminative: it must fire when a
//! declared fact is dropped and stay silent when the same fact is carried.
//! A gate that only ever fires proves nothing about the facts it passes.

use super::support::{glimmer_shaped_target_with, known_dense_config, known_dense_with_config};
use crate::format::vindex3::plan::carriage::{carriage_rules, rule_for, Carriage};
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding, SemanticClass};

/// Plan the Glimmer-shaped fixture with `mutate` applied to its config.
fn plan_with(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        glimmer_shaped_target_with(dir.path(), mutate),
    )];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

/// The finding whose subject ends with `suffix`.
fn finding_for<'a>(findings: &'a [PlannedFinding], suffix: &str) -> &'a PlannedFinding {
    findings
        .iter()
        .find(|f| f.subject.ends_with(suffix))
        .unwrap_or_else(|| panic!("no finding for `{suffix}`"))
}

/// A dense Llama-family plan with one extra declared key — for the two
/// vestigial keys below, which are checked against a *recognised* family
/// rather than the Glimmer shape.
fn dense_plan_with(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let mut config = known_dense_config();
    mutate(&mut config);
    let named = vec![(
        "target-artifact".to_string(),
        known_dense_with_config(dir.path(), config),
    )];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

/// The CARRIAGE finding for `subject` — the one that asks whether the
/// container holds the fact. `sliding_window` also raises a compare
/// finding on the same subject, and picking that one by suffix reads
/// "declared and resolved agree" no matter what carriage did.
fn carriage_finding_for<'a>(findings: &'a [PlannedFinding], subject: &str) -> &'a PlannedFinding {
    findings
        .iter()
        .find(|f| f.subject == subject && f.carriage.is_some())
        .unwrap_or_else(|| panic!("no carriage finding for {subject}"))
}

mod carriage_basics;
mod carriage_basics_2;
