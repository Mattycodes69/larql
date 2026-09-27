//! Reconciling expectations against stored and touched bytes.

use super::super::cpu::ledger::{PlanTally, ProjectionLedger};
use super::super::cpu::physical::PhysicalProjectionPlan;
use super::super::realization::RealizationRecord;
use super::super::weights::DEVICE_PAGE_ALIGN;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::OperandRef;
use std::collections::{BTreeMap, BTreeSet};

#[allow(unused_imports)]
use super::*;

/// Every expectation must meet one observation for its operand and
/// operation instance, in the pinned representation, holding the declared
/// bytes plus at most a page of padding per allocation; nothing observed
/// may be unexpected.
///
/// Instances are counted, not deduplicated: the same stored operand bound
/// twice — a tied head under two operations, Gemma-4's layer that binds
/// one tensor as both its key and value projection — is two expectations
/// meeting two objects, and the loader really does hold it twice.
pub fn reconcile(
    expected: &[Expectation],
    observed: &[Observed],
) -> Result<Reconciliation, VindexError> {
    let key = |op: &OperandRef, operation: Operation, layer: Option<usize>| {
        (
            op.object.clone(),
            op.tensor.clone(),
            operation.name(),
            layer,
        )
    };
    let mut pool: BTreeMap<(String, String, &'static str, Option<usize>), Vec<&Observed>> =
        BTreeMap::new();
    for o in observed {
        pool.entry(key(&o.operand, o.operation, o.layer))
            .or_default()
            .push(o);
    }
    let mut out = Reconciliation::default();
    for e in expected {
        let k = key(&e.operand, e.operation, e.layer);
        let Some(o) = pool.get_mut(&k).and_then(Vec::pop) else {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): pinned {} but nothing is resident for it",
                e.operand.tensor,
                e.operation.name(),
                e.realization.name()
            )));
        };
        let pinned = e.realization.format();
        if o.format != pinned {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): pinned {pinned:?} but {:?} is resident",
                e.operand.tensor,
                e.operation.name(),
                o.format
            )));
        }
        // A mapping is held to its ADDRESS SPACE exactly; the pages of it
        // resident at this moment are a fact reported beside the
        // declaration, never reconciled against it.
        if e.resources().mapped > 0 {
            if o.mapped_bytes != e.declared_resident {
                return Err(VindexError::Parse(format!(
                    "operand `{}` ({}): {} declares {} mapped bytes over {} elements; {} are \
                     mapped — the declaration and the loader disagree",
                    e.operand.tensor,
                    e.operation.name(),
                    e.realization.name(),
                    e.declared_resident,
                    e.logical_elements,
                    o.mapped_bytes
                )));
            }
            out.matched += 1;
            out.mapped += e.declared_resident;
            out.mapped_resident += o.resident_bytes;
            continue;
        }
        if o.mapped_bytes != 0 {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): pinned {} but a mapping of {} bytes is bound for it",
                e.operand.tensor,
                e.operation.name(),
                e.realization.name(),
                o.mapped_bytes
            )));
        }
        // Exact for an object held in plain vectors; up to a page of
        // padding per page-aligned allocation, and never less than declared.
        let ceiling = e.declared_resident + (o.allocations as u64) * DEVICE_PAGE_ALIGN as u64;
        let within = if o.allocations == 0 {
            o.resident_bytes == e.declared_resident
        } else {
            o.resident_bytes >= e.declared_resident && o.resident_bytes < ceiling
        };
        if !within {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): {} declares {} resident bytes over {} elements; {} are \
                 resident in {} allocation(s) — the declaration and the loader disagree",
                e.operand.tensor,
                e.operation.name(),
                e.realization.name(),
                e.declared_resident,
                e.logical_elements,
                o.resident_bytes,
                o.allocations
            )));
        }
        out.matched += 1;
        out.declared_resident += e.declared_resident;
        out.observed_resident += o.resident_bytes;
        out.padding += o.resident_bytes - e.declared_resident;
    }
    if let Some(stray) = pool.values().flatten().next() {
        return Err(VindexError::Parse(format!(
            "operand `{}` ({}) is resident but nothing was pinned for it",
            stray.operand.tensor,
            stray.operation.name()
        )));
    }
    Ok(out)
}

/// The stored footprint: each stored operand counted once, however many
/// operations read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StoredFootprint {
    pub bytes: u64,
    pub operands: usize,
}

pub fn stored_footprint(expected: &[Expectation]) -> StoredFootprint {
    let mut seen = BTreeSet::new();
    let mut out = StoredFootprint::default();
    for e in expected {
        if seen.insert((e.operand.object.clone(), e.operand.tensor.clone())) {
            out.bytes += e.stored_bytes;
            out.operands += 1;
        }
    }
    out
}

/// The execution touch: the stored bytes read, once per operation.
pub fn execution_touch(expected: &[Expectation]) -> u64 {
    expected.iter().map(Expectation::touch).sum()
}

/// The ledger's account of what ran, held against the pins: every CPU
/// plan the ledger tallied is a pinned realization, and every pinned CPU
/// realization ran. Returned per plan with the number of operands pinned
/// to it and its tally, so a caller can hold the tally's POSITIONS against
/// the operands — calls are not the unit, because the executor batches a
/// projection's positions differently per site (per position at the
/// attention, all at once in the FFN), while every position a pinned
/// operand processed is counted exactly once.
pub fn ledger_correspondence(
    records: &[RealizationRecord],
    ledger: &ProjectionLedger,
) -> Result<Vec<(PhysicalProjectionPlan, usize, PlanTally)>, VindexError> {
    let mut pinned: Vec<(PhysicalProjectionPlan, usize)> = Vec::new();
    for r in records {
        if let Some(plan) = r.selection.realization.cpu_plan() {
            match pinned.iter_mut().find(|(p, _)| *p == plan) {
                Some((_, n)) => *n += 1,
                None => pinned.push((plan, 1)),
            }
        }
    }
    let mut out = Vec::new();
    for (plan, tally) in ledger.all() {
        let pins = pinned.iter().find(|(p, _)| *p == plan).map(|(_, n)| *n);
        match (pins, tally.calls) {
            (None, 0) => {}
            (None, calls) => {
                return Err(VindexError::Parse(format!(
                    "{plan:?} ran {calls} call(s) but no operand was pinned to it"
                )))
            }
            (Some(n), 0) => {
                return Err(VindexError::Parse(format!(
                    "{n} operand(s) pinned to {plan:?} but it never ran"
                )))
            }
            (Some(n), _) => out.push((plan, n, tally)),
        }
    }
    Ok(out)
}

/// A selection summary for a report: presentation over the structured
/// records, one line per realization with the operands it serves.
pub fn render_selection_summary(records: &[RealizationRecord]) -> String {
    let mut groups: Vec<(String, String, String, usize, usize)> = Vec::new();
    for r in records {
        let key = (
            r.representation.clone(),
            r.selection.realization.name(),
            r.selection.reason.name().to_string(),
        );
        match groups
            .iter_mut()
            .find(|g| g.0 == key.0 && g.1 == key.1 && g.2 == key.2)
        {
            Some(g) => {
                g.3 += 1;
                g.4 += r.planned.logical_elements;
            }
            None => groups.push((key.0, key.1, key.2, 1, r.planned.logical_elements)),
        }
    }
    let mut out = String::from("realizations:\n");
    for (representation, realization, reason, operands, elements) in groups {
        out.push_str(&format!(
            "  {representation:<10} → {realization:<32} {operands:>4} operand(s) {:>12} weights  ({reason})\n",
            elements
        ));
    }
    out
}
