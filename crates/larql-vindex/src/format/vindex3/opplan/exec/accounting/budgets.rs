//! Throughput and residency budgets and their deficits.

use super::super::backend::WeightFormat;
use super::super::realization::{DependencyLifetime, RealizationForm, RealizationRecord};
use super::super::weights::LoadedWeight;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::ResidencyProfile;
use std::collections::BTreeSet;

#[allow(unused_imports)]
use super::*;

impl ThroughputBudget {
    /// Bytes a token may touch at the target rate.
    pub fn bytes_per_token(&self) -> u64 {
        (self.bytes_per_second as f64 / self.target_tokens_per_second).round() as u64
    }
}

impl ResidencyBudget {
    /// No constraint on either dimension — every selection is the
    /// backend's own preference, exactly as before a budget existed.
    pub const UNBOUNDED: Self = Self {
        physical_bytes: None,
        throughput: None,
        expert_access: super::super::realization::MappedAccess::Demand,
        prepare_bytes: None,
        fidelity: RepresentationFloor::TerminalExtent,
    };

    /// This machine's physical memory as the budget, read from the OS;
    /// unconstrained where the OS does not say.
    pub fn machine() -> Self {
        Self {
            physical_bytes: physical_memory_bytes(),
            throughput: None,
            expert_access: super::super::realization::MappedAccess::Demand,
            prepare_bytes: None,
            fidelity: RepresentationFloor::TerminalExtent,
        }
    }

    pub fn physical(bytes: u64) -> Self {
        Self {
            physical_bytes: Some(bytes),
            throughput: None,
            expert_access: super::super::realization::MappedAccess::Demand,
            prepare_bytes: None,
            fidelity: RepresentationFloor::TerminalExtent,
        }
    }

    pub fn with_throughput(mut self, throughput: ThroughputBudget) -> Self {
        self.throughput = Some(throughput);
        self
    }

    /// Stored bytes the plan may open to prepare itself.
    pub fn with_prepare_bytes(mut self, bytes: u64) -> Self {
        self.prepare_bytes = Some(bytes);
        self
    }

    /// The reconstruction the plan requires — the quality half of a
    /// budget, without which selection would take the cheapest extent
    /// every time and call it feasibility.
    pub fn with_fidelity(mut self, floor: RepresentationFloor) -> Self {
        self.fidelity = floor;
        self
    }

    pub fn with_expert_access(mut self, access: super::super::realization::MappedAccess) -> Self {
        self.expert_access = access;
        self
    }

    /// Whether `ledger` fits, and by how much it does not: the physical
    /// deficit and the per-token touch deficit, zero where it fits.
    pub fn deficit(&self, ledger: &ResourceLedger) -> BudgetDeficit {
        BudgetDeficit {
            physical: self
                .physical_bytes
                .map(|b| ledger.physical_working_set().saturating_sub(b))
                .unwrap_or(0),
            touch_per_token: self
                .throughput
                .map(|t| ledger.touch_per_token.saturating_sub(t.bytes_per_token()))
                .unwrap_or(0),
            prepare: self
                .prepare_bytes
                .map(|b| ledger.read_to_prepare.saturating_sub(b))
                .unwrap_or(0),
        }
    }
}

/// How far a ledger overshoots a budget, per constrained dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BudgetDeficit {
    pub physical: u64,
    pub touch_per_token: u64,
    /// Stored bytes the preparation would open over its budget.
    pub prepare: u64,
}

impl BudgetDeficit {
    pub fn is_zero(&self) -> bool {
        self.physical == 0 && self.touch_per_token == 0 && self.prepare == 0
    }
}

/// The machine's physical memory, from the OS.
pub fn physical_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf reads two process-independent constants.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page > 0 {
            return Some(pages as u64 * page as u64);
        }
        None
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: MEMORYSTATUSEX is a plain C struct, so all-zero is a valid
        // value; `dwLength` is set as the API requires before the call, and
        // the struct outlives it.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok != 0 && status.ullTotalPhys > 0 {
            return Some(status.ullTotalPhys);
        }
        None
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

/// One expectation's demand on each resource a budget decision reads.
/// Seven numbers because they aggregate by SEVEN different rules — see
/// [`ResourceLedger::aggregate`] — and a single "resident" figure had
/// conflated a 98 GB mapping with 8 GB of committed memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Resources {
    /// Bytes the container stores for the operand.
    pub stored: u64,
    /// Address space a mapping of the stored bytes occupies; pages become
    /// resident only as touched.
    pub mapped: u64,
    /// Committed memory the realization keeps physically resident.
    pub resident: u64,
    /// Bytes materialised transiently on the way to residency.
    pub transient: u64,
    /// Bytes the host streams for this operation per position: the
    /// resident image, or the touched fraction of a mapping.
    pub touch_per_token: u64,
    /// Bytes a token is expected to page in cold from a mapping.
    pub page_in_per_token: u64,
    /// Bytes held on a device target.
    pub device: u64,
    /// Stored bytes opened once to prepare the operand at its pinned
    /// extent — every plane for a terminal representation, the extent's
    /// planes for a progressive one.
    pub read_to_prepare: u64,
}

/// Where a plan's preparation I/O actually goes.
///
/// D1's rule: **preparation accounting reports bytes actually read,
/// including attestation verification, counted according to physical
/// reads — not inferred from metadata.**
///
/// Three causes, kept apart because they answer different questions and
/// respond to different fixes: materialising the representation is what
/// the extent costs, resolving an auxiliary is what the dependency
/// closure costs, and verifying an attestation is what a GUARANTEE
/// costs. A single figure hid the third entirely, which let a plan price
/// verified fidelity as though evidence were free.
///
/// The aggregate is retained ([`Self::total`], and
/// [`ResourceLedger::read_to_prepare`] beside it) so nothing that asked
/// one preparation-I/O question has to learn three. This invents no new
/// residency resource: these bytes are read and dropped, and none of
/// them is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrepareReads {
    /// Opening the operand's own stored planes at the selected extent.
    pub representation_materialisation: u64,
    /// Opening the auxiliaries a codec's closure requires, once per
    /// dependency object however many owners resolve it.
    pub auxiliary_resolution: u64,
    /// Hashing the payload an attestation binds to. Counted per physical
    /// read: an operand attested at two depths is read twice, because
    /// nothing shares that materialisation.
    pub attestation_verification: u64,
}

impl PrepareReads {
    /// The one aggregate preparation-I/O figure.
    pub fn total(&self) -> u64 {
        self.representation_materialisation
            + self.auxiliary_resolution
            + self.attestation_verification
    }
}

/// A plan's demand on each resource, each aggregated by its own rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceLedger {
    /// Stored footprint: once per physical object (an operand bound
    /// under two operations is stored once).
    pub stored: u64,
    /// Mapped address space: once per mapping.
    pub mapped: u64,
    /// Persistent resident memory: every committed allocation, summed.
    pub resident: u64,
    /// Transient decode or staging: the MAXIMUM overlapping lifetime —
    /// the loader stages one operand at a time, so the peak is the
    /// largest, never the total.
    pub transient_peak: u64,
    /// Execution touch per token: summed over every operation instance.
    pub touch_per_token: u64,
    /// Expected cold page-in per token from mappings, summed.
    pub page_in_per_token: u64,
    /// Device memory, summed per target.
    pub device: u64,
    /// Stored bytes opened to prepare the plan: once per stored operand,
    /// like the footprint, because an operand bound once is read once
    /// however many operations it serves.
    ///
    /// The aggregate, and always equal to `prepare_reads.total()`.
    pub read_to_prepare: u64,
    /// The same figure, by cause.
    pub prepare_reads: PrepareReads,
}

impl ResourceLedger {
    /// Aggregate every expectation by the rule its resource carries.
    pub fn aggregate(expected: &[Expectation]) -> Self {
        let mut ledger = Self::default();
        let mut stored_seen = BTreeSet::new();
        let mut mapped_seen = BTreeSet::new();
        for e in expected {
            let r = e.resources();
            let object = (e.operand.object.clone(), e.operand.tensor.clone());
            if stored_seen.insert(object.clone()) {
                ledger.stored += r.stored;
                ledger.prepare_reads.representation_materialisation += r.read_to_prepare;
                // Verification is attributed to the operand that incurred
                // it, on the same once-per-operand rule: an operand bound
                // under two operations was verified once.
                ledger.prepare_reads.attestation_verification += e.verified_bytes;
            }
            // A dependency is ONE object however many owners resolve it:
            // its footprint and the reading that prepares it count once,
            // and only a realization that RETAINS it pays residency and
            // per-token touch for it.
            for dependency in &e.dependencies {
                let address = dependency.address();
                let first_time = stored_seen.insert(address);
                let bytes = dependency.stored_bytes.unwrap_or(0);
                if first_time {
                    ledger.stored += bytes;
                    ledger.prepare_reads.auxiliary_resolution += bytes;
                }
                if dependency.lifetime == DependencyLifetime::Retained {
                    // Resident once, whoever keeps it; touched once per
                    // OPERATION that reads it, which is per owner.
                    let image = (dependency.elements as f64 * F32_WIDTH).round() as u64;
                    if first_time {
                        ledger.resident += image;
                    }
                    ledger.touch_per_token += image;
                }
            }
            if r.mapped > 0 && mapped_seen.insert(object) {
                ledger.mapped += r.mapped;
            }
            ledger.resident += r.resident;
            ledger.transient_peak = ledger.transient_peak.max(r.transient);
            ledger.touch_per_token += r.touch_per_token;
            ledger.page_in_per_token += r.page_in_per_token;
            ledger.device += r.device;
        }
        // The aggregate is DERIVED from the breakdown rather than summed
        // beside it, so the two cannot disagree about what preparation
        // costs — a second accumulator is a second answer.
        ledger.read_to_prepare = ledger.prepare_reads.total();
        ledger
    }

    /// The physical working set a token needs on the host: committed
    /// memory, the staging peak, and the mapped bytes a token pages in.
    pub fn physical_working_set(&self) -> u64 {
        self.resident + self.transient_peak + self.page_in_per_token
    }
}

/// Price every record. `stored_len` answers with the container's recorded
/// length for an operand, or `None` for one the container does not hold.
pub fn expectations(
    records: &[RealizationRecord],
    stored_len: impl Fn(&OperandRef) -> Option<u64>,
    geometry: BlockGeometry,
) -> Vec<Expectation> {
    records
        .iter()
        .map(|r| {
            let logical = r.planned.logical_elements;
            let realization = r.selection.realization;
            // Direct and decode carry the codec's own declaration; the
            // executor's forms are priced from its geometry, so a change
            // to that geometry re-prices them here and nowhere else.
            let profile = match realization.form {
                RealizationForm::Direct(_) | RealizationForm::Decode(_) => r.selection.residency,
                RealizationForm::Requantise(_)
                | RealizationForm::SliceStored { .. }
                | RealizationForm::DeviceResident(_) => {
                    resident_profile_with(realization.format(), geometry)
                }
                RealizationForm::DecodedGather => ResidencyProfile::DECODED_F32,
                // Mapped as stored: resident exactly as the container
                // holds it, nothing staged on the way.
                RealizationForm::MappedStored { format, .. } => {
                    resident_profile_with(format, geometry)
                }
            };
            let staging = match realization.form {
                RealizationForm::Direct(_) | RealizationForm::MappedStored { .. } => 0,
                RealizationForm::Decode(_)
                | RealizationForm::Requantise(_)
                | RealizationForm::DecodedGather => (logical as f64 * F32_WIDTH).round() as u64,
                RealizationForm::SliceStored { convert }
                | RealizationForm::DeviceResident(convert) => {
                    if convert == WeightFormat::F32 {
                        0
                    } else {
                        (logical as f64 * F32_WIDTH).round() as u64
                    }
                }
            };
            let stored_bytes = stored_len(&r.planned.operand).unwrap_or(0);
            Expectation {
                operand: r.planned.operand.clone(),
                operation: r.planned.operation,
                layer: r.planned.layer,
                realization,
                stored_bytes,
                logical_elements: logical,
                declared_resident: declared_resident_for(
                    &r.planned,
                    realization,
                    profile,
                    geometry,
                ),
                staging,
                // What the pin OPENS: the extent's own price where the
                // codec gives one, and otherwise the whole footprint —
                // an unpriced extent is read whole, never assumed cheap.
                read_to_prepare: r.extent.touch_bytes().unwrap_or(stored_bytes),
                // Observed, not inferred: what verification actually read.
                verified_bytes: r.verified_bytes,
                dependencies: r.dependencies.clone(),
            }
        })
        .collect()
}

/// One operand paired with the object(s) the loader bound for it: a
/// matrix is one object, a packed bank is one per expert. Every loader
/// names its own pairing, field by field, so the observation cannot be
/// derived from the plan's order.
pub struct Bound<'a> {
    pub operand: &'a OperandRef,
    pub weights: Vec<&'a LoadedWeight>,
}

impl<'a> Bound<'a> {
    pub fn one(operand: &'a OperandRef, weight: &'a LoadedWeight) -> Self {
        Self {
            operand,
            weights: vec![weight],
        }
    }

    pub fn observed(
        &self,
        operation: Operation,
        layer: Option<usize>,
    ) -> Result<Observed, VindexError> {
        let Some(first) = self.weights.first() else {
            return Err(VindexError::Parse(format!(
                "operand `{}`: bound to no object",
                self.operand.tensor
            )));
        };
        let format = first.format();
        if let Some(other) = self.weights.iter().find(|w| w.format() != format) {
            return Err(VindexError::Parse(format!(
                "operand `{}`: bound objects disagree on their representation ({format:?} vs {:?})",
                self.operand.tensor,
                other.format()
            )));
        }
        Ok(Observed {
            operand: self.operand.clone(),
            operation,
            layer,
            format,
            resident_bytes: self.weights.iter().map(|w| w.resident_bytes() as u64).sum(),
            mapped_bytes: self.weights.iter().map(|w| w.mapped_bytes() as u64).sum(),
            allocations: self.weights.iter().map(|w| w.padded_allocations()).sum(),
        })
    }
}

/// What the loader actually made resident for one planned operand.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    pub operand: OperandRef,
    pub operation: Operation,
    pub layer: Option<usize>,
    pub format: WeightFormat,
    /// Bytes physically held, allocation padding included: committed
    /// allocations in full, a mapping's pages resident at the moment of
    /// observation.
    pub resident_bytes: u64,
    /// Address space held as a mapping of the container's segment; zero
    /// for an owned object.
    pub mapped_bytes: u64,
    /// Allocations the bytes live in: one for a matrix, one per expert
    /// for a bank.
    pub allocations: usize,
}

/// The result of a reconciliation that held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Reconciliation {
    pub matched: usize,
    /// Committed bytes declared and observed, over the owned objects.
    pub declared_resident: u64,
    pub observed_resident: u64,
    /// Observed minus declared — allocation padding, bounded per
    /// allocation by the page.
    pub padding: u64,
    /// Address space declared for mappings, held exactly against the
    /// mappings observed.
    pub mapped: u64,
    /// Pages of those mappings physically resident at observation — a
    /// fact about this moment, reported beside the declaration and never
    /// reconciled against it.
    pub mapped_resident: u64,
}
