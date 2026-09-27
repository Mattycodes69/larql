//! **Accounting bound AGAINST the census, never into it.**
//!
//! Rung 3c of the representation/execution contract. Two instruments,
//! kept apart on purpose:
//!
//! * an [`Expectation`] is what a pinned realization DECLARES — derived from
//!   the pin, the codec's declared residency, the executor's own block
//!   geometry and the container's RECORDED operand length. It never reads
//!   a loaded object.
//! * an [`Observed`] is what the loader actually made resident — read off
//!   the bound `LoadedWeight`s, paired with the operand each one binds.
//!
//! [`reconcile`] compares the two. If both came from one declaration the
//! comparison would be circular; because they do not, a wrong block
//! constant, a wrong codec residency, or a loader that drifted from the
//! selector shows up as a mismatch. The residency census is a third
//! reading — the loaded objects summed by site — and the plan-level
//! witness checks it against the observed pairs.
//!
//! Three quantities, each with one definition:
//!
//! ```text
//! stored footprint   Σ recorded length over DISTINCT stored operands
//!                    (a tied head and its embedding are one object)
//! execution touch    Σ recorded length over OPERATIONS
//!                    (the same object read once per operation)
//! resident / staging what the realization holds, and what it
//!                    materialises transiently on the way there
//! ```

use super::backend::WeightFormat;
use super::cpu::integer::weight_index_enabled;
use super::quantise::{Q4_BLOCK, Q8_BLOCK, SUM_BLOCK};
use super::realization::{DependencyPin, RealizationForm, RealizationId};
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::planned::PlannedOperand;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{ResidencyClass, ResidencyProfile};

mod budgets;
mod reconcile;
pub use budgets::*;
pub use reconcile::*;

/// Width of the canonical decode target.
const F32_WIDTH: f64 = std::mem::size_of::<f32>() as f64;
/// Width of a half-precision resident element.
const HALF_WIDTH: f64 = std::mem::size_of::<u16>() as f64;
/// Bits in a byte, for the stored-bit pricings below.
const BITS_PER_BYTE: f64 = 8.0;
/// NVFP4's stored rate — 4-bit codes and one 8-bit scale per 16 elements.
const NVFP4_BITS_PER_WEIGHT: f64 = 4.5;
/// MXFP4's stored rate — 4-bit codes and one 8-bit scale per 32 elements.
const MXFP4_BITS_PER_WEIGHT: f64 = 4.25;
/// The widest of the three K-quant codecs; the bound operand carries which.
const KQUANT_WIDEST_BITS_PER_WEIGHT: f64 = 8.5;
/// One f32 scale per block of a re-quantised image.
const SCALE_WIDTH: f64 = F32_WIDTH;
/// Fine-grained FP8's element width: E4M3 is one byte, exactly.
const FP8_BITS_PER_WEIGHT: f64 = 8.0;
/// One i16 code sum per block, when the weight-code index is on.
const SUM_WIDTH: f64 = std::mem::size_of::<i16>() as f64;

/// The executor's own resident geometry: what a re-quantised image costs
/// per weight depends on these, and they are the executor's declarations,
/// not the codec's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGeometry {
    pub q8_block: usize,
    pub q4_block: usize,
    /// Whether the Q8 image also carries one code sum per block.
    pub q8_indexed: bool,
}

impl BlockGeometry {
    /// The geometry this process's executor uses.
    pub fn executor() -> Self {
        Self {
            q8_block: Q8_BLOCK,
            q4_block: Q4_BLOCK,
            q8_indexed: weight_index_enabled(),
        }
    }
}

/// What the executor makes resident for `format`, priced from `geometry`:
/// a widened image is f32 per weight; a re-quantised image is its codes
/// plus one f32 scale (and one i16 sum, when indexed) per block; a
/// device's half-precision image is two bytes per weight and, being
/// rounded from the source, a re-quantisation; a compact pack bound as
/// stored is its stored bits.
pub fn resident_profile_with(format: WeightFormat, geometry: BlockGeometry) -> ResidencyProfile {
    match format {
        WeightFormat::F32 => ResidencyProfile::DECODED_F32,
        WeightFormat::Bf16 => ResidencyProfile::rebound(HALF_WIDTH * BITS_PER_BYTE),
        WeightFormat::F16 => ResidencyProfile {
            class: ResidencyClass::TransientRequantised,
            bytes_per_weight: HALF_WIDTH,
        },
        WeightFormat::Q8 => ResidencyProfile {
            class: ResidencyClass::TransientRequantised,
            bytes_per_weight: 1.0
                + (SCALE_WIDTH + if geometry.q8_indexed { SUM_WIDTH } else { 0.0 })
                    / geometry.q8_block as f64,
        },
        WeightFormat::Q4 => ResidencyProfile {
            class: ResidencyClass::TransientRequantised,
            bytes_per_weight: 0.5 + SCALE_WIDTH / geometry.q4_block as f64,
        },
        WeightFormat::Nvfp4 => ResidencyProfile::rebound(NVFP4_BITS_PER_WEIGHT),
        // The same pack, copied into aligned buffers the same way; only the
        // activation it runs against differs.
        WeightFormat::Nvfp4Q8 => ResidencyProfile::rebound(NVFP4_BITS_PER_WEIGHT),
        WeightFormat::Mxfp4 => ResidencyProfile::stored(MXFP4_BITS_PER_WEIGHT),
        // The Q8_K binding is the same stored blocks; only the activation
        // it runs against differs.
        WeightFormat::KQuant | WeightFormat::KQuantQ8k => {
            ResidencyProfile::stored(KQUANT_WIDEST_BITS_PER_WEIGHT)
        }
        // Bound AS STORED, like a K-quant pack: the checkpoint's own
        // bytes, never widened at rest. That is the whole reason the
        // format is carried natively — a widened GLM-5.3-Flash would be
        // 612 GB of a 306 GB checkpoint, and the residency question this
        // ledger exists to answer would have no subject.
        //
        // The scales are counted with the codes: 8 bits per weight plus
        // one f32 per tile, which at the 128x128 grid GLM ships is
        // 32/16384 of a bit and rounds to nothing — but it is derived,
        // not waved away, because a [1, 32] grid (which the same scheme
        // permits) costs a full bit per weight.
        //
        // Priced at the CODES alone. The scale grid is a per-TENSOR fact
        // from the checkpoint — one scheme legally ships `[128, 128]` and
        // `[1, 32]` grids in one file — and `BlockGeometry` is by its own
        // definition the executor's geometry, not the codec's, so the
        // tile is not knowable here. The scales are accounted where the
        // tile IS known, on the bound operand
        // (`WeightRows::Fp8Block::bytes`, which counts them).
        //
        // The gap this leaves is stated rather than hidden: at GLM's
        // 128x128 grid it is one f32 per 16,384 weights — 0.02 bits per
        // weight, 0.2 % — but at a `[1, 32]` grid it would be a full bit,
        // and a forecast that silently omitted it would be 12 % light.
        WeightFormat::Fp8Block => ResidencyProfile::stored(FP8_BITS_PER_WEIGHT),
        // Bound AS STORED, like every other native compact format — but
        // this function only ever sees `WeightFormat`, never which codec
        // produced the bytes, so it cannot look up a real bits/weight
        // figure the way `KQuant`/`Fp8Block` do from a constant. The
        // codec's own declared residency (`Acceleration`'s
        // `ResidencyProfile`, set where the codec is registered) is the
        // real number for this path; this generic entry exists only so
        // the format is accounted for at all, not to price it.
        WeightFormat::CodecOwned => ResidencyProfile {
            class: ResidencyClass::Stored,
            bytes_per_weight: 0.0,
        },
    }
}

/// The stored width a block runs along: the operand's inner dimension,
/// which for a packed bank of `[experts, rows, ...]` is the logical
/// elements per expert row rather than the packed shape's last axis.
fn inner_width(operation: Operation, shape: &[usize], logical: usize) -> usize {
    match operation {
        Operation::ExpertBankSlice if shape.len() >= 2 => logical / (shape[0] * shape[1]).max(1),
        _ => shape.last().copied().unwrap_or(logical),
    }
}

/// Bytes the executor's re-quantised image of `format` occupies over a
/// matrix of `rows × k` — EXACT, by the loader's own rule: codes per
/// element, one f32 scale per block, one i16 sum per sum-block when the
/// weight index is on, and blocks that never straddle a row, so a row
/// whose width is not a whole number of blocks carries a short last
/// block with its own scale. `None` for a format that is not a
/// re-quantised image, whose profile prices it per weight.
pub fn requantised_image_bytes(
    format: WeightFormat,
    rows: usize,
    k: usize,
    geometry: BlockGeometry,
) -> Option<u64> {
    let elements = (rows * k) as u64;
    match format {
        WeightFormat::Q8 => {
            let scales = (rows * k.div_ceil(geometry.q8_block)) as u64 * SCALE_WIDTH as u64;
            let sums = if geometry.q8_indexed {
                (rows * k.div_ceil(SUM_BLOCK)) as u64 * SUM_WIDTH as u64
            } else {
                0
            };
            Some(elements + scales + sums)
        }
        WeightFormat::Q4 => {
            let scales = (rows * k.div_ceil(geometry.q4_block)) as u64 * SCALE_WIDTH as u64;
            Some(elements / 2 + scales)
        }
        WeightFormat::F32
        | WeightFormat::Bf16
        | WeightFormat::F16
        | WeightFormat::Nvfp4
        | WeightFormat::Nvfp4Q8
        | WeightFormat::Mxfp4
        | WeightFormat::KQuant
        | WeightFormat::KQuantQ8k
        | WeightFormat::CodecOwned
        // Stored as-is: there is no re-quantised image, so no bytes to price.
        | WeightFormat::Fp8Block => None,
    }
}

/// What `realization` declares it makes resident for `planned`: the
/// profile's bytes per weight over the logical elements, except that a
/// re-quantised image is priced by the loader's exact per-row rule. The
/// ONE pricing the ledger and the budget's re-selection share.
pub fn declared_resident_for(
    planned: &PlannedOperand,
    realization: RealizationId,
    profile: ResidencyProfile,
    geometry: BlockGeometry,
) -> u64 {
    let logical = planned.logical_elements;
    let exact = match realization.form {
        RealizationForm::Requantise(_)
        | RealizationForm::SliceStored { .. }
        | RealizationForm::DeviceResident(_) => {
            let k = inner_width(planned.operation, &planned.operand.shape, logical);
            let rows = logical.checked_div(k).unwrap_or(0);
            requantised_image_bytes(realization.format(), rows, k, geometry)
        }
        RealizationForm::Direct(_)
        | RealizationForm::Decode(_)
        | RealizationForm::DecodedGather
        | RealizationForm::MappedStored { .. } => None,
    };
    exact.unwrap_or_else(|| (profile.bytes_per_weight * logical as f64).round() as u64)
}

/// What one pinned realization declares it will cost. Every field is
/// derived from the pin and the container's record; none from a loaded
/// object.
#[derive(Debug, Clone, PartialEq)]
pub struct Expectation {
    pub operand: OperandRef,
    pub operation: Operation,
    pub layer: Option<usize>,
    pub realization: RealizationId,
    /// The container's recorded length for the stored operand — an
    /// instance fact, which for an entropy-coded operand is not a
    /// function of its shape.
    pub stored_bytes: u64,
    pub logical_elements: usize,
    /// The declared resident image: the realization's residency profile
    /// over the logical elements — ADDRESSABLE bytes, whatever their
    /// physical form. [`Expectation::resources`] splits it into what is
    /// committed and what is merely mapped.
    pub declared_resident: u64,
    /// Bytes materialised transiently on the way to residency: the f32
    /// image a decode or re-quantisation passes through, none for a
    /// realization that binds the stored bytes.
    pub staging: u64,
    /// Bytes of the stored operand this pin OPENS to prepare it — the
    /// streams its selected extent reads, which for a terminal
    /// representation is everything the container holds and for a
    /// progressive one is the planes the extent reaches.
    ///
    /// Distinct from `stored_bytes`, which is the whole footprint on disk
    /// and does not move when an extent does, and from `touch_per_token`,
    /// which is the image the executor streams once the operand is
    /// resident. Under canonical decode this is the ONLY dimension a
    /// shallower extent moves.
    pub read_to_prepare: u64,
    /// The other represented objects this pin resolves, and what its
    /// realization does with each.
    ///
    /// Priced by the LEDGER rather than folded in here, because a
    /// dependency shared by many owners is one object: summing it per
    /// owner would count a codebook once per tensor that indexes it.
    /// Bytes physically read to VERIFY this operand's attestations, and
    /// zero whenever nothing was hashed for it — no attestation, or one
    /// refused from metadata before any payload was opened.
    pub verified_bytes: u64,
    pub dependencies: Vec<DependencyPin>,
}

impl Expectation {
    /// The stored bytes read once for this operation.
    pub fn touch(&self) -> u64 {
        self.stored_bytes
    }

    /// The peak the realization holds while preparing: resident plus
    /// whatever it staged to get there.
    pub fn working_set(&self) -> u64 {
        self.declared_resident + self.staging
    }

    /// This expectation's demand on each resource, in the vocabulary a
    /// budget decision needs. A mapped realization's bytes are ADDRESS
    /// SPACE and page in per token as touched; a decoded, re-quantised,
    /// rebound or direct-copied realization's bytes are COMMITTED memory
    /// kept resident; a device realization's bytes live on its target.
    /// Touch is per operation instance per position, and it is the IMAGE
    /// the executor streams — the resident bytes of a decoded or
    /// re-quantised projection, the mapped bytes of a bank — not the
    /// stored bytes read once at load: a whole matrix for a projection,
    /// `top_k / experts` of an expert's matrix for a bank access, because
    /// a token selects that fraction of the bank.
    pub fn resources(&self) -> Resources {
        let per_token = match self.operation {
            Operation::ExpertProject { experts, top_k } if experts > 0 => {
                self.declared_resident as f64 * top_k as f64 / experts as f64
            }
            _ => self.declared_resident as f64,
        };
        let touch_per_token = per_token.round() as u64;
        match self.realization.form {
            RealizationForm::MappedStored { .. } => Resources {
                stored: self.stored_bytes,
                mapped: self.declared_resident,
                resident: 0,
                transient: self.staging,
                touch_per_token,
                page_in_per_token: touch_per_token,
                device: 0,
                read_to_prepare: self.read_to_prepare,
            },
            // On-device traffic is the device's; the host streams nothing.
            RealizationForm::DeviceResident(_) => Resources {
                stored: self.stored_bytes,
                mapped: 0,
                resident: 0,
                transient: self.staging,
                touch_per_token: 0,
                page_in_per_token: 0,
                device: self.declared_resident,
                read_to_prepare: self.read_to_prepare,
            },
            RealizationForm::Direct(_)
            | RealizationForm::Decode(_)
            | RealizationForm::Requantise(_)
            | RealizationForm::SliceStored { .. }
            | RealizationForm::DecodedGather => Resources {
                stored: self.stored_bytes,
                mapped: 0,
                resident: self.declared_resident,
                transient: self.staging,
                touch_per_token,
                page_in_per_token: 0,
                device: 0,
                read_to_prepare: self.read_to_prepare,
            },
        }
    }
}

/// What a preparation may hold and stream: the constraint selection
/// answers to. `None` on a dimension means unconstrained there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidencyBudget {
    /// Physical host memory the plan's working set — committed
    /// allocations, the staging peak, and a token's page-in — may reach.
    /// Never the address space a mapping occupies.
    pub physical_bytes: Option<u64>,
    /// Bytes the host may stream per token to reach a target rate.
    pub throughput: Option<ThroughputBudget>,
    /// Stored bytes the plan may OPEN to prepare itself — the cold cost of
    /// getting ready, as against the steady cost of running. The dimension
    /// a shallower extent moves: reading less of an artifact is what an
    /// extent buys under a realization that decodes.
    pub prepare_bytes: Option<u64>,
    /// How a mapped bank's selected experts are brought in per token —
    /// a policy on the ACCESS realization, stamped on every mapped pin
    /// the selection makes.
    pub expert_access: super::realization::MappedAccess,
    /// The reconstruction execution requires of the representations it
    /// selects. Representation-independent: a floor, never a depth.
    pub fidelity: RepresentationFloor,
}

/// What execution requires of a representation's reconstruction.
///
/// A quality REQUIREMENT, stated without naming a codec or a depth, so a
/// plan can carry it and any representation can answer it.
///
/// # Two questions one word used to answer
///
/// The variant now called [`Self::TerminalExtent`] was called `Exact`,
/// and it has ALWAYS meant "the deepest extent this codec declares" —
/// `admits` returned true for the terminal extent and false for
/// everything else, without ever consulting a radius. For a lossless
/// progressive codec the two readings coincide, so the name was
/// harmless. For a lossy one they are not the same claim at all:
/// `VQ8_SHARED`'s terminal extent is every byte the artifact holds AND
/// an unbounded error against the tensor it was fitted to. Left alone,
/// this build would eventually print "exact reconstruction" while
/// planning terminal Q4 or VQ data, possessing no such guarantee.
///
/// So the two questions are separated, and neither answers the other:
///
/// * **How much of the artifact must be read?** — [`Self::TerminalExtent`].
///   A STRUCTURAL requirement. It makes no claim about error, because
///   reading every stored byte says nothing about what encoding those
///   bytes cost.
/// * **How close must the result be to the source?** —
///   [`Self::CertifiedExact`] and [`Self::Within`]. EVIDENCE
///   requirements, met only by a certificate in a compatible metric and
///   domain. Neither is satisfied by being terminal: an operand with no
///   certificate has no error claim, and the whole point of this plane
///   is that an absent claim is unavailable rather than optimistic.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RepresentationFloor {
    /// The codec's COMPLETE stored representation: every plane the
    /// artifact holds is read, and no extent short of the terminal one
    /// will do.
    ///
    /// The default, so a plan that asks for nothing gets no silent
    /// quality change and a budget it cannot meet refuses rather than
    /// degrading. It states NO source-error claim: a terminal VQ or
    /// K-quant extent satisfies this floor and is still lossy against
    /// the checkpoint it came from.
    #[default]
    TerminalExtent,
    /// A verified, compatible certificate whose radius is `0.0` — the
    /// exact reconstruction's honest answer, stated rather than assumed.
    ///
    /// Deliberately NOT satisfied by the terminal extent as such. A
    /// codec that reconstructs its source exactly can say so
    /// (`F32_PLANES` certifies `0.0` at its terminal depth); one that
    /// cannot is refused, which is the entire difference between this
    /// and [`Self::TerminalExtent`].
    CertifiedExact,
    /// A verified, composed certificate at or under this bound.
    ///
    /// The radius judged here is the one the planner DERIVED — the
    /// attested claim where a measurement was verified, the codec's
    /// declaration otherwise, composed with the certificates of the
    /// dependency extents actually selected. An extent that declares no
    /// radius does not satisfy it at any depth: an undeclared error is
    /// not a small one.
    Within(f64),
}

impl RepresentationFloor {
    /// Whether `option` satisfies this floor, given the representation's
    /// terminal extent.
    pub fn admits(
        self,
        option: &super::realization::ExtentOption,
        terminal: super::super::super::represent::codec::RepresentationExtent,
    ) -> bool {
        match self {
            Self::TerminalExtent => option.certificate.extent == terminal,
            Self::CertifiedExact => self.certified_within(option, 0.0),
            Self::Within(bound) => self.certified_within(option, bound),
        }
    }

    /// A certificate in THIS build's metric and domain, at or under
    /// `bound`.
    ///
    /// v1 compares like with like: a bound stated in another metric or
    /// over another domain does not satisfy this floor, and is not
    /// converted into one that would.
    fn certified_within(self, option: &super::realization::ExtentOption, bound: f64) -> bool {
        option.certificate.radius.as_ref().is_some_and(|r| {
            *r.metric() == super::super::super::represent::codec::MetricId::relative_rms()
                && *r.domain() == super::super::super::represent::codec::DomainId::finite_normals()
                && r.radius() <= bound
        })
    }

    pub fn describe(self) -> String {
        match self {
            Self::TerminalExtent => "the complete stored representation".to_string(),
            Self::CertifiedExact => "a certified exact reconstruction (radius 0.0)".to_string(),
            Self::Within(bound) => format!("relative RMS at or under {bound:.3e}"),
        }
    }
}

/// A rate constraint: a plan can fit in memory and still be unusably
/// slow, so bytes touched per token are held against what the machine
/// moves per token at the rate the caller wants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThroughputBudget {
    pub bytes_per_second: u64,
    pub target_tokens_per_second: f64,
}
