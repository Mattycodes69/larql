//! Expert encodings, encoded regions and expert-bank bindings.

use crate::error::VindexError;

#[allow(unused_imports)]
use super::*;

impl ExpertEncoding {
    pub fn name(self) -> &'static str {
        match self {
            ExpertEncoding::Bf16 => "BF16",
            ExpertEncoding::Q80 => "Q8_0",
            ExpertEncoding::Q6K => "Q6_K",
            ExpertEncoding::Q4K => "Q4_K",
        }
    }

    /// The encoding a precision map named, or `None` if no grouped
    /// kernel reads it — refused rather than approximated.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(ExpertEncoding::Bf16),
            "Q8_0" => Some(ExpertEncoding::Q80),
            "Q6_K" => Some(ExpertEncoding::Q6K),
            "Q4_K" => Some(ExpertEncoding::Q4K),
            _ => None,
        }
    }

    /// The codec this encoding IS.
    ///
    /// Each arm names a shipped codec directly rather than resolving its
    /// label through a registry: this enum enumerates the four encodings
    /// the grouped expert kernels read, every one of them built in, so
    /// there is no registry to consult and a hidden built-in lookup here
    /// would be a second registry that registration cannot reach.
    pub fn codec(self) -> &'static dyn super::super::codec::RepresentationCodec {
        use super::super::codec::codecs::{float, kquant};
        match self {
            ExpertEncoding::Bf16 => &float::BF16,
            ExpertEncoding::Q80 => &kquant::Q8_0,
            ExpertEncoding::Q6K => &kquant::Q6_K,
            ExpertEncoding::Q4K => &kquant::Q4_K,
        }
    }

    /// Bytes an `[n, k]` matrix occupies in this encoding.
    ///
    /// Priced by the codec the encoding names, so the block geometry has
    /// one home: a table here that repeated it was the drift the codec
    /// contract exists to remove.
    pub fn matrix_bytes(self, n: usize, k: usize) -> Result<u64, VindexError> {
        use super::super::codec::RepresentationExtent;
        Ok(self
            .codec()
            .stored_bytes(&[n, k], RepresentationExtent::BASE, EXPERT_BANK_OPERAND)?)
    }
}

/// A region together with what its bytes ARE.
///
/// Per projection, not per bank, because a precision map can already
/// name `gate/up at Q6_K, down at BF16` — the scope vocabulary supports
/// it, so the physical vocabulary must too or the next experiment
/// changes this type again.
///
/// The encoding is not a caller's declaration: it comes from the
/// resolver, which knows the precision map that decided it. The map says
/// Q6_K, the resolver returns Q6_K bytes, the binding says Q6_K, and the
/// kernel follows that one fact.
#[derive(Clone)]
pub struct EncodedRegion {
    pub region: WeightRegion,
    pub encoding: ExpertEncoding,
}

impl std::fmt::Debug for EncodedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} as {}", self.region, self.encoding.name())
    }
}

impl EncodedRegion {
    /// Which physical store these bytes are in.
    pub fn store_id(&self) -> &str {
        self.region.store_id()
    }

    /// Refuse a region too small to hold what it claims to be.
    ///
    /// The execution analogue of the no-silent-fallback rule: bytes that
    /// are BF16 dispatched as Q6_K would decode to plausible garbage,
    /// and the failure would read as "quantisation is catastrophic"
    /// rather than "the wrong kernel ran". A size check catches every
    /// mismatch where the declared encoding is LARGER than the bytes;
    /// the reverse is caught by [`ExpertBankBinding::validate`], which
    /// knows the bank's extent.
    pub fn check_room(&self, top_offset: u64, n: usize, k: usize) -> Result<(), VindexError> {
        let need = top_offset + self.encoding.matrix_bytes(n, k)?;
        if need > self.region.len() {
            return Err(VindexError::Parse(format!(
                "a {} bank of [{n}, {k}] needs {need} bytes to reach its last expert, but the                  bound region ({:?}) is {} — the bytes are not what this encoding claims",
                self.encoding.name(),
                self.region,
                self.region.len()
            )));
        }
        Ok(())
    }
}

/// The shared expert's three projections, each its own region under its
/// own encoding.
///
/// A separate binding from the routed bank because `Shared` vs `Routed`
/// is SEMANTIC identity and must not imply physical co-location: a
/// source container keeps the shared expert in the decoder stack while
/// the routed experts live in an expert bank, and a candidate overlay
/// may compile the routed bank to Q6_K while the shared branch stays
/// source BF16. Placing the shared bytes next to the routed ones is a
/// layout an artifact MAY choose — the regions can be subranges of one
/// store — never something execution may assume.
#[derive(Clone)]
pub struct SharedExpertBinding {
    pub gate: EncodedRegion,
    pub up: EncodedRegion,
    pub down: EncodedRegion,
}

/// **Logical expert id → byte coordinate, for ONE projection.**
///
/// The owned form of the vocabulary the grouped kernel already speaks
/// (`larql_compute_metal::trait_impl::kimi_layer::ExpertAddressing`,
/// which borrows). One per projection rather than one per bank, and
/// that is not tidiness: a projection-scoped candidate needs `gate`
/// addressed by IDENTITY over a compiled Q6_K bank while `up` and
/// `down` stay TABLE-addressed over the arbitrarily-ordered source
/// segment, in the same layer, in the same forward pass. Metal could
/// express that from rung C onward; a bank-wide layout could not, and
/// a projection sweep is what proved the gap real rather than tidy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionAddressing {
    /// `offset = expert_id * stride`. Nothing is tabulated, and no
    /// selection can be unaddressable — a compiled full bank.
    Identity { experts: u32, stride: u32 },
    /// Byte offset per expert, indexed by logical expert id. What an
    /// arbitrarily-ordered source segment or a packed subset needs.
    Table(Vec<u32>),
}

impl ProjectionAddressing {
    /// How many experts this projection can address.
    pub fn experts(&self) -> u32 {
        match self {
            Self::Identity { experts, .. } => *experts,
            Self::Table(t) => t.len() as u32,
        }
    }

    /// The constant per-expert stride, when this projection has one.
    ///
    /// `None` for a table: a tabulated projection's entries need not be
    /// evenly spaced, and inventing a stride from two of them would be
    /// a claim about the layout nobody made.
    pub fn identity_stride(&self) -> Option<u32> {
        match self {
            Self::Identity { stride, .. } => Some(*stride),
            Self::Table(_) => None,
        }
    }

    /// The highest byte offset this projection can be asked to read
    /// from, or `None` if it addresses nothing.
    ///
    /// The quantity an extent check needs, and NOT the entry count: a
    /// table has one entry per SCORED expert while the bank behind it
    /// holds only the addressable subset — the fixture's is 256 entries
    /// over 65 blocks. Sizing the check by entries demanded a bank four
    /// times the size of the real one and refused a valid binding.
    pub fn max_offset(&self) -> Option<u64> {
        match self {
            Self::Identity { experts, stride } => {
                (*experts > 0).then(|| u64::from(experts - 1) * u64::from(*stride))
            }
            Self::Table(t) => t
                .iter()
                .filter(|o| **o != NOT_ADDRESSABLE)
                .map(|o| u64::from(*o))
                .max(),
        }
    }

    /// Byte offset of `expert`'s payload, or `None` when this projection
    /// cannot address it at all.
    pub fn offset_of(&self, expert: u32) -> Option<u64> {
        match self {
            Self::Identity { experts, stride } => {
                (expert < *experts).then(|| u64::from(expert) * u64::from(*stride))
            }
            Self::Table(t) => t
                .get(expert as usize)
                .copied()
                .filter(|o| *o != NOT_ADDRESSABLE)
                .map(u64::from),
        }
    }
}

/// One projection of a routed bank: its bytes, how they are addressed,
/// and whether the region IS the bank or a window onto a larger one.
///
/// The three travel together because they are one fact — where this
/// projection's bytes came from. A compiled candidate is `Exact` and
/// addressed by identity; a source view is a `ContainingView` addressed
/// by table. Splitting them across the bank, as this type replaced,
/// made a mixed binding inexpressible: an experiment compiling gate to
/// Q6_K while up and down stayed source-backed had to declare the whole
/// bank a `ContainingView`, which silently disabled the surplus-byte
/// check on the one projection under test — the only check that catches
/// BF16 bytes mislabelled as a smaller encoding.
#[derive(Clone)]
pub struct RoutedProjection {
    pub region: EncodedRegion,
    pub addressing: ProjectionAddressing,
    pub extent: ExtentPolicy,
}

impl RoutedProjection {
    /// Which physical store these bytes are in — the attribution a
    /// quality run asserts rather than infers from the numbers.
    pub fn store_id(&self) -> &str {
        self.region.store_id()
    }

    /// What these bytes ARE.
    pub fn encoding(&self) -> ExpertEncoding {
        self.region.encoding
    }
}

/// One layer's expert bank: three independently-bound routed
/// projections and — independently again — the shared expert's binding.
///
/// Deliberately one level above `DeviceLayer`, so execution never infers
/// addressing from the fact that it happens to hold regions. The same
/// binding covers an owned packed fixture, an mmap'd compiled bank, a
/// candidate overlay over some projections only, and eventually a
/// resident view over a K3 cold bank — one code path, several physical
/// stories.
///
/// Nothing physical is left at bank level. The only bank-wide fact is
/// the semantic one: these three projections constitute one routed MoE.
#[derive(Clone)]
pub struct ExpertBankBinding {
    pub gate: RoutedProjection,
    pub up: RoutedProjection,
    pub down: RoutedProjection,
    /// The shared expert, when the architecture declares one. `None` is
    /// a claim that the model HAS no shared branch, not that its bytes
    /// were not found — a loader that cannot find a declared shared
    /// expert must refuse, never construct a `None`.
    pub shared: Option<SharedExpertBinding>,
}

/// Why a region may be larger than the bank it addresses.
///
/// Two very different facts would otherwise be indistinguishable:
/// *these regions ARE the bank* (a compiled overlay), and *these regions
/// are windows onto a much larger segment* (a source container view).
/// Only the first can conclude that surplus bytes mean the declared
/// encoding is wrong — and that conclusion is the only thing that
/// catches BF16 bytes mislabelled Q6_K, since those are LARGER than the
/// claim and every room check passes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtentPolicy {
    /// The regions are exactly this bank. Surplus bytes are a defect.
    Exact,
    /// The regions are windows onto a larger backing. Surplus bytes are
    /// expected and say nothing about the encoding.
    ContainingView,
}

impl ExpertBankBinding {
    /// Byte offset of `expert`'s payload within the gate bank.
    ///
    /// Reads gate's OWN addressing. The stride is no longer a parameter
    /// because it is no longer a caller's guess: it belongs to the
    /// projection, and a caller that supplied a sibling's would have
    /// been believed.
    pub fn gate_offset(&self, expert: u32) -> Option<u64> {
        self.gate.addressing.offset_of(expert)
    }

    pub fn up_offset(&self, expert: u32) -> Option<u64> {
        self.up.addressing.offset_of(expert)
    }

    pub fn down_offset(&self, expert: u32) -> Option<u64> {
        self.down.addressing.offset_of(expert)
    }

    /// The three projections, labelled, for checks that must cover all
    /// of them. `(label, projection, n, k)` — down is the transpose.
    pub(super) fn projections(
        &self,
        hidden: usize,
        inter: usize,
    ) -> [(&'static str, &RoutedProjection, usize, usize); 3] {
        [
            ("routed gate", &self.gate, inter, hidden),
            ("routed up", &self.up, inter, hidden),
            ("routed down", &self.down, hidden, inter),
        ]
    }

    /// Which physical store backs the gate projection.
    ///
    /// Gate's, not "the bank's": after per-projection binding the three
    /// may legitimately come from different stores, and a single answer
    /// is only meaningful where they agree. [`Self::stores_agree`] is
    /// the question to ask when that matters.
    pub fn store_id(&self) -> &str {
        self.gate.region.region.store_id()
    }

    /// Whether all three projections are backed by the same store.
    ///
    /// False is legitimate, not a defect: an experiment that compiles
    /// one projection to a candidate store and leaves the other two
    /// source-backed is exactly the asymmetry this binding exists to
    /// express. Callers that need a single provenance answer must ask
    /// this before believing [`Self::store_id`].
    pub fn stores_agree(&self) -> bool {
        let g = self.gate.region.region.store_id();
        g == self.up.region.region.store_id() && g == self.down.region.region.store_id()
    }

    /// Every projection has room for every addressable expert at the
    /// encoding it claims.
    ///
    /// [`ExtentPolicy::Exact`] additionally refuses a region LARGER than
    /// the encoding implies; a [`ExtentPolicy::ContainingView`] cannot
    /// make that claim and checks room only.
    pub fn validate(&self, hidden: usize, inter: usize) -> Result<(), VindexError> {
        // Extent is read PER PROJECTION. A mixed binding — a candidate
        // gate compiled Exact beside source-backed up/down windows — is
        // the whole point of the per-projection split, and the exact
        // check must still bite on the compiled one. Reading a single
        // bank-wide flag here meant one source-backed sibling disabled
        // the check on every projection, including the one under test.
        for (what, proj, n, k) in self.projections(hidden, inter) {
            let enc = &proj.region;
            // The highest OFFSET this projection can be asked to read,
            // from its own addressing. Not the expert count: a table
            // has an entry per scored expert while its bank holds only
            // the addressable subset.
            let top = proj.addressing.max_offset().unwrap_or(0);
            let per = enc.encoding.matrix_bytes(n, k)?;
            enc.check_room(top, n, k)?;
            if proj.extent == ExtentPolicy::Exact {
                let want = top + per;
                if enc.region.len() != want {
                    return Err(VindexError::Parse(format!(
                        "{what}: a {} bank reaching offset {top} at [{n}, {k}] is {want} bytes,                          but the bound region ({:?}) is {} — the bytes are not this encoding",
                        enc.encoding.name(),
                        enc.region,
                        enc.region.len()
                    )));
                }
            }
        }
        // The shared expert's regions, each under its own encoding.
        // Room-checked only: a shared region is typically a whole named
        // tensor or a window into a store whose extent semantics belong
        // to that store, so surplus bytes say nothing here.
        if let Some(shared) = &self.shared {
            for (enc, n, k) in [
                (&shared.gate, inter, hidden),
                (&shared.up, inter, hidden),
                (&shared.down, hidden, inter),
            ] {
                enc.check_room(0, n, k)?;
            }
        }
        self.check_bindable()
    }

    /// Every region sits at an offset a backend can bind zero-copy.
    ///
    /// A segment whose payload starts at an odd byte puts every tensor
    /// in it at an odd address. A backend that binds the mapping
    /// zero-copy then hands its kernel a misaligned pointer — on Metal,
    /// a `device const ushort*` at an odd address reads garbage and the
    /// command buffer still reports success. The backend declines such
    /// a binding and stages a copy instead, which is correct but means
    /// silently copying gigabytes per dispatch, so the condition is
    /// refused HERE, where the cause can be named.
    ///
    /// Measured: the Kimi container's `decoder_stack` payload began at
    /// 56,925, and every dense/shared-expert dispatch returned NaN.
    pub(super) fn check_bindable(&self) -> Result<(), VindexError> {
        let mut regions: Vec<(&str, &EncodedRegion)> = vec![
            ("routed gate", &self.gate.region),
            ("routed up", &self.up.region),
            ("routed down", &self.down.region),
        ];
        if let Some(s) = &self.shared {
            regions.extend([
                ("shared gate", &s.gate),
                ("shared up", &s.up),
                ("shared down", &s.down),
            ]);
        }
        for (what, r) in regions {
            if !r.region.is_bindable_at(WEIGHT_BINDING_ALIGN) {
                return Err(VindexError::Parse(format!(
                    "{what} sits at byte {} of {:?}, which is not a multiple of                      {WEIGHT_BINDING_ALIGN} — a backend cannot bind it zero-copy, and the                      usual cause is a segment whose payload does not start on an aligned                      boundary. Re-write the container's segments with an aligned payload                      (`encode::segment::SEGMENT_PAYLOAD_ALIGN`); the payload bytes and their                      hash do not change.",
                    r.region.store_offset(),
                    r.region
                )));
            }
        }
        Ok(())
    }
}
