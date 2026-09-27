//! The backend seam (V3-G5b-3b): what *executes* a plan, versus what the
//! plan *means*.
//!
//! One [`ComponentOpPlan`](super::super::ComponentOpPlan), one interpreter,
//! many backends. The interpreter in [`super`] owns every decision that is
//! semantics — operation ordering, residual ordering, layer traversal,
//! whether an optional operation exists at all, and how position and span
//! policy dispatch. A [`PlanBackend`] owns only the numerical realisation
//! of work it is handed.
//!
//! **Nothing in this file mentions a model family, and nothing in it takes
//! a plan type.** Backends receive primitives, judged enums, and *already
//! loaded* weight slices — never a `LayerPlan`, an `OperandRef`, or the
//! `OperandStore`. That is deliberate and load-bearing: a backend that
//! could resolve its own operands by name, or read the layer structure,
//! could quietly grow into a second implementation of the model and
//! disagree with the IR while still passing. It cannot reach the bytes,
//! so it cannot reinterpret them.
//!
//! The corollary for anyone adding a method: if a backend needs to ask
//! *whether* to do something, the seam is in the wrong place. It should
//! only ever be told what to compute.

use crate::format::vindex3::represent::kquant::KQuant;

mod calls;
mod plan_backend;
mod step_calls;
pub use calls::*;
pub use plan_backend::*;
pub use step_calls::*;

/// V3-INTERVENE-2: a per-head intervention applier — `(head, ctx_h)`,
/// mutating the head's mixed value in place.
pub type HeadIntervene<'a> = dyn FnMut(usize, &mut [f32]) + 'a;

/// The numerical representation a backend wants matrix operands in.
///
/// Asked once by the interpreter (a capability, like [`PlanBackend::name`],
/// not a per-call decision): the interpreter loads every matrix operand in
/// the declared format and the backend receives what it asked for. Norm
/// and QK-norm weights and the embedding table are always f32 — they are
/// elementwise glue, not matrix traffic, and narrowing them buys nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightFormat {
    /// Widened f32 — the constitutional representation.
    F32,
    /// Stored bf16, kept EXACTLY as the checkpoint holds it.
    ///
    /// Not a conversion and not a quantisation: bf16 is the top 16 bits
    /// of the f32 it denotes, so a consumer widens with `(bits as u32) <<
    /// 16` — no rounding, no table, no loss. Declaring this removes the
    /// artificial F32 materialisation (107.6 GB resident against a 53.8
    /// GB checkpoint) rather than introducing a new numeric format.
    ///
    /// Only worth declaring for matrices large enough to STREAM. A
    /// cache-resident matrix has no RAM traffic to halve, and the
    /// measured `48 x 5120` case runs 3.8x faster through BLAS f32 — see
    /// `exec::cpu::kernels::FusedBf16`.
    Bf16,
    /// Symmetric int8, one f32 scale per [`Q8_BLOCK`] elements along the
    /// input axis.
    ///
    /// **The first LOSSY residency format.** `Bf16` keeps the
    /// checkpoint's own bytes and changes no value; this one quantises at
    /// load and the model it decodes is not quite the model that was
    /// stored. Everything about it is therefore judged on logits, KL, a
    /// trajectory and recurrent-state drift, not on residency alone.
    ///
    /// Blocked along the input axis so a kernel accumulates a block and
    /// scales once. 8.5 bits/weight with the scales counted.
    ///
    /// Worth declaring only where the BF16 image is too big for cache —
    /// measured, `1024 x 5120` runs 0.81x through Q8 because at 10.5 MB
    /// it is already L2-resident and the extra unpacking is pure cost.
    Q8,
    /// Symmetric int4, two codes per byte, one f32 scale per
    /// [`Q4_BLOCK`](crate::format::vindex3::opplan::exec::quantise::Q4_BLOCK)
    /// elements along the input axis. 4.5 bits/weight.
    ///
    /// **The second lossy residency format, and by far the larger
    /// perturbation.** Its step is `peak / 7` against Q8's `peak / 127` —
    /// 18.1x coarser at the same block — so it is not "Q8 with fewer
    /// bytes", it is a different numerical proposition that has to earn
    /// its place on logits, KL, a trajectory and recurrent-state drift.
    ///
    /// Worth declaring only where the arithmetic can consume it. CPU-4A
    /// measured Q4 against f32 activations at 1.08x — SLOWER than Q8 —
    /// because the kernel was already conversion-bound and Q4 adds a
    /// nibble split on top; CPU-4Y measured the same bytes against an
    /// int8 activation at 3.12x. The format is only ever as good as the
    /// domain it is multiplied in.
    Q4,
    /// IEEE 754 half, little-endian. Exactly representable from stored
    /// bf16 for all normal-range values (bf16's 7 mantissa bits fit in
    /// f16's 10); conversion fails closed on overflow. A device backend
    /// declares this so weights can stay resident in half the bytes.
    F16,
    /// OCP microscaling 4-bit float: e2m1 codes two-per-byte plus one
    /// e8m0 scale per 32-element group, in separate streams. A lossy
    /// realisation — quantised at load, judged by the parity gates —
    /// that quarters the bytes every decoded token must read.
    Mxfp4,
    /// The same e2m1 elements under a different scale geometry: 16-element
    /// groups with **E4M3** scales, plus one f32 per matrix. 4.5 bpw
    /// against MXFP4's 4.25.
    ///
    /// Present as its own format rather than a parameter of [`Self::Mxfp4`]
    /// because the difference is the point: E8M0 forces a group's scale to
    /// a power of two, and a weight-reconstruction sweep over Muse-Glimmer
    /// with an equal-bit-budget control (E8M0 at group 16) found the group
    /// size worth nothing and the scale format worth 1.27x in relative RMS
    /// and 1.7x in worst-element error.
    Nvfp4,
    /// The same stored NVFP4 pack as [`Self::Nvfp4`], bound to run against a
    /// **Q8 activation** (NVFP4-Q8-1, `docs/nvfp4-q8-1.md`).
    ///
    /// Byte-for-byte the same residency, and like [`Self::KQuantQ8k`] a
    /// separate format only because the executor OBSERVES its kernel from
    /// what is resident: the activation form is fixed at load, by the
    /// realization the provider pinned. Stored packs only; nothing is
    /// quantised at load.
    Nvfp4Q8,
    /// A stored ggml K-quant pack — Q8_0, Q6_K or Q4_K — kept as the
    /// container holds it and executed in place by the codec's kernel.
    ///
    /// Like [`Self::Bf16`] and unlike [`Self::Q8`], NOT a conversion: the
    /// resident bytes are the stored bytes. The lossy step, if any,
    /// happened when `vindex represent` compiled the pack, and it is the
    /// artifact under measurement, not this loader's doing. Which codec
    /// is a property of the bytes, carried with them — the format names
    /// the family, the operand's stored dtype names the member.
    ///
    /// Only ever declared for an operand the container holds as a
    /// K-quant; a backend asking for it over anything else is refused at
    /// load rather than served a manufactured pack.
    KQuant,
    /// The same stored K-quant blocks as [`Self::KQuant`], bound to run
    /// against a **Q8_K activation** (Q8K-ACT-1, `docs/q8k-act-1.md`).
    ///
    /// Byte-for-byte the same residency. It is a separate format because
    /// the executor OBSERVES its kernel from what is resident and never
    /// chooses again: the activation form has to be fixed at load, by the
    /// realization the provider pinned, or the plan that ran would not be
    /// the plan that was selected. Only for members with a Q8_K kernel
    /// (Q4_K, Q6_K); any other member is refused at load.
    KQuantQ8k,
    /// Fine-grained (block-wise) FP8: the checkpoint's own E4M3 codes
    /// against a two-dimensional grid of f32 scales.
    Fp8Block,
    /// The stored bytes themselves, uninterpreted, for whichever codec
    /// the operand's representation names — carried with a copy of that
    /// name, never decoded, widened, or otherwise judged here.
    ///
    /// Like [`Self::KQuant`], the format names a capability, not a
    /// member: which codec produced the bytes is a property of the
    /// operand, read from the container and handed back alongside them.
    /// This loader has no registry of what any codec's bytes mean and
    /// asks none — a backend requesting this format is the only party
    /// that can interpret what comes back, by matching the returned name
    /// itself. Only ever declared by a backend prepared to do that; a
    /// backend that says nothing here never receives it.
    CodecOwned,
}

/// The activation form a stored K-quant is bound to run against — fixed
/// at load from the pinned realization, so the kernel is read back off
/// the resident operand rather than chosen a second time.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum KQuantActivation {
    /// f32 activation: the codec's own dequantise-and-FMA kernel.
    #[default]
    F32,
    /// The activation quantised once per call to Q8_K (int8, one scale
    /// per 256), multiplied by the integer-dot `q4k_q8k` family. Lossy in
    /// the activation, by declaration.
    Q8k,
}

/// The activation form a stored NVFP4 pack is bound to run against, fixed
/// at load from the pinned realization.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Nvfp4Activation {
    /// f32 activation: FP4 codes widened and multiplied in f32.
    #[default]
    F32,
    /// The activation quantised once per call to int8, one scale per
    /// 16-element NVFP4 group, and multiplied by an integer dot product.
    /// The weight side stays exact; lossy in the activation, by declaration.
    Q8,
}

/// Which matrix a format question is about. Formats are declared per
/// class because the classes have different numerical stakes: the
/// output head feeds logits directly, attention feeds the softmax, and
/// the FFN is the bulk of the bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixClass {
    AttentionProjection,
    FfnProjection,
    OutputHead,
    /// One stored tensor holding every expert's matrix, sliced at load.
    ///
    /// Its own class because it is not one matrix: the bank is split into
    /// `experts` matrices and may be quantised on the way, so the path has
    /// already widened to f32 by the time a format could be applied. A
    /// question about "how big is this matrix" has no answer here, which
    /// is exactly why answering it as an `FfnProjection` would be wrong.
    RoutedExpertBank,
}

/// A backend's declared format per matrix class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WeightFormats {
    pub attention: WeightFormat,
    pub ffn: WeightFormat,
    pub head: WeightFormat,
}

impl WeightFormats {
    /// The same format everywhere.
    pub fn uniform(format: WeightFormat) -> Self {
        Self {
            attention: format,
            ffn: format,
            head: format,
        }
    }

    pub fn for_class(&self, class: MatrixClass) -> WeightFormat {
        match class {
            MatrixClass::AttentionProjection => self.attention,
            // A device backend places the bank exactly as it places any
            // other FFN matrix — the distinction the class draws is about
            // the HOST load path, not about device residency.
            MatrixClass::FfnProjection | MatrixClass::RoutedExpertBank => self.ffn,
            MatrixClass::OutputHead => self.head,
        }
    }
}

/// One matrix operand, in the representation the backend declared.
///
/// An `F16` slice may be longer than the matrix needs (page-padded for
/// zero-copy device wrapping); geometry always travels separately.
#[derive(Clone, Copy)]
pub enum WeightSlice<'a> {
    F32(&'a [f32]),
    /// Stored bf16 code units, still compact.
    Bf16(&'a [u16]),
    /// Symmetric int8 codes and their per-block f32 scales.
    Q8 {
        codes: &'a [i8],
        scales: &'a [f32],
        /// Per-`SUM_BLOCK` code sums; empty where no arm consumes them.
        sums: &'a [i16],
        block: usize,
    },
    /// Symmetric int4 codes packed two per byte, and their per-block f32
    /// scales. Byte `j` of a block holds elements `j` and `j + block/2`.
    Q4 {
        packed: &'a [u8],
        scales: &'a [f32],
        block: usize,
    },
    /// Little-endian IEEE f16 bytes.
    F16(&'a [u8]),
    /// MXFP4: packed e2m1 codes (`[n, k/32, 16]`, lo nibble first) and
    /// e8m0 scales (`[n, k/32]`) as two streams.
    Mxfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
    },
    /// NVFP4: packed e2m1 codes (`[n, k/16, 8]`, lo nibble first), E4M3
    /// group scales (`[n, k/16]`), and the single f32 both scale levels
    /// are expressed relative to.
    Nvfp4 {
        packed: &'a [u8],
        scales: &'a [u8],
        tensor_scale: f32,
        activation: Nvfp4Activation,
    },
    /// A stored ggml K-quant block stream, still compact, with the codec
    /// that names its layout. Scales live inside the blocks, so this is
    /// one stream, not two.
    KQuant {
        blocks: &'a [u8],
        codec: KQuant,
        activation: KQuantActivation,
    },
    /// Fine-grained FP8: E4M3 codes and the f32 scale grid, both the
    /// checkpoint's own bytes. TWO streams, and unlike every other pair
    /// here the second is indexed in two dimensions.
    Fp8Block {
        codes: &'a [u8],
        scales: &'a [f32],
        block_rows: usize,
        block_cols: usize,
        scale_cols: usize,
    },
    /// The stored bytes themselves, exactly as [`WeightFormat::CodecOwned`]
    /// promises: one stream, one name, zero interpretation. `label` is
    /// the operand's own stored representation name (a container's
    /// `dtype`/encoding field) — a plain string this loader read and
    /// passed through, not a type it knows the meaning of.
    CodecOwned {
        bytes: &'a [u8],
        label: &'a str,
    },
}
