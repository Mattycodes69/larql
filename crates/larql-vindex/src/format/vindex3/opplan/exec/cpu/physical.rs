//! One decision — format, kernel, and threading together.
//!
//! The hazard this file exists to remove is a loader that chooses BF16
//! while an executor separately guesses which kernel to run. Those are
//! two derivations of one fact, and two derivations drift. With two
//! formats the drift is a bug; with Q8 and Q4 as well it is a state
//! space nobody can hold in their head.
//!
//! So there is exactly one value, [`PhysicalProjectionPlan`], and both
//! halves are read off it. The loader asks [`PhysicalProjectionPlan::choose`]
//! what to make resident; the executor asks
//! [`PhysicalProjectionPlan::for_resident`] what is resident. The second
//! is not a second decision — it is an OBSERVATION of the first, total
//! over the representations a CPU kernel can consume, so the two cannot
//! disagree about a matrix even in principle.
//!
//! [`project_matrix`] and [`ExecutorProjections`] live here rather than
//! beside the backend for the same reason: they are the only readers of
//! the observation, and a projection helper that sat somewhere else would
//! be one refactor away from choosing its own kernel again.

use super::projector::{CpuParallelism, DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::backend::MatrixClass;

mod projection;
pub use projection::*;

/// Q8_K's activation block: one f32 scale per this many elements. A
/// property of the ggml format the `q4k_q8k` kernels read, not a policy.
pub const Q8K_ACTIVATION_BLOCK: usize = 256;

/// Default performance-cluster L2, used where the machine does not
/// report one. The value this rung measured against (Apple M3 Max).
const DEFAULT_L2_BYTES: usize = 16 * 1024 * 1024;

/// How a dense projection is physically realised on the CPU.
///
/// A single enum rather than a `(format, kernel)` pair because the
/// pairing is not free: `FusedBf16` consumes [`WeightRows::Bf16`] and
/// nothing else, and a pair type would let a caller build the one
/// combination that cannot run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PhysicalProjectionPlan {
    /// The literal scalar transcription over f32. The oracle: chosen by
    /// the reference backend, never by the policy.
    ScalarF32,
    /// Q8 resident, widened and scaled in registers, executor-threaded.
    ///
    /// The first LOSSY plan: the values it decodes are not the values the
    /// checkpoint stores. Worth 1.28x on the projections a token runs —
    /// half the bytes returning a third of the time, because at 8.5 bits
    /// the kernel stops waiting for memory and starts waiting for the
    /// widen.
    FusedQ8,
    /// Q4 resident, unpacked and scaled in registers.
    ///
    /// Reachable by OBSERVATION and not by [`Self::choose`]: CPU-4A asks
    /// only whether Q4 x f32 is worth making a model representation, and
    /// no `WeightFormat` names it, so a policy answering Q4 would refuse
    /// at load. Listed so `for_resident` stays total.
    FusedQ4,
    /// NVFP4 resident, decoded in registers.
    ///
    /// Reached by OBSERVATION, like [`Self::FusedQ4`]: the residency
    /// policy never chooses NVFP4: a pack is compiled deliberately by
    /// `vindex represent` and the loader binds what the container holds.
    /// This exists so that a compiled pack has somewhere to execute — a
    /// representation with no execution path cannot be measured, and
    /// before this arm every NVFP4 backend was a device backend.
    FusedNvfp4,
    /// The same stored NVFP4 pack against a **Q8 activation**, one scale
    /// per 16-element group, with an exact int32 sum per group
    /// (NVFP4-Q8-1). Reached by OBSERVATION of the resident binding, which
    /// the `production-nvfp4-q8` provider pins; never chosen by a policy.
    FusedNvfp4Q8,
    /// A stored ggml K-quant — Q8_0, Q6_K or Q4_K — executed in place by
    /// the kernel its codec names. PARETO-1's v3 arm.
    ///
    /// Reached by OBSERVATION, like [`Self::FusedNvfp4`], and for the
    /// same reason: nobody manufactures a K-quant at load. A pack is
    /// compiled deliberately by `vindex represent`, the loader binds the
    /// bytes the container holds, and this is where they execute. The
    /// policy's only say is whether a stored pack executes in place at
    /// all — [`kquant_execution`] — never which codec.
    FusedKQuant,
    /// The same stored K-quant blocks against a **Q8_K activation**: the
    /// integer-dot kernel V2's CPU decode uses (Q8K-ACT-1). Reached by
    /// OBSERVATION of the resident binding, which the
    /// `production-q4k-q8k` provider pins; never chosen by a policy.
    FusedKQuantQ8k,
    /// Fine-grained FP8 resident, decoded and tile-scaled in registers.
    ///
    /// Reached by OBSERVATION like [`Self::FusedKQuant`], and with less
    /// ambiguity than any arm here: these are the CHECKPOINT's bytes in
    /// the checkpoint's own format, so there is no policy that could have
    /// produced them and none that can choose otherwise. Not lossy — it
    /// decodes what the loader would have materialised, later.
    FusedFp8Block,
    /// f32 resident, BLAS `sgemv`, threaded by the library.
    ///
    /// The right answer for a matrix whose widened image still fits
    /// cache — see [`compact_threshold_bytes`].
    BlasF32,
    /// bf16 resident, widened in registers, threaded by the executor.
    ///
    /// Halves the bytes a decoded token streams AND halves what the
    /// model occupies, because they are the same bytes.
    FusedBf16,
    /// Q8 resident, **int8 activation, i32 accumulator**, rescaled once
    /// per row. 224.75 ms/token at 118.0 GB/s.
    Q8xQ8,
    /// Q4 resident, int8 activation, i32 accumulator. 135.10 ms/token at
    /// 106.6 GB/s — the CPU-4Y frontier.
    Q4xQ8,
    /// bf16 resident and EXACT, int8 activation, f32 dot. The control
    /// arm: it isolates activation quantisation from weight
    /// quantisation, and is never chosen for speed.
    Bf16xQ8,
    /// A kernel this crate does not implement, over
    /// [`WeightFormat::CodecOwned`](super::super::backend::WeightFormat::CodecOwned)
    /// bytes. Nothing in `larql-vindex` ever pins or executes this
    /// variant — [`PlanBackend::select`](super::super::backend::PlanBackend::select)
    /// exists precisely so an external backend can pin its OWN kernel
    /// without this crate knowing what it is; this arm exists only so
    /// the enum has somewhere to name that possibility, the same reason
    /// [`WeightFormat::CodecOwned`](super::super::backend::WeightFormat::CodecOwned)
    /// does.
    CodecOwned,
}

/// **Which arithmetic the projections run in**, for the whole process.
///
/// A weight representation does NOT determine this. Q8 bytes can be
/// consumed by a widening f32 kernel or by `SDOT`, and the two are the
/// same residency with different numerics — which is exactly why
/// [`PhysicalProjectionPlan::for_resident`] cannot answer from the bytes
/// alone any more and has to consult the same policy the loader did.
///
/// Process-wide and read ONCE, because it is the arm of an experiment:
/// a bank run is one process per arm, and a value that could change
/// mid-decode would make the resulting distribution describe no single
/// representation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize)]
pub enum ArithmeticArm {
    /// Compact weights against an f32 activation — what ships today.
    #[default]
    FloatActivation,
    /// Exact bf16 weights against an int8 activation. CPU-5 arm A1.
    Bf16TimesQ8,
    /// Q8 weights against an int8 activation. CPU-5 arm A3.
    Q8TimesQ8,
    /// Q4 weights against an int8 activation. CPU-5 arm A4.
    Q4TimesQ8,
}

/// **Which matrix classes a Q4 arm is permitted to reach.**
///
/// Blanket Q4 is a hypothesis, not a plan. If it fails, the question
/// becomes the smallest set of operands that must be RESTORED to a
/// higher precision to recover quality — and the only axis today's seam
/// can express is the matrix class, because
/// [`super::super::prepared`] deliberately refuses to hand the policy an
/// `OperandRef`: resolving operands by name is the one thing the seam
/// forbids, and a name-based exception set would be a per-model recipe
/// rather than a policy.
///
/// Class is therefore a FIRST CUT, not the final axis. It cannot say
/// "the last five layers' FFN", which is where a 4-bit knee has already
/// been found once on another model. Saying so is a scope limit, not a
/// result.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q4Classes {
    pub attention: bool,
    pub ffn: bool,
    pub head: bool,
}

impl Q4Classes {
    /// Blanket Q4 — every eligible operand, which is arm R0.
    pub const ALL: Self = Self {
        attention: true,
        ffn: true,
        head: true,
    };

    /// Whether this class may go to Q4. A class that may not falls back
    /// to Q8 **in the same arithmetic domain**, never to an f32
    /// activation: a rescue that also changed the activation treatment
    /// would move two things at once, which is the mistake CPU-4A made.
    pub fn admits(self, class: MatrixClass) -> bool {
        match class {
            MatrixClass::AttentionProjection => self.attention,
            MatrixClass::FfnProjection => self.ffn,
            MatrixClass::OutputHead => self.head,
            // The bank is widened to f32 on the way in; no compact bytes
            // remain by the time a format could apply.
            MatrixClass::RoutedExpertBank => false,
        }
    }
}

/// Names which classes a Q4 arm reaches: a comma-separated subset of
/// `attn`, `ffn`, `head`, or `all`.
pub const Q4_CLASSES_ENV: &str = "LARQL_CPU_Q4_CLASSES";

impl Q4Classes {
    /// The class set a value of [`Q4_CLASSES_ENV`] names.
    ///
    /// Unset, empty and `all` mean [`Self::ALL`] — the blanket arm — so
    /// an arm run without the variable is the hypothesis rather than a
    /// silent exception set. Tokens are exact: `ATTN` names nothing, and
    /// a set that names nothing restores every class to Q8.
    pub fn from_env_value(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            None | Some("") | Some("all") => Self::ALL,
            Some(v) => {
                let named = |k: &str| v.split(',').any(|t| t.trim() == k);
                Self {
                    attention: named("attn"),
                    ffn: named("ffn"),
                    head: named("head"),
                }
            }
        }
    }
}

/// The Q4 class set, resolved once per process.
pub fn q4_classes() -> Q4Classes {
    static CLASSES: std::sync::OnceLock<Q4Classes> = std::sync::OnceLock::new();
    *CLASSES
        .get_or_init(|| Q4Classes::from_env_value(std::env::var(Q4_CLASSES_ENV).ok().as_deref()))
}

/// Names the arithmetic arm. See [`ArithmeticArm`].
pub const ARITHMETIC_ARM_ENV: &str = "LARQL_CPU_ARITHMETIC";

impl ArithmeticArm {
    /// The arm a value of [`ARITHMETIC_ARM_ENV`] names.
    ///
    /// An unrecognised value is the DEFAULT rather than an error, matching
    /// [`MAX_FORMAT_ENV`]: a typo must not silently invent a fourth
    /// numerical regime that then gets reported as a measurement.
    pub fn from_env_value(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            // The `b` suffix selects a per-BLOCK activation scale; the
            // arm itself is the same arithmetic either way, which is why
            // the scale geometry is a separate value and not a fourth arm.
            Some("bf16xq8") | Some("bf16xq8b") => Self::Bf16TimesQ8,
            Some("q8xq8") | Some("q8xq8b") => Self::Q8TimesQ8,
            Some("q4xq8") | Some("q4xq8b") => Self::Q4TimesQ8,
            _ => Self::FloatActivation,
        }
    }
}

/// The arm, resolved once per process.
pub fn arithmetic_arm() -> ArithmeticArm {
    static ARM: std::sync::OnceLock<ArithmeticArm> = std::sync::OnceLock::new();
    *ARM.get_or_init(|| {
        ArithmeticArm::from_env_value(std::env::var(ARITHMETIC_ARM_ENV).ok().as_deref())
    })
}

/// [`PhysicalProjectionPlan::CodecOwned`]'s kernel slot. This crate never
/// selects that plan (only an external `PlanBackend` pins it, and such a
/// backend runs its own arithmetic in its own `project`/`ffn`/
/// `output_head` methods without ever calling `PhysicalProjectionPlan::
/// kernel`), so this exists only to keep [`PhysicalProjectionPlan::kernel`]
/// total; reaching it is an interpreter bug, not a codec's doing, and it
/// says so rather than guessing at an answer.
struct NoInTreeKernel;

impl DenseProjector for NoInTreeKernel {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::Serial
    }

    fn project_rows(&self, _weight_rows: WeightRows<'_>, _x: &[f32], _out: &mut [f32]) {
        unreachable!(
            "PhysicalProjectionPlan::CodecOwned has no in-tree kernel; an external PlanBackend \
             must pin this plan and run its own arithmetic without calling `kernel()`"
        );
    }
}
