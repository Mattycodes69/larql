//! The reference backend: naive f32, the semantic anchor.
//!
//! Shares **nothing** with `larql-compute`'s kernels. That is the whole
//! point of it — a reference that called the production kernels would
//! agree with them by construction, and the agreement would prove
//! nothing. Plain loops, row-major `[out, in]` weights, no BLAS, no
//! SIMD, no fusion.
//!
//! When the production backend disagrees with this one, this one is
//! right about *meaning* and may well be wrong about speed. Divergence
//! is a bug in the production backend or a hole in the seam, never a
//! licence to change what the plan means.

use rayon::prelude::*;

mod ops;
mod plan_backend;
pub(super) use plan_backend::*;

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "reference-f32";
/// The provider's family ([`PlanBackend::identity`]): the literal f32
/// transcription that defines correctness. Its revision moves only if the
/// transcription itself is corrected — which is a change to what every
/// other provider is judged against, and is recorded as one.
pub const IDENTITY_FAMILY: &str = "reference";
pub const IDENTITY_REVISION: u32 = 1;

/// Naive f32 realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceBackend;
