//! `larql vindex3 ops` — the generic operation plan (V3-G5b-1).
//!
//! Given only a container, answer: **what exact generic program does this
//! component mean?** Every argument comes from the persisted graph — the
//! execution surface, the per-layer attention policy, the operand roles —
//! and every operand is a logical-object reference plus a segment-relative
//! tensor. No family name, no layer-pattern arithmetic, no HF tensor name
//! appears anywhere in a plan.
//!
//! **Operand closure** is the hard gate this rung adds (the invariant G4
//! cannot state): four-authority equivalence proves *consistency*; closure
//! proves *sufficiency*.
//!
//! ```text
//! for every tensor of an executable object:
//!     tensor → classified operand role → consumed by a generic op
//! and for every op the surface implies:
//!     its operands exist, with the geometry the surface states
//! ```
//!
//! A tensor the roles cannot classify, an operand implying an op the
//! surface does not carry (the attention-gate discovery), a missing
//! operand, or a wrong shape each block the plan — itemised, before a
//! single matmul.

pub mod build;
pub mod conv_qkv;
pub mod exec;
pub mod gated_delta;
pub mod kda;
pub mod mamba2;
pub mod mla;
pub mod planned;

#[cfg(test)]
pub(crate) mod tests;

pub use build::plan_component_ops;
pub use gated_delta::GatedDeltaOp;
pub use kda::{KdaOp, KdaOutputGate};
pub use mamba2::Mamba2Op;
pub use mla::{MlaOp, MlaQueryProjection};

mod layer;
mod ops;
mod outcome;
pub use layer::*;
pub use ops::*;
pub use outcome::*;
