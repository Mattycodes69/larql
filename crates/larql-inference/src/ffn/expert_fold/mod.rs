//! Shard-side MoE expert fold: the router-weighted sum over a set of
//! experts a client has already selected.
//!
//! This is what a remote expert shard computes per request, and it is
//! independent of how the request arrived: [`fold`] holds the CPU fold
//! (pre-experts norm hoisted, per-thread scratch, rayon fold) and
//! [`packed`] the checked slicing of the packed BF16 expert table.

pub mod fold;
pub mod packed;

pub use fold::{
    count_nonzero_weights, fold_experts_cpu, fold_experts_q8k_prenormed, ExpertFoldOptions,
};
