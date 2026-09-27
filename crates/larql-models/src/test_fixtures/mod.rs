//! Shared test fixtures for `ModelWeights` consumers.
//!
//! Gated behind the `test-utils` feature so production builds never
//! pull in the synthetic builders. Downstream test crates
//! (`larql-compute`, `larql-inference`, `larql-vindex`, `larql-kv`)
//! depend on `larql-models` with `features = ["test-utils"]` under
//! `[dev-dependencies]` to construct realistic `ModelWeights` without
//! disk I/O.
//!
//! Architecture-specific fixtures (Gemma 3, StarCoder2, Q4K, MoE, E2B)
//! still live in `crates/larql-inference/src/test_utils.rs` because
//! they pull in inference-side concepts (vindex, tokenizer, mock GPU
//! backends). Only the generic `TinyModel` builder lives here — it's
//! the one the moved-down forward-pass tests in `larql-compute` need.

mod gemma3;
mod gemma4_e2b;
mod gemma4_moe;
mod q4k;
mod rng;
mod starcoder2;
mod tiny;

pub use gemma3::*;
pub use gemma4_e2b::*;
pub use gemma4_moe::*;
pub use q4k::*;
use rng::*;
pub use starcoder2::*;
pub use tiny::*;

#[cfg(test)]
mod tests;
