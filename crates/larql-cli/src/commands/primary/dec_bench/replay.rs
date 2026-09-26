//! Pure replay logic for the DEC loadgen: sweep-plan expansion, wire-frame
//! construction, and per-point statistics. Everything transport-shaped lives
//! in `replay_runtime.rs`; every byte on the wire goes through the SAME codec
//! functions the production client uses (parity discipline).

mod denominators;
mod frames;
mod summary;
mod wire;
pub use denominators::*;
pub use frames::*;
pub use summary::*;
pub use wire::*;

#[cfg(test)]
mod tests;
