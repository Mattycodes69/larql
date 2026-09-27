//! One complete KDA attention operation in ONE GPU ownership interval.
//!
//! Rung 5c. Rung 5b measured the shape this replaces: KDA's projections
//! run 2.8x faster on device (0.25 ms against 0.70), and the two
//! CPU↔GPU crossings the host-side recurrence forces cost 0.40 ms and
//! give 89% of that back. The crossings exist only because the stages
//! between the projections live on the host. This encodes all of them —
//! convolution, q/k norms, low-rank gates, decay, beta, the delta-rule
//! recurrence, the gated norm — into one command buffer with the
//! projections, so a layer's attention costs **one** crossing.
//!
//! ```text
//! upload normalised hidden          <- the only host->device transfer
//!   grouped q|k|v                       (one dispatch, three slots)
//!   conv+silu x3, q/k L2 norm
//!   f_a -> f_b -> decay, g_a -> g_b, b_proj -> beta
//!   recurrence            (reads and writes device-resident state)
//!   gated RMS norm
//!   o_proj
//! read the attention output         <- the only device->host transfer
//! ```
//!
//! **The recurrent state and the three convolution windows stay on
//! device between calls.** Reading them back to keep the host's
//! representation authoritative would reintroduce the crossing this rung
//! exists to remove; [`KdaDeviceState`] owns them for the life of a
//! sequence and [`KdaDeviceState::read_back`] exists only so a gate can
//! check them against the CPU path.
//!
//! Deliberately NOT done: nothing is fused beyond what
//! `exec::kda::step` already fuses, and no stage is reordered. Rung 4
//! is the standing warning — a fusion whose traffic saving is a fraction
//! of a percent can lose by perturbing an access pattern, and none of
//! these stages is where the bytes are.

// Names the test modules reach through `use super::*`.
#[cfg(test)]
use super::grouped_experts::{ExpertOffset, GroupedError};
#[cfg(test)]
use super::kimi_layer::ExpertEncoding;
#[cfg(test)]
use crate::shaders::kda as kda_shader;

mod encode;
mod types;
pub use types::*;

pub mod trajectory;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gate_form;

#[cfg(test)]
mod q4_trajectory;
