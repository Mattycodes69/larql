//! One complete Kimi decoder layer in ONE GPU ownership interval.
//!
//! Rung 5d. Rung 5c closed the attention; this closes the layer, and the
//! part that decides whether it IS closed is not the residuals or the
//! norms — it is whether Metal can compute the routing decision and
//! consume it in the grouped MoE **without the host ever seeing a
//! selected expert id**.
//!
//! ```text
//! hidden ─┬──────────────────────────────────────────┐  residual
//!         ↓                                          │
//!   input RMSNorm                                    │
//!         ↓                                          │
//!   KDA attention (device-resident state)            │
//!         ↓                                          ↓
//!   after_attention = hidden + attn ─────────────────┘─┐  residual
//!         ↓                                            │
//!   post-attention RMSNorm                             │
//!         ↓                                            │
//!   router: logits → sigmoid → +bias → top-k → renorm  │
//!         ↓  (GPU-written offset table AND weights)    │
//!   grouped MoE: top-k routed + shared                 │
//!         ↓                                            ↓
//!   layer output = after_attention + Σ w·expert ───────┘
//! ```
//!
//! One command buffer. One host→device upload (the layer input), one
//! device→host read (the layer output). Everything between — including
//! which experts ran — stays on device.
//!
//! **Residency is checked, never guessed.** The router writes offsets
//! out of a caller-supplied table mapping each expert to where it lives
//! in the resident bank; a selection of a non-resident expert is counted
//! on device and refused by the host after the wait. Which experts are
//! resident is the next problem, not this one's — what matters here is
//! that reading the wrong expert's weights is impossible rather than
//! merely unlikely.

mod ffn;
mod head;
mod traced;
pub use ffn::{FfnSpec, KimiDenseFfn};
pub use head::KimiHead;
pub use traced::KimiLayerPlanes;

mod encode;
mod types;
mod validate;
pub use types::*;
use validate::*;

#[cfg(test)]
mod tests;
