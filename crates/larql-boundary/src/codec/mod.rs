//! Phase 1 — residual compression codecs.
//!
//! Two schemes ship in v0.1:
//!
//! | Module  | Bytes (d=2560) | Ratio | Contract |
//! |---------|----------------|-------|----------|
//! | [`bf16`]  | 5 120         | 1×    | Exact    |
//! | [`int8`]  | 2 564         | 2×    | D-       |
//!
//! All functions are pure: no allocations beyond the returned value.
//! No model or MLX dependency.

pub mod bf16;
pub mod int8;

/// A wire payload that cannot be decoded under its codec's framing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CodecError {
    /// The payload length is not a whole number of encoded elements.
    #[error("{codec} payload of {len} bytes is not a multiple of {elem_bytes}")]
    RaggedPayload {
        codec: &'static str,
        len: usize,
        elem_bytes: usize,
    },
    /// The payload is shorter than the codec's fixed header.
    #[error("{codec} payload of {len} bytes is shorter than its {header_bytes}-byte header")]
    TruncatedHeader {
        codec: &'static str,
        len: usize,
        header_bytes: usize,
    },
}
