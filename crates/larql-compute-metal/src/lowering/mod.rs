//! Encoder-level primitives for lowering a VINDEX3 `ComponentOpPlan`.
//!
//! **VINDEX3-G6.** The interpreter path in `larql-vindex` returns to the
//! host between every matrix operation, so it commits and waits 209 times
//! per Glimmer token. Measured consequence: the command queue is empty
//! for 215-271 us before each dispatch begins, flat across a 50x range of
//! weight bytes, and a queue-depth A/B collapses per-dispatch cost from
//! 408 us at depth 1 to 57 us at depth 32. That is queue starvation, the
//! same defect `tests/test_cb_queue_starvation.rs` convicted in the
//! serving decoder — which answered it by encoding a whole token into one
//! command buffer with the elementwise glue on the GPU.
//!
//! The functions here are the pieces that let VINDEX3 adopt that shape
//! **without** calling the serving decoder as a black box. Each one
//! *encodes* into a caller-supplied encoder and touches device buffers
//! only: no command buffer, no commit, no wait, no readback. Scheduling
//! becomes the caller's decision, which is the whole point — VINDEX3's
//! plan stays the authority on *what* happens, and lowering owns *how* it
//! is scheduled.
//!
//! The serving path's own encoders (`decode::encode_qkv`, `encode_attn`,
//! `encode_ffn`) are deliberately not reused as-is: they are keyed to
//! `FullPipelineLayer` and to a `QuantFormat` enum that has no NVFP4
//! variant, so adapting a plan into them would be the bypass this rung
//! exists to avoid. The intended end state is that both frontends share
//! primitives at *this* level.

pub mod attention;
pub mod ffn;
pub mod head;
pub mod nvfp4;
pub mod profile;
pub mod stack;

pub use nvfp4::{
    nvfp4_fusion_enabled, nvfp4_kernel_choice, nvfp4_residual_fusion_enabled, nvfp4_segment,
    NormOutput, Nvfp4Kernel, Nvfp4Segment, PreNorm, NVFP4_FUSE_ENV, NVFP4_KERNEL_ENV,
    NVFP4_MAX_SEGMENTS, RMS_NORM_MAX_OUTPUTS,
};

/// A device buffer, re-exported so callers can hold lowering state
/// without linking `metal` themselves. The CLI is the only place a plan
/// and a device meet, and it should not need the graphics API in its
/// dependency list to say "this is resident".
pub use metal::Buffer as DeviceBuffer;
/// A command buffer, re-exported for the same reason: a caller holding
/// an encoded-but-uncommitted token should not need to link `metal`.
pub use metal::CommandBuffer as DeviceCommandBuffer;

mod elementwise;
mod matmul;
mod matvec;
mod operands;
pub use elementwise::*;
pub use operands::*;
