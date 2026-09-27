//! Position, weights, scratch and shape for one lowered attention op.

use super::super::{LoweredMatrix, PostNorm};
use metal::Buffer;

#[allow(unused_imports)]
use super::*;

/// Position encoding for this layer, from its own policy entry — never a
/// model-wide default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoweredPosition {
    /// Rotary at this base, unit amplitude.
    Rope { theta: f64 },
    /// Rotary at frequencies the caller's `inv_freq` table already
    /// carries (YaRN's ramped blend), with this amplitude on `cos`/`sin`
    /// — the part of YaRN that rescales every logit at every position.
    Scaled { theta: f64, amplitude: f32 },
    /// The layer attends position-agnostically (NoPE).
    None,
}

impl LoweredPosition {
    /// The `cos`/`sin` scalar this policy applies; `None` for a layer
    /// that does not rotate.
    pub(super) fn amplitude(self) -> Option<f32> {
        match self {
            Self::Rope { .. } => Some(1.0),
            Self::Scaled { amplitude, .. } => Some(amplitude),
            Self::None => None,
        }
    }
}

/// Everything attention reads.
pub struct AttnWeights<'a> {
    pub q: LoweredMatrix<'a>,
    pub k: LoweredMatrix<'a>,
    pub v: LoweredMatrix<'a>,
    pub o: LoweredMatrix<'a>,
    /// The judged attention output gate. `None` = no gate op — which is
    /// a different claim from a gate that happens to be near 1.
    pub gate: Option<LoweredMatrix<'a>>,
    /// Q/K/V/O projection biases (f32), from the plan's operands: added
    /// right after each projection — before QK-norm/RoPE for Q and K,
    /// before the cache holds V, after `o` before the residual. `None`
    /// = the plan carries no bias for that projection.
    pub q_bias: Option<&'a Buffer>,
    pub k_bias: Option<&'a Buffer>,
    pub v_bias: Option<&'a Buffer>,
    pub o_bias: Option<&'a Buffer>,
    /// Per-query-head attention-sink logits (f32), when the plan's
    /// judged sink semantics apply; `None` = ordinary softmax.
    pub sinks: Option<&'a Buffer>,
    /// Weighted per-head Q/K norms (Gemma `q_norm` / `k_norm`), applied
    /// after the projections and before the query scale and rotation,
    /// with the plan's weight offset. `None` = the op is absent.
    pub qk_norm: Option<QkNormWeights<'a>>,
    /// Pre-attention norm weight (f32).
    pub norm_weight: &'a Buffer,
    /// The post-attention norm, under four-norm placement. `None` =
    /// absent.
    pub post_norm: Option<PostNorm<'a>>,
}

/// Caller-owned device scratch and cache.
pub struct AttnScratch<'a> {
    /// `hidden` floats.
    pub normed: &'a Buffer,
    /// `num_q_heads * head_dim` floats.
    pub q: &'a Buffer,
    /// `[T, num_kv_heads, head_dim]` — K and V caches, written in place.
    pub k_cache: &'a Buffer,
    pub v_cache: &'a Buffer,
    /// `num_q_heads * head_dim` floats each.
    pub gate: &'a Buffer,
    pub concat: &'a Buffer,
    pub gated: &'a Buffer,
    /// `hidden` floats — o_proj output, before the residual.
    pub attn_out: &'a Buffer,
    /// `head_dim / 2` floats, host-computed to match the interpreter's
    /// `theta^(-2i/head_dim)`.
    pub inv_freq: &'a Buffer,
    /// SPLITK-1 partials. `None` = split-K is never dispatched and the
    /// planner's seqpar/serial choice runs unchanged.
    pub splitk: Option<&'a crate::ops::kv_splitk::SplitKScratch<'a>>,
}

/// The weighted QK-norm operands: one `[head_dim]` vector each.
pub struct QkNormWeights<'a> {
    pub q: &'a Buffer,
    pub k: &'a Buffer,
    /// Centred-norm convention (`1 + w`); a plan fact.
    pub weight_offset: f32,
}

/// Geometry and judged semantics, straight off the plan.
#[derive(Clone)]
pub struct AttnShape {
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub norm_weight_offset: f32,
    pub qk_norm_eps: f32,
    pub parameter_free_q: bool,
    pub parameter_free_k: bool,
    /// Parameter-free per-head RMS norm on V (Gemma 4 `v_norm`), applied
    /// to the raw value projection in its cache slot — before anything
    /// reads it, and on a K≡V layer before the key's own norm/rotation
    /// touch the separately-projected K.
    pub parameter_free_v: bool,
    /// `None` = the op is absent, not a multiply by one.
    pub query_scale: Option<f32>,
    /// The canonical score-time multiply, kept separate from
    /// `query_scale` because folding them is algebra-equivalent and not
    /// fp-equivalent.
    pub score_scale: f32,
    pub position: LoweredPosition,
    /// Sliding window; `None` = attends the whole prefix.
    pub window: Option<usize>,
    /// `None` = the softcap op is absent.
    pub softcap: Option<f32>,
    /// The plan's residual-scale op (`LayerPlan::residual_scale`): the
    /// branch output (after any post-norm) is multiplied by this before
    /// its residual add. `None` = the op is absent, not a multiply by one.
    pub residual_scale: Option<f32>,
    /// Absolute position of the token being decoded.
    pub position_index: usize,
    /// Cache length **including** this position.
    pub kv_len: usize,
}

impl AttnShape {
    pub(super) fn q_rows(&self) -> usize {
        self.num_q_heads * self.head_dim
    }
    pub(super) fn kv_rows(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// Byte offset of this position's slot in a `[T, num_kv, head_dim]`
    /// cache.
    pub(super) fn kv_slot_offset(&self) -> u64 {
        (self.position_index * self.kv_rows() * std::mem::size_of::<f32>()) as u64
    }
}
