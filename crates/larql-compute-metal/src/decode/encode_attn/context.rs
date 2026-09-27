//! Per-layer scalars every attention stage reads.
//!
//! Derived once, before the first dispatch, from the layer's structured
//! views, the backend's cached decode flags and the KV cache's counters.
//! Every field is a pure read — building the context encodes nothing and
//! touches no buffer cache, so the stage files see exactly the values the
//! single-function block used to hold in locals.

use super::{AttnBufs, AttnDims};
use crate::ops;
use crate::ops::kv_cache::MAX_HEAD_DIM_SINGLE_SG;
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;

pub(super) struct AttnCtx<'l, 'w, 'b> {
    pub layer: &'l FullPipelineLayer<'w>,
    pub bufs: AttnBufs<'b>,
    pub layer_idx: usize,
    pub hidden: usize,
    pub hidden_val: u32,
    pub layer_q_dim: usize,
    pub ffn_uses_kquant: bool,
    pub q_norm_enabled: bool,
    pub norm_offset: f32,
    pub qk_norm_offset: f32,
    pub eps: f32,
    pub scale: f32,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// Rotary width: the declared `rotary_dim`, or the full head when 0.
    pub rotary_dim: usize,
    /// Narrower of the architecture's per-layer SWA and any window the
    /// engine imposed on this sequence.
    pub window_size: u32,
    pub kv_shared_source: Option<usize>,
    /// Cache the attend reads: the shared source, or this layer's own.
    pub attend_cache_idx: usize,
    /// ABSOLUTE stream position to RoPE at.
    pub pos: u32,
    /// Rows the attend actually spans after the window clamp.
    pub attn_span: u32,
    pub use_fused_qkn_rope: bool,
    pub use_fused_kv_aa: bool,
    pub use_fused_post_attn: bool,
}

impl<'l, 'w, 'b> AttnCtx<'l, 'w, 'b> {
    pub(super) fn new(
        backend: &MetalBackend,
        layer: &'l FullPipelineLayer<'w>,
        kv_cache: &ops::kv_cache::KVCache,
        layer_idx: usize,
        bufs: AttnBufs<'b>,
        dims: AttnDims,
    ) -> Self {
        let AttnDims {
            hidden,
            layer_q_dim,
            ffn_uses_kquant,
        } = dims;
        // M2: extract per-layer params via the structured views.
        // `attn_spec` carries the attention shape (head_dim, head counts,
        // RoPE, sliding_window, qk_norm flags); `norms` carries the eps
        // and offsets. The compiler optimises these getter methods to
        // plain field reads, so this is a clarity win — the function
        // signature can later narrow to `(AttentionSpec, LayerNorms,
        // ...)` once vindex/inference callers stop passing a flat layer.
        let attn_spec = layer.attention_spec();
        let norms_view = layer.norms();
        let head_dim = attn_spec.head_dim;
        let rotary_dim = if attn_spec.rotary_dim > 0 {
            attn_spec.rotary_dim
        } else {
            head_dim
        };
        // Narrower of the architecture's per-layer SWA and any window the
        // engine imposed on this sequence. The kernel attends
        // `[T - window_size, T)`, so this both bounds attention and lets
        // the cache hold more rows than the window between compactions.
        let window_size = backend.effective_window_for(attn_spec.sliding_window as u32);

        // KV-cache sharing (Gemma 4 E2B): later layers reuse K/V from a
        // "source" layer that already ran this token's attention. Shared
        // layers compute their own Q (against their own residual + W_q)
        // but skip K/V append + read from the source's cache. The fused
        // attention paths write the layer's own cache, so they're disabled
        // for shared layers — the unfused encode_kv_attend branch
        // handles the source-cache binding.
        //
        // Shared layers' position counter is pinned to the source's
        // (`source.current_len` is the count of stored positions
        // including the new token, since source has already finished its
        // block for this token by the time we reach the shared layer).
        let kv_shared_source = layer.kv_shared_source;
        let attend_cache_idx = kv_shared_source.unwrap_or(layer_idx);
        // `pos` is the ABSOLUTE stream position to RoPE at; `t_val` is how
        // many cached rows to attend. They are equal-and-offset only while
        // nothing has been evicted — once a window slides, occupancy falls
        // and position keeps climbing, so they must be read from different
        // fields. Deriving `t_val` from `pos` (as `pos + 1`) is what tied
        // them together before.
        let pos = if let Some(src) = kv_shared_source {
            // Source has already advanced past the row it just wrote;
            // RoPE at that row's position.
            (kv_cache.layers[src].abs_position.saturating_sub(1)) as u32
        } else {
            kv_cache.layers[layer_idx].abs_position as u32
        };
        let t_val = if kv_shared_source.is_some() {
            // Source's current_len already counts this token.
            kv_cache.layers[attend_cache_idx].current_len as u32
        } else {
            (kv_cache.layers[layer_idx].current_len + 1) as u32
        };
        let attn_span = ops::kv_cache::attention_span(t_val, window_size);

        // Env flags governing kernel-level fusion. Cached at backend
        // startup (see `metal::flags::DecodeFlags`) so the decode hot
        // path doesn't pay 4× `getenv` per layer.  Defaults preserve
        // the proven-win May-2026 fusion wave; opts-out are diagnostic
        // only.
        //
        // kv_append_attend_fused uses a fixed tg_scores[SHORT_ATTENTION_SPAN]
        // threadgroup array. Spans beyond that overflow it — global-attention
        // layers (window_size=0) grow unboundedly and must fall back to
        // encode_kv_attend, which auto-selects kv_attention_long past the threshold.
        //
        // Additionally, the kernel is designed for head_dim <= MAX_HEAD_DIM_SINGLE_SG
        // (it dispatches exactly head_dim threads per group and assumes head_dim fits
        // in a single simdgroup). Layers with larger head_dim must use the unfused
        // encode_kv_append + encode_kv_attend path which handles arbitrary head_dim.
        let use_fused_kv_aa = attn_span <= ops::kv_cache::SHORT_ATTENTION_SPAN
            && head_dim <= MAX_HEAD_DIM_SINGLE_SG
            && backend.decode_flags.fused_kv_append_attend
            // Shared layers don't append to their own cache; the fused
            // append+attend kernel writes its argument cache, so it can't
            // be reused on the source's cache without corrupting the
            // source. Fall through to the unfused attend-only branch.
            && kv_shared_source.is_none();

        Self {
            layer,
            bufs,
            layer_idx,
            hidden,
            hidden_val: hidden as u32,
            layer_q_dim,
            ffn_uses_kquant,
            q_norm_enabled: attn_spec.q_norm_enabled,
            norm_offset: norms_view.norm_offset,
            qk_norm_offset: norms_view.qk_norm_offset,
            eps: norms_view.eps,
            scale: attn_spec.attn_scale,
            head_dim,
            num_q_heads: attn_spec.num_q_heads,
            num_kv_heads: attn_spec.num_kv_heads,
            rotary_dim,
            window_size,
            kv_shared_source,
            attend_cache_idx,
            pos,
            attn_span,
            use_fused_qkn_rope: backend.decode_flags.fused_qk_norm_rope,
            use_fused_kv_aa,
            use_fused_post_attn: backend.decode_flags.fused_post_attn_norm,
        }
    }
}
