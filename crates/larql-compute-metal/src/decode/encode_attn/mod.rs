//! Per-layer attention block — Steps 1.5 through 5 of the decode loop.
//!
//! Inputs (already populated by `encode_input_norm_and_qkv`):
//! - `q_out`, `k_out`, `v_out`: raw Q/K/V projections (pre-norm, pre-RoPE).
//! - `h_buf`: layer-input residual.
//!
//! Outputs:
//! - `ffn_norm_out`: RMS-normed `h_buf + o_out` (FFN gate/up input).
//! - `h_post_attn`: raw `h_buf + o_out` (post-FFN residual base).
//! - `kv_cache.layers[l].current_len += 1` (the new token's K/V row is appended).
//!
//! Path selection (env-gated, defaults preserve the proven-win May-2026 fusion wave):
//! - `LARQL_FUSED_ATTN=1` (opt-in) — single `attn_fused` kernel covers QK-norm +
//!   RoPE + KV append + attend. Currently regresses on Gemma 3 4B (parallelism
//!   collapse 12 TGs → 8); kept registered for the multi-TG-per-head retry.
//! - `LARQL_FUSED_QK_NORM_ROPE=0` — opt out of the fused QK-norm + RoPE path.
//! - `LARQL_FUSED_KV_APPEND_ATTEND=0` — opt out of the fused KV append + attend.
//! - `LARQL_FUSED_POST_ATTN_NORM=0` — opt out of the triple-fused
//!   `post_attn_norm + residual + ffn_norm + store`.
//!
//! The block is split into one file per stage, each encoding into the same
//! compute encoder in the order `block.rs` sequences them:
//!
//! | file          | stage                                                   |
//! |---------------|---------------------------------------------------------|
//! | `gate.rs`     | the `attn_fused_will_fire` authority                    |
//! | `context.rs`  | per-layer scalars shared by every stage (`AttnCtx`)     |
//! | `block.rs`    | `encode_attention_block` — the stage sequence           |
//! | `fused.rs`    | Path 1: the single-dispatch `attn_fused` kernel         |
//! | `qk_rope.rs`  | Steps 1.5 + 2: QK-norm + RoPE (fused or unfused)        |
//! | `v_norm.rs`   | Step 3: optional V-norm                                 |
//! | `kv_attend.rs`| Step 4: KV append + attend (fused / shared / unfused)   |
//! | `o_proj.rs`   | Step 5a: O projection + optional O bias                 |
//! | `residual.rs` | Step 5b: residual + post-attn norm + ffn-input norm     |

use metal::Buffer;

mod block;
mod context;
mod fused;
mod gate;
mod kv_attend;
mod o_proj;
mod qk_rope;
mod residual;
mod v_norm;

/// First of the two consecutive `attn_fused` slots carrying attention
/// sinks; the `has_sinks` flag follows in slot 19. See `stages::sinks`.
const ATTN_FUSED_SINKS_INDEX: u64 = 18;

/// `inv_freq` slot on `attn_fused` — took over the old `rope_base` index in
/// place, so no other binding moved.
const ATTN_FUSED_INV_FREQ_INDEX: u64 = 16;

/// `amplitude` slot on `attn_fused`, appended after the sinks pair.
const ATTN_FUSED_AMPLITUDE_INDEX: u64 = 20;

/// `softcap` slot on `attn_fused` (0.0 = disabled).
const ATTN_FUSED_SOFTCAP_INDEX: u64 = 22;

/// `has_qk_norm` slot on `attn_fused` — 0 lets no-QK-norm archs
/// (GPT-OSS) take the single-dispatch path with Q/K passed through raw.
const ATTN_FUSED_HAS_QK_NORM_INDEX: u64 = 23;

/// First of the four consecutive `attn_fused` slots carrying the Q/K/V
/// projection biases: q_bias(24), k_bias(25), v_bias(26), then the
/// `has_qkv_bias` flag (27). Sinks convention — stub buffers bound when
/// absent, flag gates the read.
const ATTN_FUSED_QKV_BIAS_INDEX: u64 = 24;

/// `softcap` slot on `kv_append_attend_fused` (0.0 = disabled).
const KV_APPEND_ATTEND_SOFTCAP_INDEX: u64 = 14;

/// Absolute stream position for RoPE on `attn_fused`. The kernel's cache
/// row stays occupancy-indexed; only the rotation angle uses this. The
/// two diverge once a sliding window compacts (audit F11).
const ATTN_FUSED_ABS_POS_INDEX: u64 = 21;

/// First of the two consecutive `kv_append_attend_fused` slots carrying
/// attention sinks; the `has_sinks` flag follows in slot 13.
const KV_APPEND_ATTEND_SINKS_INDEX: u64 = 12;

pub(super) struct AttnBufs<'a> {
    /// Layer-input residual (read).
    pub h_buf: &'a Buffer,
    pub q_out: &'a Buffer,
    pub k_out: &'a Buffer,
    pub v_out: &'a Buffer,
    pub attn_out_buf: &'a Buffer,
    pub o_out_buf: &'a Buffer,
    /// FFN gate/up input (written).
    pub ffn_norm_out: &'a Buffer,
    /// Post-FFN residual base (written).
    pub h_post_attn: &'a Buffer,
    /// Scratch for Q8 quantize on the legacy O-proj path.
    pub o_q8_scratch: &'a Buffer,
    pub o_q8s_scratch: &'a Buffer,
    /// Scratch for the Q8-input residual+norm path.
    pub ffn_q8: &'a Buffer,
    pub ffn_q8s: &'a Buffer,
    /// Scratch for the unfused post-attn norm chain.
    pub normed_scratch: &'a Buffer,
    pub wo: &'a Buffer,
    pub wo_scales: Option<&'a Buffer>,
    pub post_attn_norm: &'a Buffer,
}

pub(super) struct AttnDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    /// True iff the FFN side will run Q4_K family (selects the fused
    /// `residual_norm_store` path that mirrors the FFN's input dtype).
    pub ffn_uses_kquant: bool,
}
