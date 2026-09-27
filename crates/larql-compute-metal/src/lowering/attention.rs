//! Lowering a plan's attention op into one encoder (VINDEX3-G6b-3).
//!
//! The delicate fragment. Unlike the FFN, attention is an **ordered**
//! program whose steps approximately commute, so a lowering can contain
//! every operation, produce plausible numbers, and still represent a
//! different model. The order below is the interpreter's
//! `condition_qk_in_place`, transcribed rather than reconstructed:
//!
//! ```text
//! h ─ pre-attn norm ─┬─ Q proj ─ param-free QK norm ─ query scale ─ RoPE ─┐
//!                    ├─ K proj ─ param-free QK norm ─────────────  RoPE ──┤ (into KV cache)
//!                    ├─ V proj ───────────────────────────────────────────┤ (into KV cache)
//!                    └─ gate proj ────────────────────────┐               │
//!                                                          │      attention
//!                                                          │          │
//!                                            sigmoid gate ─┴──────────┘
//!                                                          │
//!                                            o_proj ─ post_attn_norm ─ residual ─ h'
//! ```
//!
//! **Query scale applies to Q only, after QK norm and before RoPE.** All
//! three touch Q, and swapping any pair changes the model while leaving
//! magnitudes plausible — the parity test carries an explicit ordering
//! control for exactly this.
//!
//! K and V project **directly into their KV-cache slots** rather than
//! into scratch that is later copied: the cache is `[T, num_kv,
//! head_dim]` position-major, so the current position's slot is a plain
//! byte offset, and the in-place QK norm and RoPE then operate on the
//! cache through the same offset. Removing the copy also removes the
//! chance of the cached K diverging from the K that was normed.

mod rows;
mod single;
mod types;
pub use types::*;
