//! The decode token loop: one token through every layer, in one command
//! buffer wherever the layer's shape allows it.
//!
//! This is the implementation every entry point in `entry.rs` reaches, and
//! the place the per-token command-buffer count is decided. The stages a
//! layer encodes live in the `encode_*` modules beside it; what stays here
//! is the sequencing between them, the MoE fire/collect split, and the
//! single commit + wait at the bottom that TOKEN-B1 rung 2's fused head
//! rides (see `head.rs`).
//!
//! | file         | responsibility                                            |
//! |--------------|-----------------------------------------------------------|
//! | `run.rs`     | `decode_token_with_moe_split_fn` — setup and layer loop   |
//! | `cmd.rs`     | the live command buffer / encoder pair (`TokenCmd`)       |
//! | `ctx.rs`     | per-token read-only inputs shared by the stages           |
//! | `staging.rs` | state-dump staging buffers: allocate, blit, drain         |
//! | `attn.rs`    | Steps 1–5: input norm + QKV, then the attention block     |
//! | `ffn.rs`     | Steps 6–8: dense FFN, post-FFN residual, PLE, MoE tail    |
//! | `hooks.rs`   | env-gated per-layer diagnostics (NaN, dumps, early exit)  |
//! | `finish.rs`  | fused head, final commit + wait, drains, timing records   |

mod attn;
mod cmd;
mod ctx;
mod ffn;
mod finish;
mod hooks;
mod run;
mod staging;
