//! The instruction format a family's chat checkpoints were tuned on.
//!
//! A fallback for when no checkpoint template (`chat_template.jinja` /
//! `tokenizer_config.json`) is available. A family that declares none has
//! no single agreed format — DeepSeek and gpt-oss (Harmony) are examples —
//! and must be rendered from its own template, never guessed.

/// A family's canonical chat turn format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    /// `<start_of_turn>user … <end_of_turn>` (Gemma 1–4).
    GemmaTurns,
    /// `[INST] … [/INST]` (Mistral, Mixtral).
    MistralInst,
    /// `<|start_header_id|>…<|end_header_id|>` (Llama 3).
    Llama3Headers,
    /// `<|im_start|>… <|im_end|>` (Qwen).
    ChatMl,
}
