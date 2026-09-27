//! The instruction format a family's chat checkpoints were tuned on.
//!
//! A fallback for when no checkpoint template (`chat_template.jinja` /
//! `tokenizer_config.json`) is available. A family that declares none has
//! no single agreed format — DeepSeek and gpt-oss (Harmony) are examples —
//! and must be rendered from its own template, never guessed.

/// A family's canonical chat turn format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    /// `<start_of_turn>user … <end_of_turn>` (Gemma 1–3).
    GemmaTurns,
    /// `<|turn>user … <turn|>` (Gemma 4).
    ///
    /// The turn syntax is common to every Gemma 4 checkpoint; how the model
    /// turn opens is not. The 12B, 26B-A4B and 31B templates open it with an
    /// empty thought channel (`<|channel>thought\n<channel|>`) when thinking
    /// is off, and E2B does not. That is a checkpoint fact, read from the
    /// checkpoint's own template when one exists; this fallback renders the
    /// common syntax only.
    Gemma4Turns,
    /// `[INST] … [/INST]` (Mistral, Mixtral).
    MistralInst,
    /// `<|start_header_id|>…<|end_header_id|>` (Llama 3).
    Llama3Headers,
    /// `<|im_start|>… <|im_end|>` (Qwen).
    ChatMl,
}

/// The system prompt a Gemma 4 turn renderer adds when the caller gives
/// none.
const GEMMA4_DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer questions concisely.";

/// Gemma 4's common turn syntax as a Jinja template, for checkpoints whose
/// extract predates snapshotting `chat_template.jinja` (older 31B dense
/// extracts). System and user turns plus the generation prompt; no tools,
/// no multimodal, and no thought-channel tail (see [`ChatFormat::Gemma4Turns`]).
const GEMMA4_FALLBACK_TEMPLATE: &str = "{{- bos_token -}}\
{%- if messages[0]['role'] in ['system', 'developer'] -%}\
{{- '<|turn>system\n' -}}{{- messages[0]['content'] | trim -}}{{- '<turn|>\n' -}}\
{%- set loop_messages = messages[1:] -%}\
{%- else -%}\
{%- set loop_messages = messages -%}\
{%- endif -%}\
{%- for message in loop_messages -%}\
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}\
{{- '<|turn>' + role + '\n' -}}\
{%- if message['content'] is string -%}{{- message['content'] | trim -}}{%- endif -%}\
{{- '<turn|>\n' -}}\
{%- endfor -%}\
{%- if add_generation_prompt -%}{{- '<|turn>model\n' -}}{%- endif -%}";

impl ChatFormat {
    /// The system prompt to add when the caller gives none, for formats
    /// whose checkpoints are tuned to expect one.
    pub fn default_system_prompt(self) -> Option<&'static str> {
        match self {
            Self::Gemma4Turns => Some(GEMMA4_DEFAULT_SYSTEM_PROMPT),
            Self::GemmaTurns | Self::MistralInst | Self::Llama3Headers | Self::ChatMl => None,
        }
    }

    /// A Jinja chat template for this format, used only when the checkpoint
    /// ships none. `None` where the format has no template fallback here.
    pub fn fallback_template(self) -> Option<&'static str> {
        match self {
            Self::Gemma4Turns => Some(GEMMA4_FALLBACK_TEMPLATE),
            Self::GemmaTurns | Self::MistralInst | Self::Llama3Headers | Self::ChatMl => None,
        }
    }
}
