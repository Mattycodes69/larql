//! GGUF format constants — magic number, type IDs, key names, replacement table.

pub(crate) const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian

// Metadata value types
pub(crate) const GGUF_TYPE_UINT8: u32 = 0;
pub(crate) const GGUF_TYPE_INT8: u32 = 1;
pub(crate) const GGUF_TYPE_UINT16: u32 = 2;
pub(crate) const GGUF_TYPE_INT16: u32 = 3;
pub(crate) const GGUF_TYPE_UINT32: u32 = 4;
pub(crate) const GGUF_TYPE_INT32: u32 = 5;
pub(crate) const GGUF_TYPE_FLOAT32: u32 = 6;
pub(crate) const GGUF_TYPE_BOOL: u32 = 7;
pub(crate) const GGUF_TYPE_STRING: u32 = 8;
pub(crate) const GGUF_TYPE_ARRAY: u32 = 9;
pub(crate) const GGUF_TYPE_UINT64: u32 = 10;
pub(crate) const GGUF_TYPE_INT64: u32 = 11;
pub(crate) const GGUF_TYPE_FLOAT64: u32 = 12;

pub(crate) const GGUF_GENERAL_ARCHITECTURE: &str = "general.architecture";
/// Data-section alignment key, and the value GGUF specifies when it is absent.
pub(crate) const GGUF_GENERAL_ALIGNMENT: &str = "general.alignment";
pub(crate) const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
pub(crate) const GGUF_EMBEDDING_LENGTH: &str = "embedding_length";
pub(crate) const GGUF_BLOCK_COUNT: &str = "block_count";
pub(crate) const GGUF_FEED_FORWARD_LENGTH: &str = "feed_forward_length";
// MoE-only architectures (DeepSeek-V4 family) omit the global
// `feed_forward_length` and emit only the per-expert size — fall back
// to it so config validation doesn't reject the model with
// `intermediate_size: must be greater than 0`.
pub(crate) const GGUF_EXPERT_FEED_FORWARD_LENGTH: &str = "expert_feed_forward_length";
pub(crate) const GGUF_ATTENTION_HEAD_COUNT: &str = "attention.head_count";
pub(crate) const GGUF_ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";
pub(crate) const GGUF_ATTENTION_KEY_LENGTH: &str = "attention.key_length";
pub(crate) const GGUF_ROPE_FREQ_BASE: &str = "rope.freq_base";
// MLA-specific metadata keys emitted by llama.cpp for DeepSeek-V2/V3/Kimi-K2
// family models. `_mla` variants carry the pre-absorption per-head dims;
// non-`_mla` variants carry the (possibly larger) absorbed/effective sizes.
// `rope.dimension_count` is the RoPE-positional portion of each Q/K head
// (qk_rope_head_dim in the HF config).
pub(crate) const GGUF_ATTENTION_KEY_LENGTH_MLA: &str = "attention.key_length_mla";
pub(crate) const GGUF_ATTENTION_VALUE_LENGTH: &str = "attention.value_length";
pub(crate) const GGUF_ATTENTION_VALUE_LENGTH_MLA: &str = "attention.value_length_mla";
pub(crate) const GGUF_ATTENTION_Q_LORA_RANK: &str = "attention.q_lora_rank";
pub(crate) const GGUF_ATTENTION_KV_LORA_RANK: &str = "attention.kv_lora_rank";
pub(crate) const GGUF_ROPE_DIMENSION_COUNT: &str = "rope.dimension_count";
pub(crate) const GGUF_VOCAB_SIZE: &str = "vocab_size";

pub(crate) const HF_MODEL_TYPE: &str = "model_type";
pub(crate) const HF_HIDDEN_SIZE: &str = "hidden_size";
pub(crate) const HF_NUM_HIDDEN_LAYERS: &str = "num_hidden_layers";
pub(crate) const HF_INTERMEDIATE_SIZE: &str = "intermediate_size";
pub(crate) const HF_NUM_ATTENTION_HEADS: &str = "num_attention_heads";
pub(crate) const HF_NUM_KEY_VALUE_HEADS: &str = "num_key_value_heads";
pub(crate) const HF_HEAD_DIM: &str = "head_dim";
pub(crate) const HF_ROPE_THETA: &str = "rope_theta";
pub(crate) const HF_VOCAB_SIZE: &str = "vocab_size";

pub(crate) const TOKENIZER_JSON: &str = "tokenizer.json";
pub(crate) const TOKENIZER_MODEL: &str = "model";
pub(crate) const TOKENIZER_VOCAB: &str = "vocab";

pub(crate) const GGUF_OUTPUT_WEIGHT: &str = "output.weight";

pub(crate) const GGUF_TO_HF_KEY_REPLACEMENTS: &[(&str, &str)] = &[
    ("blk.", "layers."),
    ("attn_qkv.", "self_attn.qkv_proj."),
    ("attn_q.", "self_attn.q_proj."),
    ("attn_k.", "self_attn.k_proj."),
    ("attn_v.", "self_attn.v_proj."),
    ("attn_output.", "self_attn.o_proj."),
    ("ffn_gate.", "mlp.gate_proj."),
    ("ffn_up.", "mlp.up_proj."),
    ("ffn_down.", "mlp.down_proj."),
    ("attn_norm.", "input_layernorm."),
    ("ffn_norm.", "post_attention_layernorm."),
    ("token_embd.", "embed_tokens."),
    ("position_embd.", "wpe."),
    ("output_norm.", "norm."),
    ("output.", "lm_head."),
];

// Tensor type constants moved to format::quant::ggml
