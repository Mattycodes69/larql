//! Model-architecture config carried in `index.json` so the
//! architecture can be reconstructed without the original
//! `config.json`.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup.
//!
//! ## This struct is a lossy projection, and the loss is load-bearing
//!
//! Every field a checkpoint declares that reaches the forward pass has
//! to appear here or the served model silently differs from the
//! checkpoint. That is not hypothetical: `rope_scaling` was absent
//! until 2026-08-06, so `gemma-3-4b-it` — whose `config.json` says
//! `{"factor": 8.0, "rope_type": "linear"}` — was served with a
//! position divisor of 1.0 on its five global layers instead of 8.0.
//! CPU and Metal read the same `index.json`, so both were wrong in the
//! same way and the CPU-vs-Metal parity suite stayed green. A parity
//! gate cannot see a defect in the config both of its arms share.
//!
//! `model_config_persists_every_forward_affecting_field` pins the
//! inventory. When you add a field to `larql_models::ModelConfig`, that
//! test tells you to either persist it here or record why it does not
//! need persisting. `embedding_multiplier` is the standing example of
//! the second case: it round-trips through the top-level
//! `VindexConfig.embed_scale` instead, and duplicating it here would
//! create a second source of truth for one number.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// The window's explicit enable flag, persisted separately from the
    /// window itself because they are separate declarations: Qwen2.5
    /// ships a 32768 window beside `use_sliding_window: false`. Dropping
    /// the flag on the way into a vindex would serve the window against
    /// the checkpoint's instruction.
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    /// How far up the stack an enabled window applies.
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    /// The declared positional scheme, verbatim.
    ///
    /// Persisted because on `granitemoehybrid` it is what turns rotation
    /// on at all — a container that dropped it would rebuild the model as
    /// a NoPE model, or as a rotating one, depending only on which side of
    /// the boundary the question was asked.
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    /// The per-layer rotary schedule, in the checkpoint's own polarity:
    /// `1` rotates, `0` is NoPE.
    ///
    /// Persisted because it decides which layers encode position at all.
    /// Dropping it would rebuild SmolLM3 as a model that rotates
    /// everywhere — fluent, and wrong on 9 of 36 layers.
    #[serde(default)]
    pub no_rope_layers: Option<Vec<i64>>,
    /// The interval fallback, persisted so a container that carried no
    /// explicit mask still knows the schedule it was compiled under.
    #[serde(default)]
    pub no_rope_layer_interval: Option<usize>,
    /// The declared rotary pairing, persisted so the container records
    /// what the checkpoint claimed rather than what this build does.
    #[serde(default)]
    pub rope_interleaved: Option<bool>,
    /// The declared multi-axis flag, persisted for the same reason.
    #[serde(default)]
    pub use_mrope: Option<bool>,
    /// Falcon's one-word FFN shape (`activation`), persisted so the
    /// container records the claim the boundary checked.
    #[serde(default)]
    pub ffn_shape_name: Option<String>,
    /// The declared `is_llama_config` flag, persisted for the same reason.
    #[serde(default)]
    pub is_llama_config: Option<bool>,
    /// MoE configuration (None for dense models).
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // ── Gemma 4 per-layer attention geometry ──
    // All optional for backward compatibility with existing vindexes.
    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    /// Gemma 4: 512 for global layers, head_dim (256) for sliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim to apply RoPE to (0.0–1.0). If None, full rotation.
    /// Gemma 4 global layers: 0.25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// Gemma 4: 6 (layers 5, 11, 17, ... are full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V).
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Number of layers at the end that share KV from earlier layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Per-layer embedding dimension (PLE). 0 or None = no PLE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_embed_dim: Option<usize>,
    /// Gemma 3n/4-E: double-wide MLP on the KV-shared layers, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_double_wide_mlp: Option<bool>,
    /// Gemma 3n/4-E: the per-layer-input embedding vocabulary, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size_per_layer_input: Option<u64>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Per-layer declared rope theta, verbatim from `layer_rope_theta` —
    /// `0.0` entries are the upstream NoPE sentinel, interpreted only by
    /// `ModelArchitecture::position_policy_for_layer` after the round-trip.
    /// Dropping this served every NoPE layer with full-strength rotation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_rope_theta: Option<Vec<f64>>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Extra attention-score multiplier on top of 1/sqrt(head_dim)
    /// (`qk_scale_factor`). Distinct from `query_pre_attn_scalar`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_scale_factor: Option<f64>,
    /// Multiplier on the final hidden state before the vocab projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_multiplier: Option<f64>,
    /// Post-norm epsilon when it differs from `norm_eps` (1e-8 vs 1e-5 on
    /// the same checkpoint is a real shape).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_norm_eps: Option<f64>,
    /// Whether attention projections carry biases, when declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    /// Whether Q/K/V (not the output) carry biases, when declared
    /// (`qkv_bias`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qkv_bias: Option<bool>,
    /// FFN activation name, verbatim (`hidden_act`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_act: Option<String>,
    /// SiTU-GLU's two softcaps (`activation_situ_beta`,
    /// `activation_situ_linear_beta`), verbatim.
    ///
    /// They are parameters of the combine `hidden_act: "situ"` names, and
    /// a container that carried the name without them would rebuild the
    /// FFN at the reference's `beta or 1.0` fallback — a different
    /// function, with every shape still closing and no parity arm able to
    /// see it, since both arms would read the same index.json.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_situ_beta: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation_situ_linear_beta: Option<f64>,
    /// One FFN intermediate width per layer (`larql_ffn_intermediate_size_by_layer`),
    /// verbatim, for checkpoints whose gate/up/down projections were sliced
    /// to different widths in different layers. A vindex that dropped it
    /// would rebuild every layer at the dense width and refuse its own tensors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_intermediate_size_by_layer: Option<Vec<usize>>,
    /// Declared context bound (`max_position_embeddings`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<usize>,
    /// Final-logit tanh softcap (Gemma 2/3/4: 30.0). Applied to logits
    /// immediately before softmax in `logits_to_predictions`. Omitting it
    /// leaves logits uncapped — on E2B this peaked the softmax on the
    /// wrong token (observed: "Paris" → "hyperparameters").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,

    // ── Granite-family scaling multipliers ──
    // None on every other arch. Captured at vindex-build time so the
    // reconstructed `ModelArchitecture` knows about them at load time;
    // without these the vindex Metal forward path silently runs with
    // all three at 1.0 and Granite emits gibberish (the safetensors
    // detect path picks them up from config.json directly, which is why
    // `shannon verify` was clean while `larql run` on a Granite vindex
    // was not). `embedding_multiplier` is already captured at the top
    // level of `VindexConfig` as `embed_scale`.
    /// Attention score multiplier (Granite 4.1: 1/64 on 3B, 1/128 on
    /// 8B/30B). Applied on top of 1/sqrt(head_dim) — see
    /// [`larql_models::ModelArchitecture::attention_multiplier`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_multiplier: Option<f64>,
    /// Residual-stream scaling factor applied after attention and FFN
    /// additions (Granite 4.1: 0.22 on 3B/8B, 0.175 on 30B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_multiplier: Option<f64>,
    /// Logits scaling factor — final logits are divided by this before
    /// softmax (Granite 4.1: 10 on 3B, 16 on 8B/30B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits_scaling: Option<f64>,
    /// RMS-norm / LayerNorm epsilon parsed from `rms_norm_eps` (or
    /// `layer_norm_eps`). Llama 3, Mistral, Gemma 3, and Granite 4.1 all
    /// ship 1e-5; older default was 1e-6. Captured here so the vindex
    /// load path doesn't silently fall back to the arch-class default —
    /// same regression mode that broke the safetensors path before the
    /// fix in `docs/diagnoses/shannon-cross-engine-divergence.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_eps: Option<f64>,

    // ── Fields that were dropped until 2026-08-06 ──
    // Each is read by the forward pass and each was absent from this
    // struct, so no vindex-served model ever saw it. All are
    // `#[serde(default)]`, so vindexes written before this lands still
    // load — they just keep answering `None`, which is what they
    // already did. Re-extract to pick the values up.
    /// RoPE scaling block, in the `config.json` shape
    /// (`larql_models::RopeScaling::to_config_json`). Carried as raw
    /// JSON rather than a typed mirror so the detector's parser stays
    /// the single definition of how each family is read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<serde_json::Value>,
    /// Gemma 2 attention-logit softcapping. Note `final_logit_softcapping`
    /// was already persisted and this one was not — the pair splits
    /// across the same model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_logit_softcapping: Option<f64>,
    /// GPT-OSS clamp on both halves of the fused gate/up projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swiglu_limit: Option<f64>,
    /// OLMoE / Mixtral router top-k renormalisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_topk_prob: Option<bool>,
    /// Whether `lm_head` is tied to the embedding matrix. ROADMAP H5a
    /// made an untied-but-missing `lm_head` a hard error in
    /// `larql-models`; without this field that fix could not reach a
    /// vindex-served model, which always answered `None` (= "absent, no
    /// claim either way").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_word_embeddings: Option<bool>,
}

/// MoE (Mixture of Experts) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token (top-K routing).
    pub top_k: usize,
    /// Whether there's a shared expert always active (DeepSeek V2/V3).
    #[serde(default)]
    pub shared_expert: bool,
    /// That branch's intermediate width, where the judgment declares one.
    /// Carried beside the boolean rather than derived from
    /// [`Self::moe_intermediate_size`]: the DeepSeek/Kimi lineage sizes
    /// the branch as one wider FFN at `moe_intermediate_size * count`
    /// while Qwen sizes it from its own key, and on Qwen1.5-MoE the two
    /// answers differ fourfold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_intermediate_size: Option<usize>,
    /// Router type (e.g., "top_k_softmax", "gemma4_top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
    /// Per-expert intermediate (hidden) dimension.
    /// Differs from the dense FFN intermediate_size in hybrid models (Gemma 4 A4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Hybrid MoE: dense MLP and expert block coexist in each layer, outputs summed.
    /// True for Gemma 4 A4B. False for pure MoE (Mixtral, DeepSeek).
    #[serde(default)]
    pub hybrid: bool,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

impl VindexModelConfig {
    /// Build the serialisable vindex architecture config from the detected
    /// model architecture. Keeping this mapping in one place prevents vector
    /// imports, f32 writers, and Q4K writers from drifting.
    pub fn from_arch(arch: &dyn larql_models::ModelArchitecture) -> Self {
        let cfg = arch.config();
        Self {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            use_sliding_window: cfg.use_sliding_window,
            max_window_layers: cfg.max_window_layers,
            position_embedding_type: cfg.position_embedding_type.clone(),
            no_rope_layers: cfg.no_rope_layers.clone(),
            no_rope_layer_interval: cfg.no_rope_layer_interval,
            rope_interleaved: cfg.rope_interleaved,
            use_mrope: cfg.use_mrope,
            ffn_shape_name: cfg.ffn_shape_name.clone(),
            is_llama_config: cfg.is_llama_config,
            moe: if arch.is_moe() {
                Some(MoeConfig {
                    num_experts: arch.num_experts(),
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    shared_expert_intermediate_size: arch.shared_expert_intermediate_size(),
                    router_type: arch.moe_router_type().into(),
                    moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                        Some(arch.moe_intermediate_size())
                    } else {
                        None
                    },
                    hybrid: arch.is_hybrid_moe(),
                })
            } else {
                None
            },
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            use_double_wide_mlp: cfg.use_double_wide_mlp,
            vocab_size_per_layer_input: cfg.vocab_size_per_layer_input,
            rope_local_base: cfg.rope_local_base,
            layer_rope_theta: cfg.layer_rope_theta.clone(),
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            qk_scale_factor: cfg.qk_scale_factor,
            output_multiplier: cfg.output_multiplier,
            post_norm_eps: cfg.post_norm_eps,
            attention_bias: cfg.attention_bias,
            qkv_bias: cfg.qkv_bias,
            hidden_act: cfg.hidden_act.clone(),
            activation_situ_beta: cfg.activation_situ_beta,
            activation_situ_linear_beta: cfg.activation_situ_linear_beta,
            ffn_intermediate_size_by_layer: cfg.ffn_intermediate_size_by_layer.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            final_logit_softcapping: cfg.final_logit_softcapping,
            attention_multiplier: cfg.attention_multiplier,
            residual_multiplier: cfg.residual_multiplier,
            logits_scaling: cfg.logits_scaling,
            norm_eps: cfg.norm_eps,
            rope_scaling: cfg.rope_scaling.as_ref().map(|rs| rs.to_config_json()),
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            swiglu_limit: cfg.swiglu_limit,
            norm_topk_prob: cfg.norm_topk_prob,
            tie_word_embeddings: cfg.tie_word_embeddings,
        }
    }
}

#[cfg(test)]
mod tests;
