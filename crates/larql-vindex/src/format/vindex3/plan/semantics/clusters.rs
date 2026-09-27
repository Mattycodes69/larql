//! Semantic clusters and how a config subject is assigned one.

use super::super::report::SemanticClass;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

/// The model concept a finding is about.
///
/// This is the census's third axis, and it answers a different question
/// from [`SemanticClass`]. The class says how much losing a subject would
/// matter and decides whether it blocks; the cluster says *which idea of
/// the model* it belongs to. Forty `rope_*` spellings across ten
/// organisations are forty subjects, one idea — and the unit of
/// remediation work is the idea, never the spelling.
///
/// It lives here, keyed by exact leaf name, rather than in a script's
/// regex, because a regex over finding text is a fourth authority on a
/// question three already answer. `num_experts_per_tok` and a sampling
/// `top_k` share no substring rule that separates them; an exact table
/// does.
///
/// [`Self::Unclustered`] is honest and expected: a subject no table has
/// judged. It is a *finding about the taxonomy*, not a bucket to grow by
/// pattern-matching, and the conformance report counts it as its own row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticCluster {
    PositionRope,
    PositionMrope,
    AttentionSchedule,
    AttentionBias,
    AttentionSparseIndexer,
    AttentionLogitSoftcapping,
    MoeRouting,
    FfnActivation,
    NormGeometry,
    ShapeAndTensorNaming,
    MixerSsmGeometry,
    DecodeMultiTokenPrediction,
    ResidualHyperConnections,
    ResidualWiring,
    ScaleMultipliers,
    RepresentationQuantization,
    ModalityVision,
    ModalityAudio,
    /// The graph could not be completed — a consequence finding, whose
    /// cause is one of the others.
    ExecutionSurface,
    /// Who the checkpoint says it is.
    ArchitectureIdentity,
    InertTrainingOnly,
    InertGenerationPolicy,
    Unclustered,
}

/// Leaf names whose concept is known, grouped by concept.
///
/// Exact names throughout. The contract is the same one every table in
/// this module carries: an entry is a claim about a specific key, and a
/// key nobody has judged stays [`SemanticCluster::Unclustered`] rather
/// than being pattern-matched into a neighbour.
pub(super) const CLUSTER_KEYS: &[(SemanticCluster, &[&str])] = &[
    (
        SemanticCluster::PositionRope,
        &[
            "rope_theta",
            "rope_type",
            "type",
            "factor",
            "rope_scaling",
            "rope_parameters",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "short_factor",
            "long_factor",
            "short_mscale",
            "long_mscale",
            "partial_rotary_factor",
            "rotary_pct",
            "rotary_emb_base",
            "rope_interleave",
            "rope_local_base_freq",
            "no_rope_layers",
            "no_rope_layer_interval",
            "position_embedding_type",
            "n_positions",
            "n_ctx",
            "interpolate_factor",
            "llama_4_scaling_beta",
            "rope_scaling_factor",
            "use_mrope",
        ],
    ),
    (
        SemanticCluster::PositionMrope,
        &["mrope_section", "mrope_interleaved"],
    ),
    (
        SemanticCluster::AttentionSchedule,
        &[
            "sliding_window",
            "use_sliding_window",
            "sliding_window_size",
            "layer_types",
            "local_layer_ids",
            "full_attention_interval",
            "attn_layer_indices",
            "attn_layer_offset",
            "attn_layer_period",
            "attention_chunk_size",
            "full_attn_mod",
            "attention_policy",
            "max_window_layers",
            "num_kv_shared_layers",
        ],
    ),
    (
        SemanticCluster::AttentionBias,
        &[
            "attention_bias",
            "qkv_bias",
            "use_bias",
            "bias",
            "clip_qkv",
            "o_proj_bias",
            "attention_out_bias",
        ],
    ),
    (
        SemanticCluster::AttentionSparseIndexer,
        &[
            "index_topk",
            "index_n_heads",
            "index_head_dim",
            "index_topk_freq",
            "index_topk_pattern",
            "index_skip_topk_offset",
            "index_kpool",
            "index_kpool_compress",
            "index_kpool_always_select_tail",
            "indexer_rope_interleave",
            "compress_ratios",
            "compress_rope_theta",
            "o_lora_rank",
            "o_groups",
        ],
    ),
    (
        SemanticCluster::AttentionLogitSoftcapping,
        &[
            "attn_logit_softcapping",
            "final_logit_softcapping",
            "query_pre_attn_scalar",
        ],
    ),
    (
        SemanticCluster::MoeRouting,
        &[
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
            "n_routed_experts",
            "n_shared_experts",
            "shared_expert_intermediate_size",
            "moe_shared_expert_intermediate_size",
            "moe_intermediate_size",
            "expert_intermediate_size",
            "top_k_experts",
            "topk_method",
            "topk_group",
            "n_group",
            "scoring_func",
            "norm_topk_prob",
            // The latent routed branch: where the routed experts run, and
            // whether their aggregate is normalised. Clustered with
            // routing rather than with norm geometry, because the leverage
            // they describe is the ROUTED BRANCH's shape — the norm is a
            // parameter of that branch, not a norm the model has anywhere
            // else.
            "routed_expert_hidden_size",
            "latent_moe_use_norm",
            "routed_scaling_factor",
            "decoder_sparse_step",
            "mlp_only_layers",
            "moe_layer_freq",
            "first_k_dense_replace",
            "num_experts_shared",
            "use_qk_norm",
        ],
    ),
    (
        SemanticCluster::FfnActivation,
        &[
            "hidden_act",
            "hidden_activation",
            "activation",
            "activation_function",
            "swiglu_limit",
            // SiTU-GLU's two softcaps. Beside `hidden_act`, which is the
            // key that selects them: a fact and its parameters in
            // different clusters would make the cluster map lie about
            // where the leverage is.
            "activation_situ_beta",
            "activation_situ_linear_beta",
            "mlp_type",
            "mlp_expansion_factor",
            "use_double_wide_mlp",
        ],
    ),
    (
        SemanticCluster::NormGeometry,
        &[
            "rms_norm_eps",
            "layer_norm_eps",
            "layer_norm_epsilon",
            "norm_eps",
            "normalization_type",
            "qk_layernorm",
            "use_qk_layernorm",
        ],
    ),
    (
        SemanticCluster::MixerSsmGeometry,
        &[
            "mamba_d_conv",
            "mamba_d_state",
            "mamba_expand",
            "mamba_n_heads",
            "mamba_n_groups",
            "mamba_conv_bias",
            "mamba_proj_bias",
            "mamba_d_head",
            "mamba_chunk_size",
            "conv_kernel",
            "state_size",
            "linear_conv_kernel_dim",
            "linear_attn_config",
            "time_step_limit",
        ],
    ),
    (
        SemanticCluster::DecodeMultiTokenPrediction,
        &[
            "num_nextn_predict_layers",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "layers",
            "fc",
            "norm",
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
        ],
    ),
    (
        SemanticCluster::ResidualHyperConnections,
        &[
            "hc_mult",
            "hc_eps",
            "hc_sinkhorn_iters",
            "hc_count",
            "hc_lowrank",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
            "mhc",
        ],
    ),
    (
        SemanticCluster::ResidualWiring,
        &[
            "use_parallel_residual",
            "attn_res_block_size",
            "output_attn_res_proj",
        ],
    ),
    (
        SemanticCluster::ScaleMultipliers,
        &[
            "attention_multiplier",
            "embedding_multiplier",
            "logits_scaling",
            "residual_multiplier",
            "attention_in_multiplier",
            "attention_out_multiplier",
            "key_multiplier",
            "lm_head_multiplier",
            "mlp_multipliers",
        ],
    ),
    (SemanticCluster::ArchitectureIdentity, &["is_llama_config"]),
    (
        SemanticCluster::ShapeAndTensorNaming,
        &[
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "n_head",
            "n_embd",
            "n_layer",
            "n_inner",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "kv_lora_rank",
            "q_lora_rank",
            "architectures",
        ],
    ),
    (
        SemanticCluster::ModalityAudio,
        &[
            "audio_token_id",
            "boa_token_id",
            "eoa_token_id",
            "eoa_token_index",
            "audio_embed_dim",
            "audio_config",
        ],
    ),
];

/// Clusters carried by a whole flattened path rather than a leaf name.
///
/// A structural finding names a rule, not a config key — `layer_census`
/// is not a leaf anyone declared — and a subject that is a *nested
/// section* (`quantization_config.…`) is owned by the section rather than
/// by whatever its last segment happens to be called.
pub(super) fn cluster_by_path(subject: &str) -> Option<SemanticCluster> {
    if subject.contains("quantization_config") {
        return Some(SemanticCluster::RepresentationQuantization);
    }
    if subject.contains("vision") || subject.contains("image") || subject.contains("video") {
        return Some(SemanticCluster::ModalityVision);
    }
    if subject.contains("execution_surface") || subject == "layer_census" {
        return Some(SemanticCluster::ExecutionSurface);
    }
    if subject == "architecture_identity" || subject == "architecture_family" {
        return Some(SemanticCluster::ArchitectureIdentity);
    }
    if subject.contains("mtp") {
        return Some(SemanticCluster::DecodeMultiTokenPrediction);
    }
    None
}

/// The concept a finding's subject belongs to.
///
/// One derivation, used by the plan document and by anything reporting
/// over it. Path-scoped ownership is asked first: a key nested under a
/// section belongs to that section even when its leaf name would match a
/// general table.
pub fn cluster_for(subject: &str) -> SemanticCluster {
    if let Some(cluster) = cluster_by_path(subject) {
        return cluster;
    }
    let leaf = leaf_of(subject);
    if let Some((cluster, _)) = CLUSTER_KEYS.iter().find(|(_, keys)| keys.contains(&leaf)) {
        return *cluster;
    }
    // Fall back to what the key IS, when the registry already judged it
    // inert. These are concepts too — "this is decoding policy" is a
    // statement about the model, and reporting them as unclustered would
    // hide the largest cheap win in the corpus behind a shrug.
    match classify_key(leaf) {
        SemanticClass::GenerationPolicy => SemanticCluster::InertGenerationPolicy,
        SemanticClass::TrainingOnly => SemanticCluster::InertTrainingOnly,
        _ => SemanticCluster::Unclustered,
    }
}

/// Classify an unconsumed config key by its leaf name.
pub fn classify_key(leaf: &str) -> SemanticClass {
    if EXECUTION_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::ExecutionSemantic
    } else if TENSOR_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::TensorSemantic
    } else if INTERFACE_SEMANTIC_KEYS.contains(&leaf) {
        SemanticClass::InterfaceSemantic
    } else if METADATA_KEYS.contains(&leaf) {
        SemanticClass::MetadataOnly
    } else if TRAINING_ONLY_KEYS.contains(&leaf) || DROPOUT_KEYS.contains(&leaf) {
        SemanticClass::TrainingOnly
    } else if GENERATION_POLICY_KEYS.contains(&leaf) {
        SemanticClass::GenerationPolicy
    } else if alias_canonical(leaf).is_some() {
        SemanticClass::Alias
    } else if IGNORED_SAFE_KEYS.contains(&leaf) {
        SemanticClass::IgnoredSafe
    } else if unsupported_component(leaf).is_some() {
        SemanticClass::UnsupportedComponent
    } else {
        SemanticClass::Unknown
    }
}

/// Logical component a flattened config path belongs to.
///
/// `<name>_config.<rest>` attributes to `<name>` (`text_config.x` → `text`);
/// everything else is the artifact root — including a bare leaf that
/// happens to end in `_config`. SmolLM2's `is_llama_config: true` is a
/// boolean at the root, not the section of a component called `is_llama`,
/// and reading it as one sent its probe to a component the graph never
/// builds. A section is a segment with something after it.
pub fn component_of(path: &str) -> String {
    const CONFIG_SUFFIX: &str = "_config";
    const ROOT_COMPONENT: &str = "root";
    match path.split_once('.') {
        // A section that parameterises an operator of the main stack is
        // not a component of its own, so its keys belong to the stack that
        // runs that operator. Naming a component here that the graph never
        // builds sends every probe looking for it and finding nothing —
        // which reads as "not carried" for facts that are carried
        // perfectly well. `linear_attn_config` is the case.
        Some((section, _rest))
            if section.ends_with(CONFIG_SUFFIX)
                && !larql_models::inventory::is_operator_config_section(section) =>
        {
            section[..section.len() - CONFIG_SUFFIX.len()].to_string()
        }
        _ => ROOT_COMPONENT.to_string(),
    }
}

/// Last dot-separated segment of a flattened path.
pub fn leaf_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}
