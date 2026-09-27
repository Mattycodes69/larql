//! Config key aliases and typed field readers for the parser.

// Different model families use different JSON keys for the same concept.
// Ordering is priority: first match wins.

/// Total routed expert count: DeepSeek, Qwen MoE, Mixtral variants.
pub(super) const NUM_EXPERTS_KEYS: &[&str] =
    &["n_routed_experts", "num_local_experts", "num_experts"];

/// Experts activated per token: llama.cpp / HF spelling variants.
pub(super) const NUM_EXPERTS_PER_TOK_KEYS: &[&str] =
    &["num_experts_per_tok", "num_experts_per_token"];

/// Shared-expert count. DeepSeek-lineage checkpoints write
/// `n_shared_experts`; Kimi Linear writes `num_shared_experts`. One fact,
/// and reading only the first spelling silently drops the always-on branch.
pub(super) const NUM_SHARED_EXPERTS_KEYS: &[&str] = &["n_shared_experts", "num_shared_experts"];

/// The always-on branch's OWN intermediate width, where a family sizes it
/// independently of the routed experts. Qwen2-MoE and Qwen3.5-MoE write
/// `shared_expert_intermediate_size`; Nemotron-H writes
/// `moe_shared_expert_intermediate_size`. One fact, two spellings.
///
/// Not interchangeable with `moe_intermediate_size * shared experts`,
/// which is how the DeepSeek/Kimi lineage sizes one wider shared FFN:
/// Qwen1.5-MoE declares 5632 against a routed width of 1408, and
/// Nemotron-3 Nano declares 3712 against 1856 with one shared expert.
/// Deriving it would have built the branch four times too narrow.
pub(super) const SHARED_EXPERT_INTERMEDIATE_SIZE_KEYS: &[&str] = &[
    "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size",
];

/// Whether the router renormalises its selected top-k probabilities.
/// `norm_topk_prob` in the DeepSeek lineage, `moe_renormalize` on Kimi
/// Linear. The two settings differ by a rescale of the whole expert
/// branch, so a default here is a quiet numerical change.
pub(super) const NORM_TOPK_PROB_KEYS: &[&str] = &["norm_topk_prob", "moe_renormalize"];

/// Router scoring function: `scoring_func` (DeepSeek, GLM-5.3-Flash) or
/// `moe_router_activation_func` (Kimi Linear).
pub(super) const ROUTER_ACTIVATION_KEYS: &[&str] = &["scoring_func", "moe_router_activation_func"];

/// Expert-group count: `n_group` (DeepSeek, GLM-5.3-Flash) or
/// `num_expert_group` (Kimi Linear).
pub(super) const EXPERT_GROUP_KEYS: &[&str] = &["n_group", "num_expert_group"];

/// Return the first `u64` found under any of `keys` in `config`.
pub(super) fn field_u64(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| config[k].as_u64())
}

/// Read a topology field by alias list as `usize`, preferring `text_config`
/// (multimodal nesting) and falling back to the top-level object. The first
/// alias to resolve wins. Returns 0 when no alias is present; the configured
/// field validators reject 0 at the next layer, so the magic-number guess
/// defaults (e.g. 2048) don't leak in and masquerade as a real model topology.
///
/// Alias lists live in `config_io.rs` so the loader's `require_config_fields`
/// validator and this parser agree on what names are acceptable for each
/// canonical field — see [`crate::detect::config_io::CONFIG_KEY_HIDDEN_SIZE_ALIASES`]
/// (GPT-2's `n_embd` etc.).
pub(super) fn topology_field(
    config: &serde_json::Value,
    text_config: &serde_json::Value,
    aliases: &[&str],
) -> usize {
    crate::detect::config_io::read_aliased_u64(config, text_config, aliases).unwrap_or(0) as usize
}
