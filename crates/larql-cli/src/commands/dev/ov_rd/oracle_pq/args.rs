//! `oracle-pq` arguments.

use clap::Args;
use std::path::PathBuf;

#[allow(unused_imports)]
use super::*;

#[derive(Args)]
pub(in super::super) struct OraclePqArgs {
    /// Self-contained Q4K vindex directory.
    #[arg(long)]
    pub(super) index: PathBuf,

    /// JSONL prompt file. Each line must include at least {"prompt": "..."}.
    #[arg(long)]
    pub(super) prompts: PathBuf,

    /// Output directory.
    #[arg(long)]
    pub(super) out: PathBuf,

    /// Explicit heads as layer:head comma list, e.g. 0:6.
    #[arg(long)]
    pub(super) heads: String,

    /// Comma-separated PQ configs as K:groups:bits, e.g. 128:16:4,192:24:4.
    #[arg(long)]
    pub(super) configs: String,

    /// Relative singular value cutoff for retained W_O-visible directions.
    #[arg(long, default_value_t = 1e-6)]
    pub(super) sigma_rel_cutoff: f64,

    /// Lloyd iterations per product-codebook group.
    #[arg(long, default_value_t = 25)]
    pub(super) pq_iters: usize,

    /// Also materialize residual-space additive tables and compare Mode D injection.
    #[arg(long)]
    pub(super) mode_d_check: bool,

    /// Fit and evaluate graph-native discrete address probes.
    ///
    /// The probes use only prompt metadata and token ids, not residual vectors.
    /// Requires --mode-d-check because predicted addresses are evaluated through
    /// the materialized residual-space tables.
    #[arg(long)]
    pub(super) address_probes: bool,

    /// Add a mixed simple-key address probe that picks the best discrete key
    /// independently for each PQ group on the training split.
    #[arg(long)]
    pub(super) address_mixed_key_probe: bool,

    /// Evaluate simple discrete keys on selected PQ groups only. Selected
    /// groups are predicted from each key; unselected groups are evaluated as
    /// either oracle-correct or majority/default.
    #[arg(long)]
    pub(super) address_key_group_probe: bool,

    /// Comma-separated PQ groups for --address-key-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_key_groups: String,

    /// Optional comma-separated simple-key probe names for
    /// --address-key-group-probe. Empty evaluates all simple-key probes.
    #[arg(long, default_value = "")]
    pub(super) address_key_group_probe_names: String,

    /// Evaluate selected PQ groups by replacing them with train-set majority
    /// codes while all unselected groups remain oracle-correct.
    #[arg(long)]
    pub(super) address_majority_group_probe: bool,

    /// Comma-separated PQ groups for --address-majority-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_majority_groups: String,

    /// Evaluate code-level behavioral substitution for selected PQ groups.
    ///
    /// Positions whose oracle group code equals a selected from-code are
    /// substituted to each selected to-code while all other groups and
    /// positions remain oracle-correct.
    #[arg(long)]
    pub(super) address_code_substitution_group_probe: bool,

    /// Comma-separated PQ groups for --address-code-substitution-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_code_substitution_groups: String,

    /// Optional comma-separated source codes. Empty means all codes.
    #[arg(long, default_value = "")]
    pub(super) address_code_substitution_from_codes: String,

    /// Target codes. Use "majority" or a comma-separated list of codes.
    #[arg(long, default_value = "majority")]
    pub(super) address_code_substitution_to_codes: String,

    /// Evaluate simultaneous behavioral class-collapse substitutions.
    ///
    /// Spec format:
    ///   name=6+10+13:13
    ///   name=6+10+13:13|7:10
    /// Multiple specs are separated by semicolons.
    #[arg(long)]
    pub(super) address_code_class_collapse_group_probe: bool,

    /// Comma-separated PQ groups for --address-code-class-collapse-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_code_class_collapse_groups: String,

    /// Semicolon-separated class-collapse specs.
    #[arg(long, default_value = "")]
    pub(super) address_code_class_collapse_specs: String,

    /// Probe position-local interactions for one prompt and one PQ group.
    ///
    /// This is a targeted diagnostic for quotient failures: selected primary
    /// and secondary source codes are changed to one target code only within
    /// the requested prompt, while all other positions/groups remain oracle.
    #[arg(long)]
    pub(super) address_code_position_interaction_probe: bool,

    /// Prompt id for --address-code-position-interaction-probe.
    #[arg(long, default_value = "")]
    pub(super) address_code_position_prompt_id: String,

    /// PQ group for --address-code-position-interaction-probe.
    #[arg(long, default_value_t = 0)]
    pub(super) address_code_position_group: usize,

    /// Primary source codes for --address-code-position-interaction-probe.
    #[arg(long, default_value = "10")]
    pub(super) address_code_position_primary_codes: String,

    /// Secondary source codes for --address-code-position-interaction-probe.
    #[arg(long, default_value = "6")]
    pub(super) address_code_position_secondary_codes: String,

    /// Target code for --address-code-position-interaction-probe.
    #[arg(long, default_value_t = 13)]
    pub(super) address_code_position_target_code: usize,

    /// Evaluate split-wide conditional quotient rules for one PQ group.
    ///
    /// Primary codes are mapped to the target unconditionally. Secondary codes
    /// are mapped to the target except where a built-in guard preserves the
    /// oracle code. This tests whether a quotient plus local exception guard
    /// clears the held-out gate.
    #[arg(long)]
    pub(super) address_code_conditional_quotient_group_probe: bool,

    /// PQ group for --address-code-conditional-quotient-group-probe.
    #[arg(long, default_value_t = 0)]
    pub(super) address_code_conditional_quotient_group: usize,

    /// Primary source codes for the conditional quotient probe.
    #[arg(long, default_value = "10")]
    pub(super) address_code_conditional_quotient_primary_codes: String,

    /// Secondary source codes for the conditional quotient probe.
    #[arg(long, default_value = "6")]
    pub(super) address_code_conditional_quotient_secondary_codes: String,

    /// Target code for the conditional quotient probe.
    #[arg(long, default_value_t = 13)]
    pub(super) address_code_conditional_quotient_target_code: usize,

    /// Max early position guarded by early-prose conditional quotient variants.
    #[arg(long, default_value_t = 1)]
    pub(super) address_code_conditional_quotient_early_position_max: usize,

    /// Conditional quotient guards to evaluate.
    ///
    /// Supported: early_prose_position, early_prose_bos_prev, prose_bos_prev.
    #[arg(
        long,
        default_value = "early_prose_position,early_prose_bos_prev,prose_bos_prev"
    )]
    pub(super) address_code_conditional_quotient_guards: String,

    /// Extra source:target mappings layered on top of the conditional quotient.
    ///
    /// Spec format matches class-collapse specs. Empty adds only the base
    /// conditional quotient. Example:
    ///   code4_to13=4:13;code7_to10=7:10
    #[arg(long, default_value = "")]
    pub(super) address_code_conditional_quotient_extra_specs: String,

    /// Export per-position occurrences for selected PQ group codes.
    #[arg(long)]
    pub(super) address_code_occurrences: bool,

    /// Comma-separated PQ groups for --address-code-occurrences.
    #[arg(long, default_value = "0")]
    pub(super) address_code_occurrence_groups: String,

    /// Optional comma-separated codes for --address-code-occurrences.
    /// Empty means all codes.
    #[arg(long, default_value = "")]
    pub(super) address_code_occurrence_codes: String,

    /// Occurrence split to export: train, eval, or all.
    #[arg(long, default_value = "eval")]
    pub(super) address_code_occurrence_split: String,

    /// Evaluate a hard-coded code7 fallback rule for L0H6-style probes.
    ///
    /// For selected groups, predict special code when attention argmax is BOS
    /// and stratum is not arithmetic; otherwise predict the train majority
    /// code. Unselected groups remain oracle-correct.
    #[arg(long)]
    pub(super) address_code7_bos_rule_group_probe: bool,

    /// Comma-separated PQ groups for --address-code7-bos-rule-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_code7_bos_rule_groups: String,

    /// Special code used by --address-code7-bos-rule-group-probe.
    #[arg(long, default_value_t = 7)]
    pub(super) address_code7_bos_rule_code: usize,

    /// Evaluate oracle upper bounds for a binary code7-vs-default address.
    ///
    /// Selected groups use the special code only where the oracle address has
    /// that code and the requested structural filter matches; all other
    /// positions use the train majority code. Unselected groups remain
    /// oracle-correct.
    #[arg(long)]
    pub(super) address_code7_oracle_binary_group_probe: bool,

    /// Comma-separated PQ groups for --address-code7-oracle-binary-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_code7_oracle_binary_groups: String,

    /// Special code used by --address-code7-oracle-binary-group-probe.
    #[arg(long, default_value_t = 7)]
    pub(super) address_code7_oracle_binary_code: usize,

    /// Comma-separated filters for oracle binary code7 upper bounds.
    ///
    /// Supported: all, natural_prose_bos, natural_prose_bos_or_prev.
    #[arg(
        long,
        default_value = "all,natural_prose_bos,natural_prose_bos_or_prev"
    )]
    pub(super) address_code7_oracle_binary_filters: String,

    /// Evaluate how sensitive Mode D is to address corruption.
    ///
    /// This keeps a prefix of oracle PQ groups and replaces the rest with
    /// per-group majority codes learned from the training split. It estimates
    /// how many groups must be addressed correctly before predicted addressing
    /// can pass the KL gate.
    #[arg(long)]
    pub(super) address_corruption_sweep: bool,

    /// Evaluate one-group-at-a-time address importance by replacing each group
    /// with its train-set majority code while all other groups remain oracle.
    #[arg(long)]
    pub(super) address_group_importance: bool,

    /// Fit and evaluate fixed random-hyperplane LSH probes for selected PQ
    /// groups. The selected groups are predicted from the residual entering the
    /// target layer; other groups are evaluated both oracle-correct and
    /// majority/default.
    #[arg(long)]
    pub(super) address_lsh_group_probe: bool,

    /// Comma-separated PQ groups for --address-lsh-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_lsh_groups: String,

    /// Number of LSH bits per selected group. For a 4-bit PQ group, 4 LSH bits
    /// creates 16 buckets.
    #[arg(long, default_value_t = 4)]
    pub(super) address_lsh_bits: usize,

    /// Number of deterministic random-hyperplane seeds to try per selected
    /// group. The best seed is selected by train code accuracy.
    #[arg(long, default_value_t = 32)]
    pub(super) address_lsh_seeds: usize,

    /// Fit and evaluate supervised binary-hyperplane address probes for
    /// selected PQ groups. The selected groups are predicted from the residual
    /// entering the target layer; other groups are evaluated both
    /// oracle-correct and majority/default.
    #[arg(long)]
    pub(super) address_supervised_group_probe: bool,

    /// Comma-separated PQ groups for --address-supervised-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_supervised_groups: String,

    /// SGD epochs for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 16)]
    pub(super) address_supervised_epochs: usize,

    /// SGD learning rate for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 0.05)]
    pub(super) address_supervised_lr: f32,

    /// L2 weight decay for supervised binary-hyperplane group address probes.
    #[arg(long, default_value_t = 1e-4)]
    pub(super) address_supervised_l2: f32,

    /// Fit and evaluate supervised group address probes after a diagonal
    /// affine gamma-alignment projection from the layer input toward later
    /// post-layer residual snapshots.
    #[arg(long)]
    pub(super) address_gamma_projected_group_probe: bool,

    /// Comma-separated PQ groups for --address-gamma-projected-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_gamma_projected_groups: String,

    /// Comma-separated post-layer residual snapshots used as gamma-alignment
    /// targets, e.g. 20,26,29,33. The raw layer-input supervised probe is
    /// always included as gamma_raw for comparison.
    #[arg(long, default_value = "20,26,29,33")]
    pub(super) address_gamma_projected_layers: String,

    /// Comma-separated random projection ranks for the gamma bridge control,
    /// e.g. 64,128. These are fixed Rademacher low-rank projections of the
    /// layer input followed by the same supervised bit probes.
    #[arg(long, default_value = "")]
    pub(super) address_gamma_random_ranks: String,

    /// Comma-separated deterministic seeds for random projection ranks.
    #[arg(long, default_value = "0")]
    pub(super) address_gamma_random_seeds: String,

    /// Comma-separated learned bridge ranks for the gamma bridge test. These
    /// fit a low-rank target-PCA proxy from layer input to later residual
    /// snapshots before training the same supervised group-bit probes.
    #[arg(long, default_value = "")]
    pub(super) address_gamma_learned_ranks: String,

    /// SGD epochs for learned low-rank gamma bridge fitting.
    #[arg(long, default_value_t = 8)]
    pub(super) address_gamma_learned_epochs: usize,

    /// Normalized LMS learning rate for learned low-rank gamma bridge fitting.
    #[arg(long, default_value_t = 0.5)]
    pub(super) address_gamma_learned_lr: f32,

    /// L2 weight decay for learned low-rank gamma bridge fitting.
    #[arg(long, default_value_t = 1e-5)]
    pub(super) address_gamma_learned_l2: f32,

    /// Power-iteration steps for the learned bridge target PCA basis.
    #[arg(long, default_value_t = 8)]
    pub(super) address_gamma_learned_pca_iters: usize,

    /// Report train/eval PQ code distribution stability for selected groups.
    #[arg(long)]
    pub(super) address_code_stability: bool,

    /// Comma-separated PQ groups for --address-code-stability.
    #[arg(long, default_value = "0")]
    pub(super) address_code_stability_groups: String,

    /// Fit and evaluate selected PQ groups from previous-layer FFN top-feature
    /// keys. This is the first model-native discrete-state address probe for
    /// non-layer-0 dynamic heads.
    #[arg(long)]
    pub(super) address_prev_ffn_feature_group_probe: bool,

    /// Comma-separated PQ groups for --address-prev-ffn-feature-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_prev_ffn_feature_groups: String,

    /// Number of previous-layer FFN activation features retained for feature
    /// hash keys.
    #[arg(long, default_value_t = 4)]
    pub(super) address_prev_ffn_feature_top_k: usize,

    /// Fit and evaluate selected PQ groups from an FFN-first diagnostic state:
    /// run the target layer's FFN on the pre-attention residual, use top
    /// activation features as keys, but leave the real forward ordering
    /// unchanged. This tests whether computed L0 FFN features would bootstrap
    /// attention addressability under an FFN-first reorder.
    #[arg(long)]
    pub(super) address_ffn_first_feature_group_probe: bool,

    /// Comma-separated PQ groups for --address-ffn-first-feature-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_ffn_first_feature_groups: String,

    /// Number of FFN-first activation features retained for feature hash keys.
    #[arg(long, default_value_t = 4)]
    pub(super) address_ffn_first_feature_top_k: usize,

    /// Fit and evaluate selected PQ groups from discrete attention/relation
    /// state keys. This tests whether the dominant address is carried by QK
    /// routing structure rather than token or FFN-feature state.
    #[arg(long)]
    pub(super) address_attention_relation_group_probe: bool,

    /// Comma-separated PQ groups for --address-attention-relation-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_attention_relation_groups: String,

    /// Fit and evaluate selected PQ groups from learned attention-pattern
    /// cluster IDs. This is a discrete relation-catalogue probe over fixed
    /// features derived from the full attention distribution.
    #[arg(long)]
    pub(super) address_attention_cluster_group_probe: bool,

    /// Comma-separated PQ groups for --address-attention-cluster-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_attention_cluster_groups: String,

    /// Comma-separated k values for attention-pattern clustering.
    #[arg(long, default_value = "16,32")]
    pub(super) address_attention_cluster_ks: String,

    /// Optional comma-separated attention-cluster probe names. Empty evaluates
    /// all cluster probe names for the selected k values.
    #[arg(long, default_value = "")]
    pub(super) address_attention_cluster_probe_names: String,

    /// Fit/evaluate selected PQ groups from attention-pattern clusters where
    /// the attention distribution is recomputed from only the first r Q/K
    /// dimensions. Use rank 0 for the full-QK control.
    #[arg(long)]
    pub(super) address_reduced_qk_cluster_group_probe: bool,

    /// Comma-separated PQ groups for --address-reduced-qk-cluster-group-probe.
    #[arg(long, default_value = "0")]
    pub(super) address_reduced_qk_cluster_groups: String,

    /// Comma-separated QK ranks. Rank 0 means full QK; positive ranks are
    /// clamped to the layer head dimension.
    #[arg(long, default_value = "0,128,64,32,16")]
    pub(super) address_reduced_qk_ranks: String,

    /// Comma-separated k values for reduced-QK attention-pattern clustering.
    #[arg(long, default_value = "16,32")]
    pub(super) address_reduced_qk_cluster_ks: String,

    /// Optional comma-separated reduced-QK cluster probe names. Empty evaluates
    /// all generated names.
    #[arg(long, default_value = "")]
    pub(super) address_reduced_qk_cluster_probe_names: String,

    /// Comma-separated PQ groups whose centroids are fit separately per
    /// prompt stratum. This is a codebook-layout diagnostic for cases where a
    /// single global PQ group carries a hard prose/structured tail.
    #[arg(long, default_value = "")]
    pub(super) stratum_conditioned_pq_groups: String,

    /// Limit prompts for bounded oracle runs.
    #[arg(long)]
    pub(super) max_prompts: Option<usize>,

    /// Keep at most N prompts per stratum after loading. Useful for balanced
    /// held-out smoke runs from a larger ordered corpus.
    #[arg(long)]
    pub(super) max_per_stratum: Option<usize>,

    /// Evaluate only prompts where prompt_index % eval_mod == eval_offset.
    /// The remaining prompts are used to fit static means, PCA, and PQ.
    #[arg(long)]
    pub(super) eval_mod: Option<usize>,

    /// Held-out modulo offset used with --eval-mod.
    #[arg(long, default_value_t = 0)]
    pub(super) eval_offset: usize,
}
