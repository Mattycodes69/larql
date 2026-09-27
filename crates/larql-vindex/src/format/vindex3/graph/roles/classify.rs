//! Classifying a stack or expert tensor into its operand role.

use super::super::policy::LayerOperator;

#[allow(unused_imports)]
use super::*;

/// Classify a layer-relative suffix as one `PerExpert`-format family's
/// indexed operand, or `None` — never a guess: the prefix must match
/// exactly, the segment between prefix and leaf must be *only* decimal
/// digits (so `experts.10.w1.weight` cannot be mistaken for `experts.1`'s
/// `0.w1.weight`, and a non-numeric segment refuses rather than silently
/// classifying), and the leaf must match one of the family's declared
/// spellings exactly.
pub(super) fn classify_indexed_expert(suffix: &str) -> Option<OperandRole> {
    for family in INDEXED_EXPERT_FAMILIES {
        let Some(rest) = suffix.strip_prefix(family.prefix) else {
            continue;
        };
        let Some((index, leaf)) = rest.split_once('.') else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(expert_id) = index.parse::<u16>() else {
            continue;
        };
        if let Some((_, role)) = family.leaves.iter().find(|(name, _)| *name == leaf) {
            return Some(role(expert_id));
        }
    }
    None
}

/// Classify one stack tensor by its object-relative name
/// (`{layer}.{suffix}`). `None` when the name is not layer-shaped or the
/// suffix matches no judged role — callers treat that as a blocking fact.
/// Suffix → role **on a KDA layer**, consulted before [`ROLE_TABLE`].
///
/// Only the suffixes KDA claims are listed; everything else (norms, FFN,
/// router) falls through, because those mean the same thing whatever the
/// attention operator is. Five of these collide with softmax spellings and
/// are the reason this table exists.
pub(super) const KDA_ROLE_TABLE: &[(&str, OperandRole)] = &[
    // Collides with the softmax set — same suffix, different operator.
    ("self_attn.q_proj.weight", OperandRole::KdaQProj),
    ("self_attn.k_proj.weight", OperandRole::KdaKProj),
    ("self_attn.v_proj.weight", OperandRole::KdaVProj),
    ("self_attn.o_proj.weight", OperandRole::KdaOutProj),
    // KDA-only spellings.
    ("self_attn.q_conv1d.weight", OperandRole::KdaQConv1d),
    ("self_attn.k_conv1d.weight", OperandRole::KdaKConv1d),
    ("self_attn.v_conv1d.weight", OperandRole::KdaVConv1d),
    ("self_attn.f_a_proj.weight", OperandRole::KdaFAProj),
    ("self_attn.f_b_proj.weight", OperandRole::KdaFBProj),
    ("self_attn.g_a_proj.weight", OperandRole::KdaGAProj),
    ("self_attn.g_b_proj.weight", OperandRole::KdaGBProj),
    // The full-rank form of the same gate. Which form the layer is
    // EXPECTED to ship is the declaration's question, answered at closure;
    // the table only says what the spelling is on this operator.
    ("self_attn.g_proj.weight", OperandRole::KdaGProj),
    ("self_attn.b_proj.weight", OperandRole::KdaBProj),
    ("self_attn.A_log", OperandRole::KdaALog),
    ("self_attn.dt_bias", OperandRole::KdaDtBias),
    ("self_attn.o_norm.weight", OperandRole::KdaONorm),
];

/// Suffix → role **on an MLA layer**, consulted before [`ROLE_TABLE`] —
/// the same reason [`KDA_ROLE_TABLE`] exists: `q_proj`/`o_proj` collide in
/// SPELLING with the softmax set at a DIFFERENT shape, so only the
/// layer's operator can tell them apart.
pub(super) const MLA_ROLE_TABLE: &[(&str, OperandRole)] = &[
    // Collides with the softmax set — same suffix, different geometry.
    ("self_attn.q_proj.weight", OperandRole::MlaQProj),
    // Kimi-K3's factorised query. Named unconditionally here — the table
    // answers for the OPERATOR, as its siblings do — and held to the
    // declared form by closure, which is where "declared Direct but
    // shipped a triple" is refused.
    ("self_attn.q_a_proj.weight", OperandRole::MlaQAProj),
    ("self_attn.q_a_layernorm.weight", OperandRole::MlaQANorm),
    ("self_attn.q_b_proj.weight", OperandRole::MlaQBProj),
    ("self_attn.o_proj.weight", OperandRole::MlaOutProj),
    // MLA-only spellings.
    (
        "self_attn.kv_a_proj_with_mqa.weight",
        OperandRole::MlaKvAProj,
    ),
    ("self_attn.kv_b_proj.weight", OperandRole::MlaKvBProj),
    ("self_attn.kv_a_layernorm.weight", OperandRole::MlaKvANorm),
    // Same spelling as the KDA full-rank gate, different operator, different
    // operation. Expected only under `mla_use_output_gate`, which closure
    // checks; the table answers for the operator alone.
    ("self_attn.g_proj.weight", OperandRole::MlaOutputGate),
];

/// Suffix → role **on a Mamba2 layer**, consulted before [`ROLE_TABLE`]
/// for the same reason [`KDA_ROLE_TABLE`] is: the roles are
/// operator-gated, so an unknown stack shipping a bare `norm.weight`
/// still classifies as *nothing* and blocks rather than acquiring the
/// mixer's placement vocabulary.
pub(super) const MAMBA2_ROLE_TABLE: &[(&str, OperandRole)] = &[
    ("mixer.in_proj.weight", OperandRole::Mamba2InProj),
    ("mixer.conv1d.weight", OperandRole::Mamba2Conv1d),
    ("mixer.conv1d.bias", OperandRole::Mamba2Conv1dBias),
    ("mixer.A_log", OperandRole::Mamba2ALog),
    ("mixer.D", OperandRole::Mamba2D),
    ("mixer.dt_bias", OperandRole::Mamba2DtBias),
    ("mixer.norm.weight", OperandRole::Mamba2GatedNorm),
    ("mixer.out_proj.weight", OperandRole::Mamba2OutProj),
    ("norm.weight", OperandRole::Mamba2PreMixerNorm),
];

/// Suffix → role **on a conv-QKV attention layer**, consulted before
/// [`ROLE_TABLE`]. Every spelling here collides with
/// [`MAMBA2_ROLE_TABLE`] at a different shape — the hybrid stack wraps
/// both block kinds in the same `mixer.`/`norm.` estate — so the layer's
/// operator is the only authority that can separate them, exactly the
/// per-layer-table argument [`classify_stack_tensor_on`] documents.
/// The pre-mixer norm role is shared deliberately: it is the SAME
/// declaration (one bare `norm.weight` wrapping the block) on both
/// layer kinds of this lineage.
pub(super) const CONV_QKV_ROLE_TABLE: &[(&str, OperandRole)] = &[
    ("mixer.in_proj.weight", OperandRole::ConvQkvInProj),
    ("mixer.conv1d.weight", OperandRole::ConvQkvConv1d),
    ("mixer.conv1d.bias", OperandRole::ConvQkvConv1dBias),
    ("mixer.out_proj.weight", OperandRole::ConvQkvOutProj),
    ("norm.weight", OperandRole::Mamba2PreMixerNorm),
];

/// Split a stack tensor's object-relative name into its layer index and
/// the suffix every role table matches on. `None` when the name is not
/// layer-shaped, which is what makes a bare top-level tensor under a
/// stack binding a blocking fact rather than a layer-0 operand.
pub(super) fn layer_and_suffix(relative_name: &str) -> Option<(usize, &str)> {
    let (layer, suffix) = relative_name.split_once('.')?;
    Some((layer.parse().ok()?, suffix))
}

/// Classify one stack tensor under the component's declared residual
/// topology as well as its layer's operator.
///
/// **The op plan's entry point.** Two authorities gate a role here and
/// they answer different questions: the OPERATOR separates spellings two
/// attention families share (`self_attn.o_proj.weight` on a KDA layer
/// and on an MLA layer of the same checkpoint), and the TOPOLOGY decides
/// whether a residual programme's operands exist at all.
///
/// The topology cannot be inferred from the operands, and this is the
/// site where that rule is enforced: a checkpoint shipping
/// `mlp_res_norm.weight` without declaring `attn_res_block_size`
/// classifies as NOTHING here, exactly as it did before this rung, and
/// the op plan reports it as an operand implying an absent op. Grading
/// it a site role would let a component acquire a residual topology by
/// spelling — the same failure as reading hyper-connections off an
/// `hc_`-prefixed name.
pub fn classify_stack_tensor_under(
    relative_name: &str,
    operator: LayerOperator,
    topology: larql_models::config::ResidualTopology,
) -> Option<(usize, OperandRole)> {
    if matches!(
        topology,
        larql_models::config::ResidualTopology::AttentionResidual { .. }
    ) {
        if let Some((layer, suffix)) = layer_and_suffix(relative_name) {
            if let Some((_, role)) = ATTENTION_RESIDUAL_ROLE_TABLE
                .iter()
                .find(|(name, _)| *name == suffix)
            {
                return Some((layer, *role));
            }
        }
    }
    classify_stack_tensor_on(relative_name, operator)
}

/// Classify one stack tensor, given the operator its layer runs.
///
/// The operator is required, not optional, because a name alone cannot
/// answer for every checkpoint: Kimi Linear's `self_attn.o_proj.weight` is
/// `[2304, 4096]` on its KDA layers, `[2304, 4096]` on its MLA layers
/// (heads·v_head_dim = 32·128 = 4096, coincidentally equal to KDA's width
/// on THIS checkpoint — a shape check alone would not even prove the
/// two apart here), and would answer a THIRD width on a genuine softmax
/// layer. The graph's per-layer table is the only authority that can
/// separate them, which is what makes the interleave carriage (P3a) a
/// precondition for KDA/MLA operand binding rather than a nicety.
pub fn classify_stack_tensor_on(
    relative_name: &str,
    operator: LayerOperator,
) -> Option<(usize, OperandRole)> {
    let (layer, suffix) = layer_and_suffix(relative_name)?;
    if operator.is_kda() {
        if let Some((_, role)) = KDA_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if operator.is_mla() {
        if let Some((_, role)) = MLA_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if operator.is_mamba2() {
        if let Some((_, role)) = MAMBA2_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if operator.is_conv_qkv() {
        if let Some((_, role)) = CONV_QKV_ROLE_TABLE.iter().find(|(name, _)| *name == suffix) {
            return Some((layer, *role));
        }
    }
    if let Some(role) = ROLE_TABLE
        .iter()
        .find(|(name, _)| *name == suffix)
        .map(|(_, role)| *role)
    {
        return Some((layer, role));
    }
    // Indexed `PerExpert` operands last: every fixed spelling above is
    // tried first, so an exact-match family entry always wins over the
    // pattern-matcher on the same suffix.
    classify_indexed_expert(suffix).map(|role| (layer, role))
}

/// [`classify_stack_tensor_on`] for a layer running softmax attention.
///
/// Correct only for roles that cannot collide across operators — the norms
/// this crate's norm-placement evidence reads. Do **not** reach for it to
/// classify an attention operand: on a KDA layer it answers `AttnQ` for
/// the recurrence's query projection.
pub fn classify_stack_tensor(relative_name: &str) -> Option<(usize, OperandRole)> {
    classify_stack_tensor_on(relative_name, LayerOperator::Softmax)
}

/// Norm placement for a stack, from the roles present across its layers.
///
/// Fail-closed: both FFN-wrap norms or neither; per-layer norms must
/// exist at all. The error names what the evidence actually shows.
pub fn norm_placement_evidence<'a>(
    relative_names: impl Iterator<Item = &'a str>,
) -> Result<NormPlacement, String> {
    let mut pre_attention = false;
    let mut post_attention = false;
    let mut pre_ffn = false;
    let mut post_ffn = false;
    for name in relative_names {
        match classify_stack_tensor(name).map(|(_, role)| role) {
            Some(OperandRole::PreAttentionNorm) => pre_attention = true,
            Some(OperandRole::PostAttentionNorm) => post_attention = true,
            Some(OperandRole::PreFfnNorm) => pre_ffn = true,
            Some(OperandRole::PostFfnNorm) => post_ffn = true,
            _ => {}
        }
    }
    match (pre_attention, post_attention, pre_ffn, post_ffn) {
        (true, true, true, true) => Ok(NormPlacement::PrePost),
        (true, true, false, false) => Ok(NormPlacement::PreOnly),
        // Both wrap norms present, neither pre-norm: the sublayer reads
        // the raw residual. See [`NormPlacement::PostOnly`].
        (false, true, false, true) => Ok(NormPlacement::PostOnly),
        (false, false, false, false) => Err("stack carries no per-layer norm operands".to_string()),
        _ => Err(format!(
            "norm operand set is neither two-norm nor four-norm \
             (pre_attn {pre_attention}, post_attn {post_attention}, \
             pre_ffn {pre_ffn}, post_ffn {post_ffn})"
        )),
    }
}

/// Norm placement for a **mixer-only** (pure-SSM) stack, from its own
/// evidence: the single pre-mixer norm per layer, and no transformer
/// wrap norms beside it.
///
/// A separate function rather than a fourth tuple arm in
/// [`norm_placement_evidence`]: that function is operator-blind and would
/// have to read `(pre_attn, ..)` from a vocabulary the mixer's estate
/// never uses — a transformer stack that lost its FFN norms must keep
/// reading as the defect it is, never as a valid mixer placement. The
/// caller chooses this path only when the component's declared program is
/// mixer-only.
pub fn mixer_norm_placement_evidence<'a>(
    relative_names: impl Iterator<Item = &'a str>,
) -> Result<NormPlacement, String> {
    let mut pre_mixer = false;
    let mut transformer_norms = false;
    for name in relative_names {
        match classify_stack_tensor_on(name, LayerOperator::Mamba2).map(|(_, role)| role) {
            Some(OperandRole::Mamba2PreMixerNorm) => pre_mixer = true,
            Some(
                OperandRole::PreAttentionNorm
                | OperandRole::PostAttentionNorm
                | OperandRole::PreFfnNorm
                | OperandRole::PostFfnNorm,
            ) => transformer_norms = true,
            _ => {}
        }
    }
    match (pre_mixer, transformer_norms) {
        (true, false) => Ok(NormPlacement::PreMixer),
        (true, true) => Err(
            "stack carries a pre-mixer norm AND attention/FFN wrap norms — \
             not a mixer-only placement"
                .to_string(),
        ),
        (false, _) => Err("mixer-only stack carries no per-layer norm operands".to_string()),
    }
}
