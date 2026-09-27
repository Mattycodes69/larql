//! Hybrid linear-attention, Mamba2/SSD and conv-QKV carriage rules.

use super::*;

pub(super) const HYBRID_RULES: &[CarriageRule] = &[
    // ── Hybrid linear-attention + multi-token-prediction (declared, not
    //    yet executed — R2/Kimi-Linear rung, see docs/k3-funnel.md) ──
    //
    // No `AttentionOp` variant computes a linear-attention layer and no
    // MTP-head object exists in the schema, so every one of these always
    // refuses via the shared `probe_unrepresented` — the same idiom
    // `norm_topk_prob`/`high_freq_factor` above use for "no schema field
    // yet". Each still gets its own rule (rather than falling through
    // `carriage_finding`'s generic no-rule message) so
    // `every_execution_semantic_leaf_has_a_carriage_rule` covers it: a
    // future field added to the registry without a rule fails there
    // before it fails on a checkpoint.
    CarriageRule {
        leaf: "linear_conv_kernel_dim",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.conv_kernel → GatedDeltaOp.conv_kernel",
        probe: Some(probe_linear_conv_kernel),
    },
    CarriageRule {
        leaf: "linear_key_head_dim",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.key_head_dim → GatedDeltaOp.key_head_dim",
        probe: Some(probe_linear_key_head_dim),
    },
    CarriageRule {
        leaf: "linear_value_head_dim",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.value_head_dim → GatedDeltaOp.value_head_dim",
        probe: Some(probe_linear_value_head_dim),
    },
    CarriageRule {
        leaf: "linear_num_key_heads",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.key_heads → GatedDeltaOp.num_key_heads",
        probe: Some(probe_linear_key_heads),
    },
    CarriageRule {
        leaf: "linear_num_value_heads",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.value_heads → GatedDeltaOp.num_value_heads",
        probe: Some(probe_linear_value_heads),
    },
    CarriageRule {
        leaf: "mamba_ssm_dtype",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.state_dtype → GatedDeltaState precision",
        probe: Some(probe_linear_state_dtype),
    },
    // ── Mamba2/SSD mixer geometry and switches (schema 6). Represented,
    //    not Lowered: the surface holds every fact and no executor
    //    consumes it yet — claiming Lowered would assert an operator that
    //    does not exist (the same honesty `mamba_ssm_dtype` held to until
    //    QW-2's reference operator landed). ──
    CarriageRule {
        leaf: "state_size",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.state_size",
        probe: Some(probe_mamba2_state_size),
    },
    CarriageRule {
        leaf: "expand",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.expand",
        probe: Some(probe_mamba2_expand),
    },
    CarriageRule {
        leaf: "conv_kernel",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.conv_kernel",
        probe: Some(probe_mamba2_conv_kernel),
    },
    CarriageRule {
        leaf: "n_groups",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.n_groups",
        probe: Some(probe_mamba2_n_groups),
    },
    CarriageRule {
        leaf: "chunk_size",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.chunk_size",
        probe: Some(probe_mamba2_chunk_size),
    },
    CarriageRule {
        leaf: "time_step_limit",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.dt_limit_{min,max} — the judged \
               non-finite boundary: a bare `Infinity` is carried as a declared \
               unbounded side, never a fabricated float",
        probe: Some(probe_mamba2_time_step_limit),
    },
    CarriageRule {
        leaf: "rms_norm",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.rms_norm — the mixer's gated RMSNorm",
        probe: Some(probe_mamba2_rms_norm),
    },
    CarriageRule {
        leaf: "use_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.use_bias (closure-paired with the \
               in/out projection bias operands)",
        probe: Some(probe_mamba2_use_bias),
    },
    CarriageRule {
        leaf: "use_conv_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.use_conv_bias (closure-paired with \
               the conv bias operand)",
        probe: Some(probe_mamba2_use_conv_bias),
    },
    // ── The mamba_ssm key dialect (OuteAI Mamba2Attn): three renamed
    //    geometry keys and the projection-bias switch, read into the SAME
    //    `Mamba2Geometry` fields their HF twins fill — so each probe
    //    answers from the same surface site. ──
    CarriageRule {
        leaf: "mamba2_num_heads",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.num_heads",
        probe: Some(probe_mamba2_num_heads),
    },
    CarriageRule {
        leaf: "mamba2_head_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.head_dim",
        probe: Some(probe_mamba2_head_dim),
    },
    CarriageRule {
        leaf: "mamba2_conv_kernel",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.conv_kernel",
        probe: Some(probe_mamba2_conv_kernel),
    },
    CarriageRule {
        leaf: "use_mamba2_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.use_bias (closure-paired with the \
               in/out projection bias operands)",
        probe: Some(probe_mamba2_use_bias),
    },
    // ── The hybrid's conv-QKV attention block. Represented, not
    //    Lowered: the surface holds every fact and no executor consumes
    //    it yet — the same honesty the Mamba2 rules held to until the
    //    reference operator landed. ──
    CarriageRule {
        leaf: "attention_head_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.head_dim",
        probe: Some(probe_conv_qkv_head_dim),
    },
    CarriageRule {
        leaf: "attention_conv_kernel",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.conv_kernel",
        probe: Some(probe_conv_qkv_conv_kernel),
    },
    CarriageRule {
        leaf: "rope_emb_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.rotary_dim — the partial-rotary width, also \
               carried per layer as PositionPolicy::PartialRope",
        probe: Some(probe_conv_qkv_rotary_dim),
    },
    CarriageRule {
        leaf: "use_attention_qkv_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.qkv_bias — a declared-FALSE is carried; a \
               declared-TRUE has no judged bias role yet and must block",
        probe: Some(probe_conv_qkv_qkv_bias),
    },
    CarriageRule {
        leaf: "use_attention_out_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.out_bias — same contract as the QKV bias \
               switch",
        probe: Some(probe_conv_qkv_out_bias),
    },
    CarriageRule {
        leaf: "attention_layers_idx",
        reaches: Carriage::Represented,
        site: "Component.attention[] — the per-layer operator table; the declared set \
               is echoed only when the table's conv-QKV layers correspond to it \
               under a consistent index base",
        probe: Some(probe_attention_layer_idx),
    },
    CarriageRule {
        leaf: "attn_layer_idx",
        reaches: Carriage::Represented,
        site: "Component.attention[] — the state-spaces spelling of the same set",
        probe: Some(probe_attention_layer_idx),
    },
    // ── The mamba_ssm lineage's MLP declaration. ──
    CarriageRule {
        leaf: "mlp_intermediate_size",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn presence per layer — 0 declares NO MLP blocks, \
               carried as every layer's absent FFN op; a non-zero width has no \
               judged lowering yet and must block",
        probe: Some(probe_mlp_intermediate_size),
    },
    CarriageRule {
        leaf: "mlp_padding_size",
        reaches: Carriage::Represented,
        site: "no schema field — pads an MLP width; inert exactly when \
               mlp_intermediate_size declares 0 (no MLP exists to pad), blocking \
               otherwise",
        probe: Some(probe_mlp_padding_size),
    },
    CarriageRule {
        leaf: "use_mlp_bias",
        reaches: Carriage::Represented,
        site: "no schema field — biases an MLP; inert exactly when \
               mlp_intermediate_size declares 0, blocking otherwise",
        probe: Some(probe_mlp_padding_size),
    },
    // ── The mamba_ssm-native nested spellings. ──
    CarriageRule {
        leaf: "layer",
        reaches: Carriage::Represented,
        site: "Component.attention[].operator — `ssm_cfg.layer` names the layer class; \
               \"Mamba2\" is represented as the mixer operator, and any other class \
               finds no surface and blocks",
        probe: Some(probe_ssm_layer_class),
    },
    CarriageRule {
        leaf: "d_conv",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.{conv_qkv,mamba2}.conv_kernel — whichever block declared \
               it; a width matching neither blocks",
        probe: Some(probe_declared_conv_kernel),
    },
    CarriageRule {
        leaf: "d_state",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.state_size — the ssm_cfg spelling",
        probe: Some(probe_mamba2_state_size),
    },
    CarriageRule {
        leaf: "headdim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.head_dim — the ssm_cfg spelling",
        probe: Some(probe_mamba2_head_dim),
    },
    CarriageRule {
        leaf: "ngroups",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.n_groups — the ssm_cfg spelling",
        probe: Some(probe_mamba2_n_groups),
    },
    CarriageRule {
        leaf: "rotary_emb_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.rotary_dim",
        probe: Some(probe_conv_qkv_rotary_dim),
    },
    CarriageRule {
        leaf: "qkv_proj_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.qkv_bias — declared-FALSE carried; TRUE blocks",
        probe: Some(probe_conv_qkv_qkv_bias),
    },
    CarriageRule {
        leaf: "out_proj_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.out_bias — same contract",
        probe: Some(probe_conv_qkv_out_bias),
    },
    CarriageRule {
        leaf: "causal",
        reaches: Carriage::Represented,
        site: "the conv-QKV operator's masking — causal by construction; a declared \
               non-causal block has no operator and blocks",
        probe: Some(probe_attn_causal),
    },
    CarriageRule {
        leaf: "d_intermediate",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn presence per layer — mamba_ssm's own spelling of \
               mlp_intermediate_size; 0 declares NO MLP blocks",
        probe: Some(probe_mlp_intermediate_size),
    },
    CarriageRule {
        leaf: "residual_in_fp32",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.residual_in_fp32 — residual-stream precision, declared",
        probe: Some(probe_residual_in_fp32),
    },
    CarriageRule {
        leaf: "attn_output_gate",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.output_gate → GateOp → the gated attention op",
        probe: Some(probe_attn_output_gate),
    },
    // MLA's output gate (K3-REP-GATE-1): the same generic gate the softmax
    // rule above carries, on the MLA surface. Lowered: the op carries the
    // gate operand, and the executor gates the aggregated value before
    // `o_proj`. The probe answers from the BUILT surface, as the softmax
    // one does — a declared `false` reads as "no gate", which is what the
    // surface says, so declaration and carriage agree on both values.
    CarriageRule {
        leaf: "mla_use_output_gate",
        reaches: Carriage::Lowered,
        site: "MlaSurface.output_gate (AttentionGateSpec) → MlaOp.output_gate → exec::mla gated_value",
        probe: Some(probe_mla_use_output_gate),
    },
    CarriageRule {
        leaf: "output_gate_type",
        reaches: Carriage::Represented,
        site: "no schema field — the gate IS represented (see attn_output_gate); \
               what is unresolved is whether THIS key describes it",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mtp_num_hidden_layers",
        reaches: Carriage::Represented,
        site: "no schema field — the multi-token-prediction head is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mtp_use_dedicated_embeddings",
        reaches: Carriage::Represented,
        site: "no schema field — the multi-token-prediction head is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mrope_interleaved",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::MRope.interleaved) → mrope_axis_table → mrope_rotate",
        probe: Some(probe_mrope_interleaved),
    },
    CarriageRule {
        leaf: "mrope_section",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::MRope.section) → mrope_axis_table → mrope_rotate",
        probe: Some(probe_mrope_section),
    },
];
