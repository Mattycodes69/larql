//! Residual topology, position, span policy and MoE carriage rules.

use super::*;

pub(super) const STRUCTURE_RULES: &[CarriageRule] = &[
    // ── Residual topology (wave 19) ─────────────────────────────────
    //
    // The three Sinkhorn hyper-connection parameters are one declared
    // component fact, read together (a partial declaration refuses the
    // surface, not these rules), carried to the component's residual
    // topology, and lowered: the op plan carries it, and the decode step
    // and batch traversal both run the bundle it declares.
    CarriageRule {
        leaf: "hc_mult",
        reaches: Carriage::Lowered,
        site: "Component.execution.residual_topology (ResidualTopology::HyperConnection.streams) → ComponentOpPlan.residual_topology → the executor's bundle carrier",
        probe: Some(probe_hc_streams),
    },
    CarriageRule {
        leaf: "hc_sinkhorn_iters",
        reaches: Carriage::Lowered,
        site: "Component.execution.residual_topology (ResidualTopology::HyperConnection.sinkhorn_iters) → hc_split_sinkhorn's pass count",
        probe: Some(probe_hc_sinkhorn_iters),
    },
    CarriageRule {
        leaf: "hc_eps",
        reaches: Carriage::Lowered,
        site: "Component.execution.residual_topology (ResidualTopology::HyperConnection.sinkhorn_eps) → hc_split_sinkhorn's epsilon",
        probe: Some(probe_hc_eps),
    },
    // The attention-residual period (K3-ATTNRES-1). ONE declared
    // component fact, carried to the component's residual topology and
    // now all the way to a traversal that reads it. `Lowered` since
    // K3-ATTNRES-1: the executor's history carrier asks this period
    // which layers take a block-boundary snapshot, on the decode path
    // (2a) and the batch path (2b), each witnessed against a Torch
    // oracle transcribed from the reference. Before that it stopped at
    // `Represented` on purpose, because a `Lowered` claim would have
    // said a backend receives this period when nothing did.
    //
    // The COUNT does not move on this reader — the leaf was already
    // Representable/ExecutionSemantic and non-blocking — and that is
    // the point: a count that changed here would mean the stage name was
    // doing work it should not. The probe still reads the BUILT surface,
    // so a checkpoint whose surface does not build answers nothing here
    // and keeps its blocker, the lesson wave 19 learned on DeepSeek-V4.
    CarriageRule {
        leaf: "attn_res_block_size",
        reaches: Carriage::Lowered,
        site: "Component.execution.residual_topology (ResidualTopology::AttentionResidual.block_size) → the executor's attention-residual history carrier, which reads the period to decide which layers take a block-boundary snapshot (opplan::exec::attention_residual::is_block_boundary, on the decode and batch traversals alike)",
        probe: Some(probe_attn_res_block_size),
    },
    // ── Position ────────────────────────────────────────────────────
    CarriageRule {
        leaf: "rope_theta",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::Rope) → AttentionOp.position",
        probe: Some(probe_rope_theta),
    },
    CarriageRule {
        leaf: "partial_rotary_factor",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::{PartialRope,MRope}.rotary_fraction) → AttentionOp.position",
        probe: Some(probe_partial_rotary_factor),
    },
    CarriageRule {
        leaf: "layer_rope_theta",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position, per layer → AttentionOp.position",
        probe: Some(probe_layer_rope_theta),
    },
    CarriageRule {
        leaf: "rope_type",
        reaches: Carriage::Represented,
        // PositionPolicy is `Rope { theta } | Linear { theta, factor } |
        // Yarn { theta, scaling } | Llama3 { theta, scaling } | None`:
        // unscaled rotary, positions divided before rotation, YaRN-scaled
        // rotary (frequencies AND the attention amplitude), Llama-3
        // wavelength bands, or no position encoding. Any other declared
        // rope class (dynamic, ...) still has no variant and mismatches
        // here — represented, not lowered: the interpreter and the
        // lowering refuse a YaRN layer until A-9.3/A-9.4 execute it.
        site: "Component.attention[].position (PositionPolicy::Rope | Linear | Yarn | Llama3)",
        probe: Some(probe_rope_type),
    },
    // The scaling block's own leaves, each carried on the policy variant
    // its `rope_type` selects and answered from it. A checkpoint that
    // declares them without declaring a scaling `rope_type` gets no
    // answer, which is right — the leaves mean nothing outside a block.
    CarriageRule {
        leaf: "factor",
        reaches: Carriage::Represented,
        site: "Component.attention[].position ({Yarn,Llama3}.scaling.factor | Linear.factor)",
        probe: Some(probe_scaling_factor),
    },
    CarriageRule {
        leaf: "beta_fast",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.beta_fast)",
        probe: Some(probe_yarn_beta_fast),
    },
    CarriageRule {
        leaf: "beta_slow",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.beta_slow)",
        probe: Some(probe_yarn_beta_slow),
    },
    CarriageRule {
        leaf: "truncate",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.truncate)",
        probe: Some(probe_yarn_truncate),
    },
    CarriageRule {
        leaf: "original_max_position_embeddings",
        reaches: Carriage::Represented,
        site: "Component.attention[].position ({Yarn,Llama3}.scaling.original_max_position_embeddings)",
        probe: Some(probe_scaling_original_max),
    },
    CarriageRule {
        leaf: "type",
        reaches: Carriage::Represented,
        // The older HF spelling of `rope_type` (same discriminator, same
        // block) — same claim, same probe: `PositionPolicy` can only
        // express the unscaled class under this name too.
        site: "Component.attention[].position — PositionPolicy expresses unscaled rope only",
        probe: Some(probe_rope_type),
    },
    CarriageRule {
        leaf: "low_freq_factor",
        reaches: Carriage::Represented,
        // Llama-3 wavelength-band scaling, carried on its own policy
        // variant. It is NOT the YaRN convention `beta_fast`/`beta_slow`
        // above represent: llama3 adjusts frequencies by wavelength band
        // and leaves the amplitude at unity, where YaRN also rescales
        // every logit. Two conventions, two blocks, one probe each.
        site: "Component.attention[].position (PositionPolicy::Llama3.scaling.low_freq_factor)",
        probe: Some(probe_llama3_low_freq),
    },
    CarriageRule {
        leaf: "high_freq_factor",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Llama3.scaling.high_freq_factor)",
        probe: Some(probe_llama3_high_freq),
    },
    CarriageRule {
        leaf: "mscale",
        reaches: Carriage::Represented,
        // DeepSeek-style YaRN mscale extension — a different scaling
        // convention from HF's generic YaRN block above. No field exists;
        // always refuses.
        site: "no schema field — DeepSeek's mscale extension is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mscale_all_dim",
        reaches: Carriage::Represented,
        site: "no schema field — DeepSeek's mscale extension is not represented yet",
        probe: Some(probe_unrepresented),
    },
    // ── Span policy ─────────────────────────────────────────────────
    CarriageRule {
        leaf: "layer_types",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{GatedDelta,Softmax}",
        probe: Some(probe_layer_types),
    },
    CarriageRule {
        leaf: "sliding_window",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
    },
    // The window's ENABLE flag and its layer bound. Both are read by
    // `ModelArchitecture::sliding_window_size`, which resolves all three
    // declarations into one effective per-layer policy, and both are
    // persisted by the vindex config round-trip — so the container does
    // not lose them.
    //
    // `Parsed`, and that is the honest stage rather than a weak one: the
    // effect of both facts is fully ABSORBED into the resolved per-layer
    // window before a graph exists. `sliding_window_size` returns `None`
    // for a disabled window and `is_sliding_window_layer` applies the
    // bound, so what the container carries is the effective policy —
    // there is no separate flag downstream to read back, and a deeper
    // claim would need a probe the schema cannot answer.
    CarriageRule {
        leaf: "use_sliding_window",
        reaches: Carriage::Parsed,
        site: "absorbed by ModelArchitecture::sliding_window_size into the resolved \
               per-layer window the graph carries; also persisted by the vindex config \
               round-trip",
        probe: None,
    },
    // The positional scheme, answered from the graph rather than the
    // config, because on `granitemoehybrid` this key is the SWITCH: HF
    // builds a rotary embedding only when it reads `rope`, so a
    // checkpoint that omits it is a NoPE model. `Represented` and not
    // `Parsed` — the effect is visible on every layer's carried
    // PositionPolicy, so the container can be asked what it believes
    // rather than trusted to have read the key.
    // The rotary schedule, answered from the graph in the checkpoint's
    // own polarity so a declared mask and a carried one are comparable
    // term by term. `Represented`: each layer's PositionPolicy is what
    // the schedule produced, so the container can be asked rather than
    // trusted.
    CarriageRule {
        leaf: "no_rope_layers",
        reaches: Carriage::Represented,
        site: "Component.attention[].position — 1 where the layer rotates, 0 where it is NoPE,                the same polarity the checkpoint declares",
        probe: Some(probe_no_rope_layers),
    },
    // The fallback generator. `Parsed`, and honestly so: both references
    // consult it only when the mask is absent, so on a checkpoint that
    // declares both it is SUPERSEDED and contributes nothing to the
    // graph. Same shape as `max_window_layers` being inert while the
    // window is disabled.
    // The rotary PAIRING. No reference implementation reads this key, so
    // there is no upstream behaviour to match — only this build's, which
    // is split-half and uniform. `Represented` because the answer comes
    // from the executor's own declared pairing rather than from the
    // config that was just read.
    CarriageRule {
        leaf: "rope_interleaved",
        reaches: Carriage::Represented,
        site: "larql-compute rotates (x[i], x[i + half]) — split-half, so an interleaved \
               pairing is a different operator and mismatches",
        probe: Some(probe_rope_interleaved),
    },
    // The multi-axis flag, checked against the policy actually resolved
    // from `mrope_section` + `mrope_interleaved` rather than against
    // itself.
    CarriageRule {
        leaf: "use_mrope",
        reaches: Carriage::Represented,
        site: "Component.attention[].position — PositionPolicy::MRope when the axis geometry                resolves one, false otherwise",
        probe: Some(probe_use_mrope),
    },
    // A claim about which family serves the checkpoint. Answered from the
    // registry's resolution of the declared identity, never from the flag.
    CarriageRule {
        leaf: "is_llama_config",
        reaches: Carriage::Represented,
        site: "the registry entry the declared model_type resolved to — true when it is the \
               Llama family, false for any other or none",
        probe: Some(probe_is_llama_config),
    },
    CarriageRule {
        leaf: "no_rope_layer_interval",
        reaches: Carriage::Parsed,
        site: "absorbed by ModelArchitecture::position_policy_for_layer as the schedule when                no_rope_layers is absent; superseded by an explicit mask, as upstream supersedes it",
        probe: None,
    },
    CarriageRule {
        leaf: "position_embedding_type",
        reaches: Carriage::Represented,
        site: "Component.attention[].position — a rotating policy answers `rope`, a stack \
               that rotates nowhere answers null",
        probe: Some(probe_position_embedding_type),
    },
    CarriageRule {
        leaf: "max_window_layers",
        reaches: Carriage::Parsed,
        site: "absorbed by ModelArchitecture::is_sliding_window_layer as the bound on an \
               enabled window; also persisted by the vindex config round-trip",
        probe: None,
    },
    // Inkling-Small's spelling of the same window. One site, because it
    // is one fact: the graph carries a window per layer whichever key
    // stated it.
    CarriageRule {
        leaf: "sliding_window_size",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
    },
    // The index-set spelling of the per-layer topology, carried to the
    // same place `layer_types` is — which is the claim worth testing: two
    // very different declarations reaching one canonical policy.
    CarriageRule {
        leaf: "local_layer_ids",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{Kda,GatedDelta,Softmax}",
        // An index SET, compared by cardinality against the resolved
        // table — the array probe would render a `layer_types` array and
        // never equal the declared set of indices.
        probe: Some(probe_sliding_layer_set),
    },
    CarriageRule {
        leaf: "d_rel",
        reaches: Carriage::Represented,
        site: "Component.attention[].position → PositionPolicy::Relative",
        probe: Some(probe_relative_d_rel),
    },
    CarriageRule {
        leaf: "rel_extent",
        reaches: Carriage::Represented,
        site: "Component.attention[].position → PositionPolicy::Relative",
        probe: Some(probe_relative_extent),
    },
    // ── MoE facts, in every spelling that reaches one surface ────────
    CarriageRule {
        leaf: "moe_renormalize",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.routing_policy",
        probe: Some(probe_moe_routing_policy),
    },
    CarriageRule {
        leaf: "num_shared_experts",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.shared_experts",
        probe: Some(probe_moe_shared_experts),
    },
    // The same branch's WIDTH, which two lineages state two ways. The
    // container carries the resolved width, so this is checked against
    // what the branch will actually be built at — not echoed back.
    CarriageRule {
        leaf: "shared_expert_intermediate_size",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.shared_expert_intermediate_size → SharedExpertOp.intermediate_size (and the shared-expert operand shapes)",
        probe: Some(probe_shared_expert_width),
    },
    CarriageRule {
        leaf: "moe_shared_expert_intermediate_size",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.shared_expert_intermediate_size → SharedExpertOp.intermediate_size (and the shared-expert operand shapes)",
        probe: Some(probe_shared_expert_width),
    },
    // Kimi-K3's latent routed branch (K3-LATENTMOE-1). `Lowered`, and
    // the claim is exact: the width reaches `LatentBranchOp.width`, where
    // it is BOTH the geometry every routed-bank shape contract is sized
    // from and the width the two wrapper projections are bound at — so a
    // build that stored the number and kept sizing the bank from
    // `hidden_size` would fail this rule's probe and the op plan
    // together, rather than reporting a fact it does not honour.
    //
    // The domain was measured before the rule was promised: exactly one
    // of the 117 conformance rows declares either leaf. The previous
    // rung's `q_lora_rank` rule was withdrawn for the opposite reason —
    // it reached eighteen rows of which only six built the surface it
    // named — and that withdrawal is why this one is checked first.
    CarriageRule {
        leaf: "routed_expert_hidden_size",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.latent.width → LatentBranchOp.width, and through MoeSurface::routed_expert_input_width every routed expert-bank shape contract",
        probe: Some(probe_routed_expert_width),
    },
    CarriageRule {
        leaf: "latent_moe_use_norm",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.latent.norm → LatentBranchOp.norm — the RMS norm on the weighted aggregate, between summation and the up-projection",
        probe: Some(probe_latent_moe_use_norm),
    },
    CarriageRule {
        leaf: "moe_router_activation_func",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.router_kind",
        probe: Some(probe_moe_router_kind),
    },
    CarriageRule {
        leaf: "scoring_func",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.router_kind",
        probe: Some(probe_moe_router_kind),
    },
    CarriageRule {
        leaf: "moe_layer_freq",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — every layer after the dense prefix is routed",
        probe: Some(probe_identity_valued),
    },
    // Expert grouping. At one group the router selects over every expert,
    // which is what an ungrouped router does — so the schema represents
    // its effect exactly, by having none. Any other value is a real
    // grouping this schema cannot state, and refuses.
    CarriageRule {
        leaf: "num_expert_group",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — one group is ungrouped routing",
        probe: Some(probe_identity_valued),
    },
    CarriageRule {
        leaf: "n_group",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — one group is ungrouped routing",
        probe: Some(probe_identity_valued),
    },
    CarriageRule {
        leaf: "topk_group",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — one group is ungrouped routing",
        probe: Some(probe_identity_valued),
    },
    CarriageRule {
        leaf: "use_grouped_topk",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — grouping is a no-op at one group",
        probe: Some(probe_grouping_is_a_no_op),
    },
];
