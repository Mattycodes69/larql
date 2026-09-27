//! How far a declared fact travels — the VINDEX3-boundary authority gate.
//!
//! The inventory answers *did a parser read this key?*
//! ([`KeyStatus::Consumed`](larql_models::inventory::KeyStatus::Consumed)).
//! The plan used to treat that answer as *can VINDEX3 represent this
//! fact?* — a different question about a different object, and the gap
//! between them is silent by construction: a fact the parser reads into
//! `ModelConfig` and VINDEX3 then drops looks fully covered from the
//! plan's side.
//!
//! GPT-OSS is the witness. It declares `rope_scaling = {rope_type:
//! "yarn", factor: 32}` for a 131k context. Every one of those leaves
//! classifies `consumed` — the parser genuinely reads them. But
//! [`PositionPolicy`] expresses `Rope { theta } | None` and nothing
//! else, and no other field under `format/vindex3/` carries a scaling
//! block, so the model would plan, encode and execute as **plain rope at
//! θ=150000**, with the plan reporting no defect at all. (VINDEX1/2 do
//! carry it, as raw JSON — so this is a regression the older path does
//! not have.)
//!
//! ```text
//! config.json fact
//!    ↓  parsed        larql-models' parser stored it in ModelConfig
//!    ↓  represented   the VINDEX3 system graph persists it
//!    ↓  lowered       it reaches the generic op plan as an op parameter
//!    ↓  executed      an executor reads that op parameter
//! ```
//!
//! Each execution-semantic key needs a [`CarriageRule`] declaring which
//! of those stages it reaches. Rules claiming [`Carriage::Represented`]
//! or deeper carry a **probe** that reads the value back off the built
//! graph, so the claim is checked against the schema rather than
//! trusted; a probe that disagrees with the declaration blocks. Rules
//! that honestly stop at [`Carriage::Parsed`] must say why, and are
//! reported rather than hidden. A key with **no rule at all** blocks —
//! that is the state this module exists to abolish.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use larql_models::config::score_scale_from_query_pre_attn_scalar;

use super::super::graph::policy::AttentionSpan;
use super::super::graph::Component;

mod rules_core;
mod rules_hybrid;
mod rules_surface;
use rules_core::STRUCTURE_RULES;
use rules_hybrid::HYBRID_RULES;
use rules_surface::SURFACE_RULES;

mod probes_attention;
mod probes_layers;
use probes_attention::*;
use probes_layers::*;

/// How far a declared fact travels from `config.json` into execution.
///
/// Ordered: a deeper stage implies every shallower one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Carriage {
    /// A registered parser read the key into `ModelConfig`. This is what
    /// the inventory's `consumed` status means, and on its own it is not
    /// evidence of anything downstream.
    Parsed,
    /// The VINDEX3 system graph persists it: a container round-trips the
    /// fact, so encoding does not lose it.
    Represented,
    /// It reaches the generic op plan as an op parameter, so a backend
    /// receives it rather than re-deriving it.
    Lowered,
    /// An executor reads that op parameter on the path under test.
    Executed,
}

impl Carriage {
    /// The stage name as the report prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Parsed => "parsed",
            Self::Represented => "represented",
            Self::Lowered => "lowered",
            Self::Executed => "executed",
        }
    }
}

/// What VINDEX3 claims about one execution-semantic config leaf, and the
/// means of checking the claim.
pub struct CarriageRule {
    /// Flattened config leaf name this rule governs (`rope_type`), matched
    /// after the container path — `text_config.rope_parameters.rope_type`
    /// and `rope_scaling.rope_type` share one rule, because they are the
    /// same fact under two spellings.
    pub leaf: &'static str,
    /// The deepest stage VINDEX3 carries this fact to.
    pub reaches: Carriage,
    /// Where in the schema it lands (or why it stops), printed in the
    /// finding so a reader never has to grep for the answer.
    pub site: &'static str,
    /// Reads the carried value back off the built component. `None` when
    /// the component cannot answer (no surface, no attention table); the
    /// gate then reports carriage without a value comparison rather than
    /// inventing a disagreement.
    ///
    /// Required for [`Carriage::Represented`] and deeper, and unused for
    /// [`Carriage::Parsed`] — a rule that stops at the parser has nothing
    /// to read back.
    pub probe: Option<fn(&Component, &ProbeContext<'_>) -> Option<Value>>,
}

/// What a probe may know about the fact it is answering for, beyond the
/// component: the attention span the fact's path names, when a family
/// declares a fact per layer TYPE (`rope_parameters.full_attention.*` vs
/// `rope_parameters.sliding_attention.*` — Gemma 3/4), and the declared
/// value, so a probe can answer in the checkpoint's own spelling when
/// several spellings name one judged variant (`gelu_pytorch_tanh` and
/// `gelu_new` are both `Activation::GeluTanh`). A probe never lets the
/// declared value *choose* what it reports — it only resolves aliases of
/// what the schema already holds.
pub struct ProbeContext<'a> {
    pub span: Option<AttentionSpan>,
    pub declared: &'a Value,
    /// The registry label the component's declared identity resolved to,
    /// `None` when no family matched. For the probes that judge a claim
    /// about WHICH family serves the checkpoint, so they answer from the
    /// resolution rather than from the flag.
    pub family: Option<&'a str>,
}

impl ProbeContext<'_> {
    /// The per-layer-type scope a flattened config path names, if any.
    pub fn span_of(path: &str) -> Option<AttentionSpan> {
        [
            AttentionSpan::Full,
            AttentionSpan::Sliding,
            AttentionSpan::Windowed,
        ]
        .into_iter()
        .find(|span| {
            path.split('.')
                .any(|segment| segment == span.declared_name())
        })
    }
}

/// A declaration whose effect another declaration switches off.
///
/// Distinct from [`super::semantics::INERT_AT_VALUE`], which asks whether a
/// key holds an inert value of its own. Here the key holds a perfectly
/// real value and a *companion* says not to use it — so no probe of the
/// graph can settle it, because the graph is right to carry nothing.
///
/// Qwen2.5 is the witness: `sliding_window: 32768` beside
/// `use_sliding_window: false`. VINDEX3 resolves no window, which is
/// correct, and the carriage rule read that agreement as a dropped fact
/// and refused the checkpoint over a window it had been told not to
/// apply.
pub struct CompanionGate {
    /// The leaf whose value is inert while the switch is off.
    pub leaf: &'static str,
    /// The leaf that switches it off, at the same nesting level.
    pub switch: &'static str,
    /// The switch value that disables it.
    pub off: bool,
}

/// The gates. Each entry is a claim that one declaration cancels another,
/// and needs the same justification as any other rule in this file.
pub const COMPANION_GATES: &[CompanionGate] = &[CompanionGate {
    leaf: "sliding_window",
    switch: "use_sliding_window",
    off: false,
}];

/// The switch that disables `path`, if one is declared beside it and set
/// to its off value.
///
/// The companion is looked up at the SAME nesting level — `text_config.
/// sliding_window` is gated by `text_config.use_sliding_window`, never by
/// a root-level switch belonging to another component. A gate that
/// reached across components would let one tower's flag silence another's
/// window.
pub fn disabled_by_companion<'a>(
    path: &str,
    keys: impl IntoIterator<Item = (&'a str, &'a Value)>,
) -> Option<&'static str> {
    let gate = COMPANION_GATES
        .iter()
        .find(|g| super::semantics::leaf_of(path) == g.leaf)?;
    let prefix = &path[..path.len() - gate.leaf.len()];
    let switch_path = format!("{prefix}{}", gate.switch);
    keys.into_iter()
        .find(|(p, _)| *p == switch_path)
        .filter(|(_, v)| v.as_bool() == Some(gate.off))
        .map(|_| gate.switch)
}

/// The rules. Every leaf classified
/// [`ExecutionSemantic`](super::report::SemanticClass::ExecutionSemantic)
/// must appear here or block.
///
/// Adding a key here is a claim about the VINDEX3 schema, not about the
/// parser — which is the whole point of the module.
pub const CARRIAGE_RULE_GROUPS: &[&[CarriageRule]] =
    &[STRUCTURE_RULES, SURFACE_RULES, HYBRID_RULES];

/// Every rule, in table order.
pub fn carriage_rules() -> impl Iterator<Item = &'static CarriageRule> {
    CARRIAGE_RULE_GROUPS.iter().flat_map(|group| group.iter())
}

/// The rule governing a config leaf, if any.
pub fn rule_for(leaf: &str) -> Option<&'static CarriageRule> {
    carriage_rules().find(|rule| rule.leaf == leaf)
}

/// Canonicalises a declared config value into the vocabulary a probe's
/// carried value uses, for leaves where VINDEX3 legitimately stores a
/// *derived* form of the same fact rather than the checkpoint's own
/// spelling.
///
/// This is not a tolerance knob: the one arm here reuses the identical
/// formula the runtime already applies
/// ([`score_scale_from_query_pre_attn_scalar`]), so agreement means the
/// same fact was recognised twice by the same rule, not that comparison
/// was loosened. A leaf with no arm here falls through unchanged, so
/// [`super::values_agree`] still requires byte-for-byte (or f32-precision)
/// identity — this function only ever narrows a `mismatched` finding to
/// `representable`, never the reverse, and callers still show the raw
/// declared value in the finding regardless of what this returns.
///
/// `hidden_act`/`hidden_activation` used to have an arm here too, but
/// [`probe_activation`] now resolves that alias itself (via
/// [`ProbeContext::declared`], returning the checkpoint's own spelling on
/// a match) — canonicalising *both* sides at once made them disagree in
/// opposite directions (`"gelu_pytorch_tanh"` vs `"gelu_tanh"`) rather
/// than agree. One rule owns each fact's normalisation, never two.
pub fn canonical_declared(leaf: &str, declared: &Value) -> Value {
    match leaf {
        // The checkpoint declares the raw scalar; VINDEX3's execution
        // surface stores the score scale execution actually reads —
        // `scalar.powf(-0.5)`, the identical formula
        // `ModelArchitecture::attention_scale` applies at runtime, called
        // through the one shared function rather than re-derived here.
        "query_pre_attn_scalar" => declared
            .as_f64()
            .map(|scalar| json!(score_scale_from_query_pre_attn_scalar(scalar)))
            .unwrap_or_else(|| declared.clone()),
        _ => declared.clone(),
    }
}

// ── Probes ──────────────────────────────────────────────────────────
//
// Each reads what the *built graph* holds, so a rule's claim is checked
// against the schema rather than believed. They return `None` when the
// component has no surface or table to answer from.

/// The layers a per-layer-type fact speaks for: those of the span the
/// fact's path names, or every layer for a checkpoint-wide fact.
fn layers_in_scope<'a>(
    component: &'a Component,
    ctx: &ProbeContext<'_>,
) -> Option<impl Iterator<Item = &'a super::super::graph::AttentionLayerPolicy>> {
    let table = component.attention.as_ref()?;
    let span = ctx.span;
    Some(
        table
            .iter()
            .filter(move |l| span.is_none_or(|s| l.span == Some(s))),
    )
}

/// Shared by every rule for a fact VINDEX3 has no schema field for yet:
/// always refuses, so the fact honestly blocks (`Unrepresented`, with the
/// rule's own `site` text naming why) rather than falling through the
/// generic no-rule message. Never returns `Some` — a rule using this probe
/// makes no claim this function could get wrong.
fn probe_unrepresented(_component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    None
}

/// The uniform rope base across the layers in scope, when there is one:
/// the whole table for `rope_theta`, one layer type for
/// `rope_parameters.full_attention.rope_theta` (Gemma 4 declares 1e6 on
/// its full layers and 1e4 on its sliding ones — two facts, two probes).
/// A per-layer split (Muse-Glimmer's `layer_rope_theta`) answers `None`
/// here and is checked by [`probe_layer_rope_theta`] instead.
/// The declared hyper-connection topology, when the component carries
/// one; `None` on a single stream (the leaf would not be declared) and
/// on a component with no surface.
fn probe_hc(component: &Component) -> Option<larql_models::config::HyperConnection> {
    match component.execution.as_ref()?.residual_topology {
        larql_models::config::ResidualTopology::HyperConnection(hc) => Some(hc),
        larql_models::config::ResidualTopology::SingleStream
        | larql_models::config::ResidualTopology::AttentionResidual { .. } => None,
    }
}

fn probe_hc_streams(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(probe_hc(component)?.streams))
}

fn probe_hc_sinkhorn_iters(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(probe_hc(component)?.sinkhorn_iters))
}

fn probe_hc_eps(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(probe_hc(component)?.sinkhorn_eps))
}

/// The declared attention-residual period, read back off the BUILT
/// surface. `None` on any other topology (the leaf would not be
/// declared) and on a component with no surface — which is the honest
/// answer, and the one that keeps a row blocked until its surface builds.
fn probe_attn_res_block_size(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    match component.execution.as_ref()?.residual_topology {
        larql_models::config::ResidualTopology::AttentionResidual { block_size } => {
            Some(json!(block_size))
        }
        larql_models::config::ResidualTopology::SingleStream
        | larql_models::config::ResidualTopology::HyperConnection(_) => None,
    }
}

fn probe_rope_theta(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    // Nothing in scope rotates: the declared base is inert, and reporting
    // it as uncarried would demand a rotation the model does not perform.
    // See the matching arm in `plan::compare::rope_theta_findings`.
    if layers_in_scope(component, ctx)?
        .all(|l| l.position == larql_models::config::PositionPolicy::None)
    {
        return Some(ctx.declared.clone());
    }
    let mut thetas = layers_in_scope(component, ctx)?.filter_map(|l| l.position.rope_theta());
    let first = thetas.next()?;
    thetas.all(|t| t == first).then(|| json!(first))
}

/// Every layer's rope base in layer order, with NoPE layers as `0` —
/// the same sentinel spelling the checkpoints use.
fn probe_layer_rope_theta(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.position.rope_theta().unwrap_or(0.0)))
            .collect(),
    ))
}

/// This build's rotary pairing.
///
/// Constant because the executor has exactly one pairing — split-half,
/// `(x[i], x[i + half])` — and the point of answering at all is that a
/// checkpoint declaring the interleaved pairing gets a mismatch instead
/// of a rotation performed against different partners in silence. The
/// value comes from [`ROPE_PAIRING_INTERLEAVED`], which
/// `larql-compute`'s own gate pins to the executor, so this cannot drift
/// away from what actually runs.
fn probe_rope_interleaved(_component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(larql_models::config::ROPE_PAIRING_INTERLEAVED))
}

/// Whether the resolved policy is multi-axis rotary.
///
/// Answered from the policy the axis geometry produced, never from the
/// flag itself — a probe that echoed `use_mrope` back would agree with
/// every checkpoint including one declaring `true` with no
/// `mrope_section` to build it from.
fn probe_use_mrope(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let mut layers = layers_in_scope(component, ctx)?;
    Some(json!(layers.any(|l| matches!(
        l.position,
        larql_models::config::PositionPolicy::MRope { .. }
    ))))
}

/// The rotary schedule the graph carries, in the checkpoint's polarity.
///
/// `1` where the layer rotates and `0` where it does not — deliberately
/// the declared spelling and not a boolean, so the probe's answer and
/// the declaration are comparable element by element. Emitting the
/// natural-language polarity instead would make every SmolLM3 look
/// mismatched while being right.
fn probe_no_rope_layers(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(i64::from(l.position.rope_theta().is_some())))
            .collect(),
    ))
}

/// The positional scheme the graph actually carries.
///
/// `rope` when any layer in scope rotates; `null` when none does, which
/// is an ANSWER and not a failure to answer — "this stack encodes no
/// position" is exactly what a `granitemoehybrid` without the opt-in
/// means, and reporting it as unknown would hide the case the rule
/// exists for. Mirrors `probe_sliding_window`, which answers null the
/// same way for a stack with no windowed layer.
fn probe_position_embedding_type(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let mut layers = layers_in_scope(component, ctx)?;
    Some(match layers.any(|l| l.position.rope_theta().is_some()) {
        true => json!(larql_models::config::POSITION_EMBEDDING_TYPE_ROPE),
        false => Value::Null,
    })
}

/// The rope *class* the layers in scope carry, in the checkpoint's own
/// spelling: `yarn` when any rotating layer holds a YaRN block,
/// `proportional` when any holds a head-width-basis partial rotary
/// (Gemma 4's full layers), else `default`. Within one scope the class
/// is uniform, so the first classed layer answers for all.
fn probe_rope_type(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let mut layers = layers_in_scope(component, ctx)?;
    let class = layers
        .find_map(|l| l.position.declared_rope_type())
        .unwrap_or(larql_models::config::ROPE_TYPE_DEFAULT);
    Some(json!(class))
}

/// The KV-sharing count the table represents: none. Every layer in the
/// graph projects its own K/V, so the only declaration the schema agrees
/// with is `0`.
fn probe_kv_shared_layers(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component.attention.as_ref()?;
    Some(json!(0))
}

/// Whether any layer takes V from its K projection.
fn probe_k_eq_v(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(json!(table.iter().any(|l| l.v_from_k)))
}

fn probe_moe_enabled(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .ffn
        .as_ref()?
        .moe
        .is_some()))
}

fn probe_moe_top_k(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.ffn.as_ref()?.moe?.top_k
    ))
}

/// The head width the full-attention layers carry — the fact
/// `global_head_dim` declares — when every full layer agrees. A layer
/// without its own geometry has the surface's (that is what the absence
/// means), so a uniform tower answers with its surface head width.
fn probe_full_layer_head_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let surface = component.execution.as_ref()?;
    let mut dims = table
        .iter()
        .filter(|l| l.span == Some(AttentionSpan::Full))
        .map(|l| {
            l.geometry
                .map_or(surface.attention.as_ref().map(|a| a.head_dim), |g| {
                    Some(g.head_dim)
                })
        });
    let first = dims.next()??;
    dims.all(|d| d == Some(first)).then(|| json!(first))
}

/// The KV-head count the full-attention layers carry.
fn probe_full_layer_kv_heads(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let surface = component.execution.as_ref()?;
    let mut heads = table
        .iter()
        .filter(|l| l.span == Some(AttentionSpan::Full))
        .map(|l| {
            l.geometry
                .map_or(surface.attention.as_ref().map(|a| a.num_kv_heads), |g| {
                    Some(g.num_kv_heads)
                })
        });
    let first = heads.next()??;
    heads.all(|h| h == Some(first)).then(|| json!(first))
}
