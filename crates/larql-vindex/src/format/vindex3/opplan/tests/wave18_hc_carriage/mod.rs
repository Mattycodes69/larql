//! **Wave 18 — hyper-connection addressability, the carriage witness.**
//!
//! The claim, kept narrow on purpose: LARQL can assign semantic operand
//! identity to the Sinkhorn hyper-connection vocabulary and carry those
//! operands through its normal execution-planning machinery. Six site
//! operands per layer classify to roles, are required by closure, are
//! checked against the DECLARED stream count's geometry, and are bound
//! into the plan; the head's three bare operands are placed as their own
//! object under the declaration and bound beside them. Since wave 19 the
//! executor traverses the bundle; what it still refuses — a whole-stack
//! image with no head object — it refuses at its door by the head's
//! name, through the same fact the plan report reads.
//!
//! **What this file does NOT claim.** It does not make DeepSeek-V4
//! plannable: DeepSeek remains blocked by an independently unsupported
//! base tensor dialect (`attn.wq_a`, `attn.wkv`, `attn_norm`,
//! `ffn.experts.N.w1`), and its reference supplied wave 17's arithmetic
//! oracle, not this wave's addressability witness. It does not execute a
//! hyper-connected checkpoint; no payload for one exists on this machine.
//!
//! # Three kinds of evidence from three checkpoints
//!
//! ```text
//! synthetic          a two-layer dense stack that DECLARES the topology
//!                    and ships every site operand at the declared
//!                    geometry — the one place closure can be watched
//!                    holding, and each of its refusals can be made to
//!                    fire on its own
//! DeepSeek-V4-Flash  REAL headers: the head's three bare groups gain an
//!                    owner under the declaration and lose it without
//!                    one; `mtp.0`'s eighteen hyper-connection tensors
//!                    stay external. The dialect-blocked control.
//! Kimi-K3            REAL headers: the four `*_res_*` operands the K3
//!                    programme expected this wave to address are a
//!                    `[hidden]` norm and a `[1, hidden]` projection —
//!                    not a Sinkhorn site under ANY stream count. The
//!                    transfer question, answered by shape.
//! ```
//!
//! The synthetic fixture's geometry is small (hidden 64, four streams)
//! and asymmetric: `(2 + 4) · 4 = 24` mix rows against `4 · 64 = 256`
//! columns, so no dimension equals another and a transposed or
//! misassigned operand cannot pass the shape check by coincidence.

use crate::format::vindex3::encode::{encode_graph, encode_system_unenforced};
use crate::format::vindex3::graph::roles::classify_stack_tensor_on;
use crate::format::vindex3::graph::{
    build_from_inventories, LayerOperator, ObjectKind, OperandRole,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::execute_text;
use crate::format::vindex3::opplan::exec::hyper_connection::{HC_HEAD_SCALE_LEN, HC_SCALE_LEN};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, OpPlanOutcome};
use crate::format::vindex3::plan::plan_system;
use crate::format::vindex3::plan::tests_support::{custom_artifact, header_only_shards};
use larql_models::config::{HyperConnection, HyperConnectionWeights, ResidualTopology};

/// The synthetic component's geometry.
const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const STREAMS: usize = 4;
const SINKHORN_ITERS: usize = 20;
const SINKHORN_EPS: f64 = 1e-6;
/// `(2 + 4) · 4`.
const MIX_ROWS: usize = 24;
/// `4 · 64`.
const BUNDLE_WIDTH: usize = 256;

/// Hy4-preview's site shape, `[2·hc, hc·hidden]`, at this fixture's
/// geometry — the Sinkhorn-free form that must not bind to a Sinkhorn
/// role.
const PREPOST_MIX_ROWS: usize = 8;

/// A dense two-layer stack, with or without the topology declared.
fn config(hyper_connected: bool) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    if hyper_connected {
        config["hc_mult"] = serde_json::json!(STREAMS);
        config["hc_sinkhorn_iters"] = serde_json::json!(SINKHORN_ITERS);
        config["hc_eps"] = serde_json::json!(SINKHORN_EPS);
    }
    config
}

type Tensors = Vec<(String, Vec<usize>)>;

/// The ordinary two-norm estate: 9 stack operands per layer, plus
/// embedding, final norm and head.
fn dense_tensors() -> Tensors {
    let mut tensors: Tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![128, HIDDEN]),
        ("model.norm.weight".to_string(), vec![HIDDEN]),
        ("lm_head.weight".to_string(), vec![128, HIDDEN]),
    ];
    for layer in 0..LAYERS {
        let stack = format!("model.layers.{layer}");
        for (leaf, shape) in [
            ("self_attn.q_proj.weight", vec![64, HIDDEN]),
            ("self_attn.k_proj.weight", vec![16, HIDDEN]),
            ("self_attn.v_proj.weight", vec![16, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, 64]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("mlp.gate_proj.weight", vec![256, HIDDEN]),
            ("mlp.up_proj.weight", vec![256, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, 256]),
        ] {
            tensors.push((format!("{stack}.{leaf}"), shape));
        }
    }
    tensors
}

/// The six site operands per layer, at the declared geometry, spelled as
/// DeepSeek-V4 and GLM-5.3-Flash both spell them.
fn site_tensors() -> Tensors {
    let mut tensors = Tensors::new();
    for layer in 0..LAYERS {
        let stack = format!("model.layers.{layer}");
        for site in ["attn", "ffn"] {
            tensors.push((
                format!("{stack}.hc_{site}_fn"),
                vec![MIX_ROWS, BUNDLE_WIDTH],
            ));
            tensors.push((format!("{stack}.hc_{site}_base"), vec![MIX_ROWS]));
            tensors.push((format!("{stack}.hc_{site}_scale"), vec![HC_SCALE_LEN]));
        }
    }
    tensors
}

/// The head's three bare operands, at the head's own geometry.
fn head_tensors() -> Tensors {
    vec![
        ("hc_head_fn".to_string(), vec![STREAMS, BUNDLE_WIDTH]),
        ("hc_head_base".to_string(), vec![STREAMS]),
        ("hc_head_scale".to_string(), vec![HC_HEAD_SCALE_LEN]),
    ]
}

/// The encoded container and its plan outcome, sources kept alive.
struct Planned {
    _source: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: crate::format::vindex3::inspect::SystemInspection,
    outcome: OpPlanOutcome,
}

/// Encode through the doctored-write seam: the plan is built and its
/// graph encoded WITHOUT the admissibility gate, so a headless
/// hyper-connected fixture — inadmissible by the head's own finding — can
/// still be constructed and its closure and refusals proven downstream.
/// (A head-bearing estate is admissible since wave 19; see
/// [`the_production_encode_gate_admits_a_head_bearing_container_and_refuses_a_headless_one`].)
fn plan(config: serde_json::Value, tensors: Tensors) -> Planned {
    let source = tempfile::tempdir().unwrap();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config, &borrowed);
    let named = vec![("hc-artifact".to_string(), inventory)];
    let container = tempfile::tempdir().unwrap();
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    Planned {
        _source: source,
        container,
        inspection,
        outcome,
    }
}

fn hyper_connected() -> Planned {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    plan(config(true), tensors)
}

const HEADERS: &str = include_str!("../fixtures/hc_operand_headers.json");
const DEEPSEEK: &str = "deepseek-ai/DeepSeek-V4-Flash";

/// DeepSeek-V4-Flash's config, trimmed to what the builder reads, every
/// value from the checkpoint's own `config.json`.
fn deepseek_config(mutate: impl FnOnce(&mut serde_json::Value)) -> serde_json::Value {
    let mut config = serde_json::json!({
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20
    });
    mutate(&mut config);
    config
}

/// The fixture's flat `{name: {dtype, shape, bytes, shard}}` census,
/// regrouped into per-shard safetensors headers with sequential offsets
/// — what `header_only_shards` writes. Byte counts are the real ones.
fn deepseek_shards() -> serde_json::Map<String, serde_json::Value> {
    let fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    let mut shards: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut offsets: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for (name, tensor) in fixture[DEEPSEEK].as_object().unwrap() {
        let shard = tensor["shard"].as_str().unwrap().to_string();
        let bytes = tensor["bytes"].as_u64().unwrap();
        let offset = offsets.entry(shard.clone()).or_insert(0);
        let entry = shards
            .entry(shard.clone())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        entry.as_object_mut().unwrap().insert(
            name.clone(),
            serde_json::json!({
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": [*offset, *offset + bytes],
            }),
        );
        *offset += bytes;
    }
    shards
}

fn deepseek_graph(
    mutate: impl FnOnce(&mut serde_json::Value),
) -> crate::format::vindex3::graph::BuiltGraph {
    let dir = tempfile::tempdir().unwrap();
    let inventory = header_only_shards(dir.path(), &deepseek_config(mutate), &deepseek_shards());
    build_from_inventories(&[("deepseek".to_string(), inventory)])
}

const K3_HEADERS: &str = include_str!("../../../plan/tests/fixtures/k3_two_layer_headers.json");
const K3_HIDDEN: usize = 7168;

/// The fact the arm reports, as `ClosureDefect::UnjudgedSemantic` spells
/// it. Matched as a fragment so the layer suffix can vary.
const HC_ON_MIXER_FACT_FRAGMENT: &str = "hyper-connection sites on a mixer-only layer";

trait RefusedLayers {
    /// The layer index named by every `UnjudgedSemantic` defect whose
    /// fact carries `fragment` and the required-by names the traversal.
    fn outcome_layers_refused_for(&self, fragment: &str) -> impl Iterator<Item = usize> + '_;
}

impl RefusedLayers for OpPlanOutcome {
    fn outcome_layers_refused_for(&self, fragment: &str) -> impl Iterator<Item = usize> + '_ {
        let fragment = fragment.to_string();
        self.defects.iter().filter_map(move |d| match d {
            ClosureDefect::UnjudgedSemantic {
                fact, required_by, ..
            } if fact.contains(&fragment) && required_by.contains("traversal") => fact
                .rsplit("(layer ")
                .next()
                .and_then(|tail| tail.trim_end_matches(')').parse().ok()),
            _ => None,
        })
    }
}

mod real_headers_deepseek_v4_flash;
mod real_headers_kimi_k3_the_transfer_questi;
mod the_mixer_on_hyper_connection_arm_reache;
mod wave18_hc_carriage_basics;
