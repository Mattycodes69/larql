//! Probes answer only for what the surface carries: an optional fact the
//! surface leaves unset answers nothing (and so blocks as dropped), and a
//! value-dependent probe answers exactly for the values it can represent.

use serde_json::{json, Value};

use super::super::support::glimmer_shaped_target;
use crate::format::vindex3::graph::{build_from_inventories, Component};
use crate::format::vindex3::plan::carriage::{rule_for, ProbeContext};

/// A carried output multiplier, and the `logits_scaling` divisor it
/// expresses.
const MULTIPLIER: f64 = 0.25;
const DIVISOR: f64 = 4.0;
const LOWER_BOUND: f32 = 0.5;

fn glimmer_component() -> Component {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("target".to_string(), glimmer_shaped_target(dir.path()))];
    build_from_inventories(&named)
        .graph
        .components
        .into_iter()
        .find(|c| c.execution.is_some())
        .expect("the fixture builds a surface")
}

/// The glimmer component with every optional attention and head fact
/// unset, and neither an FFN nor a mixer to name an activation.
fn stripped() -> Component {
    let mut component = glimmer_component();
    let surface = component.execution.as_mut().unwrap();
    let attention = surface.attention.as_mut().expect("the fixture attends");
    attention.attention_bias = None;
    attention.qkv_bias = None;
    attention.query_scale = None;
    attention.logit_softcapping = None;
    let head = surface.head.as_mut().expect("the fixture has a head");
    head.embed_scale = None;
    head.output_multiplier = None;
    head.final_logit_softcapping = None;
    surface.ffn = None;
    surface.mamba2 = None;
    surface.linear_attention = None;
    surface.kda_gate_lower_bound = None;
    component
}

fn probe(leaf: &str, component: &Component, declared: &Value) -> Option<Value> {
    let probe = rule_for(leaf)
        .and_then(|rule| rule.probe)
        .unwrap_or_else(|| panic!("`{leaf}` has a probe"));
    probe(
        component,
        &ProbeContext {
            span: None,
            declared,
            family: None,
        },
    )
}

#[test]
fn unset_optional_surface_facts_answer_nothing() {
    let component = stripped();
    for leaf in [
        "mamba_ssm_dtype",
        "hidden_act",
        "attention_bias",
        "qkv_bias",
        "qk_scale_factor",
        "attn_logit_softcapping",
        "final_logit_softcapping",
        "output_multiplier",
        "logits_scaling",
        "embedding_multiplier",
        "gate_lower_bound",
    ] {
        assert_eq!(probe(leaf, &component, &Value::Null), None, "{leaf}");
    }
}

#[test]
fn logits_scaling_answers_in_the_divisor_it_declares() {
    let mut component = stripped();
    let head = component.execution.as_mut().unwrap().head.as_mut().unwrap();
    head.output_multiplier = Some(MULTIPLIER);
    assert_eq!(
        probe("logits_scaling", &component, &Value::Null),
        Some(json!(DIVISOR))
    );
    // A zero multiplier has no divisor to express.
    let head = component.execution.as_mut().unwrap().head.as_mut().unwrap();
    head.output_multiplier = Some(0.0);
    assert_eq!(probe("logits_scaling", &component, &Value::Null), None);
}

#[test]
fn a_carried_kda_lower_bound_answers_as_carried() {
    let mut component = stripped();
    component.execution.as_mut().unwrap().kda_gate_lower_bound = Some(LOWER_BOUND);
    assert_eq!(
        probe("gate_lower_bound", &component, &Value::Null),
        Some(json!(LOWER_BOUND))
    );
}

#[test]
fn zero_is_the_only_count_the_absence_represents() {
    let component = glimmer_component();
    let leaf = "num_nextn_predict_layers";
    assert_eq!(probe(leaf, &component, &json!(0)), Some(json!(0)));
    assert_eq!(probe(leaf, &component, &json!(1)), None);
}

#[test]
fn mla_nope_declared_false_is_declined_not_answered() {
    assert_eq!(
        probe("mla_use_nope", &glimmer_component(), &json!(false)),
        None
    );
}

#[test]
fn an_unrecognised_activation_word_reads_back_in_the_schemas_spelling() {
    let component = glimmer_component();
    let declared = json!("not-an-activation");
    let activation = probe("hidden_act", &component, &declared).expect("the FFN answers");
    assert_ne!(activation, declared, "a disagreement must read as one");
    let shape = probe("activation", &component, &declared);
    assert_ne!(shape, Some(declared));
}

#[test]
fn the_sliding_layer_set_is_probed_against_the_resolved_table() {
    let component = glimmer_component();
    // Answers or declines; either way it never echoes a set the table
    // does not hold.
    let declared = json!([usize::MAX]);
    assert_ne!(
        probe("local_layer_ids", &component, &declared),
        Some(declared)
    );
}
