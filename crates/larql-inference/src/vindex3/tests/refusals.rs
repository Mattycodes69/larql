//! Refusals

use super::*;

#[test]
fn an_empty_prompt_is_refused() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let mut session = runtime.session(&row(runtime.plan())).unwrap();
    let err = session.prefill(&[]).unwrap_err();
    assert!(
        err.to_string().contains("at least one prompt token"),
        "{err}"
    );
}

#[test]
fn an_unknown_component_refuses_to_open() {
    let container = container_with(miniature_glimmer);
    let err = match Vindex3Runtime::open(
        container.path(),
        "no-such-component",
        ReferenceBackend::new(),
    ) {
        Ok(_) => panic!("an unknown component must refuse to open"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("no-such-component"), "{err}");
}

/// An unclosed component must refuse to open with the defects in the
/// error — never "best-effort" execute. Provoked the same way the
/// vindex closure gates do it: strip the component's execution surface
/// from the container's own graph, which planning reports as a
/// `MissingSurface` defect.
#[test]
fn an_unclosed_component_refuses_to_open_naming_its_defects() {
    let container = container_with(miniature_glimmer);
    let graph_path = container
        .path()
        .join(larql_vindex::format::vindex3::encode::SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&graph_path).unwrap()).unwrap();
    for component in graph["components"].as_array_mut().unwrap() {
        component.as_object_mut().unwrap().remove("execution");
    }
    std::fs::write(&graph_path, graph.to_string()).unwrap();

    let err = match Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()) {
        Ok(_) => panic!("an unclosed component must refuse to open"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("does not close"), "{err}");
}

/// The step path's defensive refusal (a step yielding no logits) is
/// unreachable while construction refuses headless plans — pinned here
/// so the fail-closed message stays a work item, not a mystery, if a
/// future change ever reaches it.
#[test]
fn the_defensive_missing_logits_refusal_names_the_invariant() {
    let err = super::super::session::missing_logits_error();
    assert!(err.to_string().contains("output head"), "{err}");
}

/// Same discipline for the prefill path's defensive refusal.
#[test]
fn the_defensive_headless_prefill_refusal_names_the_invariant() {
    let err = super::super::runtime::headless_prefill_error();
    assert!(err.to_string().contains("no output head"), "{err}");
}

#[test]
fn a_plan_without_an_output_head_is_refused() {
    let container = container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let mut plan = outcome.plan.unwrap();
    plan.output = None;
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let backend = ReferenceBackend::new();
    let err = match Vindex3Session::new(
        &plan,
        &store,
        &backend,
        Box::new(larql_vindex::format::vindex3::opplan::exec::kv::RowKvState::default()),
    ) {
        Ok(_) => panic!("a headless plan must be refused"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("no output head"), "{err}");
}
