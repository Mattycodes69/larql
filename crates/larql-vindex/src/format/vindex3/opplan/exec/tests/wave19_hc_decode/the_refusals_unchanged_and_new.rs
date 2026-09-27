//! The refusals, unchanged and new
//! The loader's own refusals, on doctored plans
//! The bundle entry's own refusals

use super::*;

/// The public loader prepares what the witness proved: a head-bearing
/// whole stack prepares and executes end to end, a headless component
/// prepares as a layer range, and a headless whole stack is refused by
/// the head's name — never by the topology's.
#[test]
fn the_public_loader_prepares_what_the_witness_proved() {
    let backend = ReferenceBackend::new();
    let sub = substrate::build(Variant::HeadBearing);
    let operands = store(&sub);
    PreparedOperands::load(&sub.plan, &operands, &backend, ExecutionSlice::Full)
        .expect("a head-bearing whole stack prepares");
    let trace = execute_text(&sub.plan, &operands, &[1, 2]).expect("and executes");
    assert_eq!(trace.logits.unwrap().len(), VOCAB);
    assert!(
        trace.embedded.bundles().is_some(),
        "the embedding entered as bundles"
    );

    for variant in [Variant::Headless, Variant::Hybrid] {
        let sub = substrate::build(variant);
        let operands = store(&sub);
        PreparedOperands::load(&sub.plan, &operands, &backend, layer_range())
            .unwrap_or_else(|e| panic!("{variant:?}: a headless layer range prepares: {e}"));
        let err = PreparedOperands::load(&sub.plan, &operands, &backend, ExecutionSlice::Full)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| panic!("{variant:?}: a headless whole stack must refuse"));
        assert!(err.contains("hyper_connection_head"), "{variant:?}: {err}");
        assert!(!err.contains("traversal"), "{variant:?}: {err}");
    }
}

/// A6. Headless: a whole-stack image refuses at preparation, naming the
/// head; a layer-range image prepares and runs, and produces no logits.
#[test]
fn a_headless_component_prepares_as_a_layer_range_and_not_as_a_whole_stack() {
    let sub = substrate::build(Variant::Headless);
    let err = prepare_err(&sub, ExecutionSlice::Full);
    assert!(err.contains("hyper_connection_head"), "{err}");
    assert!(err.contains("layer-range"), "{err}");
    assert!(
        !err.contains("traversal"),
        "the head reason is not the topology reason: {err}"
    );
    let run = run_from_oracle(&sub, layer_range(), Mutation::None);
    for step in &run.steps {
        assert!(step.logits.is_none(), "a layer range produces no logits");
        assert!(step.exit.is_none(), "a layer range reduces nothing");
        assert!(
            step.bundle.is_some(),
            "the bundle is the layer range's output"
        );
    }
}

/// A layer scale under the topology is unjudged and refused by name.
#[test]
fn a_layer_scale_under_the_topology_is_refused_at_preparation() {
    let sub = substrate::build(Variant::HybridWithLayerScale);
    let err = prepare_err(&sub, layer_range());
    assert!(err.contains("layer scale"), "{err}");
    assert!(err.contains("unjudged"), "{err}");
}

/// P4. The single stream is untouched: on the sibling with no topology,
/// the decode step's logits equal the batch traversal's, bit for bit —
/// and the batch path did not change in this wave.
#[test]
fn the_single_stream_sibling_decodes_exactly_as_it_batches() {
    let sub = substrate::single_stream_sibling();
    let store = store(&sub);
    let backend = ReferenceBackend::new();
    let tokens = [4u32, 1, 6, 2];
    let batch = execute_text(&sub.plan, &store, &tokens).unwrap();
    let mut session = DecodeSession::new(
        &sub.plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut witness = Witness::default();
    let mut last = None;
    for &token in &tokens {
        last = session.step_observed(token, &mut witness).unwrap().logits;
    }
    assert_eq!(last.unwrap(), batch.logits.unwrap());
    assert!(
        witness.records.is_empty(),
        "a single-stream step emits no hyper-connection record"
    );
    assert!(!store_carries_hc(&sub));
}

/// The residency census counts the sites and the head as glue, by the
/// decision that made the mix projection f32.
#[test]
fn the_sites_and_the_head_are_counted_as_glue() {
    let headless = prepare(&substrate::build(Variant::Headless), layer_range()).unwrap();
    let head_bearing = prepare(
        &substrate::build(Variant::HeadBearing),
        ExecutionSlice::Full,
    )
    .unwrap();
    let site_bytes =
        LAYERS * SITES * (MIX_ROWS * STREAMS * HIDDEN + MIX_ROWS + 3) * std::mem::size_of::<f32>();
    let head_bytes = (STREAMS * STREAMS * HIDDEN + STREAMS) * std::mem::size_of::<f32>();
    let norms = |ops: &PreparedOperands| -> usize { ops.residency_census().glue.widened_f32 };
    // Headless: the layer norms plus the sites (a layer range carries no
    // final norm). Head-bearing: the same, plus the final norm and the
    // head's two vectors.
    let headless_glue = norms(&headless);
    let head_bearing_glue = norms(&head_bearing);
    assert!(
        headless_glue >= site_bytes,
        "{headless_glue} < {site_bytes}"
    );
    assert_eq!(
        head_bearing_glue - headless_glue,
        head_bytes + HIDDEN * std::mem::size_of::<f32>()
    );
}

#[test]
fn a_layer_without_sites_under_the_topology_is_refused() {
    let (sub, store) = head_bearing_plan();
    let mut plan = sub.plan.clone();
    plan.layers[1].hyper_connection = None;
    let err = load_err(&plan, &store);
    assert!(err.contains("layer 1 carries no"), "{err}");
}

#[test]
fn sites_on_a_single_stream_component_are_refused_by_the_public_loader() {
    let (sub, store) = head_bearing_plan();
    let mut plan = sub.plan.clone();
    plan.residual_topology = larql_models::config::ResidualTopology::SingleStream;
    let err = PreparedOperands::load(
        &plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("a single-stream plan with sites is refused");
    assert!(err.contains("single"), "{err}");
    assert!(err.contains("never produces"), "{err}");
}

#[test]
fn a_head_at_a_sites_geometry_is_refused() {
    let (sub, store) = head_bearing_plan();
    let site = sub.plan.layers[0].hyper_connection.clone().unwrap();
    let mut plan = sub.plan.clone();
    plan.hyper_connection_head.as_mut().unwrap().reduce_fn = site.attention.mix_fn.clone();
    let err = load_err(&plan, &store);
    assert!(err.contains("head's geometry"), "{err}");

    let mut plan = sub.plan.clone();
    plan.hyper_connection_head.as_mut().unwrap().scale = site.attention.scale.clone();
    let err = load_err(&plan, &store);
    assert!(err.contains("scale holds 3 values"), "{err}");
}

#[test]
fn a_site_at_the_heads_geometry_is_refused() {
    let (sub, store) = head_bearing_plan();
    let head = sub.plan.hyper_connection_head.clone().unwrap();
    let mut plan = sub.plan.clone();
    plan.layers[0].hyper_connection.as_mut().unwrap().ffn.mix_fn = head.reduce_fn;
    let err = load_err(&plan, &store);
    assert!(err.contains("layer 0 ffn site: mix_fn holds"), "{err}");
}

#[test]
fn layers_that_disagree_on_norm_eps_leave_the_head_no_epsilon() {
    let (sub, store) = head_bearing_plan();
    let mut plan = sub.plan.clone();
    plan.layers[1].declared_norm_eps = 2.0 * NORM_EPS;
    let err = load_err(&plan, &store);
    assert!(err.contains("one component value"), "{err}");
}

#[test]
fn a_bundle_cannot_enter_a_single_stream_session() {
    let sub = substrate::single_stream_sibling();
    let store = store(&sub);
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(
        &sub.plan,
        &store,
        &backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let err = session
        .step_from_bundle(Oracle::load().input(0), &mut NoopObserver, Mutation::None)
        .err()
        .map(|e| e.to_string())
        .expect("a bundle has no meaning on one stream");
    assert!(err.contains("single-stream component"), "{err}");
}

#[test]
fn a_bundle_of_the_wrong_shape_is_refused_at_entry() {
    let sub = substrate::build(Variant::HeadBearing);
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let narrow = Bundle::replicate(&[0.0; HIDDEN - 1], STREAMS);
    let err = session
        .step_from_bundle(narrow, &mut NoopObserver, Mutation::None)
        .err()
        .map(|e| e.to_string())
        .expect("the entering bundle must match the component");
    assert!(err.contains("entering bundle"), "{err}");
}
