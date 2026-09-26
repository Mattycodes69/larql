//! EXPLAIN (LQL-2): the structured explanation IS the authority

use super::*;

#[test]
fn explain_is_stable_and_reads_the_plan() {
    let container = container_with(miniature_glimmer);
    let a = Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let b = Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let explain = super::super::ExplainPlan::from_runtime(&a);
    // Stability: same container, identical structured explanation.
    assert_eq!(explain, super::super::ExplainPlan::from_runtime(&b));

    assert_eq!(explain.generation, 3);
    assert_eq!(explain.component, COMPONENT);
    assert!(explain.execution_closed);
    assert_eq!(explain.layers.len(), 2);
    // The miniature's sliding(3)/full split, from the plan alone.
    assert_eq!(explain.layers[0].attention.mode, "sliding");
    assert_eq!(explain.layers[0].attention.window, Some(3));
    assert_eq!(explain.layers[1].attention.mode, "full");
    // Four-norm placement shows as explicit ops, in execution order.
    assert_eq!(
        explain.layers[0].ops,
        vec![
            "pre_attention_norm",
            "attention",
            "post_attention_norm",
            "residual_add",
            "pre_ffn_norm",
            "ffn",
            "post_ffn_norm",
            "residual_add",
        ]
    );
    // Continuation geometry matches what a provider is prepared with.
    let geometry = plan_kv_geometry(a.plan());
    assert_eq!(explain.continuation.len(), geometry.len());
    for (e, g) in explain.continuation.iter().zip(&geometry) {
        assert_eq!(e.kv_dim, g.kv_dim);
        assert_eq!(e.window, g.window);
    }
    assert!(explain.output.is_some());
    assert!(explain.final_norm);
}

/// The negative control: mutate the plan and the explanation must
/// change with it — the instrument sees the program, not a cached
/// family notion.
#[test]
fn explain_changes_when_the_plan_changes() {
    let container = container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let plan = outcome.plan.unwrap();
    let baseline = super::super::ExplainPlan::from_plan(&plan, "m");

    let mut widened = plan.clone();
    widened.layers[0].attention.softmax_mut().unwrap().window = None;
    let explained = super::super::ExplainPlan::from_plan(&widened, "m");
    assert_ne!(baseline, explained);
    assert_eq!(explained.layers[0].attention.mode, "full");

    let mut headless = plan.clone();
    headless.output = None;
    assert!(super::super::ExplainPlan::from_plan(&headless, "m")
        .output
        .is_none());
}

/// Provenance closure: the coordinates the explanation quotes are
/// sufficient, alone, to reach the exact bytes execution loads — the
/// explain chain and the execution chain cannot name different
/// operands.
#[test]
fn explain_operand_coordinates_resolve_to_the_executed_bytes() {
    use larql_vindex::format::vindex3::opplan::OperandRef;
    let container = container_with(miniature_glimmer);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), COMPONENT).unwrap();
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let explain = super::super::ExplainPlan::from_plan(&plan, "m");

    // One attention operand and one FFN operand, per the gate.
    let q = &explain.layers[0].attention.operands[0];
    assert_eq!(q.role, "q");
    let ffn_up = explain.layers[0]
        .ffn
        .operands
        .iter()
        .find(|o| o.role == "up")
        .unwrap();
    for (quoted, executed) in [
        (q, &plan.layers[0].attention.softmax().unwrap().q),
        (
            ffn_up,
            &plan.layers[0].ffn.as_ref().unwrap().dense().unwrap().up,
        ),
    ] {
        let rebuilt = OperandRef {
            object: quoted.object.clone(),
            tensor: quoted.tensor.clone(),
            dtype: quoted.dtype.clone(),
            shape: quoted.shape.clone(),
        };
        let via_explain = store.load(&rebuilt).unwrap();
        let via_execution = store.load(executed).unwrap();
        assert_eq!(via_explain, via_execution, "{} bytes diverge", quoted.role);
        assert!(!via_explain.is_empty());
    }
}

/// The session-level observation seam: observed and unobserved steps
/// are bit-identical, and the recorder sees the step's boundaries —
/// one execution path, many consumers.
#[test]
fn an_observed_session_step_is_bit_identical_and_records_boundaries() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    let mut plain = runtime.session(&row(runtime.plan())).unwrap();
    let mut observed = runtime.session(&row(runtime.plan())).unwrap();
    let mut recorder = RecordingObserver::default();
    for &token in G_TOKENS.iter() {
        let a = plain.step(token).unwrap();
        let b = observed.step_observed(token, &mut recorder).unwrap();
        assert_eq!(a, b, "observation changed the arithmetic");
    }
    // Per position: embed + (write, attention, write, ffn) per layer +
    // logits — the two carrier writes are V3-OBS-1's events.
    let per_position = 1 + 4 * runtime.plan().layers.len() + 1;
    assert_eq!(recorder.events.len(), G_TOKENS.len() * per_position);
    assert!(matches!(
        recorder.events[0],
        StepEvent::Embedded { position: 0 }
    ));
}

/// The analysis pass (V3-LQL-3B): `execute_streaming` runs the SAME
/// traversal prefill does — bit-identical logits — while streaming
/// each layer's residual taps to the sink. Observation is
/// subscription, never a second executor.
#[test]
fn execute_streaming_matches_prefill_and_taps_every_layer() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    let mut kv = RowKvState::default();
    let prefill_logits = runtime.prefill_into(&G_TOKENS, &mut kv).unwrap();

    let mut tapped: Vec<(usize, usize)> = Vec::new();
    let output = runtime
        .execute_streaming(&G_TOKENS, &row(runtime.plan()), &mut |event| {
            if let super::super::PlaneEvent::Layer { index, trace } = event {
                tapped.push((index, trace.post_layer.rows().len()));
            }
            Ok(())
        })
        .unwrap();

    assert_eq!(
        output.logits.as_deref(),
        Some(&prefill_logits[..]),
        "the observed pass must price the same logits bit-for-bit"
    );
    let layers = runtime.plan().layers.len();
    let expected: Vec<(usize, usize)> = (0..layers).map(|l| (l, G_TOKENS.len())).collect();
    assert_eq!(
        tapped, expected,
        "one tap per layer, one residual row per position"
    );
}

/// A sink error aborts the pass and surfaces — the observer can stop
/// an analysis, though it can never change what execution computes.
#[test]
fn execute_streaming_surfaces_a_sink_error() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let err = runtime
        .execute_streaming(&G_TOKENS, &row(runtime.plan()), &mut |_| {
            Err(larql_vindex::VindexError::Parse("stop here".into()))
        })
        .expect_err("the sink's error must surface");
    assert!(err.to_string().contains("stop here"), "{err}");
}

/// The browse view binds through the runtime, off the same plan and
/// operand store execution uses.
#[test]
fn knowledge_view_binds_from_the_runtime() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    let tok_json = crate::test_utils::synthetic_tokenizer_json(
        larql_vindex::format::vindex3::fixtures::G_VOCAB,
    );
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let view = runtime.knowledge_view(&tokenizer).unwrap();
    assert_eq!(view.num_layers(), runtime.plan().layers.len());
    assert!(view.max_features() > 0, "the miniature carries dense FFNs");
}

/// The overlaid runtime surface (V3-LQL-3B compose): with EMPTY
/// overrides every overlaid entry point is its plain counterpart bit
/// for bit, and a real edit changes what execution computes — the
/// runtime-level statement of the operand-source seam's contract.
#[test]
fn overlaid_entry_points_are_bit_identical_when_empty_and_observe_edits() {
    use super::super::{OperandEdit, OperandOverrides};
    use larql_vindex::format::vindex3::opplan::LayerFfn;

    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    // Plain arms.
    let mut kv = RowKvState::default();
    let prefill = runtime.prefill_into(&G_TOKENS, &mut kv).unwrap();
    let streamed = runtime
        .execute_streaming(&G_TOKENS, &row(runtime.plan()), &mut |_| Ok(()))
        .unwrap();

    // Empty overlay: bit-identical on every entry point.
    let empty = OperandOverrides::new();
    let mut kv2 = RowKvState::default();
    assert_eq!(
        runtime
            .prefill_into_overlaid(&G_TOKENS, &empty, &mut kv2)
            .unwrap(),
        prefill
    );
    assert_eq!(
        runtime
            .execute_streaming_overlaid(&G_TOKENS, &empty, &row(runtime.plan()), &mut |_| Ok(()))
            .unwrap()
            .logits,
        streamed.logits
    );
    let step_plain = {
        let mut session = runtime.session(&row(runtime.plan())).unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    let step_overlaid = {
        let mut session = runtime
            .session_overlaid(&empty, &row(runtime.plan()))
            .unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    assert_eq!(step_plain, step_overlaid);
    let resumed = {
        let mut session = runtime.session_with_kv_overlaid(&mut kv2, &empty).unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    let resumed_plain = {
        let mut session = runtime.session_with_kv(&mut kv).unwrap();
        session.step(G_TOKENS[0]).unwrap()
    };
    assert_eq!(resumed, resumed_plain);

    // A real edit is observed.
    let Some(LayerFfn::Dense(ffn)) = &runtime.plan().layers[0].ffn else {
        panic!("miniature layer 0 is dense");
    };
    let gate = ffn.gate.as_ref().unwrap().clone();
    let mut edited = OperandOverrides::new();
    edited.push(
        &gate,
        OperandEdit::Row {
            index: 0,
            values: vec![5.0; gate.shape[1]],
        },
    );
    let out = runtime
        .execute_streaming_overlaid(&G_TOKENS, &edited, &row(runtime.plan()), &mut |_| Ok(()))
        .unwrap();
    assert_ne!(out.logits, streamed.logits, "the edit must be observed");
    // …and the resolver serves the effective row.
    let effective = runtime.operands();
    let base_gate = effective.load(&gate).unwrap();
    assert_ne!(&base_gate[..gate.shape[1]], &vec![5.0; gate.shape[1]][..]);
}

/// LOWERING-PLUGIN-1, L3: the runtime — the path the server and LQL open
/// through — resolves its provider from the registry it is handed, and
/// constructs none.
mod via_tests {
    use std::path::Path;

    use super::super::super::*;
    use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
    use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
    use larql_vindex::format::vindex3::opplan::exec::kv::RowKvState;
    use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
    use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;

    /// F4 through a production path. A registry that omits the
    /// production provider, asked to open as it, is refused by identity
    /// naming what the registry holds — and the container path does not
    /// even exist, so the refusal provably happened before any I/O and no
    /// provider was constructed underneath to answer anyway.
    #[test]
    fn a_registry_without_the_provider_refuses_before_touching_the_container() {
        let without_production = LoweringRegistry::new()
            .register(Box::new(ReferenceBackend::new()))
            .unwrap();
        let err = Vindex3Runtime::open_via(
            Path::new("/definitely/not/a/container"),
            "target",
            std::sync::Arc::new(without_production),
            &LoweringIdentity::cpu_production(),
        )
        .err()
        .expect("refused")
        .to_string();
        assert!(err.contains("cpu-production/v1"), "{err}");
        assert!(err.contains("registered: reference/v1"), "{err}");
        assert!(
            !err.contains("not/a/container") && !err.to_lowercase().contains("no such file"),
            "the refusal is the registry's, not the filesystem's: {err}"
        );
    }

    /// Through the shipped registry the runtime opens on the same
    /// provider the direct constructor would have handed it, by identity
    /// and by name, and prepares.
    #[test]
    fn the_shipped_registry_opens_the_production_provider_the_direct_path_did() {
        let tmp = tempfile::tempdir().unwrap();
        let checkpoint = tmp.path().join("ckpt");
        std::fs::create_dir_all(&checkpoint).unwrap();
        let container = tmp.path().join("out.vindex3");
        encode_fixture_container(dense_f32_model, &checkpoint, &container, "target");

        let via = Vindex3Runtime::open_via(
            &container,
            "target",
            std::sync::Arc::new(LoweringRegistry::shipped()),
            &LoweringIdentity::cpu_production(),
        )
        .expect("the shipped registry holds the production provider");
        let direct = Vindex3Runtime::open(&container, "target", ProductionBackend::new()).unwrap();
        assert_eq!(via.backend().identity(), direct.backend().identity());
        assert_eq!(via.backend().name(), direct.backend().name());
        assert_eq!(via.model_name(), direct.model_name());
        via.prepare().expect("prepares through the shared provider");
    }

    /// LOWERING-PLUGIN-1, L4, through the production path that HOLDS
    /// prepared state: a served model prepares once and answers requests
    /// for the life of the process, so it is the one place where a pin
    /// can outlive the authority that made it.
    ///
    /// The image records which provider decided its pins; the runtime
    /// carries which providers exist now. Re-pointed at an authority
    /// without that provider, the model refuses at its next use — every
    /// use, sessions and batch prefill alike — naming the provider the
    /// pins recorded and the ones on offer. The reference provider is
    /// registered and perfectly able to execute this plan, and is not a
    /// substitute for the provider that pinned it.
    #[test]
    fn a_served_model_refuses_when_the_provider_that_pinned_it_is_gone() {
        let tmp = tempfile::tempdir().unwrap();
        let checkpoint = tmp.path().join("ckpt");
        std::fs::create_dir_all(&checkpoint).unwrap();
        let container = tmp.path().join("out.vindex3");
        encode_fixture_container(dense_f32_model, &checkpoint, &container, "target");

        let served = Vindex3Runtime::open_via(
            &container,
            "target",
            std::sync::Arc::new(LoweringRegistry::shipped()),
            &LoweringIdentity::cpu_production(),
        )
        .unwrap()
        .prepare()
        .unwrap();
        assert!(served.lowerings().is_some(), "opened through an authority");

        let tokens = [3u32, 17, 28];
        let mut kv = RowKvState::default();
        served
            .session_with_kv(&mut kv)
            .expect("the provider that pinned the image is still registered");
        let mut kv = RowKvState::default();
        served
            .prefill_into(&tokens, &mut kv)
            .expect("and batch prefill runs on it");

        let stale = served.with_lowerings(std::sync::Arc::new(
            LoweringRegistry::new()
                .register(Box::new(ReferenceBackend::new()))
                .unwrap(),
        ));
        for err in [
            {
                let mut kv = RowKvState::default();
                stale
                    .session_with_kv(&mut kv)
                    .err()
                    .expect("a session over a pin whose provider is gone is refused")
                    .to_string()
            },
            {
                let mut kv = RowKvState::default();
                stale
                    .prefill_into(&tokens, &mut kv)
                    .expect_err("and so is a prefill")
                    .to_string()
            },
        ] {
            assert!(err.contains("cpu-production/v1"), "{err}");
            assert!(err.contains("reference/v1"), "{err}");
            assert!(err.contains("re-prepare"), "{err}");
            assert!(
                err.contains("no other provider stands in for it"),
                "an available provider is not a fallback: {err}"
            );
        }
    }
}
