use super::*;

/// **The positive witness.** A component that declares the topology and
/// ships every site operand closes, and the plan carries the bundle: the
/// topology on the component, six bound operands per layer, at the
/// geometry the declaration implies.
///
/// Every binding is checked by NAME as well as by presence. A plan that
/// bound `hc_ffn_fn` into the attention site would satisfy a count and
/// run the wrong site's weights at every layer.
#[test]
fn a_declared_topology_with_every_site_operand_closes_and_is_carried() {
    let planned = hyper_connected();
    assert!(planned.outcome.closed(), "{:?}", planned.outcome.defects);
    let plan = planned.outcome.plan.as_ref().unwrap();

    assert_eq!(
        plan.residual_topology,
        ResidualTopology::HyperConnection(HyperConnection {
            streams: STREAMS,
            sinkhorn_iters: SINKHORN_ITERS,
            sinkhorn_eps: SINKHORN_EPS,
        })
    );
    assert_eq!(plan.layers.len(), LAYERS);
    for layer in &plan.layers {
        let sites = layer
            .hyper_connection
            .as_ref()
            .unwrap_or_else(|| panic!("layer {} carries no sites", layer.layer));
        let l = layer.layer;
        assert_eq!(sites.attention.mix_fn.tensor, format!("{l}.hc_attn_fn"));
        assert_eq!(sites.attention.base.tensor, format!("{l}.hc_attn_base"));
        assert_eq!(sites.attention.scale.tensor, format!("{l}.hc_attn_scale"));
        assert_eq!(sites.ffn.mix_fn.tensor, format!("{l}.hc_ffn_fn"));
        assert_eq!(sites.ffn.base.tensor, format!("{l}.hc_ffn_base"));
        assert_eq!(sites.ffn.scale.tensor, format!("{l}.hc_ffn_scale"));
        assert_eq!(sites.attention.mix_fn.shape, [MIX_ROWS, BUNDLE_WIDTH]);
        assert_eq!(sites.ffn.base.shape, [MIX_ROWS]);
        assert_eq!(sites.ffn.scale.shape, [HC_SCALE_LEN]);
        assert_eq!(sites.attention.mix_fn.object, "target.decoder_stack");
        // 9 ordinary + 6 site operands, all consumed.
        assert_eq!(layer.operands_accounted, 15);
        assert_eq!(layer.operands_present, 15);
    }
    // No head was shipped, and none is invented: GLM-5.3-Flash's shape.
    assert!(plan.hyper_connection_head.is_none());
}

/// **The control that makes the witness mean something.** The same six
/// operands on a component that declares ONE stream are strays, each
/// naming the topology as the primitive it would need. Without this a
/// vocabulary that classified the six and consumed them anywhere would
/// pass the test above.
#[test]
fn site_operands_on_a_single_stream_component_are_refused_as_strays() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    let planned = plan(config(false), tensors);
    assert!(planned.outcome.plan.is_none());

    let mut refused: Vec<String> = planned
        .outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                tensor,
                required_primitive,
                ..
            } => {
                assert!(
                    required_primitive.contains("hyper-connection residual topology"),
                    "{required_primitive}"
                );
                Some(tensor.clone())
            }
            _ => None,
        })
        .collect();
    refused.sort();
    let mut expected: Vec<String> = site_tensors()
        .into_iter()
        .map(|(name, _)| name.trim_start_matches("model.layers.").to_string())
        .collect();
    expected.sort();
    assert_eq!(refused, expected, "{:?}", planned.outcome.defects);
    // And nothing else went wrong: the ordinary operands are fine.
    assert_eq!(planned.outcome.defects.len(), expected.len());
}

/// A declared topology whose layer is missing one site operand refuses
/// with that operand's ROLE named — a bundle the traversal could not
/// expand at the FFN site of layer 1, not a partially hyper-connected
/// stack.
#[test]
fn a_missing_site_operand_is_a_named_closure_defect() {
    let mut tensors = dense_tensors();
    tensors.extend(
        site_tensors()
            .into_iter()
            .filter(|(name, _)| name != "model.layers.1.hc_ffn_scale"),
    );
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    assert_eq!(
        planned.outcome.defects,
        vec![ClosureDefect::MissingOperand {
            layer: 1,
            role: OperandRole::HcFfnScale,
        }]
    );
}

/// **The Hy4 falsifier, made to fire.** A site whose mix projection has
/// the Sinkhorn-free `[2·hc, hc·hidden]` shape binds to the role by name
/// and fails its geometry: the contract is derived from the declared
/// stream count, and `[8, 256]` is not `[24, 256]`. This is the check
/// that would catch Hy4-preview's operands if a spelling ever let them
/// through, and it is written against the shape, not the name.
#[test]
fn a_sinkhorn_free_site_shape_fails_the_declared_geometry() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors().into_iter().map(|(name, shape)| {
        if name == "model.layers.0.hc_attn_fn" {
            (name, vec![PREPOST_MIX_ROWS, BUNDLE_WIDTH])
        } else {
            (name, shape)
        }
    }));
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    assert_eq!(
        planned.outcome.defects,
        vec![ClosureDefect::GeometryMismatch {
            tensor: "target.decoder_stack/0.hc_attn_fn".to_string(),
            expected: vec![MIX_ROWS, BUNDLE_WIDTH],
            actual: vec![PREPOST_MIX_ROWS, BUNDLE_WIDTH],
        }]
    );
}

/// The site scale is exactly three scalars. A checkpoint offering two
/// (Hy4-preview's count) is describing a different operation, and the
/// plan says so by geometry rather than binding two scalars into three
/// slots.
#[test]
fn a_two_entry_site_scale_fails_the_declared_geometry() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors().into_iter().map(|(name, shape)| {
        if name == "model.layers.1.hc_ffn_scale" {
            (name, vec![HC_SCALE_LEN - 1])
        } else {
            (name, shape)
        }
    }));
    let planned = plan(config(true), tensors);
    assert_eq!(
        planned.outcome.defects,
        vec![ClosureDefect::GeometryMismatch {
            tensor: "target.decoder_stack/1.hc_ffn_scale".to_string(),
            expected: vec![HC_SCALE_LEN],
            actual: vec![HC_SCALE_LEN - 1],
        }]
    );
}

/// **The head, placed and bound.** Under the declaration the three bare
/// groups become one object with three bindings, and the plan binds them
/// at the head's OWN geometry — one row per stream and a single scalar,
/// not a site's `(2 + hc)·hc` rows and three scalars.
#[test]
fn the_head_is_placed_as_its_own_object_and_bound_at_its_own_geometry() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(head_tensors());
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.closed(), "{:?}", planned.outcome.defects);

    let head_object = planned
        .inspection
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::HyperConnectionHead)
        .expect("the head object is placed");
    assert_eq!(head_object.id, "target.hyper_connection_head");
    let mut prefixes: Vec<&str> = head_object
        .source_bindings
        .iter()
        .map(|b| b.tensor_prefix.as_str())
        .collect();
    prefixes.sort_unstable();
    assert_eq!(prefixes, ["hc_head_base", "hc_head_fn", "hc_head_scale"]);

    let plan = planned.outcome.plan.as_ref().unwrap();
    let head = plan.hyper_connection_head.as_ref().expect("bound");
    assert_eq!(head.reduce_fn.object, "target.hyper_connection_head");
    assert_eq!(head.reduce_fn.tensor, "hc_head_fn");
    assert_eq!(head.reduce_fn.shape, [STREAMS, BUNDLE_WIDTH]);
    assert_eq!(head.base.tensor, "hc_head_base");
    assert_eq!(head.base.shape, [STREAMS]);
    assert_eq!(head.scale.tensor, "hc_head_scale");
    assert_eq!(head.scale.shape, [HC_HEAD_SCALE_LEN]);
}

/// A head stored at a SITE's geometry — the mistake wave 17 corrected in
/// wave 16's record, arriving as bytes — fails the head's contract rather
/// than binding a Sinkhorn split into an operation that runs none.
#[test]
fn a_head_at_a_sites_geometry_fails_the_heads_contract() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(
        head_tensors()
            .into_iter()
            .map(|(name, shape)| match name.as_str() {
                "hc_head_fn" => (name, vec![MIX_ROWS, BUNDLE_WIDTH]),
                "hc_head_scale" => (name, vec![HC_SCALE_LEN]),
                _ => (name, shape),
            }),
    );
    let planned = plan(config(true), tensors);
    assert!(planned.outcome.plan.is_none());
    let mut mismatches: Vec<(String, Vec<usize>, Vec<usize>)> = planned
        .outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::GeometryMismatch {
                tensor,
                expected,
                actual,
            } => Some((tensor.clone(), expected.clone(), actual.clone())),
            _ => None,
        })
        .collect();
    mismatches.sort();
    assert_eq!(
        mismatches,
        [
            (
                "target.hyper_connection_head/hc_head_fn".to_string(),
                vec![STREAMS, BUNDLE_WIDTH],
                vec![MIX_ROWS, BUNDLE_WIDTH],
            ),
            (
                "target.hyper_connection_head/hc_head_scale".to_string(),
                vec![HC_HEAD_SCALE_LEN],
                vec![HC_SCALE_LEN],
            ),
        ]
    );
}

/// The head's three bare names on a component that declares ONE stream
/// have no owner: the builder refuses them by name, the disagreement
/// between estate and declaration stated, and the plan is built without
/// them — the ordinary single-stream program, exactly as before.
#[test]
fn head_operands_without_the_declaration_stay_unplaced_with_the_disagreement_named() {
    let mut tensors = dense_tensors();
    tensors.extend(head_tensors());
    let source = tempfile::tempdir().unwrap();
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(false), &borrowed);
    let built = build_from_inventories(&[("hc-artifact".to_string(), inventory)]);

    assert!(
        !built
            .graph
            .objects
            .iter()
            .any(|o| o.kind == ObjectKind::HyperConnectionHead),
        "a single-stream component must not own a hyper-connection head"
    );
    for group in ["hc_head_fn", "hc_head_base", "hc_head_scale"] {
        let unplaced = built
            .unplaced
            .iter()
            .find(|u| u.prefix == group)
            .unwrap_or_else(|| panic!("{group} was placed anyway: {:?}", built.unplaced));
        assert!(
            unplaced.reason.contains("declares no Sinkhorn-split"),
            "{group}: {}",
            unplaced.reason
        );
        assert!(
            unplaced.reason.contains("recognised"),
            "{}",
            unplaced.reason
        );
    }
}

/// **The payload half runs, and the control proves what still refuses.**
/// Addressability satisfied, the executor's preparation step prepares a
/// hyper-connected stack that carries a head object and executes it
/// (wave 19); the same stack with no head object is refused at the
/// executor's door, naming the HEAD — a whole-stack image has no declared
/// reduction from the bundle — and not the topology. Both halves against
/// the same fixture family, so the refusal cannot be a broken store or a
/// broken plan wearing the topology's name.
#[test]
fn a_head_bearing_stack_executes_and_a_headless_whole_stack_is_refused_at_the_executors_door() {
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(head_tensors());
    let with_head = plan(config(true), tensors);
    let plan_with_head = with_head.outcome.plan.as_ref().unwrap();
    assert!(plan_with_head.hyper_connection_head.is_some());
    let store = OperandStore::open(with_head.container.path(), &with_head.inspection).unwrap();
    let trace = execute_text(plan_with_head, &store, &[1, 2, 3])
        .expect("a hyper-connected stack with a head executes");
    assert_eq!(trace.executed_layers, vec![0, 1]);
    assert_eq!(
        trace.logits.expect("the head prices the vocabulary").len(),
        128
    );

    let headless = hyper_connected();
    let hc_plan = headless.outcome.plan.as_ref().unwrap();
    assert!(hc_plan.hyper_connection_head.is_none());
    let store = OperandStore::open(headless.container.path(), &headless.inspection).unwrap();
    let err = execute_text(hc_plan, &store, &[1, 2, 3])
        .unwrap_err()
        .to_string();
    assert!(err.contains("hyper_connection_head"), "{err}");
    assert!(err.contains("layer-range"), "{err}");
    assert!(
        !err.contains("traversal"),
        "the topology's old refusal must not reappear: {err}"
    );

    // The control: one stream, same estate minus the sites, executes.
    let single = plan(config(false), dense_tensors());
    let single_plan = single.outcome.plan.as_ref().unwrap();
    assert_eq!(
        single_plan.residual_topology,
        ResidualTopology::SingleStream
    );
    let store = OperandStore::open(single.container.path(), &single.inspection).unwrap();
    execute_text(single_plan, &store, &[1, 2, 3]).expect("a single-stream plan executes");
}

/// **The encode boundary follows the same fact.** The production writers
/// admit a hyper-connected estate that carries a head object — its
/// `hc_*` keys are carried and its execution surface is executable — and
/// refuse the same estate without one, itemising the head's finding.
/// Wave 18 closed addressability without opening a path that wrote a
/// container it could not run; wave 19 opens exactly the path it can.
#[test]
fn the_production_encode_gate_admits_a_head_bearing_container_and_refuses_a_headless_one() {
    let source = tempfile::tempdir().unwrap();
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    tensors.extend(head_tensors());
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(true), &borrowed);
    let named = vec![("hc-with-head".to_string(), inventory)];
    // Named first, so a refusal reads as the finding it is rather than
    // as a count.
    let system = plan_system(&named);
    let blocking: Vec<String> = system
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(
        system.admissible,
        "a head-bearing hyper-connected estate is admissible: {blocking:#?}"
    );
    let out = tempfile::tempdir().unwrap();
    encode_system_unenforced(&named, out.path())
        .expect("a head-bearing hyper-connected estate encodes");

    let source = tempfile::tempdir().unwrap();
    let mut tensors = dense_tensors();
    tensors.extend(site_tensors());
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let inventory = custom_artifact(source.path(), &config(true), &borrowed);
    let out = tempfile::tempdir().unwrap();
    let err = encode_system_unenforced(&[("hc-headless".to_string(), inventory)], out.path())
        .unwrap_err()
        .to_string();
    assert!(err.contains("inadmissible"), "{err}");
    assert!(err.contains("hyper_connection_head"), "{err}");
}

/// A single-stream plan serialises exactly as it did before the topology
/// travelled on the plan: no `residual_topology`, no `hyper_connection`,
/// no `hyper_connection_head` key. Every stored plan stays byte-comparable.
#[test]
fn a_single_stream_plan_serialises_as_before_wave_18() {
    let single = plan(config(false), dense_tensors());
    let json = serde_json::to_string(single.outcome.plan.as_ref().unwrap()).unwrap();
    assert!(!json.contains("residual_topology"), "{json}");
    assert!(!json.contains("hyper_connection"), "{json}");

    let hc = hyper_connected();
    let json = serde_json::to_string(hc.outcome.plan.as_ref().unwrap()).unwrap();
    assert!(json.contains("\"residual_topology\""), "{json}");
    assert!(json.contains("\"hyper_connection\""), "{json}");
    assert!(json.contains("\"mix_fn\""), "{json}");
}
