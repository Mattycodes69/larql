//! The mixer-on-hyper-connection arm, reached

use super::*;

/// **The unjudged arm is reachable, and its `||` is load-bearing.** PR
/// #417's advisory mutation run replaced `is_mamba2() || is_conv_qkv()`
/// with `&&` in the arm that refuses a hyper-connected component's
/// mixer-only layer, and nothing noticed — no fixture declared the
/// topology on a Mamba2 stack. This one does: the pure-SSM miniature with
/// `hc_mult`, `hc_sinkhorn_iters` and `hc_eps` added to its own config.
///
/// What the arm protects against is stated by the control below: without
/// the declaration the same estate CLOSES, so under the declaration a
/// build that lost this arm would plan a hyper-connected component whose
/// every layer carries no site — a plan that disagrees with itself. The
/// refusal must name every layer, because the traversal has no judged
/// form for a one-sublayer block in a bundle, and a stack that is
/// refused on layer 0 alone would read as "layer 1 is fine".
#[test]
fn a_hyper_connected_mixer_only_stack_is_refused_as_unjudged_on_every_layer() {
    use super::super::mamba2::miniature_mamba2;
    use larql_models::inventory::build_inventory;

    let plan_mixer = |declare_topology: bool| {
        let dir = tempfile::tempdir().unwrap();
        miniature_mamba2(dir.path(), None);
        if declare_topology {
            // The fixture writes a bare `Infinity`, deliberately, so its
            // config is not JSON to serde. Declare the topology as a text
            // insertion at the opening brace instead.
            let path = dir.path().join("config.json");
            let config = std::fs::read_to_string(&path).unwrap();
            let declared = config.replacen(
                '{',
                &format!(
                    "{{\"hc_mult\":{STREAMS},\"hc_sinkhorn_iters\":{SINKHORN_ITERS},\
                     \"hc_eps\":{SINKHORN_EPS},"
                ),
                1,
            );
            std::fs::write(&path, declared).unwrap();
        }
        let inventory = build_inventory(dir.path()).unwrap();
        // The declaration reached the resolved surface — so a refusal
        // below is the arm's, not a parse that silently dropped the keys.
        let topology = inventory
            .resolved
            .execution
            .as_ref()
            .and_then(|e| e.residual_topology);
        assert_eq!(
            topology.map(|t| t.streams()),
            Some(if declare_topology { STREAMS } else { 1 })
        );
        let named = vec![("mamba2-mini".to_string(), inventory)];
        let system = plan_system(&named);
        let container = tempfile::tempdir().unwrap();
        encode_graph(&system.graph, &named, container.path()).unwrap();
        let inspection = inspect_container(container.path(), false).unwrap();
        let layers = inspection
            .graph
            .components
            .iter()
            .find(|c| c.id == "target")
            .expect("the mixer stack is the target component")
            .num_layers;
        (
            plan_component_ops(&inspection, container.path(), "target").unwrap(),
            layers,
        )
    };

    // The control first: the same estate, one stream, closes.
    let (plain, layers) = plan_mixer(false);
    assert!(plain.closed(), "{:?}", plain.defects);
    assert!(layers >= 2, "the fixture must have more than one layer");

    // Under the declaration: refused on EVERY layer, by name, and for
    // nothing else.
    let (hyper, _) = plan_mixer(true);
    assert!(hyper.plan.is_none());
    let mut refused_layers: Vec<usize> = hyper
        .outcome_layers_refused_for(HC_ON_MIXER_FACT_FRAGMENT)
        .collect();
    refused_layers.sort_unstable();
    assert_eq!(
        refused_layers,
        (0..layers).collect::<Vec<_>>(),
        "every mixer layer must be refused: {:?}",
        hyper.defects
    );
    assert_eq!(
        hyper.defects.len(),
        layers,
        "the arm is the ONLY refusal on this estate: {:?}",
        hyper.defects
    );
}
