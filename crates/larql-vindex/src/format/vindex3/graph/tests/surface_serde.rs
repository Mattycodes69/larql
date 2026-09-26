//! A surface written before `residual_topology` existed reads back as the
//! single-stream topology every such container meant.

use larql_models::config::ResidualTopology;

use crate::format::vindex3::graph::build_from_inventories;
use crate::format::vindex3::graph::surface::ExecutionSurface;
use crate::format::vindex3::plan::tests_support::glimmer_shaped_target;

const RESIDUAL_TOPOLOGY_FIELD: &str = "residual_topology";

#[test]
fn a_surface_without_a_residual_topology_reads_as_single_stream() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("target".to_string(), glimmer_shaped_target(dir.path()))];
    let built = build_from_inventories(&named);
    let surface = built
        .graph
        .components
        .iter()
        .find_map(|c| c.execution.clone())
        .expect("the fixture builds a surface");
    let mut json = serde_json::to_value(&surface).unwrap();
    let removed = json
        .as_object_mut()
        .unwrap()
        .remove(RESIDUAL_TOPOLOGY_FIELD);
    assert!(removed.is_some(), "the field is serialised when present");
    let read: ExecutionSurface = serde_json::from_value(json).unwrap();
    assert_eq!(read.residual_topology, ResidualTopology::SingleStream);
}
