//! The component-level object closures — the hyper-connection head and
//! the attention-residual exit — refuse every operand-level shortfall by
//! itemised defect, and bind only a complete, well-shaped set.

use super::*;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use larql_models::config::HyperConnection;

const COMPONENT: &str = "target";
const HIDDEN: usize = 8;
const STREAMS: usize = 4;
const SINKHORN_ITERS: usize = 20;
const SINKHORN_EPS: f64 = 1e-6;
/// The head's scale is a single element (`HC_HEAD_SCALE_LEN`).
const SCALE_LEN: usize = 1;
/// A width no operand of either object has.
const WRONG_WIDTH: usize = HIDDEN + 1;
/// A tensor name no object table classifies.
const UNCLASSIFIED: &str = "not_an_operand.weight";
const HC_FN: &str = "hc_head_fn";
const HC_BASE: &str = "hc_head_base";
const HC_SCALE: &str = "hc_head_scale";
const EXIT_NORM: &str = "output_attn_res_norm.weight";
const EXIT_PROJ: &str = "output_attn_res_proj.weight";

fn object(kind: ObjectKind, id: &str) -> LogicalObject {
    LogicalObject {
        id: format!("{COMPONENT}.{id}"),
        component: COMPONENT.to_string(),
        kind,
        source_bindings: Vec::new(),
        representations: Vec::new(),
    }
}

fn tensor(name: &str, shape: Vec<usize>) -> SegmentTensor {
    SegmentTensor {
        name: name.to_string(),
        dtype: "BF16".to_string(),
        shape,
        offset: 0,
        len: 0,
    }
}

fn hc() -> HyperConnection {
    HyperConnection {
        streams: STREAMS,
        sinkhorn_iters: SINKHORN_ITERS,
        sinkhorn_eps: SINKHORN_EPS,
    }
}

fn head_object() -> LogicalObject {
    object(ObjectKind::HyperConnectionHead, "hyper_connection_head")
}

fn head_tensors() -> Vec<SegmentTensor> {
    vec![
        tensor(HC_FN, vec![STREAMS, STREAMS * HIDDEN]),
        tensor(HC_BASE, vec![STREAMS]),
        tensor(HC_SCALE, vec![SCALE_LEN]),
    ]
}

fn head(
    tensors: &[SegmentTensor],
    declared: Option<HyperConnection>,
) -> (bool, Vec<ClosureDefect>) {
    let mut defects = Vec::new();
    let bound =
        hyper_connection_head_closure(&head_object(), tensors, declared, HIDDEN, &mut defects);
    (bound.is_some(), defects)
}

fn has_shape_defect(defects: &[ClosureDefect], needle: &str) -> bool {
    defects
        .iter()
        .any(|d| matches!(d, ClosureDefect::ObjectShape { detail, .. } if detail.contains(needle)))
}

#[test]
fn a_complete_head_binds_its_three_operands() {
    let (bound, defects) = head(&head_tensors(), Some(hc()));
    assert!(defects.is_empty(), "{defects:?}");
    assert!(bound);
}

#[test]
fn head_operands_on_a_single_stream_component_name_the_absent_topology() {
    let (bound, defects) = head(&head_tensors(), None);
    assert!(!bound);
    assert_eq!(defects.len(), head_tensors().len(), "{defects:?}");
    assert!(defects.iter().all(|d| matches!(
        d,
        ClosureDefect::OperandImpliesAbsentOp { required_primitive, .. }
            if required_primitive == HC_HEAD_ON_SINGLE_STREAM
    )));
}

#[test]
fn an_unclassified_head_operand_is_itemised_and_its_missing_role_named() {
    let mut tensors = head_tensors();
    tensors[1] = tensor(UNCLASSIFIED, vec![STREAMS]);
    let (bound, defects) = head(&tensors, Some(hc()));
    assert!(!bound);
    assert!(defects.iter().any(|d| matches!(
        d,
        ClosureDefect::UnclassifiedOperand { tensor, .. } if tensor == UNCLASSIFIED
    )));
    assert!(has_shape_defect(&defects, "no operand for the head's Base"));
}

#[test]
fn a_head_operand_at_the_wrong_geometry_is_refused() {
    let mut tensors = head_tensors();
    tensors[1] = tensor(HC_BASE, vec![WRONG_WIDTH]);
    let (_, defects) = head(&tensors, Some(hc()));
    assert!(defects.iter().any(|d| matches!(
        d,
        ClosureDefect::GeometryMismatch { actual, .. } if *actual == vec![WRONG_WIDTH]
    )));
}

#[test]
fn two_head_operands_claiming_one_role_are_refused() {
    let mut tensors = head_tensors();
    tensors.push(tensor(HC_SCALE, vec![SCALE_LEN]));
    let (_, defects) = head(&tensors, Some(hc()));
    assert!(has_shape_defect(
        &defects,
        "two operands claim the head's Scale"
    ));
}

fn exit(tensors: &[SegmentTensor]) -> (bool, Vec<ClosureDefect>) {
    let mut defects = Vec::new();
    let object = object(ObjectKind::AttentionResidualExit, "attention_residual_exit");
    let bound = attention_residual_exit_closure(&object, tensors, true, HIDDEN, &mut defects);
    (bound.is_some(), defects)
}

#[test]
fn the_exit_itemises_unclassified_misshapen_and_duplicated_operands() {
    let (bound, defects) = exit(&[
        tensor(UNCLASSIFIED, vec![HIDDEN]),
        tensor(EXIT_NORM, vec![WRONG_WIDTH]),
        tensor(EXIT_PROJ, vec![1, HIDDEN]),
        tensor(EXIT_PROJ, vec![1, HIDDEN]),
    ]);
    // The misshapen norm still binds its role; the defects refuse the plan.
    assert!(bound);
    assert!(defects.iter().any(|d| matches!(
        d,
        ClosureDefect::UnclassifiedOperand { tensor, .. } if tensor == UNCLASSIFIED
    )));
    assert!(defects.iter().any(|d| matches!(
        d,
        ClosureDefect::GeometryMismatch { expected, .. } if *expected == vec![HIDDEN]
    )));
    assert!(has_shape_defect(
        &defects,
        "two operands claim the exit's Proj"
    ));
}

#[test]
fn an_object_without_a_representation_has_no_tensor_table() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        COMPONENT,
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let bare = head_object();
    let err = object_tensors(&inspection, container.path(), &bare)
        .unwrap_err()
        .to_string();
    assert!(err.contains("carries no representation"), "{err}");
}
