//! Attention operand loading refuses a bias set operand closure never
//! emits, the layer-scale reader refuses anything but one value, and the
//! interpreter's projection helper routes through the backend.

use super::super::super::attention_ops::{layer_scalar_of, project};
use super::super::super::backend::WeightSlice;
use super::super::decode::fixture;
use super::*;

/// A 2x2 identity weight and the vector it maps to itself.
const IDENTITY: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const X: [f32; 2] = [3.0, -2.0];
const WIDTH: usize = 2;

#[test]
fn a_partial_attention_bias_set_is_refused_at_load() {
    let (_dir, mut plan, store) = fixture();
    let op = plan.layers[0]
        .attention
        .softmax_mut()
        .expect("the dense fixture attends with softmax");
    // Q alone: closure emits Q/K/V together, so this set is one it never
    // produced.
    op.q_bias = Some(op.q.clone());
    op.k_bias = None;
    op.v_bias = None;
    op.o_bias = None;
    let err = match PreparedOperands::load(&plan, &store, &ReferenceBackend, ExecutionSlice::Full) {
        Ok(_) => panic!("a partial bias set must refuse"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("partial bias set"), "{err}");
}

#[test]
fn a_layer_scale_is_exactly_one_value() {
    assert_eq!(layer_scalar_of(&[X[0]]).unwrap(), X[0]);
    let err = layer_scalar_of(&X).unwrap_err().to_string();
    assert!(err.contains("holds 2 values"), "{err}");
    assert!(layer_scalar_of(&[]).is_err());
}

#[test]
fn the_projection_helper_projects_through_the_backend() {
    let y = project(
        &ReferenceBackend,
        WeightSlice::F32(&IDENTITY),
        WIDTH,
        WIDTH,
        &X,
    )
    .unwrap();
    assert_eq!(y, X.to_vec());
}
