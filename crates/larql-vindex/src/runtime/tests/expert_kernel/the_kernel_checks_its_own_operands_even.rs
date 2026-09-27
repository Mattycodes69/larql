//! The kernel checks its own operands, even unvalidated

use super::*;

#[test]
fn a_gate_up_slab_of_the_wrong_height_is_refused_by_the_kernel() {
    // A fused region holds both halves. One that is only `INTERMEDIATE` rows
    // tall would have the kernel read the gate half as gate+up, and the up half
    // from past the end of the region.
    let fixture = Fixture::new();
    let bytes = q4k_bytes(INTERMEDIATE, HIDDEN, GATE_UP_SEED);
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &bytes,
                    RegionFormat::Q4K,
                    INTERMEDIATE as u32,
                    HIDDEN as u32,
                ),
            },
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::Rows
                    && expected == FUSED_PROJECTION_HALVES * INTERMEDIATE
                    && found == INTERMEDIATE
        ),
        "{err}"
    );
}

#[test]
fn a_gate_up_slab_that_contracts_the_wrong_width_is_refused_by_the_kernel() {
    // The kernel strides gate_up by `hidden / block` super-blocks per row, so a
    // region storing a different width is read at the wrong offset from row one.
    let fixture = Fixture::new();
    let narrow = HIDDEN + Q4K_BLOCK_ELEMS;
    let bytes = q4k_bytes(FUSED_PROJECTION_HALVES * INTERMEDIATE, narrow, GATE_UP_SEED);
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &bytes,
                    RegionFormat::Q4K,
                    (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                    narrow as u32,
                ),
            },
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::InputWidth && expected == HIDDEN && found == narrow
        ),
        "{err}"
    );
}

#[test]
fn a_down_region_of_the_wrong_height_is_refused_by_the_kernel() {
    // `down` contracts the intermediate axis away and produces the bank's
    // output width. A shorter one silently produces a shorter residual delta.
    let fixture = Fixture::new();
    let short = HIDDEN - Q4K_BLOCK_ELEMS / 2;
    let bytes = q4k_bytes(short, INTERMEDIATE_PADDED, DOWN_SEED);
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            down: column_prefix(
                DOWN_REGION_SET,
                &bytes,
                RegionFormat::Q4K,
                short as u32,
                INTERMEDIATE_PADDED as u32,
                INTERMEDIATE as u32,
            ),
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::OutputWidth && expected == HIDDEN && found == short
        ),
        "{err}"
    );
}

#[test]
fn a_down_region_exposing_the_wrong_live_width_is_refused_by_the_kernel() {
    // Stored correctly, but the role claims more live columns than the bank
    // means. The kernel passes `intermediate` to the incumbent as the count of
    // activation values to compute, so this is the difference between the
    // padding being inert and being read as data.
    let fixture = Fixture::new();
    let wrong = INTERMEDIATE + 1;
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            down: column_prefix(
                DOWN_REGION_SET,
                &fixture.down,
                RegionFormat::Q4K,
                HIDDEN as u32,
                INTERMEDIATE_PADDED as u32,
                wrong as u32,
            ),
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::Columns && expected == INTERMEDIATE && found == wrong
        ),
        "{err}"
    );
}

#[test]
fn a_decomposed_projection_is_refused_at_execution_too() {
    let fixture = Fixture::new();
    let half = fixture.gate_up.len() / FUSED_PROJECTION_HALVES;
    let err = execute_unvalidated(
        &fixture,
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Decomposed {
                gate: direct(
                    GATE_UP_REGION_SET,
                    &fixture.gate_up[..half],
                    RegionFormat::Q4K,
                    INTERMEDIATE as u32,
                    HIDDEN as u32,
                ),
                up: direct(
                    GATE_UP_REGION_SET,
                    &fixture.gate_up[half..],
                    RegionFormat::Q4K,
                    INTERMEDIATE as u32,
                    HIDDEN as u32,
                ),
            },
            ..fixture.q4k_expert()
        },
    );
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::Arrangement { .. },
                ..
            }
        ),
        "{err}"
    );
}
