//! The integer projectors' contract outside the arithmetic: which weight
//! representations each consumes, how the many-position form lays its
//! output out, and what the portable definitions do with a scale list
//! that runs past the row.

use super::super::super::integer::{q4_row_portable, q8_row_portable};
use super::super::super::stationary;
use super::*;

/// Positions in a many-position call: more than one, so the interleaved
/// layout is distinguishable from a single-position one.
const POSITIONS: usize = 3;
/// Output rows of the many-position fixture.
const ROWS: usize = 4;
/// A one-block row for the portable-definition fixtures.
const ONE_BLOCK: usize = Q8_BLOCK;

/// **The many-position form is the single-position form, interleaved.**
///
/// Whichever schedule the kernel picks — stationary where its geometry
/// holds, the loop otherwise — position `i` of row `r` lands at
/// `out[r * n + i]` and equals what `project_rows` computes for that
/// position alone.
#[test]
fn a_many_position_q8_projection_is_each_position_interleaved() {
    let in_dim = Q8_BLOCK * 2;
    let w = lcg_values(ROWS * in_dim, 401);
    let (codes, scales) = q8_parts(&w, in_dim);
    let weight = WeightRows::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &[],
        block: Q8_BLOCK,
    };
    let xs: Vec<Vec<f32>> = (0..POSITIONS)
        .map(|p| lcg_values(in_dim, 410 + p as u64))
        .collect();
    let views: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();

    assert_eq!(
        Q8xQ8.is_weight_stationary(weight, in_dim, POSITIONS),
        stationary::supports(weight, in_dim, POSITIONS),
        "the kernel's stationarity claim is the stationary module's, not its own"
    );

    let mut many = vec![0.0f32; ROWS * POSITIONS];
    Q8xQ8.project_rows_many(weight, &views, &mut many, POSITIONS);
    for (p, x) in xs.iter().enumerate() {
        let mut one = vec![0.0f32; ROWS];
        Q8xQ8.project_rows(weight, x, &mut one);
        for r in 0..ROWS {
            assert_eq!(
                many[r * POSITIONS + p].to_bits(),
                one[r].to_bits(),
                "row {r} position {p}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "consumes q8 weights only")]
fn the_q8_kernel_refuses_non_q8_weights() {
    let w = vec![0.0f32; ONE_BLOCK];
    let x = vec![0.0f32; ONE_BLOCK];
    let mut out = vec![0.0f32; 1];
    Q8xQ8.project_rows(WeightRows::F32(&w), &x, &mut out);
}

#[test]
#[should_panic(expected = "consumes q4 weights only")]
fn the_q4_kernel_refuses_non_q4_weights() {
    let w = vec![0.0f32; ONE_BLOCK];
    let x = vec![0.0f32; ONE_BLOCK];
    let mut out = vec![0.0f32; 1];
    Q4xQ8.project_rows(WeightRows::F32(&w), &x, &mut out);
}

#[test]
#[should_panic(expected = "consumes bf16 weights only")]
fn the_bf16_control_refuses_non_bf16_weights() {
    let w = vec![0.0f32; ONE_BLOCK];
    let x = vec![0.0f32; ONE_BLOCK];
    let mut out = vec![0.0f32; 1];
    Bf16xQ8.project_rows(WeightRows::F32(&w), &x, &mut out);
}

/// A scale list longer than the row's blocks stops at the row's end: the
/// surplus scales pair with nothing, so they contribute nothing — never
/// an out-of-range read.
#[test]
fn the_portable_rows_ignore_scales_past_the_row() {
    const SURPLUS: f32 = 1.0e6;
    let codes: Vec<i8> = (0..ONE_BLOCK).map(|i| (i % 7) as i8 - 3).collect();
    let qx: Vec<i8> = (0..ONE_BLOCK).map(|i| (i % 5) as i8 - 2).collect();
    let packed: Vec<u8> = (0..ONE_BLOCK / 2).map(|i| (i % 256) as u8).collect();
    let one = [0.5f32];
    let with_surplus = [0.5f32, SURPLUS];

    assert_eq!(
        q8_row_portable(&codes, &with_surplus, &qx, ONE_BLOCK, ONE_BLOCK).to_bits(),
        q8_row_portable(&codes, &one, &qx, ONE_BLOCK, ONE_BLOCK).to_bits(),
    );
    assert_eq!(
        q4_row_portable(&packed, &with_surplus, &qx, ONE_BLOCK, ONE_BLOCK).to_bits(),
        q4_row_portable(&packed, &one, &qx, ONE_BLOCK, ONE_BLOCK).to_bits(),
    );
}
