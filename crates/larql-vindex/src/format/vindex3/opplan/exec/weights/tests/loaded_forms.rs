//! The loaded forms' own contracts: binding a fine-grained FP8 pair, and
//! the allocation inventory each multi-buffer form reports.

use super::*;
use crate::format::vindex3::opplan::exec::operands::RawOperand;
use crate::format::vindex3::opplan::exec::quantise::quantise_q8_indexed_for_test;
use crate::format::vindex3::opplan::OperandRef;

/// Rows and columns of the FP8 fixture: two tiles each way.
const FP8_ROWS: usize = 4;
const FP8_COLS: usize = 8;
/// Its scale grid: one scale per `2 x 4` tile.
const SCALE_ROWS: usize = 2;
const SCALE_COLS: usize = 2;
/// The grid's tensor name, carried into every refusal.
const GRID: &str = "q_proj.weight_scale_inv";
/// FP8 codes are one byte per element.
const DTYPE_FP8: &str = "F8_E4M3";

fn operand(shape: &[usize]) -> OperandRef {
    OperandRef {
        object: "layers".to_string(),
        tensor: "q_proj.weight".to_string(),
        dtype: String::new(),
        shape: shape.to_vec(),
    }
}

fn codes(len: usize) -> RawOperand {
    RawOperand {
        dtype: DTYPE_FP8.to_string(),
        bytes: vec![0u8; len],
    }
}

fn scales() -> Vec<f32> {
    vec![1.0; SCALE_ROWS * SCALE_COLS]
}

/// A well-formed pair binds with the tile DERIVED from the two shapes.
#[test]
fn an_fp8_pair_binds_with_the_tile_its_shapes_imply() {
    let loaded = load_fp8_block(
        &operand(&[FP8_ROWS, FP8_COLS]),
        codes(FP8_ROWS * FP8_COLS),
        GRID,
        &[SCALE_ROWS, SCALE_COLS],
        scales(),
    )
    .unwrap();
    let LoadedWeight::Fp8Block {
        codes,
        scales,
        block_rows,
        block_cols,
        scale_cols,
    } = loaded
    else {
        panic!("an fp8 pair must bind as the fp8 form");
    };
    assert_eq!(codes.logical_len(), FP8_ROWS * FP8_COLS);
    assert_eq!(scales.len(), SCALE_ROWS * SCALE_COLS);
    assert_eq!(
        (block_rows, block_cols),
        (FP8_ROWS / SCALE_ROWS, FP8_COLS / SCALE_COLS)
    );
    assert_eq!(scale_cols, SCALE_COLS);
}

/// Every geometry the pair can disagree on is refused, by name.
#[test]
fn an_fp8_pair_refuses_every_geometry_it_cannot_tile() {
    let refusal = |shape: &[usize], len: usize, grid: &[usize], scales: Vec<f32>| {
        load_fp8_block(&operand(shape), codes(len), GRID, grid, scales)
            .expect_err("a mismatched fp8 pair must be refused")
            .to_string()
    };
    let full = FP8_ROWS * FP8_COLS;
    let grid = [SCALE_ROWS, SCALE_COLS];

    let err = refusal(&[full], full, &grid, scales());
    assert!(err.contains("no reading of any other rank"), "{err}");

    let err = refusal(&[FP8_ROWS, FP8_COLS], full, &[SCALE_ROWS], scales());
    assert!(
        err.contains("not a two-dimensional grid") && err.contains(GRID),
        "{err}"
    );

    // A grid that does not divide the matrix has no tile.
    let err = refusal(&[FP8_ROWS, FP8_COLS], full, &[FP8_ROWS + 1, 1], scales());
    assert!(err.contains("q_proj.weight"), "{err}");

    let err = refusal(&[FP8_ROWS, FP8_COLS], full - 1, &grid, scales());
    assert!(err.contains("E4M3 bytes"), "{err}");

    let err = refusal(&[FP8_ROWS, FP8_COLS], full, &grid, vec![1.0]);
    assert!(err.contains("decodes to 1 values"), "{err}");
}

/// Q8 with its code-sum index holds THREE allocations, one per stream —
/// the index is traffic a residency ledger must see.
#[test]
fn an_indexed_q8_weight_reports_its_index_as_an_allocation() {
    const ROWS: usize = 2;
    const IN: usize = 64;
    let values: Vec<f32> = (0..ROWS * IN).map(|i| i as f32 / IN as f32).collect();
    let indexed = quantise_q8_indexed_for_test(&values, IN);
    let LoadedWeight::Q8 { sums, .. } = &indexed else {
        panic!("the indexed quantiser must produce the q8 variant");
    };
    assert!(!sums.is_empty());
    let allocations = indexed.allocations();
    assert_eq!(allocations.len(), 3, "codes, scales and the index");
    assert_eq!(allocations[2].1, std::mem::size_of_val(&sums[..]));
}

/// The two packed FP4 forms each hold codes and scales as separate
/// page-aligned buffers, and report both at their padded length.
#[test]
fn packed_fp4_forms_report_codes_and_scales_as_two_allocations() {
    const ROWS: usize = 2;
    const K: usize = 32;
    let values: Vec<f32> = (0..ROWS * K).map(|i| i as f32 / K as f32).collect();
    for weight in [
        quantize_mxfp4(&values, ROWS, K, "w").unwrap(),
        quantize_nvfp4(&values, ROWS, K, "w").unwrap(),
    ] {
        let allocations = weight.allocations();
        assert_eq!(allocations.len(), 2, "{:?}", weight.format());
        assert_eq!(allocations.len(), weight.padded_allocations());
        for (_, bytes) in allocations {
            assert!(bytes.is_multiple_of(DEVICE_PAGE_ALIGN));
        }
    }
}
