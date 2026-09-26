//! Public execution path: `q4k_q8k_matvec_parallel`

use super::*;

/// Valid shape, unchanged numerics: the parallel entry is bit-identical to
/// the scalar reference on the fixture the refusal tests below reuse.
#[test]
fn q8k_matvec_parallel_computes_exactly_the_scalar_reference_on_the_fixture() {
    let (w, q, rows, cols) = public_path_fixture();
    let mut reference = vec![0.0f32; rows];
    q4k_q8k_matvec_scalar(&mut reference, &q, &w, rows, cols).expect("valid shape");
    let mut out = vec![7.0f32; rows];
    q4k_q8k_matvec_parallel(&mut out, &q, &w, rows, cols, "Q4_K").expect("valid shape");
    assert!(
        reference.iter().any(|&v| v != 0.0),
        "control: the fixture must not be all zeros"
    );
    for r in 0..rows {
        assert_eq!(
            out[r].to_bits(),
            reference[r].to_bits(),
            "row {r}: parallel={} scalar={}",
            out[r],
            reference[r]
        );
    }
}

/// An activation that is not `cols` long is refused by name — the hand-asm
/// kernels read `qs` through a bare pointer, so this used to be an OOB
/// read hidden behind a zero-filled output.
#[test]
fn q8k_matvec_parallel_refuses_an_activation_of_the_wrong_length() {
    let (w, _, rows, cols) = public_path_fixture();
    let wrong = quantize_x_to_q8k(&vec![0.5f32; cols + 256]);
    let mut out = vec![7.0f32; rows];
    let err = q4k_q8k_matvec_parallel(&mut out, &wrong, &w, rows, cols, "Q4_K")
        .expect_err("activation length must match cols");
    assert_eq!(out, vec![7.0f32; rows], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_q8k_matvec_parallel");
    assert_eq!((err.x_len, err.cols), (cols + 256, cols));
    assert!(
        err.to_string()
            .contains("activation length 768 for 512 cols"),
        "{err}"
    );
}

/// An output shorter than `rows` is refused before any row is written.
#[test]
fn q8k_matvec_parallel_refuses_an_output_shorter_than_rows() {
    let (w, q, rows, cols) = public_path_fixture();
    let mut out = vec![7.0f32; rows - 1];
    let err = q4k_q8k_matvec_parallel(&mut out, &q, &w, rows, cols, "Q4_K")
        .expect_err("output must hold rows");
    assert_eq!(out, vec![7.0f32; rows - 1]);
    assert_eq!((err.out_len, err.rows), (rows - 1, rows));
}

/// A weight slab shorter than `rows` packed rows is refused by name. This
/// case used to panic at the entry point; it is now the same typed
/// refusal the per-row kernels return, so a caller with a channel can
/// carry it up.
#[test]
fn q8k_matvec_parallel_refuses_a_short_weight_slab() {
    let (w, q, rows, cols) = public_path_fixture();
    let short = &w[..w.len() - BLOCK_BYTES];
    let mut out = vec![7.0f32; rows];
    let err = q4k_q8k_matvec_parallel(&mut out, &q, short, rows, cols, "Q4_K")
        .expect_err("short slab must be refused");
    assert_eq!(out, vec![7.0f32; rows]);
    assert_eq!(
        (err.weight_bytes, err.needed_bytes),
        (w.len() - BLOCK_BYTES, w.len())
    );
}

/// The refusal reaches the `FormatRoute` registry's kernel pointer too:
/// the Q6_K route returns the same typed error.
#[test]
fn q8k_matvec_parallel_refuses_through_the_q6k_route_as_well() {
    let (_, q, rows, cols) = public_path_fixture();
    let short = vec![0u8; Q6K_BLOCK_BYTES];
    let mut out = vec![7.0f32; rows];
    let err = q4k_q8k_matvec_parallel(&mut out, &q, &short, rows, cols, "Q6_K")
        .expect_err("short Q6_K slab must be refused");
    assert_eq!(out, vec![7.0f32; rows]);
    assert_eq!(err.needed_bytes, rows * 2 * Q6K_BLOCK_BYTES);
}
