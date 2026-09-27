//! The per-BLOCK activation arms of the three integer projectors, reached
//! through `project_rows_under` so one process can run every arm.
//!
//! `project_rows` resolves its scaling and code from `OnceLock`-cached
//! environment reads, so a suite that only calls the trait method runs
//! the per-tensor arm and nothing else — which is exactly how the blocked
//! arms went unexercised on every target without `SDOT`. These tests run
//! on every target: each is judged against the format's portable
//! definition, with a tolerance only where the vectorised kernels are
//! documented to reassociate.

use super::super::super::arithmetic::ScaleSpan;
use super::super::super::integer::{
    q4_row_portable, q4_row_subblocked, q8_row_asym_portable, q8_row_portable,
    quantise_activation_asymmetric, quantise_activation_blocked, scaling_for_arm, ActivationCode,
};
use super::*;
use crate::format::vindex3::opplan::exec::quantise::quantise_q8_indexed_for_test;

/// Output rows per fixture: enough that a row-offset bug shows.
const OUT: usize = 3;
/// Input width: several weight blocks plus a ragged tail is not allowed
/// for Q4 (two codes a byte), so a whole number of weight blocks.
const IN: usize = Q8_BLOCK * 3;
/// Activation blocks finer than the weight block: the SDOT-sized one, a
/// middle one, and the weight block itself.
const ABLOCKS: [usize; 3] = [16, 32, Q8_BLOCK];
/// Reassociation budget: the SDOT kernels sum in a different float order
/// from the portable definition on aarch64. Portable targets are exact.
const REL_TOL: f32 = 1e-5;

fn close(got: f32, want: f32, what: &str) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= REL_TOL * scale,
        "{what}: kernel {got} vs definition {want}"
    );
}

/// Fold the weight's block scale into each finer activation block.
fn folded(weight_scales: &[f32], act: &[f32], per_weight: usize) -> Vec<f32> {
    act.iter()
        .enumerate()
        .map(|(b, a)| weight_scales[b / per_weight] * *a)
        .collect()
}

#[test]
fn a_blocked_symmetric_q8_projection_computes_the_folded_definition() {
    let w = lcg_values(OUT * IN, 301);
    let x = outlier_activation(IN, 302);
    let (codes, scales) = q8_parts(&w, IN);
    let per_row = IN.div_ceil(Q8_BLOCK);
    for ablock in ABLOCKS {
        let mut got = vec![0.0f32; OUT];
        Q8xQ8.project_rows_under(
            WeightRows::Q8 {
                codes: &codes,
                scales: &scales,
                sums: &[],
                block: Q8_BLOCK,
            },
            &x,
            &mut got,
            ScaleSpan::Block(ablock),
            ActivationCode::Symmetric,
        );
        let (qx, act_scales) = quantise_activation_blocked(&x, ablock);
        for (o, g) in got.iter().enumerate() {
            let fs = folded(
                &scales[o * per_row..(o + 1) * per_row],
                &act_scales,
                Q8_BLOCK / ablock,
            );
            let want = q8_row_portable(&codes[o * IN..(o + 1) * IN], &fs, &qx, IN, ablock);
            close(*g, want, &format!("q8 sym ablock {ablock} row {o}"));
        }
    }
}

#[test]
fn a_blocked_asymmetric_q8_projection_computes_the_offset_definition() {
    let w = lcg_values(OUT * IN, 311);
    let x = outlier_activation(IN, 312);
    let (codes, scales) = q8_parts(&w, IN);
    let per_row = IN.div_ceil(Q8_BLOCK);
    for ablock in ABLOCKS {
        let mut got = vec![0.0f32; OUT];
        Q8xQ8.project_rows_under(
            WeightRows::Q8 {
                codes: &codes,
                scales: &scales,
                sums: &[],
                block: Q8_BLOCK,
            },
            &x,
            &mut got,
            ScaleSpan::Block(ablock),
            ActivationCode::Asymmetric,
        );
        let (qx, act_scales, act_mids) = quantise_activation_asymmetric(&x, ablock);
        let per_weight = Q8_BLOCK / ablock;
        for (o, g) in got.iter().enumerate() {
            let ws = &scales[o * per_row..(o + 1) * per_row];
            let fs = folded(ws, &act_scales, per_weight);
            let fm = folded(ws, &act_mids, per_weight);
            let want =
                q8_row_asym_portable(&codes[o * IN..(o + 1) * IN], &fs, &fm, &qx, IN, ablock);
            close(*g, want, &format!("q8 asym ablock {ablock} row {o}"));
        }
    }
}

/// The weight-code index is an execution aid, never a semantic: a
/// projection that consumes it answers what the recomputing one does.
#[test]
fn the_code_sum_index_changes_no_asymmetric_projection() {
    let w = lcg_values(OUT * IN, 321);
    let x = outlier_activation(IN, 322);
    let LoadedWeight::Q8 {
        codes,
        scales,
        sums,
    } = quantise_q8_indexed_for_test(&w, IN)
    else {
        panic!("the indexed quantiser must produce the q8 variant");
    };
    assert!(!sums.is_empty(), "the fixture must carry the index");
    for ablock in ABLOCKS {
        let run = |sums: &[i16]| {
            let mut out = vec![0.0f32; OUT];
            Q8xQ8.project_rows_under(
                WeightRows::Q8 {
                    codes: &codes,
                    scales: &scales,
                    sums,
                    block: Q8_BLOCK,
                },
                &x,
                &mut out,
                ScaleSpan::Block(ablock),
                ActivationCode::Asymmetric,
            );
            out
        };
        let indexed = run(&sums);
        let recomputed = run(&[]);
        for o in 0..OUT {
            close(
                indexed[o],
                recomputed[o],
                &format!("indexed vs recomputed, ablock {ablock} row {o}"),
            );
        }
    }
}

#[test]
fn a_blocked_q4_projection_routes_whole_and_sub_blocks_to_their_definitions() {
    let w = lcg_values(OUT * IN, 331);
    let x = outlier_activation(IN, 332);
    let (packed, scales) = q4_parts(&w, IN);
    let per_row = IN.div_ceil(Q4_BLOCK);
    let bytes_per_row = IN / 2;
    for ablock in ABLOCKS {
        let mut got = vec![0.0f32; OUT];
        Q4xQ8.project_rows_under(
            WeightRows::Q4 {
                packed: &packed,
                scales: &scales,
                block: Q4_BLOCK,
            },
            &x,
            &mut got,
            ScaleSpan::Block(ablock),
        );
        let (qx, act_scales) = quantise_activation_blocked(&x, ablock);
        let per_weight = Q4_BLOCK / ablock;
        for (o, g) in got.iter().enumerate() {
            let fs = folded(
                &scales[o * per_row..(o + 1) * per_row],
                &act_scales,
                per_weight,
            );
            let row = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
            let want = if per_weight == 1 {
                q4_row_portable(row, &fs, &qx, IN, Q4_BLOCK)
            } else {
                q4_row_subblocked(row, &fs, &qx, IN, Q4_BLOCK, ablock)
            };
            close(*g, want, &format!("q4 ablock {ablock} row {o}"));
        }
    }
}

/// The bf16 control under a blocked activation is the exact kernel run on
/// the reconstructed activation, for both codes — bit for bit, since it
/// defers to that kernel rather than running its own dot.
#[test]
fn the_blocked_bf16_control_is_the_exact_kernel_on_the_reconstructed_activation() {
    let w = lcg_values(OUT * IN, 341);
    let bits: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let x = outlier_activation(IN, 342);
    for ablock in ABLOCKS {
        let (qx, s) = quantise_activation_blocked(&x, ablock);
        let sym: Vec<f32> = qx
            .iter()
            .enumerate()
            .map(|(i, c)| *c as f32 * s[i / ablock])
            .collect();
        let (qa, sa, ma) = quantise_activation_asymmetric(&x, ablock);
        let asym: Vec<f32> = qa
            .iter()
            .enumerate()
            .map(|(i, c)| *c as f32 * sa[i / ablock] + ma[i / ablock])
            .collect();
        for (code, rx) in [
            (ActivationCode::Symmetric, sym),
            (ActivationCode::Asymmetric, asym),
        ] {
            let mut got = vec![0.0f32; OUT];
            Bf16xQ8.project_rows_under(
                WeightRows::Bf16(&bits),
                &x,
                &mut got,
                ScaleSpan::Block(ablock),
                code,
            );
            let mut want = vec![0.0f32; OUT];
            FusedBf16.project_rows(WeightRows::Bf16(&bits), &rx, &mut want);
            for o in 0..OUT {
                assert_eq!(
                    got[o].to_bits(),
                    want[o].to_bits(),
                    "{code:?} ablock {ablock} row {o}: {} vs {}",
                    got[o],
                    want[o]
                );
            }
        }
    }
}

/// Each arm's `b` suffix selects per-block scaling, at the activation
/// block the process resolved; anything else — including the unsuffixed
/// arm and no value at all — is per-tensor.
#[test]
fn only_a_blocked_arm_string_selects_block_scaling() {
    use super::super::super::integer::activation_block;
    for arm in ["bf16xq8b", "q8xq8b", " q4xq8b "] {
        assert_eq!(
            scaling_for_arm(Some(arm)),
            ScaleSpan::Block(activation_block()),
            "{arm}"
        );
    }
    for arm in [Some("q8xq8"), Some("q4xq8"), Some("nonsense"), None] {
        assert_eq!(scaling_for_arm(arm), ScaleSpan::Tensor, "{arm:?}");
    }
}
