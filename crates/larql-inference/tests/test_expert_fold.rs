//! The shard-side expert fold (`ffn::expert_fold::fold`) against a
//! per-expert reference, on both expert stores: the legacy packed BF16
//! table and the per-layer Q4_K entries.

use std::io::Write;

use larql_compute::cpu::ops::moe::{
    pre_experts_norm, quantize_h_norm_for_q4k, run_single_expert, run_single_expert_into,
    run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::{ExpertMlp, QuantFormat};
use larql_inference::ffn::expert_fold::{
    count_nonzero_weights, fold_experts_cpu, fold_experts_q8k_prenormed, ExpertFoldOptions,
};
use larql_inference::ModelWeights;
use larql_models::test_fixtures::{
    make_test_gemma4_moe_weights, GEMMA4_MOE_HIDDEN, GEMMA4_MOE_INTER, GEMMA4_MOE_NUM_EXPERTS,
};
use larql_models::weights::{per_layer_ffn_key, PER_LAYER_FFN_DOWN, PER_LAYER_FFN_GATE_UP};

const LAYER: usize = 0;
const TOLERANCE: f32 = 1e-4;

fn residual() -> Vec<f32> {
    (0..GEMMA4_MOE_HIDDEN)
        .map(|i| ((i as f32) * 0.37).sin())
        .collect()
}

fn h_norm(weights: &ModelWeights, h: &[f32]) -> Vec<f32> {
    let arch = &*weights.arch;
    let pre = arch
        .moe_pre_experts_norm_key(LAYER)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    pre_experts_norm(h, pre, arch.norm_weight_offset(), arch.norm_eps())
}

fn mlp(weights: &ModelWeights) -> ExpertMlp<'static> {
    ExpertMlp::gated(larql_inference::activation_from_arch(&*weights.arch))
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= TOLERANCE, "[{i}] {g} vs {w}");
    }
}

/// Per-expert reference on the packed BF16 table: each expert run alone,
/// then summed with its router weight.
fn bf16_reference(weights: &ModelWeights, h: &[f32], picks: &[(usize, f32)]) -> Vec<f32> {
    let arch = &*weights.arch;
    let gu_all = weights
        .get_packed_bytes(&arch.packed_experts_gate_up_key(LAYER).unwrap())
        .unwrap();
    let dn_all = weights
        .get_packed_bytes(&arch.packed_experts_down_key(LAYER).unwrap())
        .unwrap();
    let norm = h_norm(weights, h);
    let mut out = vec![0.0f32; h.len()];
    for &(e, w) in picks {
        let (gu, dn) = larql_inference::ffn::expert_fold::packed::packed_bf16_expert(
            gu_all,
            dn_all,
            e,
            GEMMA4_MOE_HIDDEN,
            GEMMA4_MOE_INTER,
        )
        .unwrap();
        let y = run_single_expert(
            &norm,
            gu,
            dn,
            GEMMA4_MOE_INTER,
            QuantFormat::BF16,
            mlp(weights),
        );
        for (o, v) in out.iter_mut().zip(y) {
            *o += w * v;
        }
    }
    out
}

fn fold(
    weights: &ModelWeights,
    ids: &[usize],
    ws: &[f32],
    opts: ExpertFoldOptions,
) -> (Vec<f32>, usize) {
    fold_experts_cpu(weights, LAYER, &residual(), ids, ws, opts)
}

#[test]
fn bf16_fold_matches_the_per_expert_reference() {
    let weights = make_test_gemma4_moe_weights();
    let (out, run) = fold(
        &weights,
        &[0, 2],
        &[0.25, 0.75],
        ExpertFoldOptions::default(),
    );
    assert_eq!(run, 2);
    assert_close(
        &out,
        &bf16_reference(&weights, &residual(), &[(0, 0.25), (2, 0.75)]),
    );
}

#[test]
fn zero_weight_experts_are_skipped_and_not_counted() {
    let weights = make_test_gemma4_moe_weights();
    let ws = [0.0, 1.0];
    let (out, run) = fold(&weights, &[0, 1], &ws, ExpertFoldOptions::default());
    assert_eq!(run, 1);
    assert_eq!(count_nonzero_weights(&ws), 1);
    assert_close(&out, &bf16_reference(&weights, &residual(), &[(1, 1.0)]));
}

/// An expert outside the table contributes nothing and is not counted, so
/// the caller sees the shortfall rather than a silently partial sum.
#[test]
fn an_unresolvable_expert_shows_as_a_shortfall() {
    let weights = make_test_gemma4_moe_weights();
    let ws = [1.0, 1.0];
    let (_, run) = fold(
        &weights,
        &[0, GEMMA4_MOE_NUM_EXPERTS],
        &ws,
        ExpertFoldOptions::default(),
    );
    assert_eq!(run, 1);
    assert!(run < count_nonzero_weights(&ws));
}

#[test]
fn empty_requests_fold_to_zero() {
    let weights = make_test_gemma4_moe_weights();
    let (out, run) = fold(&weights, &[], &[], ExpertFoldOptions::default());
    assert_eq!((out, run), (vec![0.0; GEMMA4_MOE_HIDDEN], 0));
    let (out, run) = fold_experts_cpu(
        &weights,
        LAYER,
        &[],
        &[0],
        &[1.0],
        ExpertFoldOptions::default(),
    );
    assert_eq!((out.len(), run), (0, 0));
}

#[test]
fn timing_does_not_change_the_result() {
    let weights = make_test_gemma4_moe_weights();
    let quiet = fold(&weights, &[1, 3], &[0.5, 0.5], ExpertFoldOptions::default());
    let timed = fold(
        &weights,
        &[1, 3],
        &[0.5, 0.5],
        ExpertFoldOptions {
            timing: true,
            ..Default::default()
        },
    );
    assert_eq!(quiet, timed);
}

/// The fixture's experts re-stored as per-layer Q4_K entries in an mmapped
/// file, the layout the per-layer loader produces.
struct Q4kStore {
    weights: ModelWeights,
    _file: tempfile::NamedTempFile,
}

fn with_per_layer_q4k(mut weights: ModelWeights) -> Q4kStore {
    let (h, i) = (GEMMA4_MOE_HIDDEN, GEMMA4_MOE_INTER);
    let entry = |e: usize, rows: usize, cols: usize, salt: f32| -> Vec<f32> {
        (0..rows * cols)
            .map(|k| (((k + e * 7) as f32) * 0.013 + salt).sin() * 0.1)
            .collect()
    };
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let name = "layers/layer_00.weights".to_string();
    let mut offset = 0usize;
    for e in 0..GEMMA4_MOE_NUM_EXPERTS {
        for (component, bytes) in [
            (
                PER_LAYER_FFN_GATE_UP,
                quantize_q4_k(&entry(e, 2 * i, h, 0.3)),
            ),
            (PER_LAYER_FFN_DOWN, quantize_q4_k(&entry(e, h, i, 0.7))),
        ] {
            file.write_all(&bytes).unwrap();
            weights.packed_byte_ranges.insert(
                per_layer_ffn_key(LAYER, e, component),
                (name.clone(), offset, bytes.len()),
            );
            offset += bytes.len();
        }
    }
    file.flush().unwrap();
    // SAFETY: the file is private to this test, fully written and flushed,
    // and kept alive in `Q4kStore` for as long as the mapping is used.
    let mmap = unsafe { memmap2::Mmap::map(file.as_file()).unwrap() };
    weights.packed_mmaps.insert(name, mmap);
    assert!(weights.has_per_layer_ffn());
    Q4kStore {
        weights,
        _file: file,
    }
}

fn q4k_reference(weights: &ModelWeights, h: &[f32], picks: &[(usize, f32)]) -> Vec<f32> {
    let q8k = quantize_h_norm_for_q4k(&h_norm(weights, h)).unwrap();
    let padded = GEMMA4_MOE_INTER.div_ceil(256) * 256;
    let mut scratch = ExpertScratch::new(h.len(), GEMMA4_MOE_INTER, padded);
    let mut out = vec![0.0f32; h.len()];
    for &(e, w) in picks {
        let (gu, dn) = weights.get_layer_entry_bytes(LAYER, e).unwrap();
        let y = run_single_expert_q4k_q8k_into(
            &mut scratch,
            &q8k,
            gu,
            dn,
            GEMMA4_MOE_INTER,
            mlp(weights),
        );
        for (o, v) in out.iter_mut().zip(y) {
            *o += w * v;
        }
    }
    out
}

#[test]
fn per_layer_q4k_fold_matches_the_direct_kernel_reference() {
    let store = with_per_layer_q4k(make_test_gemma4_moe_weights());
    let (out, run) = fold(
        &store.weights,
        &[1, 3],
        &[0.4, 0.6],
        ExpertFoldOptions::default(),
    );
    assert_eq!(run, 2);
    assert_close(
        &out,
        &q4k_reference(&store.weights, &residual(), &[(1, 0.4), (3, 0.6)]),
    );
}

/// With the direct kernel disabled the fold dequantises instead: a
/// different arithmetic path over the same weights, so close, not equal.
#[test]
fn disabling_the_direct_kernel_takes_the_dequant_path() {
    let store = with_per_layer_q4k(make_test_gemma4_moe_weights());
    let opts = ExpertFoldOptions {
        disable_q4k_direct: true,
        ..Default::default()
    };
    let (out, run) = fold(&store.weights, &[0, 2], &[0.5, 0.5], opts);
    assert_eq!(run, 2);
    // The in-place kernel on f32 activations. (The allocating
    // `run_single_expert` would itself take the direct Q8_K path on Q4_K
    // unless `LARQL_DISABLE_Q4K_DIRECT` is set, so it is not this arm's
    // reference.)
    let norm = h_norm(&store.weights, &residual());
    let padded = GEMMA4_MOE_INTER.div_ceil(256) * 256;
    let mut scratch = ExpertScratch::new(GEMMA4_MOE_HIDDEN, GEMMA4_MOE_INTER, padded);
    let mut want = vec![0.0f32; GEMMA4_MOE_HIDDEN];
    for e in [0, 2] {
        let (gu, dn) = store.weights.get_layer_entry_bytes(LAYER, e).unwrap();
        let y = run_single_expert_into(
            &mut scratch,
            &norm,
            gu,
            dn,
            GEMMA4_MOE_INTER,
            QuantFormat::Q4_K,
            mlp(&store.weights),
        );
        for (o, v) in want.iter_mut().zip(y) {
            *o += 0.5 * v;
        }
    }
    assert_close(&out, &want);
}

/// A client that pre-norms and pre-quantises gets the same fold.
#[test]
fn q8k_prenormed_fold_matches_the_server_side_quantisation() {
    let store = with_per_layer_q4k(make_test_gemma4_moe_weights());
    let q8k = quantize_h_norm_for_q4k(&h_norm(&store.weights, &residual())).unwrap();
    let ids = [0, 3];
    let ws = [0.3, 0.7];
    let (pre, run) = fold_experts_q8k_prenormed(&store.weights, LAYER, &q8k, &ids, &ws);
    assert_eq!(run, 2);
    let (server, _) = fold(&store.weights, &ids, &ws, ExpertFoldOptions::default());
    assert_close(&pre, &server);
}

#[test]
fn q8k_prenormed_fold_skips_what_it_cannot_resolve() {
    // The packed BF16 table has no per-layer entries to resolve.
    let weights = make_test_gemma4_moe_weights();
    let q8k = quantize_h_norm_for_q4k(&h_norm(&weights, &residual())).unwrap();
    let (_, run) = fold_experts_q8k_prenormed(&weights, LAYER, &q8k, &[0, 1], &[1.0, 1.0]);
    assert_eq!(run, 0);
    let (out, run) = fold_experts_q8k_prenormed(&weights, LAYER, &q8k, &[], &[]);
    assert_eq!((out.len(), run), (GEMMA4_MOE_HIDDEN, 0));
}

/// An empty activation runs nothing, whatever the request asks for.
#[test]
fn q8k_prenormed_fold_of_an_empty_activation_runs_nothing() {
    let store = with_per_layer_q4k(make_test_gemma4_moe_weights());
    let empty = larql_compute::Q8KActivation {
        qs: Vec::new(),
        d: Vec::new(),
        sums: Vec::new(),
    };
    let (out, run) =
        fold_experts_q8k_prenormed(&store.weights, LAYER, &empty, &[0, 1], &[1.0, 1.0]);
    assert_eq!((out.len(), run), (0, 0));
}
