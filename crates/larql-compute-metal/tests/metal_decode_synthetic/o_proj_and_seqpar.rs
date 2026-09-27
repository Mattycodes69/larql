//! `decode/encode_attn/o_proj.rs`'s legacy Q8_0 arm and
//! `decode/encode_attn/kv_attend.rs`'s sequence-parallel arms, each
//! checked against an arm that computes the same thing another way.

#[allow(unused_imports)]
use crate::common::*;

use larql_compute::cpu::ops::q4_common::{
    quantize_q4_0, quantize_q4_k, quantize_q6_k, quantize_to_q8,
};
use larql_compute_metal::ops::kv_seqpar::SeqparRequest;
use larql_compute_metal::{BackendOptions, MetalBackend};

/// Tokens decoded per seqpar arm — enough rows that the attend spans more
/// than one cached position, so the slices actually split a span.
const SEQPAR_TOKENS: usize = 4;
/// Slice count requested of the sequence-parallel attend arms.
const SEQPAR_SLICES: usize = 4;
/// Max |Δ| between the serial and sequence-parallel attend: the same
/// sums in a different reduction order.
const SEQPAR_TOLERANCE: f32 = 1e-3;

/// Every weight a synthetic layer needs, owned for the whole test so no
/// buffer-cache entry keyed on a freed allocation can be reused.
struct Weights {
    wq: Vec<u8>,
    wk: Vec<u8>,
    wv: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    norm_w: Vec<f32>,
}

fn weights() -> Weights {
    Weights {
        wq: quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1)),
        wk: quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2)),
        wv: quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3)),
        gate: quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5)),
        up: quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6)),
        down: quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7)),
        norm_w: (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect(),
    }
}

impl Weights {
    /// The synthetic layer with `wo` swapped in. `build_synth_layer` takes
    /// Q4_K bytes for `wo`; the slot is overwritten before use, so `wq`'s
    /// bytes stand in there.
    fn layer<'a>(&'a self, wo: QuantWeight<'a>) -> FullPipelineLayer<'a> {
        let mut layer = build_synth_layer(
            &self.wq,
            &self.wk,
            &self.wv,
            &self.wq,
            &self.gate,
            &self.up,
            &self.down,
            &self.norm_w,
        );
        layer.wo = wo;
        layer
    }
}

fn decode(metal: &MetalBackend, layer: FullPipelineLayer<'_>, tokens: usize) -> Vec<f32> {
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let mut out = Vec::new();
    for t in 0..tokens {
        let x = synth_input(HIDDEN, 0.9 + t as f32 * 0.37);
        out = MetalBackend::decode_token(
            metal,
            &mut kv,
            std::slice::from_ref(&layer),
            &x,
            HIDDEN,
            INTER,
            Q_DIM,
            KV_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            10_000.0,
        );
    }
    out
}

/// Q8_0 in larql's split representation: int8 rows plus an f32 scale per
/// 32-element block, which the legacy `q8_matvec` reads at buffer 2.
fn q8<'a>(bytes: &'a [u8], scales: &'a [f32]) -> QuantWeight<'a> {
    let aux = larql_compute::QuantAux::ExternalScales(scales);
    QuantWeight::new(QuantFormat::Q8_0, bytes, aux)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn backend_with(opts: BackendOptions) -> MetalBackend {
    MetalBackend::with_options(opts).expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    )
}

/// The legacy Q8_0 O projection (`q8_quant` + `q8_matvec` with external
/// weight scales) against Q6_K of the SAME f32 matrix through
/// `o_proj::encode`. Both approximate one `W_o`, so they must agree far
/// more closely than either agrees with a different `W_o` — which is what
/// shows the Q8_0 arm is reading its weights and scales, not merely
/// running.
#[test]
fn q8_0_o_projection_matches_q6_k_of_the_same_matrix() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let metal = backend_with(BackendOptions::default());
    let w = weights();
    let wo_f32 = synth_weight_f32(HIDDEN * Q_DIM, 0.4);
    let other_f32 = synth_weight_f32(HIDDEN * Q_DIM, 2.9);
    let (wo_q8, wo_q8_scales) = quantize_to_q8(&wo_f32);
    let wo_q8: Vec<u8> = wo_q8.iter().map(|&v| v as u8).collect();
    let (other_q8, other_q8_scales) = quantize_to_q8(&other_f32);
    let other_q8: Vec<u8> = other_q8.iter().map(|&v| v as u8).collect();
    let wo_q6k = quantize_q6_k(&wo_f32);

    let out_q8 = decode(&metal, w.layer(q8(&wo_q8, &wo_q8_scales)), 1);
    let out_q6k = decode(
        &metal,
        w.layer(QuantWeight::new(
            QuantFormat::Q6_K,
            &wo_q6k,
            larql_compute::QuantAux::None,
        )),
        1,
    );
    let out_other = decode(&metal, w.layer(q8(&other_q8, &other_q8_scales)), 1);

    assert!(
        out_q8.iter().all(|v| v.is_finite()),
        "Q8_0 arm produced NaN/Inf"
    );
    let same = max_abs_diff(&out_q8, &out_q6k);
    let different = max_abs_diff(&out_q8, &out_other);
    assert!(
        different > 0.0 && same * 10.0 < different,
        "Q8_0 vs Q6_K of the same W_o differ by {same}, a different W_o by {different}: \
         the Q8_0 arm is not computing W_o · attn"
    );
}

/// Serial vs sequence-parallel attend, both on the UNFUSED
/// `encode_kv_append` + `encode_kv_attend[_seqpar]` path.
#[test]
fn unfused_seqpar_attend_matches_the_serial_attend() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let w = weights();
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let run = |kv_seqpar: SeqparRequest| {
        let mut opts = BackendOptions::default();
        opts.decode_flags.fused_kv_append_attend = false;
        opts.decode_flags.kv_seqpar = kv_seqpar;
        let metal = backend_with(opts);
        let wo = QuantWeight::new(QuantFormat::Q4_K, &wo, larql_compute::QuantAux::None);
        decode(&metal, w.layer(wo), SEQPAR_TOKENS)
    };
    let serial = run(SeqparRequest::Off);
    let seqpar = run(SeqparRequest::Slices(SEQPAR_SLICES));
    assert!(
        seqpar.iter().all(|v| v.is_finite()),
        "seqpar arm produced NaN/Inf"
    );
    let d = max_abs_diff(&serial, &seqpar);
    assert!(
        d < SEQPAR_TOLERANCE,
        "unfused seqpar attend drifted {d} from serial"
    );
}

/// Serial vs sequence-parallel `kv_append_attend_fused`.
#[test]
fn fused_seqpar_append_attend_matches_the_serial_kernel() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let w = weights();
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let run = |kv_seqpar: SeqparRequest| {
        let mut opts = BackendOptions::default();
        opts.decode_flags.fused_kv_append_attend = true;
        opts.decode_flags.kv_seqpar = kv_seqpar;
        let metal = backend_with(opts);
        let wo = QuantWeight::new(QuantFormat::Q4_K, &wo, larql_compute::QuantAux::None);
        decode(&metal, w.layer(wo), SEQPAR_TOKENS)
    };
    let serial = run(SeqparRequest::Off);
    let seqpar = run(SeqparRequest::Slices(SEQPAR_SLICES));
    assert!(
        seqpar.iter().all(|v| v.is_finite()),
        "seqpar arm produced NaN/Inf"
    );
    let d = max_abs_diff(&serial, &seqpar);
    assert!(
        d < SEQPAR_TOLERANCE,
        "fused seqpar append+attend drifted {d} from serial"
    );
}
