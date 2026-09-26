//! G6b, first fragment: does the GPU-lowered gated FFN compute what the
//! interpreter's CPU-glue realisation computes?
//!
//! The lowered path moves the norm, the SiLU-GLU and the residual onto
//! the GPU, so unlike G6a's scheduling-only comparison the arithmetic
//! realisation genuinely changes and float reassociation is legitimate.
//! The bar is therefore a tolerance, judged in the same units the
//! production-parity work uses — max abs, relative RMS, cosine — not
//! bit equality.
//!
//! ## Controls
//!
//! Agreement alone would not show the lowering *read the plan*. A
//! lowering that ignored the norm epsilon, dropped the centred-norm
//! offset, or silently used the wrong activation would still produce
//! finite, plausible, nearly-correct numbers. So each judged fact gets a
//! negative arm that must break parity:
//!
//! - **norm weight offset** — Glimmer's centred convention (`1 + w`).
//!   Dropping it is the single likeliest silent lowering bug.
//! - **activation** — SiLU-GLU vs plain GLU.
//! - **residual** — present vs omitted.
//!
//! If a control does *not* break parity, the corresponding assertion in
//! the positive arm is vacuous and the test says so rather than passing.
//!
//! Control strength is judged **relative to the parity residual**, not
//! against an absolute constant. A fixed threshold is a guess about the
//! fixture: the residual control below moves rel_rms to 2.4e-3, which
//! looks small next to an arbitrary 1e-2 bar and is in fact 2500x the
//! 9.6e-7 the lowering itself achieves — overwhelmingly distinguishable.
//! What makes a control meaningful is that its effect dwarfs the noise
//! the positive arm tolerates, so that is what gets asserted.

#![cfg(target_os = "macos")]

use larql_compute_metal::lowering::ffn::{FfnActivation, FfnScratch, FfnShape, FfnWeights};
use larql_compute_metal::lowering::profile::SingleEncoder;
use larql_compute_metal::lowering::LoweredMatrix;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 512;
const INTER: usize = 1408;
const EPS: f32 = 1e-5;
/// Glimmer's centred-norm convention.
const NORM_OFFSET: f32 = 1.0;
/// Muse-Glimmer's post-block epsilon — three orders of magnitude below
/// the pre-block one. Reusing the pre-norm value here is the silent
/// four-norm bug this fixture exists to catch.
const POST_EPS: f32 = 1e-8;

fn deterministic(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * 0.6
        })
        .collect()
}

/// The reference: the same program on the CPU, in f32, written straight
/// from the plan's op order. Independent of the Metal code under test.
///
/// Takes every judged fact explicitly — including the two a control
/// perturbs — because a reference that hard-coded them could not model
/// the defects the controls exist to detect.
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    h: &[f32],
    norm_w: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    offset: f32,
    silu: bool,
    residual: bool,
    post: PostNormMode<'_>,
) -> Vec<f32> {
    let ms = h.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    let normed: Vec<f32> = h
        .iter()
        .zip(norm_w)
        .map(|(x, w)| x * inv * (offset + w))
        .collect();
    let mv = |m: &[f32], x: &[f32], n: usize, k: usize| -> Vec<f32> {
        (0..n)
            .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
            .collect()
    };
    let g = mv(gate, &normed, INTER, HIDDEN);
    let u = mv(up, &normed, INTER, HIDDEN);
    let act: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(gv, uv)| {
            if silu {
                (gv / (1.0 + (-gv).exp())) * uv
            } else {
                gv * uv
            }
        })
        .collect();
    let d = mv(down, &act, HIDDEN, INTER);
    let rms_norm = |v: &[f32], w: &[f32], eps: f32| -> Vec<f32> {
        let ms = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        v.iter()
            .zip(w)
            .map(|(x, wv)| x * inv * (1.0 + wv))
            .collect()
    };
    match post {
        // The judged shape: normalise the branch, then add.
        PostNormMode::BeforeResidual(w, eps) => {
            let n = rms_norm(&d, w, eps);
            h.iter().zip(&n).map(|(a, b)| a + b).collect()
        }
        // The plausible-but-wrong shape: add, then normalise the sum.
        PostNormMode::AfterResidual(w, eps) => {
            let summed: Vec<f32> = h.iter().zip(&d).map(|(a, b)| a + b).collect();
            rms_norm(&summed, w, eps)
        }
        PostNormMode::None if residual => h.iter().zip(&d).map(|(a, b)| a + b).collect(),
        PostNormMode::None => d,
    }
}

/// How (and whether) a post-block norm joins the residual stream.
#[derive(Clone, Copy)]
enum PostNormMode<'a> {
    /// Normalise the branch output, then add — the judged semantics.
    BeforeResidual(&'a [f32], f32),
    /// Add, then normalise the sum — what the name "post-FFN norm"
    /// could plausibly be read to mean, and a different model.
    AfterResidual(&'a [f32], f32),
    None,
}

/// How far a control must exceed the parity residual to demonstrate that
/// the positive arm could have detected the corresponding defect.
const CONTROL_MARGIN: f64 = 100.0;

struct Metrics {
    max_abs: f32,
    rel_rms: f64,
    cosine: f64,
}

fn compare(reference: &[f32], got: &[f32]) -> Metrics {
    let max_abs = reference
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let (mut num, mut den, mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        let (a, b) = (*a as f64, *b as f64);
        num += (a - b) * (a - b);
        den += a * a;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    Metrics {
        max_abs,
        rel_rms: (num / den).sqrt(),
        cosine: dot / (na.sqrt() * nb.sqrt()),
    }
}

/// A control must move the result far enough above the parity residual
/// that the positive arm would have caught the defect it models.
fn assert_control(what: &str, perturbed: &[f32], got: &[f32], parity_rel_rms: f64) {
    let c = compare(perturbed, got);
    let ratio = c.rel_rms / parity_rel_rms;
    eprintln!(
        "  control `{what}`: rel_rms {:.3e} = {ratio:.0}x the parity residual",
        c.rel_rms
    );
    assert!(
        ratio > CONTROL_MARGIN,
        "control `{what}` moves the result only {ratio:.1}x the parity residual          ({:.3e} vs {parity_rel_rms:.3e}) — the positive assertion cannot          distinguish this defect, so passing it proves nothing",
        c.rel_rms
    );
}

/// Run the lowered FFN once under SiLU-GLU and return its output.
#[allow(clippy::too_many_arguments)]
fn run_lowered(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    gate: &nvfp4::Nvfp4Matrix,
    up: &nvfp4::Nvfp4Matrix,
    down: &nvfp4::Nvfp4Matrix,
    offset: f32,
    post_norm_w: Option<&[f32]>,
    post_eps: f32,
) -> Vec<f32> {
    run_lowered_act(
        gpu,
        h,
        norm_w,
        gate,
        up,
        down,
        offset,
        post_norm_w,
        post_eps,
        FfnActivation::Silu,
    )
}

/// The same run, with the plan's gate COMBINE stated explicitly.
///
/// Split out so the SiTU arm can be driven through the identical
/// lowering — the point of that test is that only `shape.activation`
/// differs, so anything else varying between the arms would confound it.
#[allow(clippy::too_many_arguments)]
fn run_lowered_act(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    gate: &nvfp4::Nvfp4Matrix,
    up: &nvfp4::Nvfp4Matrix,
    down: &nvfp4::Nvfp4Matrix,
    offset: f32,
    post_norm_w: Option<&[f32]>,
    post_eps: f32,
    activation: FfnActivation,
) -> Vec<f32> {
    run_lowered_scaled(
        gpu,
        h,
        norm_w,
        gate,
        up,
        down,
        offset,
        post_norm_w,
        post_eps,
        activation,
        None,
    )
}

/// The same run, with the plan's residual-scale op stated explicitly.
#[allow(clippy::too_many_arguments)]
fn run_lowered_scaled(
    gpu: &larql_compute_metal::MetalBackend,
    h: &[f32],
    norm_w: &[f32],
    gate: &nvfp4::Nvfp4Matrix,
    up: &nvfp4::Nvfp4Matrix,
    down: &nvfp4::Nvfp4Matrix,
    offset: f32,
    post_norm_w: Option<&[f32]>,
    post_eps: f32,
    activation: FfnActivation,
    residual_scale: Option<f32>,
) -> Vec<f32> {
    let h_in = gpu.lowering_upload(h).expect("upload");
    let norm_buf = gpu.lowering_upload(norm_w).expect("upload");
    let h_out = gpu.lowering_scratch(HIDDEN);
    let (normed, g, u, a, d) = (
        gpu.lowering_scratch(HIDDEN),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(INTER),
        gpu.lowering_scratch(HIDDEN),
    );
    let post_buf = post_norm_w.map(|w| gpu.lowering_upload(w).expect("upload"));
    let post_scratch = gpu.lowering_scratch(HIDDEN);
    let w = FfnWeights {
        gate: LoweredMatrix::Nvfp4 {
            packed: &gpu.lowering_weight(&gate.packed),
            packed_offset: 0,
            scales: &gpu.lowering_weight(&gate.scales),
            scales_offset: 0,
            tensor_scale: gate.tensor_scale,
        },
        up: LoweredMatrix::Nvfp4 {
            packed: &gpu.lowering_weight(&up.packed),
            packed_offset: 0,
            scales: &gpu.lowering_weight(&up.scales),
            scales_offset: 0,
            tensor_scale: up.tensor_scale,
        },
        down: LoweredMatrix::Nvfp4 {
            packed: &gpu.lowering_weight(&down.packed),
            packed_offset: 0,
            scales: &gpu.lowering_weight(&down.scales),
            scales_offset: 0,
            tensor_scale: down.tensor_scale,
        },
        norm_weight: &norm_buf,
        post_norm: post_buf
            .as_ref()
            .map(|b| larql_compute_metal::lowering::PostNorm {
                weight: b,
                eps: post_eps,
                weight_offset: NORM_OFFSET,
                scratch: &post_scratch,
            }),
    };
    let s = FfnScratch {
        normed: &normed,
        gate: &g,
        up: &u,
        act: &a,
        down: &d,
    };
    let shape = FfnShape {
        hidden: HIDDEN,
        intermediate: INTER,
        norm_eps: EPS,
        norm_weight_offset: offset,
        activation,
        residual_scale,
    };

    let cmd = gpu.new_lowering_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    gpu.encode_gated_ffn(&mut SingleEncoder(enc), &h_in, &h_out, &w, &s, &shape);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let out = gpu.lowering_readback(&h_out, HIDDEN).expect("readback");
    // `w` borrows the uploaded buffers; end its borrow before recycling.
    for b in [h_in, norm_buf, h_out, normed, g, u, a, d, post_scratch] {
        gpu.recycle_lowering_scratch(b);
    }
    if let Some(b) = post_buf {
        gpu.recycle_lowering_scratch(b);
    }
    out
}

mod lowering_ffn_parity_basics;
