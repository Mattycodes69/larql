//! A whole KDA attention step against a scalar reference, at a
//! hand-checkable geometry.
//!
//! The real-weight gate lives in `larql-vindex`, where the proven CPU
//! operator is — but the kernels live here, and a shader this crate
//! ships needs a gate this crate runs. The reference below is a literal
//! transcription of `exec::kda::step`, written independently of the
//! shaders it scores.

use super::*;
use crate::MetalBackend;

const HEADS: usize = 3;
const DIM: usize = 4;
const HIDDEN: usize = 6;
const KERNEL: usize = 4;
const WIDTH: usize = HEADS * DIM;

/// Loose enough for a threadgroup reduction against a serial one, tight
/// enough that any real error is orders past it.
const TOLERANCE: f32 = 1e-5;

fn shape() -> KdaShape {
    KdaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        head_dim: DIM,
        conv_kernel: KERNEL,
    }
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// `[n, k]` bf16 codes as little-endian bytes, and the exact f32 values
/// they denote — the oracle must score the STORED weights.
fn bf16_matrix(n: usize, k: usize, seed: f32) -> (Vec<u8>, Vec<f32>) {
    let values = synth(n * k, seed);
    let codes: Vec<u16> = values.iter().map(|v| narrow(*v)).collect();
    let exact: Vec<f32> = codes
        .iter()
        .map(|c| f32::from_bits((*c as u32) << 16))
        .collect();
    (codes.iter().flat_map(|c| c.to_le_bytes()).collect(), exact)
}

fn matvec(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
    let k = x.len();
    (0..out)
        .map(|r| {
            w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        v.exp().ln_1p()
    }
}

/// Everything one step needs, owned.
struct Weights {
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    qkv_exact: [Vec<f32>; 3],
    o_bytes: Vec<u8>,
    o_exact: Vec<f32>,
    conv: [Vec<f32>; 3],
    fa: Vec<f32>,
    fb: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
    bp: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    o_norm: Vec<f32>,
    eps: f32,
}

fn weights() -> Weights {
    let per = WIDTH * HIDDEN;
    let (qb, qe) = bf16_matrix(WIDTH, HIDDEN, 0.1);
    let (kb, ke) = bf16_matrix(WIDTH, HIDDEN, 1.3);
    let (vb, ve) = bf16_matrix(WIDTH, HIDDEN, 2.7);
    let (ob, oe) = bf16_matrix(HIDDEN, WIDTH, 3.9);
    let mut bank = Vec::with_capacity(3 * per * 2);
    for b in [&qb, &kb, &vb] {
        bank.extend_from_slice(b);
    }
    Weights {
        qkv_bank: bank,
        qkv_offsets: [
            ExpertOffset(0),
            ExpertOffset((per * 2) as u32),
            ExpertOffset((2 * per * 2) as u32),
        ],
        qkv_exact: [qe, ke, ve],
        o_bytes: ob,
        o_exact: oe,
        conv: [
            synth(WIDTH * KERNEL, 0.5),
            synth(WIDTH * KERNEL, 1.5),
            synth(WIDTH * KERNEL, 2.5),
        ],
        fa: synth(DIM * HIDDEN, 4.1),
        fb: synth(WIDTH * DIM, 5.2),
        ga: synth(DIM * HIDDEN, 6.3),
        gb: synth(WIDTH * DIM, 7.4),
        bp: synth(HEADS * HIDDEN, 8.5),
        a_log: synth(HEADS, 9.6),
        dt: synth(WIDTH, 10.7),
        o_norm: synth(DIM, 11.8).iter().map(|v| v + 1.0).collect(),
        eps: 1e-5,
    }
}

impl Weights {
    fn device(&self) -> KdaDeviceWeights<'_> {
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_bytes,
            projection_encoding: ExpertEncoding::Bf16,
            q_conv1d: &self.conv[0],
            k_conv1d: &self.conv[1],
            v_conv1d: &self.conv[2],
            f_a_proj: SmallMatrix::F32(&self.fa),
            f_b_proj: SmallMatrix::F32(&self.fb),
            g_a_proj: SmallMatrix::F32(&self.ga),
            g_b_proj: SmallMatrix::F32(&self.gb),
            b_proj: SmallMatrix::F32(&self.bp),
            a_log: &self.a_log,
            dt_bias: &self.dt,
            o_norm: &self.o_norm,
            norm_eps: self.eps,
            gate_form: larql_models::config::KdaGateForm::Softplus,
        }
    }
}

/// The host state the reference carries between steps.
#[derive(Clone)]
struct RefState {
    recurrent: Vec<f32>,
    conv: [Vec<f32>; 3],
}

impl RefState {
    fn zeros() -> Self {
        let tail = WIDTH * (KERNEL - 1);
        Self {
            recurrent: vec![0.0; HEADS * DIM * DIM],
            conv: [vec![0.0; tail], vec![0.0; tail], vec![0.0; tail]],
        }
    }
}

/// A literal transcription of `exec::kda::step`, in the same order.
fn reference_step(w: &Weights, st: &mut RefState, x: &[f32]) -> Vec<f32> {
    let tail = KERNEL - 1;
    let mut streams: [Vec<f32>; 3] = Default::default();
    for (i, exact) in w.qkv_exact.iter().enumerate() {
        let p = matvec(exact, x, WIDTH);
        let mut out = vec![0.0f32; WIDTH];
        for (c, (o, pc)) in out.iter_mut().zip(&p).enumerate() {
            let cw = &w.conv[i][c * KERNEL..(c + 1) * KERNEL];
            let hist = &st.conv[i][c * tail..(c + 1) * tail];
            let mut acc = 0.0f32;
            for (j, cwj) in cw.iter().enumerate().take(tail) {
                acc += cwj * hist[j];
            }
            acc += cw[tail] * pc;
            *o = silu(acc);
        }
        for (c, pc) in p.iter().enumerate() {
            let hist = &mut st.conv[i][c * tail..(c + 1) * tail];
            for j in 0..tail - 1 {
                hist[j] = hist[j + 1];
            }
            hist[tail - 1] = *pc;
        }
        streams[i] = out;
    }
    let [mut q, mut k, v] = streams;
    for stream in [&mut q, &mut k] {
        for h in 0..HEADS {
            let head = &mut stream[h * DIM..(h + 1) * DIM];
            let n = head.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = 1.0 / n.max(1e-12);
            for e in head.iter_mut() {
                *e *= inv;
            }
        }
    }

    let f_low = matvec(&w.fb, &matvec(&w.fa, x, DIM), WIDTH);
    let decay: Vec<f32> = (0..WIDTH)
        .map(|i| -w.a_log[i / DIM].exp() * softplus(f_low[i] + w.dt[i]))
        .collect();
    let gate = matvec(&w.gb, &matvec(&w.ga, x, DIM), WIDTH);
    let beta: Vec<f32> = matvec(&w.bp, x, HEADS)
        .iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect();

    let scale = (DIM as f32).powf(-0.5);
    let mut out = [0.0f32; WIDTH];
    for h in 0..HEADS {
        let s = &mut st.recurrent[h * DIM * DIM..(h + 1) * DIM * DIM];
        let (qh, kh, vh) = (&q[h * DIM..], &k[h * DIM..], &v[h * DIM..]);
        let mut pred = [0.0f32; DIM];
        for kk in 0..DIM {
            let d = decay[h * DIM + kk].exp();
            for vv in 0..DIM {
                s[kk * DIM + vv] *= d;
                pred[vv] += kh[kk] * s[kk * DIM + vv];
            }
        }
        let err: Vec<f32> = (0..DIM).map(|vv| vh[vv] - pred[vv]).collect();
        for kk in 0..DIM {
            let write = beta[h] * kh[kk];
            let qv = qh[kk] * scale;
            for vv in 0..DIM {
                let cell = &mut s[kk * DIM + vv];
                *cell += write * err[vv];
                out[h * DIM + vv] += qv * *cell;
            }
        }
    }

    let mut normed = [0.0f32; WIDTH];
    for h in 0..HEADS {
        let slice = &out[h * DIM..(h + 1) * DIM];
        let ms = slice.iter().map(|v| v * v).sum::<f32>() / DIM as f32;
        let inv = (ms + w.eps).sqrt().recip();
        for (d, (sv, nv)) in slice.iter().zip(&w.o_norm).enumerate() {
            normed[h * DIM + d] = sv * inv * nv / (1.0 + (-gate[h * DIM + d]).exp());
        }
    }
    matvec(&w.o_exact, &normed, HIDDEN)
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

//
// Its own geometry because Q8_0 blocks are 32 codes wide: the file's
// HIDDEN = 6 cannot legally encode at all (that impossibility is itself
// asserted below). WIDTH = 32 keeps o_proj's reduction axis aligned
// too, so both dispatches run the real quantised kernel.

const Q8_HEADS: usize = 2;
const Q8_DIM: usize = 16;
const Q8_HIDDEN: usize = 64;
const Q8_WIDTH: usize = Q8_HEADS * Q8_DIM;

fn q8_shape() -> KdaShape {
    KdaShape {
        hidden: Q8_HIDDEN,
        num_heads: Q8_HEADS,
        head_dim: Q8_DIM,
        conv_kernel: KERNEL,
    }
}

/// Both arms' banks from ONE set of values: the bf16 arm binds the
/// narrowed codes, the Q8_0 arm binds `quantize_q8_0` of the exact
/// widened values of those same codes. The only difference between the
/// arms is therefore the Q8_0 roundtrip itself — not a second RNG draw.
struct DualBanks {
    bf16_qkv: Vec<u8>,
    bf16_offsets: [ExpertOffset; 3],
    bf16_o: Vec<u8>,
    q8_qkv: Vec<u8>,
    q8_offsets: [ExpertOffset; 3],
    q8_o: Vec<u8>,
    f32s: Weights,
}

fn dual_banks() -> DualBanks {
    let per = Q8_WIDTH * Q8_HIDDEN;
    let (qb, qe) = bf16_matrix(Q8_WIDTH, Q8_HIDDEN, 0.1);
    let (kb, ke) = bf16_matrix(Q8_WIDTH, Q8_HIDDEN, 1.3);
    let (vb, ve) = bf16_matrix(Q8_WIDTH, Q8_HIDDEN, 2.7);
    let (ob, oe) = bf16_matrix(Q8_HIDDEN, Q8_WIDTH, 3.9);
    let mut bf16_qkv = Vec::with_capacity(3 * per * 2);
    for b in [&qb, &kb, &vb] {
        bf16_qkv.extend_from_slice(b);
    }
    let q8: Vec<Vec<u8>> = [&qe, &ke, &ve]
        .iter()
        .map(|e| larql_compute::cpu::ops::q4_common::quantize_q8_0(e))
        .collect();
    let q8_per = q8[0].len();
    let mut q8_qkv = Vec::with_capacity(3 * q8_per);
    for b in &q8 {
        assert_eq!(b.len(), q8_per);
        q8_qkv.extend_from_slice(b);
    }
    DualBanks {
        bf16_qkv,
        bf16_offsets: [
            ExpertOffset(0),
            ExpertOffset((per * 2) as u32),
            ExpertOffset((2 * per * 2) as u32),
        ],
        bf16_o: ob,
        q8_qkv,
        q8_offsets: [
            ExpertOffset(0),
            ExpertOffset(q8_per as u32),
            ExpertOffset((2 * q8_per) as u32),
        ],
        q8_o: larql_compute::cpu::ops::q4_common::quantize_q8_0(&oe),
        f32s: Weights {
            qkv_bank: Vec::new(),
            qkv_offsets: [ExpertOffset(0); 3],
            qkv_exact: [qe, ke, ve],
            o_bytes: Vec::new(),
            o_exact: oe,
            conv: [
                synth(Q8_WIDTH * KERNEL, 0.5),
                synth(Q8_WIDTH * KERNEL, 1.5),
                synth(Q8_WIDTH * KERNEL, 2.5),
            ],
            fa: synth(Q8_DIM * Q8_HIDDEN, 4.1),
            fb: synth(Q8_WIDTH * Q8_DIM, 5.2),
            ga: synth(Q8_DIM * Q8_HIDDEN, 6.3),
            gb: synth(Q8_WIDTH * Q8_DIM, 7.4),
            bp: synth(Q8_HEADS * Q8_HIDDEN, 8.5),
            a_log: synth(Q8_HEADS, 9.6),
            dt: synth(Q8_WIDTH, 10.7),
            o_norm: synth(Q8_DIM, 11.8).iter().map(|v| v + 1.0).collect(),
            eps: 1e-5,
        },
    }
}

impl DualBanks {
    fn device(&self, encoding: ExpertEncoding) -> KdaDeviceWeights<'_> {
        let f = &self.f32s;
        let (bank, offsets, o): (&[u8], _, &[u8]) = match encoding {
            ExpertEncoding::Bf16 => (&self.bf16_qkv, &self.bf16_offsets, &self.bf16_o),
            _ => (&self.q8_qkv, &self.q8_offsets, &self.q8_o),
        };
        KdaDeviceWeights {
            qkv_bank: bank,
            qkv_offsets: offsets,
            o_proj: o,
            projection_encoding: encoding,
            q_conv1d: &f.conv[0],
            k_conv1d: &f.conv[1],
            v_conv1d: &f.conv[2],
            f_a_proj: SmallMatrix::F32(&f.fa),
            f_b_proj: SmallMatrix::F32(&f.fb),
            g_a_proj: SmallMatrix::F32(&f.ga),
            g_b_proj: SmallMatrix::F32(&f.gb),
            b_proj: SmallMatrix::F32(&f.bp),
            a_log: &f.a_log,
            dt_bias: &f.dt,
            o_norm: &f.o_norm,
            norm_eps: f.eps,
            gate_form: larql_models::config::KdaGateForm::Softplus,
        }
    }
}

mod device_step_and_refusals;
mod q8_0_projections;
mod small_matrix;
