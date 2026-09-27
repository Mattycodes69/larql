//! The layer's NEW links, against a host reference.
//!
//! The attention is already gated in `trait_impl::kda::tests` and the
//! experts in `trait_impl::bf16_moe_block::tests`, so the reference here
//! composes those proven device calls and computes only what this module
//! adds — the norms, the residual, the router decision and the weighted
//! combine. That keeps the comparison pointed at the code under test
//! rather than re-deriving a whole decoder layer.

use super::*;
use crate::trait_impl::kda::SmallMatrix;

/// Reach the routed weights of a layer built by `layer_weights`, for
/// the controls that corrupt one field. Panics on a dense layer, which
/// no control here builds.
fn moe_mut<'a, 'b>(w: &'b mut KimiLayerWeights<'a>) -> &'b mut KimiMoeWeights<'a> {
    match &mut w.ffn {
        FfnSpec::Moe(m) => m,
        FfnSpec::Dense(_) => panic!("this control corrupts a routed layer"),
    }
}
use crate::shaders::kimi_layer as layer_shader;
use crate::trait_impl::bf16_moe_block::{BlockLowering, ExpertBankRef, MoeBlockCall, MoeFfnBanks};
use crate::trait_impl::grouped_experts::ExpertOffset;
use crate::trait_impl::grouped_experts::GroupedError;
use crate::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use crate::trait_impl::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use crate::MetalBackend;

const HEADS: usize = 2;
const DIM: usize = 4;
const HIDDEN: usize = 8;
const INTER: usize = 6;
const KERNEL: usize = 4;
const WIDTH: usize = HEADS * DIM;
const EXPERTS: usize = 12;
const TOP_K: usize = 3;
const RESIDENT: usize = 5;
const BRANCH_SCALE: f32 = 2.446;
const EPS: f32 = 1e-5;
const TOLERANCE: f32 = 1e-4;

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
        .map(|i| ((i as f32) * 0.41 + seed).sin() * 0.5)
        .collect()
}

fn bf16_bytes(n: usize, k: usize, seed: f32) -> Vec<u8> {
    synth(n * k, seed)
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A layer's weights, owned. The resident bank holds `RESIDENT` routed
/// experts; the shared branch lives in its OWN allocations, one per
/// projection — semantic identity, not co-location. The router scores
/// all `EXPERTS`, so the residency table is what decides whether a
/// route is servable.
struct Fixture {
    x: Vec<f32>,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    router_weight: Vec<f32>,
    router_bias: Vec<f32>,
    residency: Vec<u32>,
    bank_gate: Vec<u8>,
    bank_up: Vec<u8>,
    bank_down: Vec<u8>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    o_proj: Vec<u8>,
    kda_f32: Vec<Vec<f32>>,
}

/// One projection's bank: a routed region, a table, and the shared
/// branch's own region.
fn projection<'a>(routed: &'a [u8], table: &'a [u32], shared: &'a [u8]) -> ProjectionBank<'a> {
    ProjectionBank {
        routed: EncodedRegion {
            bytes: routed,
            encoding: ExpertEncoding::Bf16,
        },
        addressing: ExpertAddressing::Table(table),
        shared: Some(EncodedRegion {
            bytes: shared,
            encoding: ExpertEncoding::Bf16,
        }),
    }
}

fn fixture() -> Fixture {
    let per_qkv = WIDTH * HIDDEN * 2;
    let mut qkv_bank = Vec::with_capacity(3 * per_qkv);
    for (i, seed) in [0.1f32, 1.3, 2.7].into_iter().enumerate() {
        let _ = i;
        qkv_bank.extend_from_slice(&bf16_bytes(WIDTH, HIDDEN, seed));
    }
    let gate_per = INTER * HIDDEN * 2;
    let down_per = HIDDEN * INTER * 2;
    let mut bank_gate = Vec::new();
    let mut bank_up = Vec::new();
    let mut bank_down = Vec::new();
    let mut residency = vec![layer_shader::NOT_RESIDENT; EXPERTS];
    // Bias the FIRST `RESIDENT` experts into residency, and bias the
    // router towards exactly those so the default route is servable.
    for (slot, entry) in residency.iter_mut().enumerate().take(RESIDENT) {
        *entry = (slot * gate_per) as u32;
        bank_gate.extend_from_slice(&bf16_bytes(INTER, HIDDEN, 3.0 + slot as f32));
        bank_up.extend_from_slice(&bf16_bytes(INTER, HIDDEN, 9.0 + slot as f32));
        bank_down.extend_from_slice(&bf16_bytes(HIDDEN, INTER, 15.0 + slot as f32));
    }
    debug_assert_eq!(bank_down.len(), RESIDENT * down_per);

    // A correction bias that puts the resident experts on top — the
    // point of the fixture is the seam, not a route that cannot be served.
    let router_bias: Vec<f32> = (0..EXPERTS)
        .map(|e| if e < RESIDENT { 1.0 } else { 0.0 })
        .collect();

    Fixture {
        x: synth(HIDDEN, 0.7),
        input_norm: synth(HIDDEN, 2.2).iter().map(|v| v + 1.0).collect(),
        post_norm: synth(HIDDEN, 3.3).iter().map(|v| v + 1.0).collect(),
        router_weight: synth(EXPERTS * HIDDEN, 4.4),
        router_bias,
        residency,
        bank_gate,
        bank_up,
        bank_down,
        shared_gate: bf16_bytes(INTER, HIDDEN, 21.0),
        shared_up: bf16_bytes(INTER, HIDDEN, 22.0),
        shared_down: bf16_bytes(HIDDEN, INTER, 23.0),
        qkv_bank,
        qkv_offsets: [
            ExpertOffset(0),
            ExpertOffset(per_qkv as u32),
            ExpertOffset((2 * per_qkv) as u32),
        ],
        o_proj: bf16_bytes(HIDDEN, WIDTH, 5.5),
        kda_f32: vec![
            synth(WIDTH * KERNEL, 0.5),                         // q_conv1d
            synth(WIDTH * KERNEL, 1.5),                         // k_conv1d
            synth(WIDTH * KERNEL, 2.5),                         // v_conv1d
            synth(DIM * HIDDEN, 6.1),                           // f_a
            synth(WIDTH * DIM, 7.2),                            // f_b
            synth(DIM * HIDDEN, 8.3),                           // g_a
            synth(WIDTH * DIM, 9.4),                            // g_b
            synth(HEADS * HIDDEN, 10.5),                        // b_proj
            synth(HEADS, 11.6),                                 // a_log
            synth(WIDTH, 12.7),                                 // dt_bias
            synth(DIM, 13.8).iter().map(|v| v + 1.0).collect(), // o_norm
        ],
    }
}

impl Fixture {
    fn kda(&self) -> KdaDeviceWeights<'_> {
        let f = &self.kda_f32;
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_proj,
            projection_encoding: ExpertEncoding::Bf16,
            q_conv1d: &f[0],
            k_conv1d: &f[1],
            v_conv1d: &f[2],
            f_a_proj: SmallMatrix::F32(&f[3]),
            f_b_proj: SmallMatrix::F32(&f[4]),
            g_a_proj: SmallMatrix::F32(&f[5]),
            g_b_proj: SmallMatrix::F32(&f[6]),
            b_proj: SmallMatrix::F32(&f[7]),
            a_log: &f[8],
            dt_bias: &f[9],
            o_norm: &f[10],
            norm_eps: EPS,
            gate_form: larql_models::config::KdaGateForm::Softplus,
        }
    }

    fn layer<'a>(&'a self, state: &'a KdaDeviceState) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: AttentionSpec::Kda {
                weights: self.kda(),
                shape: shape(),
                state,
            },
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &self.router_weight,
                router_bias: &self.router_bias,
                gate: projection(&self.bank_gate, &self.residency, &self.shared_gate),
                up: projection(&self.bank_up, &self.residency, &self.shared_up),
                down: projection(&self.bank_down, &self.residency, &self.shared_down),
                inter: INTER,
                top_k: TOP_K,
                renormalize: true,
                branch_scale: BRANCH_SCALE,
            }),
            norm_eps: EPS,
        }
    }
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (ms + eps).sqrt().recip();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// The host reference for the router: sigmoid, correction bias, top-k
/// with ties to the lower index, renormalise, scale.
fn route(x: &[f32], w: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let logits: Vec<f32> = (0..EXPERTS)
        .map(|e| {
            w[e * HIDDEN..(e + 1) * HIDDEN]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let scores: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let sel: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut ranked: Vec<usize> = (0..EXPERTS).collect();
    ranked.sort_by(|&a, &b| {
        sel[b]
            .partial_cmp(&sel[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let ids: Vec<usize> = ranked[..TOP_K].to_vec();
    let gathered: Vec<f32> = ids.iter().map(|&i| scores[i]).collect();
    let sum: f32 = gathered.iter().sum::<f32>() + 1e-20;
    let weights = gathered.iter().map(|w| w / sum * BRANCH_SCALE).collect();
    (ids, weights)
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

/// One projection's bank reference — a free function so both the KDA
/// and MLA layer gates can build one without fighting the borrow
/// checker over a closure's return lifetime.
fn bank<'a>(w: &'a [u8], offsets: &'a [ExpertOffset]) -> ExpertBankRef<'a> {
    ExpertBankRef {
        weights: w,
        offsets,
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

const LATENT: usize = 8;
const NOPE: usize = 4;
const ROPE: usize = 2;
const V_DIM: usize = 4;

fn mla_shape() -> MlaShape {
    MlaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        kv_lora_rank: LATENT,
        qk_nope_head_dim: NOPE,
        qk_rope_head_dim: ROPE,
        v_head_dim: V_DIM,
    }
}

/// MLA's four wide matrices as bf16 codes, plus its latent norm.
struct MlaBits {
    q: Vec<u8>,
    ka: Vec<u8>,
    kb: Vec<u8>,
    o: Vec<u8>,
    norm: Vec<f32>,
}

fn mla_bits() -> MlaBits {
    let s = mla_shape();
    MlaBits {
        q: bf16_bytes(HEADS * s.q_head_dim(), HIDDEN, 31.0),
        ka: bf16_bytes(s.cache_stride(), HIDDEN, 32.0),
        kb: bf16_bytes(s.kv_row(), LATENT, 33.0),
        o: bf16_bytes(HIDDEN, s.value_width(), 34.0),
        norm: synth(LATENT, 35.0).iter().map(|v| v + 1.0).collect(),
    }
}

impl MlaBits {
    fn device(&self) -> MlaDeviceWeights<'_> {
        MlaDeviceWeights {
            q_proj: &self.q,
            kv_a_proj: &self.ka,
            kv_a_norm: &self.norm,
            kv_b_proj: &self.kb,
            o_proj: &self.o,
            kv_a_norm_eps: EPS,
            projection_encoding: ExpertEncoding::Bf16,
        }
    }
}

mod addressing;
mod dense;
mod encoding;
mod head;
mod shared;

mod gated_layer_and_refusals;
mod r6b_the_same_decoder_layer_with_mla_atte;
