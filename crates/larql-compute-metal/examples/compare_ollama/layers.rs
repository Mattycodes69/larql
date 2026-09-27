//! Synthetic Gemma-3-4B-shaped layer weights and the pipeline layers built
//! over them.

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_to_q8};
use larql_compute::{FullPipelineLayer, QuantAux, QuantFormat, QuantWeight};

/// Model width (Gemma 3 4B).
pub const HIDDEN: usize = 2560;
/// FFN intermediate width (Gemma 3 4B).
pub const INTER: usize = 10240;
/// Query heads.
pub const NUM_Q: usize = 8;
/// Key/value heads.
pub const NUM_KV: usize = 4;
/// Head dimension.
pub const HEAD_DIM: usize = 320;
pub const Q_DIM: usize = NUM_Q * HEAD_DIM;
pub const KV_DIM: usize = NUM_KV * HEAD_DIM;
/// Q4_K super-block width every quantised row is padded to.
const Q4K_BLOCK: usize = 256;
/// RoPE base for the synthetic layers.
const ROPE_BASE: f32 = 10000.0;

/// One layer's weights in every format the comparison runs.
pub struct Layer {
    pub wq: Vec<u8>,
    pub wk: Vec<u8>,
    pub wv: Vec<u8>,
    pub wo: Vec<u8>,
    pub wq_kf: Vec<u8>,
    pub wk_kf: Vec<u8>,
    pub wv_kf: Vec<u8>,
    pub wo_kf: Vec<u8>,
    pub wq8: Vec<u8>,
    pub wk8: Vec<u8>,
    pub wv8: Vec<u8>,
    pub wo8: Vec<u8>,
    pub wq8s: Vec<f32>,
    pub wk8s: Vec<f32>,
    pub wv8s: Vec<f32>,
    pub wo8s: Vec<f32>,
    pub g: Vec<u8>,
    pub u: Vec<u8>,
    pub d: Vec<u8>,
    pub norm: Vec<f32>,
}

fn pad(d: &[f32]) -> Vec<f32> {
    let p = d.len().div_ceil(Q4K_BLOCK) * Q4K_BLOCK;
    let mut o = d.to_vec();
    o.resize(p, 0.0);
    o
}

/// Deterministic trig-pattern weights for `count` layers.
pub fn build_layers(count: usize) -> Vec<Layer> {
    let (hidden, inter, q_dim, kv_dim) = (HIDDEN, INTER, Q_DIM, KV_DIM);
    (0..count)
        .map(|l| {
            let wq_f = (0..q_dim * hidden)
                .map(|i| ((i + l * 1000) as f32 * 0.0001).cos())
                .collect::<Vec<_>>();
            let wk_f = (0..kv_dim * hidden)
                .map(|i| ((i + l * 2000) as f32 * 0.0002).sin())
                .collect::<Vec<_>>();
            let wv_f = (0..kv_dim * hidden)
                .map(|i| ((i + l * 3000) as f32 * 0.0003).cos())
                .collect::<Vec<_>>();
            let wo_f = (0..hidden * q_dim)
                .map(|i| ((i + l * 4000) as f32 * 0.0004).sin())
                .collect::<Vec<_>>();
            let (q8q, q8qs) = quantize_to_q8(&wq_f);
            let (q8k, q8ks) = quantize_to_q8(&wk_f);
            let (q8v, q8vs) = quantize_to_q8(&wv_f);
            let (q8o, q8os) = quantize_to_q8(&wo_f);
            Layer {
                wq: quantize_q4_k(&pad(&wq_f)),
                wk: quantize_q4_k(&pad(&wk_f)),
                wv: quantize_q4_k(&pad(&wv_f)),
                wo: quantize_q4_k(&pad(&wo_f)),
                // The Q4_KF kernels read standard 144-byte GGUF
                // Q4_K blocks (the tag selects the llama.cpp-exact
                // inner loop, not a layout — audit F15). This
                // example previously fed them the experimental
                // 160-byte pre-baked layout, so its Q4_KF arm
                // measured garbage numerics.
                wq_kf: quantize_q4_k(&pad(&wq_f)),
                wk_kf: quantize_q4_k(&pad(&wk_f)),
                wv_kf: quantize_q4_k(&pad(&wv_f)),
                wo_kf: quantize_q4_k(&pad(&wo_f)),
                wq8: q8q.iter().map(|&x| x as u8).collect(),
                wk8: q8k.iter().map(|&x| x as u8).collect(),
                wv8: q8v.iter().map(|&x| x as u8).collect(),
                wo8: q8o.iter().map(|&x| x as u8).collect(),
                wq8s: q8qs,
                wk8s: q8ks,
                wv8s: q8vs,
                wo8s: q8os,
                g: quantize_q4_k(&pad(&(0..inter * hidden)
                    .map(|i| ((i + l * 5000) as f32 * 0.0001).cos())
                    .collect::<Vec<_>>())),
                u: quantize_q4_k(&pad(&(0..inter * hidden)
                    .map(|i| ((i + l * 6000) as f32 * 0.0002).sin())
                    .collect::<Vec<_>>())),
                d: quantize_q4_k(&pad(&(0..hidden * inter)
                    .map(|i| ((i + l * 7000) as f32 * 0.0003).cos())
                    .collect::<Vec<_>>())),
                norm: vec![1.0f32; hidden],
            }
        })
        .collect()
}

/// Which attention-projection weights a decode arm runs. The FFN is
/// Q4_KF in every arm.
#[derive(Clone, Copy)]
pub enum AttnWeights {
    /// GGUF-default Q4_K attention.
    Q4K,
    /// Q8_0 attention with external per-block scales.
    Q8,
    /// Q4_KF attention — the llama.cpp-exact `q4kf_proj` /
    /// `q4kf_qkv_proj` kernels for every projection.
    Q4KF,
}

fn attn_weight<'a>(
    attn: AttnWeights,
    q4k: &'a [u8],
    q4kf: &'a [u8],
    q8: &'a [u8],
    q8_scales: &'a [f32],
) -> QuantWeight<'a> {
    match attn {
        AttnWeights::Q4K => QuantWeight::new(QuantFormat::Q4_K, q4k, QuantAux::None),
        AttnWeights::Q8 => {
            QuantWeight::new(QuantFormat::Q8_0, q8, QuantAux::ExternalScales(q8_scales))
        }
        AttnWeights::Q4KF => QuantWeight::new(QuantFormat::Q4_KF, q4kf, QuantAux::None),
    }
}

/// Pipeline layers over `data` with the chosen attention weights.
pub fn pipeline_layers(data: &[Layer], attn: AttnWeights) -> Vec<FullPipelineLayer<'_>> {
    data.iter()
        .map(|l| FullPipelineLayer {
            attn_sinks: None,
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_o_bias: None,
            attn_softcap: 0.0,
            wq: attn_weight(attn, &l.wq, &l.wq_kf, &l.wq8, &l.wq8s),
            wk: attn_weight(attn, &l.wk, &l.wk_kf, &l.wk8, &l.wk8s),
            wv: attn_weight(attn, &l.wv, &l.wv_kf, &l.wv8, &l.wv8s),
            wo: attn_weight(attn, &l.wo, &l.wo_kf, &l.wo8, &l.wo8s),
            gate: QuantWeight::new(QuantFormat::Q4_KF, &l.g, QuantAux::None),
            up: QuantWeight::new(QuantFormat::Q4_KF, &l.u, QuantAux::None),
            down: QuantWeight::new(QuantFormat::Q4_KF, &l.d, QuantAux::None),
            input_norm: &l.norm,
            post_attn_norm: &l.norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            norm_offset: 1.0,
            has_post_norms: false,
            activation: larql_compute::Activation::Silu,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            head_dim: HEAD_DIM,
            num_q_heads: NUM_Q,
            num_kv_heads: NUM_KV,
            rope_base: ROPE_BASE,
            rotary_dim: 0,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                HEAD_DIM,
                0_usize,
                ROPE_BASE as f64,
            ),
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        })
        .collect()
}
