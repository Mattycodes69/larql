extern crate blas_src;

use larql_compute::cpu::ops::moe::cpu_moe_forward;
use larql_compute::MoeLayerWeights;
use larql_compute::{cpu_backend, default_backend, Activation};

fn bf16_fill(len: usize, val: f32) -> Vec<u8> {
    let hi = (val.to_bits() >> 16) as u16;
    let b = hi.to_le_bytes();
    let mut v = vec![0u8; len * 2];
    for i in 0..len {
        v[i * 2] = b[0];
        v[i * 2 + 1] = b[1];
    }
    v
}

fn bf16_expert_tables<'a>(
    gate_up: &'a [u8],
    down: &'a [u8],
    num_experts: usize,
    inter: usize,
    hidden: usize,
) -> (Vec<&'a [u8]>, Vec<&'a [u8]>) {
    let gu_stride = 2 * inter * hidden * 2;
    let dn_stride = hidden * inter * 2;
    let experts_gate_up = (0..num_experts)
        .map(|e| &gate_up[e * gu_stride..(e + 1) * gu_stride])
        .collect();
    let experts_down = (0..num_experts)
        .map(|e| &down[e * dn_stride..(e + 1) * dn_stride])
        .collect();
    (experts_gate_up, experts_down)
}

#[allow(clippy::too_many_arguments)]
fn make_moe_weights<'a>(
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    gate_up: &'a [u8],
    down: &'a [u8],
    router: &'a [f32],
    router_norm: &'a [f32],
    router_norm_parameter_free: bool,
) -> MoeLayerWeights<'a> {
    let (experts_gate_up, experts_down) =
        bf16_expert_tables(gate_up, down, num_experts, inter, hidden);
    MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: larql_compute::MoeRoutingPolicy::top_k_renorm_scaled(),
        weight_layout: larql_compute::MoeWeightLayout::default(),
        router_proj: router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm,
        router_norm_parameter_free,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: larql_compute::MoeGateRule::Gated(Activation::Silu),
        expert_data_format: larql_compute::QuantFormat::BF16,
    }
}

//
// Integration tests for the batched MoE prefill path introduced in
// 2026-04-26. They call through the public `DecodeBackend::prefill_kquant` API
// so they exercise the full `dispatch_full_pipeline` + `moe_fn` callback
// chain without reaching into private internals.

mod lib_rs_entry_points;
mod metal_prefill_kquant_with_moe_layers;
