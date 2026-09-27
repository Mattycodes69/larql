//! Tests for [`super`].
//!
//! Split out of `moe_interleave.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;
use larql_compute::pipeline::FullPipelineLayer;
use larql_compute::{
    Activation, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
};

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.013).sin() * 0.2)
        .collect()
}

fn pad_rows_to_256(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let padded_cols = cols.div_ceil(256) * 256;
    if padded_cols == cols {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        out[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    out
}

/// Same layout `tests/test_kernel_moe_expert_dispatch.rs` uses for
/// Q4_K experts: fused `[gate | up]` halves, block-padded down rows.
fn make_q4k_experts(hidden: usize, inter: usize, n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut gate_up = Vec::with_capacity(n);
    let mut down = Vec::with_capacity(n);
    for e in 0..n {
        let gate = synth(inter * hidden, 0.11 + e as f32 * 0.13);
        let up = synth(inter * hidden, 0.41 + e as f32 * 0.17);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(&gu));

        let raw_down = synth(hidden * inter, 0.73 + e as f32 * 0.07);
        let down_padded = pad_rows_to_256(&raw_down, hidden, inter);
        down.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(
            &down_padded,
        ));
    }
    (gate_up, down)
}

//
// The S2 GPU-route arm decides whether to SKIP attention's commit + wait by
// asking this function; the CPU fast path asks the same function whether to
// run. That shared answer is the point — two drifting copies would skip a
// wait some fallback arm still needs, which surfaces as intermittently
// wrong logits rather than as a crash.
//
// Every refusal is pinned to its own cause AND its own message, because the
// message is what `LARQL_MOE_INLINE_DIAG=1` prints: a diagnostic naming the
// wrong precondition sends the next reader to the wrong file.

/// Owns the expert bytes so a layer can borrow them for one assertion.
struct PreconditionFixture {
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
}

const P_HIDDEN: usize = 256;
const P_INTER: usize = 128;
const P_TOP_K: usize = 2;

fn precondition_fixture() -> PreconditionFixture {
    let (gate_up, down) = make_q4k_experts(P_HIDDEN, P_INTER, 4);
    PreconditionFixture { gate_up, down }
}

impl PreconditionFixture {
    /// The admitting shape. Each test mutates exactly the one field whose
    /// refusal it is checking.
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            experts_gate_up: self.gate_up.iter().map(|v| v.as_slice()).collect(),
            experts_down: self.down.iter().map(|v| v.as_slice()).collect(),
            expert_scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
            // `default()` is `gemma4_hybrid()`, which carries a post-expert
            // norm — stated explicitly here so the fixture IS the
            // identity-combine class this path serves.
            routing_policy: MoeRoutingPolicy {
                post_expert_norm: larql_compute::MoePostExpertNormPolicy::None,
                ..MoeRoutingPolicy::gemma4_hybrid()
            },
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q4_K,
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: self.gate_up.len(),
            top_k: P_TOP_K,
            intermediate_size: P_INTER,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
        }
    }
}

fn precondition_ctx() -> MoeInterleaveCtx<'static> {
    MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden: P_HIDDEN,
        inter: P_INTER,
        inter_padded: P_INTER,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    }
}

/// Assert the precondition check refuses, and that the reason it prints
/// names the actual cause.
fn assert_refuses(
    layer: &FullPipelineLayer<'_>,
    ctx: &MoeInterleaveCtx<'_>,
    scratch: &MoeScratch,
    needle: &str,
) {
    match MetalBackend::inline_moe_preconditions(layer, ctx, scratch) {
        Ok(_) => panic!("expected a refusal mentioning {needle:?}, but the layer was admitted"),
        Err(msg) => assert!(
            msg.contains(needle),
            "refused for the wrong reason: got {msg:?}, expected something mentioning {needle:?}"
        ),
    }
}

mod inline_moe_preconditions_the_single_auth;
mod without_moe_layer;
mod zero_copy_dispatch;
