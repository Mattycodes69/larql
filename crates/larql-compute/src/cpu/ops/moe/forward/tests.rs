use super::*;
use crate::cpu::ops::q4_common::quantize_q4_k;
use crate::{Activation, QuantFormat};

fn bf16_fill(len: usize, val: f32) -> Vec<u8> {
    let b = ((val.to_bits() >> 16) as u16).to_le_bytes();
    let mut bytes = vec![0u8; len * 2];
    for i in 0..len {
        bytes[i * 2] = b[0];
        bytes[i * 2 + 1] = b[1];
    }
    bytes
}

fn one_expert_moe<'a>(
    _hidden: usize,
    inter: usize,
    experts_gate_up: Vec<&'a [u8]>,
    experts_down: Vec<&'a [u8]>,
    router: &'a [f32],
    format: QuantFormat,
) -> MoeLayerWeights<'a> {
    MoeLayerWeights {
        expert_scales: crate::MoeExpertScales::Inline,
        fused_row_layout: crate::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: crate::MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: crate::MoeWeightLayout::default(),
        router_proj: router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 1,
        top_k: 1,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: crate::MoeGateRule::Gated(Activation::Silu),
        expert_data_format: format,
    }
}

#[test]
fn empty_selected_expert_weight_slices_are_skipped() {
    let hidden = 8;
    let inter = 2;
    let router = vec![1.0f32; hidden];
    let h = vec![1.0f32; hidden];
    let gate_up = bf16_fill(2 * inter * hidden, 1.0);
    let down = bf16_fill(hidden * inter, 1.0);

    let missing_gate_up = one_expert_moe(
        hidden,
        inter,
        vec![&[]],
        vec![down.as_slice()],
        &router,
        QuantFormat::BF16,
    );
    assert_eq!(
        cpu_moe_forward(&h, &missing_gate_up, 0.0, 1e-6),
        vec![0.0; hidden]
    );

    let missing_down = one_expert_moe(
        hidden,
        inter,
        vec![gate_up.as_slice()],
        vec![&[]],
        &router,
        QuantFormat::BF16,
    );
    assert_eq!(
        cpu_moe_forward(&h, &missing_down, 0.0, 1e-6),
        vec![0.0; hidden]
    );
}

#[test]
fn selected_expert_with_missing_down_table_is_skipped() {
    let hidden = 8;
    let inter = 2;
    let num_experts = 4;
    let gate_up = bf16_fill(2 * inter * hidden, 1.0);
    let down = bf16_fill(hidden * inter, 1.0);
    let experts_gate_up = vec![
        gate_up.as_slice(),
        gate_up.as_slice(),
        gate_up.as_slice(),
        gate_up.as_slice(),
    ];
    let experts_down = vec![down.as_slice()];
    let mut router = vec![0.0f32; num_experts * hidden];
    router[3 * hidden..4 * hidden].fill(10.0);
    let moe = MoeLayerWeights {
        expert_scales: crate::MoeExpertScales::Inline,
        fused_row_layout: crate::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: crate::MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: crate::MoeWeightLayout::default(),
        router_proj: &router,
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k: 1,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: crate::MoeGateRule::Gated(Activation::Silu),
        expert_data_format: QuantFormat::BF16,
    };

    assert_eq!(
        cpu_moe_forward(&vec![1.0; hidden], &moe, 0.0, 1e-6),
        vec![0.0; hidden]
    );
}

#[test]
fn q4k_cached_dequant_fallback_runs_for_non_256_hidden() {
    let hidden = 128;
    let inter = 1;
    let inter_padded = 256;
    let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let down_f32: Vec<f32> = (0..hidden * inter_padded)
        .map(|i| {
            if i % inter_padded == 0 {
                ((i / inter_padded) as f32 % 13.0 - 6.0) * 0.01
            } else {
                0.0
            }
        })
        .collect();
    let gate_up = quantize_q4_k(&gate_up_f32);
    let down = quantize_q4_k(&down_f32);
    let router = vec![1.0f32; hidden];
    let h: Vec<f32> = (0..hidden).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let moe = one_expert_moe(
        hidden,
        inter,
        vec![gate_up.as_slice()],
        vec![down.as_slice()],
        &router,
        QuantFormat::Q4_K,
    );

    let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);

    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn zero_per_expert_scale_filters_selected_expert() {
    let hidden = 8;
    let inter = 2;
    let gate_up = bf16_fill(2 * inter * hidden, 1.0);
    let down = bf16_fill(hidden * inter, 1.0);
    let router = vec![1.0f32; hidden];
    let zero_scale = [0.0f32];
    let moe = MoeLayerWeights {
        router_per_expert_scale: &zero_scale,
        ..one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        )
    };

    assert_eq!(
        cpu_moe_forward(&vec![1.0; hidden], &moe, 0.0, 1e-6),
        vec![0.0; hidden]
    );
}

// Override env flags on the current thread (`LARQL_SKIP_MOE`,
// `LARQL_MOE_DEBUG`, `LARQL_MOE_FWD_TIMING` — all read via the override-aware
// `options::env_flag`) WITHOUT `std::env::set_var`, which races concurrent
// `getenv` → SIGSEGV. Per-thread, cleared on drop → no lock, no leakage.
fn with_env<T>(vars: &[(&'static str, Option<&'static str>)], f: impl FnOnce() -> T) -> T {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            crate::options::clear_fast_path_overrides();
        }
    }
    let _clear = Clear;
    for (name, value) in vars {
        crate::options::set_env_override(name, *value);
    }
    f()
}

fn trivial_moe_inputs() -> (usize, usize, Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let hidden = 8;
    let inter = 2;
    let gate_up = bf16_fill(2 * inter * hidden, 1.0);
    let down = bf16_fill(hidden * inter, 1.0);
    let router = vec![1.0f32; hidden];
    let h = vec![1.0f32; hidden];
    (hidden, inter, gate_up, down, router, h)
}

#[test]
fn returns_zero_vec_when_num_experts_zero() {
    let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
    let moe = MoeLayerWeights {
        num_experts: 0,
        ..one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        )
    };
    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn returns_zero_vec_when_top_k_zero() {
    let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
    let moe = MoeLayerWeights {
        top_k: 0,
        ..one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        )
    };
    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn returns_zero_vec_when_intermediate_size_zero() {
    let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
    let moe = MoeLayerWeights {
        intermediate_size: 0,
        ..one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        )
    };
    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn returns_zero_vec_when_router_proj_empty() {
    let (hidden, inter, gate_up, down, _, h) = trivial_moe_inputs();
    let empty: [f32; 0] = [];
    let moe = MoeLayerWeights {
        router_proj: &empty,
        ..one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &empty,
            QuantFormat::BF16,
        )
    };
    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn returns_zero_vec_when_experts_gate_up_table_empty() {
    let (hidden, inter, _, down, router, h) = trivial_moe_inputs();
    let moe = one_expert_moe(
        hidden,
        inter,
        vec![],
        vec![down.as_slice()],
        &router,
        QuantFormat::BF16,
    );
    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn returns_zero_vec_when_experts_down_table_empty() {
    let (hidden, inter, gate_up, _, router, h) = trivial_moe_inputs();
    let moe = one_expert_moe(
        hidden,
        inter,
        vec![gate_up.as_slice()],
        vec![],
        &router,
        QuantFormat::BF16,
    );
    assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
}

#[test]
fn skip_moe_env_returns_zero_vec_before_running_experts() {
    let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
    let moe = one_expert_moe(
        hidden,
        inter,
        vec![gate_up.as_slice()],
        vec![down.as_slice()],
        &router,
        QuantFormat::BF16,
    );
    let out = with_env(&[(crate::options::ENV_SKIP_MOE, Some("1"))], || {
        cpu_moe_forward(&h, &moe, 0.0, 1e-6)
    });
    assert_eq!(out, vec![0.0; hidden]);
}

#[test]
fn moe_debug_env_exercises_diagnostic_branches() {
    // Pre-experts + router norm + scale arrays populated so the debug
    // branch reads non-empty buffers (rms calculations).
    let hidden = 8;
    let inter = 2;
    let gate_up = bf16_fill(2 * inter * hidden, 1.0);
    let down = bf16_fill(hidden * inter, 1.0);
    let router = vec![0.5f32; hidden];
    let pre = vec![1.0f32; hidden];
    let rnorm = vec![1.0f32; hidden];
    // `router_scale` is element-wise multiplied with `router_in` (zip), so
    // it must match `hidden` in length to avoid silently truncating the
    // router input vector.
    let rscale = vec![1.0f32; hidden];
    let post = vec![1.0f32; hidden];
    let moe = MoeLayerWeights {
        pre_experts_norm: &pre,
        router_norm: &rnorm,
        router_scale: &rscale,
        post_experts_norm: &post,
        ..one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        )
    };
    let h = vec![0.25f32; hidden];

    let out = with_env(&[(crate::options::ENV_MOE_DEBUG, Some("1"))], || {
        cpu_moe_forward(&h, &moe, 0.0, 1e-6)
    });
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn moe_fwd_timing_env_prints_timing_block() {
    // FWD_TIMING is a TLS-cached read of the env var, evaluated lazily on
    // first hit per thread.  Run the entire call on a fresh
    // `std::thread::spawn` so the TLS init sees the env var set to "1"
    // instead of inheriting an earlier `false` from this test process's
    // main thread.
    let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();

    let result = with_env(&[(crate::options::ENV_MOE_FWD_TIMING, Some("1"))], || {
        std::thread::spawn(move || {
            let moe = one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            );
            cpu_moe_forward(&h, &moe, 0.0, 1e-6)
        })
        .join()
        .expect("timing thread did not panic")
    });
    assert_eq!(result.len(), hidden);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn scratch_reallocates_when_dimensions_change_between_calls() {
    // Force rayon's worker TLS to hold a scratch for one shape, then call
    // again with different (hidden, inter) so the size-mismatch branch
    // (`*scratch = ExpertScratch::new(...)`) is taken.  We don't assert
    // the path runs on every worker — rayon may schedule onto a
    // freshly-warmed thread — but the par_iter is non-empty on either
    // shape, so at least one worker hits the branch.
    let first_h = 8usize;
    let first_inter = 2usize;
    let first_gate_up = bf16_fill(2 * first_inter * first_h, 1.0);
    let first_down = bf16_fill(first_h * first_inter, 1.0);
    let first_router = vec![1.0f32; first_h];
    let first_moe = one_expert_moe(
        first_h,
        first_inter,
        vec![first_gate_up.as_slice()],
        vec![first_down.as_slice()],
        &first_router,
        QuantFormat::BF16,
    );
    let out_a = cpu_moe_forward(&vec![1.0; first_h], &first_moe, 0.0, 1e-6);
    assert_eq!(out_a.len(), first_h);

    let second_h = 16usize;
    let second_inter = 4usize;
    let second_gate_up = bf16_fill(2 * second_inter * second_h, 1.0);
    let second_down = bf16_fill(second_h * second_inter, 1.0);
    let second_router = vec![1.0f32; second_h];
    let second_moe = one_expert_moe(
        second_h,
        second_inter,
        vec![second_gate_up.as_slice()],
        vec![second_down.as_slice()],
        &second_router,
        QuantFormat::BF16,
    );
    let out_b = cpu_moe_forward(&vec![1.0; second_h], &second_moe, 0.0, 1e-6);
    assert_eq!(out_b.len(), second_h);
    assert!(out_b.iter().all(|v| v.is_finite()));
}

/// `num_experts` Q4_K experts at `hidden` = `inter` = one K-quant block, with
/// only the first `gate_up_present` gate/up tables stored and the router
/// pointing at the last expert.
/// Gate/up tables, down tables, router, input.
type Q4kBank = (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<f32>, Vec<f32>);

fn q4k_bank(num_experts: usize, gate_up_present: usize) -> Q4kBank {
    const K: usize = 256;
    let gate_up: Vec<Vec<u8>> = (0..num_experts)
        .map(|e| {
            let w: Vec<f32> = (0..2 * K * K)
                .map(|i| (((i + e) % 17) as f32 - 8.0) * 0.001)
                .collect();
            quantize_q4_k(&w)
        })
        .collect();
    let down: Vec<Vec<u8>> = (0..num_experts)
        .map(|e| {
            let w: Vec<f32> = (0..K * K)
                .map(|i| (((i + 3 * e) % 13) as f32 - 6.0) * 0.001)
                .collect();
            quantize_q4_k(&w)
        })
        .collect();
    let h: Vec<f32> = (0..K).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    // The last expert's router row is `h` itself: its logit is |h|² > 0,
    // every other expert's is 0.
    let mut router = vec![0.0f32; num_experts * K];
    router[(num_experts - 1) * K..].copy_from_slice(&h);
    let gate_up = gate_up.into_iter().take(gate_up_present).collect();
    (gate_up, down, router, h)
}

fn q4k_moe<'a>(
    gate_up: &'a [Vec<u8>],
    down: &'a [Vec<u8>],
    router: &'a [f32],
    num_experts: usize,
) -> MoeLayerWeights<'a> {
    let mut moe = one_expert_moe(
        256,
        256,
        gate_up.iter().map(Vec::as_slice).collect(),
        down.iter().map(Vec::as_slice).collect(),
        router,
        QuantFormat::Q4_K,
    );
    moe.num_experts = num_experts;
    moe
}

/// Run `f` under the spin-pool schedule (`on`) or the rayon one.
fn with_spin_pool<T>(on: bool, f: impl FnOnce() -> T) -> T {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            crate::options::clear_fast_path_overrides();
        }
    }
    let _clear = Clear;
    crate::options::set_fast_path_override(crate::options::ENV_SPIN_POOL, on);
    f()
}

#[test]
fn both_schedules_compute_the_same_expert_block() {
    let (gate_up, down, router, h) = q4k_bank(2, 2);
    let moe = q4k_moe(&gate_up, &down, &router, 2);
    let rayon = with_spin_pool(false, || cpu_moe_forward(&h, &moe, 0.0, 1e-6));
    let spin = with_spin_pool(true, || cpu_moe_forward(&h, &moe, 0.0, 1e-6));
    assert!(
        rayon.iter().any(|v| *v != 0.0),
        "the routed expert contributes"
    );
    for (a, b) in rayon.iter().zip(&spin) {
        assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "{a} vs {b}");
    }
}

#[test]
fn both_schedules_skip_a_selected_expert_with_no_gate_up_table() {
    let (gate_up, down, router, h) = q4k_bank(2, 1);
    let moe = q4k_moe(&gate_up, &down, &router, 2);
    for on in [false, true] {
        let out = with_spin_pool(on, || cpu_moe_forward(&h, &moe, 0.0, 1e-6));
        assert_eq!(out, vec![0.0; h.len()], "spin pool {on}");
    }
}

#[test]
fn a_block_padded_bank_matches_the_unpadded_one() {
    let hidden = 8;
    let stored_cols = 16;
    let inter = 2;
    let router = vec![1.0f32; hidden];
    let h = vec![1.0f32; hidden];
    let down = bf16_fill(hidden * inter, 1.0);
    let plain = bf16_fill(2 * inter * hidden, 1.0);
    // Padded rows: the first `hidden` columns of each row carry the weights,
    // the pad columns hold weights the zero-padded activation must cancel.
    let padded = bf16_fill(2 * inter * stored_cols, 1.0);
    let run = |gate_up: &[u8]| {
        let moe = one_expert_moe(
            hidden,
            inter,
            vec![gate_up],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        );
        cpu_moe_forward(&h, &moe, 0.0, 1e-6)
    };
    assert_eq!(run(&padded), run(&plain));
}

#[test]
fn per_stage_timing_is_reported_without_changing_the_result() {
    let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
    let expected = {
        let moe = one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        );
        cpu_moe_forward(&h, &moe, 0.0, 1e-6)
    };
    // The timing flag is cached per thread on first use, so a fresh thread
    // is the only place the override is guaranteed to be read.
    let timed = std::thread::spawn(move || {
        with_env(&[(crate::options::ENV_MOE_FWD_TIMING, Some("1"))], || {
            let moe = one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            );
            cpu_moe_forward(&h, &moe, 0.0, 1e-6)
        })
    })
    .join()
    .unwrap();
    assert_eq!(timed, expected);
}

#[test]
fn the_latent_probe_masks_the_input_and_optionally_the_output() {
    use super::super::latent_mask::{LatentMask, Mode};
    let (gate_up, down, router, h) = q4k_bank(1, 1);
    let moe = q4k_moe(&gate_up, &down, &router, 1);
    let mask = |both_sides| LatentMask {
        retention: 0.5,
        mode: Mode::Magnitude,
        both_sides,
        block: 1,
        perm: Vec::new(),
    };
    let full = cpu_moe_forward_with_latent_mask(&h, &moe, 0.0, 1e-6, None);
    let input_only = cpu_moe_forward_with_latent_mask(&h, &moe, 0.0, 1e-6, Some(&mask(false)));
    let both = cpu_moe_forward_with_latent_mask(&h, &moe, 0.0, 1e-6, Some(&mask(true)));
    assert_ne!(input_only, full, "masking half the input changes the block");
    let zeroed = |v: &[f32]| v.iter().filter(|x| **x == 0.0).count();
    assert!(
        zeroed(&both) > zeroed(&input_only),
        "the both-sides probe also masks the output channels"
    );
}
