//! MoE parity: one expert, and a whole MoE block, across backends.

use larql_compute::cpu::ops::moe::{cpu_moe_forward, run_single_expert_with_norm};
use larql_compute::{MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat};

#[allow(unused_imports)]
use super::*;

// ── moe-expert: one expert's forward pass (proven correct in v0) ─────────────

pub(super) fn run_moe_expert(
    config: &larql_vindex::VindexConfig,
    weights: &larql_models::ModelWeights,
    args: &ParityArgs,
    backends: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let arch = &*weights.arch;
    let hidden = config.hidden_size;
    let inter = arch.moe_intermediate_size();
    let inter_padded = inter.div_ceil(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
        * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let num_experts = arch.num_experts();
    if args.expert >= num_experts {
        return Err(format!(
            "expert {} out of range (model has {num_experts})",
            args.expert
        )
        .into());
    }

    let (gu_bytes, dn_bytes) = expert_bytes(weights, args.layer, args.expert)?;
    let pre_norm = pre_experts_norm_for(weights, args.layer);
    let activation = activation_for(arch);
    let h = make_residual(hidden, args.seed);

    println!("Expert: {}", args.expert);
    println!(
        "Per-expert bytes: gate_up={} ({:.2} MB), down={} ({:.2} MB)",
        gu_bytes.len(),
        gu_bytes.len() as f64 / 1e6,
        dn_bytes.len(),
        dn_bytes.len() as f64 / 1e6,
    );
    println!();

    let mut traces: Vec<(&str, Vec<f32>)> = Vec::new();
    for backend in backends {
        let out = match *backend {
            "reference" => reference_one_expert(
                &h,
                gu_bytes,
                dn_bytes,
                hidden,
                inter,
                inter_padded,
                pre_norm,
                arch.norm_weight_offset(),
                arch.norm_eps(),
                activation,
                args.verbose,
            ),
            "cpu" => run_single_expert_with_norm(
                &h,
                gu_bytes,
                dn_bytes,
                inter,
                pre_norm,
                arch.norm_weight_offset(),
                arch.norm_eps(),
                QuantFormat::Q4_K,
                larql_compute::ExpertMlp::gated(activation),
            ),
            _ => return Err(format!("backend '{backend}' not yet wired for moe-expert").into()),
        };
        traces.push((backend, out));
    }

    println!("=== expert_output diff ===");
    diff_against_first(&traces, args.tolerance);
    Ok(())
}

// ── moe-block: full block — router + top-K + K experts + sum + post-norm ─────
//
// This is the v1 component that should localise the current Gemma 4 26B-A4B
// CPU MoE bug — per-expert compute is already proven correct (see v0
// prototype), so divergence here means routing or combination is off.

pub(super) fn run_moe_block(
    config: &larql_vindex::VindexConfig,
    weights: &larql_models::ModelWeights,
    args: &ParityArgs,
    backends: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let arch = &*weights.arch;
    let hidden = config.hidden_size;
    let inter = arch.moe_intermediate_size();
    let inter_padded = inter.div_ceil(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
        * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let num_experts = arch.num_experts();
    let top_k = arch.num_experts_per_token();

    let h = make_residual(hidden, args.seed);
    let pre_norm = pre_experts_norm_for(weights, args.layer);
    let post_norm = post_experts_norm_for(weights, args.layer);
    let router_proj = router_proj_for(weights, arch, args.layer)?;
    let router_per_expert_scale = router_per_expert_scale_for(weights, arch, args.layer);
    let router_norm = router_norm_for(weights, arch, args.layer);
    let router_norm_parameter_free = arch.moe_router_norm_parameter_free();
    let router_input_scalar = arch.moe_router_input_scalar().unwrap_or(1.0);
    let activation = activation_for(arch);
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();

    println!(
        "Block: layer {} of {}, hidden={hidden}, inter={inter} (padded {inter_padded}), \
         experts={num_experts} top_k={top_k}",
        args.layer, config.num_layers
    );
    println!();

    // Build per-expert byte tables once — both backends consume the same.
    let mut experts_gate_up: Vec<&[u8]> = Vec::with_capacity(num_experts);
    let mut experts_down: Vec<&[u8]> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let (gu, dn) = expert_bytes(weights, args.layer, e)?;
        experts_gate_up.push(gu);
        experts_down.push(dn);
    }

    let moe = MoeLayerWeights {
        // Both describe the VINDEX2 store this diagnostic reads — k-quant
        // blocks with inline scales, de-interleaved by the extraction path —
        // and match what `build_moe_weights` declares for the same bytes.
        // A parity probe that described the store differently from the route
        // it is checking would be comparing two different reads.
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: experts_gate_up.clone(),
        experts_down: experts_down.clone(),
        // Typed dispatch on `MoeRouterKind` — the string form
        // (`moe_router_type()`) is serialisation-only, and a missed
        // string arm here would silently rescale the whole expert
        // branch (the §4.7.10 failure class).
        routing_policy: match arch.moe_router_kind() {
            larql_models::MoeRouterKind::TopKSoftmax => MoeRoutingPolicy::top_k_softmax(),
            larql_models::MoeRouterKind::TopKThenSoftmax => MoeRoutingPolicy::top_k_then_softmax(),
            larql_models::MoeRouterKind::Gemma4Hybrid => MoeRoutingPolicy::gemma4_hybrid(),
            // Represented, not executable — see
            // `larql_compute::pipeline_layer::moe_build::moe_routing_policy`.
            // Every policy here normalises across experts in a way sigmoid
            // does not, so substituting one produces plausible, wrong
            // expert weights.
            larql_models::MoeRouterKind::Sigmoid => {
                unimplemented!("sigmoid expert routing is represented but not executable")
            }
        },
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q4_K,
        router_proj: &router_proj,
        router_scale: &[],
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        pre_experts_norm: pre_norm,
        post_ffn1_norm: &[],
        post_experts_norm: post_norm,
        num_experts,
        top_k,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: larql_compute::MoeGateRule::Gated(activation),
    };

    let mut traces: Vec<(&str, Vec<f32>)> = Vec::new();
    for backend in backends {
        let out = match *backend {
            "reference" => reference_moe_block(
                &h,
                &experts_gate_up,
                &experts_down,
                &router_proj,
                &router_per_expert_scale,
                &router_norm,
                router_norm_parameter_free,
                router_input_scalar,
                pre_norm,
                post_norm,
                hidden,
                inter,
                inter_padded,
                num_experts,
                top_k,
                activation,
                norm_offset,
                eps,
                args.verbose,
            ),
            "cpu" => cpu_moe_forward(&h, &moe, norm_offset, eps),
            _ => return Err(format!("backend '{backend}' not yet wired for moe-block").into()),
        };
        traces.push((backend, out));
    }

    println!("=== moe_block_output diff ===");
    diff_against_first(&traces, args.tolerance);

    // Side-by-side routing-convention check: which top-K does each
    // convention select? Per HF Gemma4TextDecoderLayer.forward, the router
    // consumes the raw post-attention residual; experts consume
    // pre_experts_norm(residual). If h_norm and raw_h pick different
    // experts, mis-routing the input is what produces "fluent but wrong"
    // generation.
    println!();
    println!("=== Routing-convention comparison ===");
    let h_norm = naive_rms_norm(&h, pre_norm, eps, norm_offset);
    let (idx_raw, w_raw) = compute_top_k(
        &h,
        &router_proj,
        &router_per_expert_scale,
        &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        num_experts,
        top_k,
        hidden,
        eps,
        norm_offset,
    );
    let (idx_norm, w_norm) = compute_top_k(
        &h_norm,
        &router_proj,
        &router_per_expert_scale,
        &router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        num_experts,
        top_k,
        hidden,
        eps,
        norm_offset,
    );
    println!("  router_in=raw_h    top_k: {idx_raw:?}");
    println!(
        "    weights:                 {}",
        w_raw
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("  router_in=h_norm   top_k: {idx_norm:?}  ← Metal/GPU convention");
    println!(
        "    weights:                 {}",
        w_norm
            .iter()
            .map(|w| format!("{w:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let same: Vec<usize> = idx_raw
        .iter()
        .filter(|&&e| idx_norm.contains(&e))
        .copied()
        .collect();
    if same.len() == top_k {
        println!("  ✓ SAME top-{top_k} experts selected — routing input choice is not the bug");
    } else {
        println!(
            "  ✗ DIFFERENT top-{top_k}: {} overlap, {} differ — expert-selection convention IS the bug surface",
            same.len(),
            top_k - same.len()
        );
    }
    Ok(())
}
