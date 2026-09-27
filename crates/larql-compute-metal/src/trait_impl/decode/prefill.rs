//! K-quant prefill bodies behind the `DecodeBackend` prefill methods.

use crate::{ops, MetalBackend};

#[allow(unused_imports)]
use super::*;

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_prefill_kquant(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );

        let has_moe = layers.iter().any(|l| l.moe.is_some());
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == larql_compute::Activation::GeluTanh)
        {
            &self.ffn.geglu_gelu_tanh_pipeline
        } else {
            &self.ffn.geglu_pipeline
        };

        // Concrete macro to avoid duplicating the 30-param dispatch call.
        // Second parameter is the optional PipelineIntervention for head replacement.
        macro_rules! run_dispatch {
            ($moe_fn:expr, $intervention:expr) => {
                ops::full_pipeline::dispatch_full_pipeline(
                    &self.queue,
                    &self.bufs,
                    &self.q4,
                    geglu,
                    &self.ffn.geglu_gelu_tanh_pipeline,
                    &self.ffn.silu_pipeline,
                    &self.ffn.gelu_tanh_pipeline,
                    &self.quant.q8_quant_pipeline,
                    Some(&self.attention.fused_attn_pipeline),
                    &self.quant.q8_matvec_pipeline.state,
                    &self.attention.q8_qkv_proj_pipeline.state,
                    &self.quant.q4k_matvec_pipeline,
                    Some(&self.quant.q4k_matmul_pipeline),
                    &self.quant.q6k_matvec_pipeline,
                    &self.norms.rms_norm_pipeline,
                    &self.norms.residual_add_pipeline,
                    &self.norms.rms_norm_q8_pipeline,
                    &self.norms.residual_norm_q8_pipeline,
                    Some(&self.attention.q4k_qkv_proj_pipeline.state),
                    Some(&self.attention.q4kf_qkv_proj_pipeline.state),
                    Some(&self.attention.q4k_q6k_qkv_proj_pipeline),
                    Some(&self.attention.q4kf_proj_pipeline.state),
                    Some(&self.attention.bias_add_pipeline),
                    Some(&self.attention.rope_at_pos_pipeline),
                    Some(&self.norms.qk_norm_pipeline),
                    Some(&self.norms.scale_vector_pipeline),
                    Some(&self.ffn.q4k_geglu_silu_down_pipeline),
                    Some(&self.ffn.q4k_geglu_gelu_tanh_down_pipeline),
                    Some(&self.ffn.q6k_geglu_silu_down_pipeline),
                    Some(&self.ffn.q6k_geglu_gelu_tanh_down_pipeline),
                    Some(kv),
                    layers,
                    x,
                    hidden,
                    inter,
                    q_dim,
                    kv_dim,
                    seq_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    rope_base,
                    use_qk_norm,
                    softcap,
                    $moe_fn,
                    $intervention,
                )
            };
        }

        if has_moe {
            // Per-layer MoE callback: runs CPU experts for all seq_len positions,
            // accumulates into new_h, then applies outer post-FFN norm + layer_scalar.
            // GPU layer_scalar step is skipped for MoE layers in dispatch_full_pipeline
            // (see `is_moe_layer` guard) so this closure owns the combine step.
            let mut moe_closure = |layer_idx: usize, h_post_attn: &[f32], new_h: &mut [f32]| {
                let layer = &layers[layer_idx];
                let moe_block = match layer.moe.as_ref() {
                    Some(m) => m,
                    None => return,
                };
                let layer_eps = layer.eps;
                let layer_norm_offset = layer.norm_offset;

                // 1. CPU MoE for each position: accumulate into new_h.
                for pos in 0..seq_len {
                    let ha = &h_post_attn[pos * hidden..(pos + 1) * hidden];
                    let moe_out = larql_compute::cpu::ops::moe::cpu_moe_forward(
                        ha,
                        moe_block,
                        layer_norm_offset,
                        layer_eps,
                    );
                    let nh = &mut new_h[pos * hidden..(pos + 1) * hidden];
                    for (i, v) in moe_out.iter().enumerate() {
                        nh[i] += v;
                    }
                }

                // 2. Outer post-FFN norm + layer_scalar per position.
                // Matches moe_combine::apply_outer_combine for batched positions.
                for pos in 0..seq_len {
                    let ha = &h_post_attn[pos * hidden..(pos + 1) * hidden];
                    let nh = &mut new_h[pos * hidden..(pos + 1) * hidden];

                    if layer.moe_combined_output_norm {
                        let outer_w = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
                        if let Some(w) = outer_w {
                            let combined: Vec<f32> =
                                nh.iter().zip(ha).map(|(h, a)| h - a).collect();
                            let rms = (combined.iter().map(|v| v * v).sum::<f32>() / hidden as f32
                                + layer_eps)
                                .sqrt();
                            for (i, (&c, &wt)) in combined.iter().zip(w.iter()).enumerate() {
                                nh[i] = ha[i] + c / rms * (wt + layer_norm_offset);
                            }
                        }
                    }

                    let ls = layer.layer_scalar;
                    if ls != 0.0 && ls != 1.0 {
                        for v in nh.iter_mut() {
                            *v *= ls;
                        }
                    }
                }
            };
            return Some(run_dispatch!(
                Some(&mut moe_closure as &mut dyn FnMut(usize, &[f32], &mut [f32])),
                None
            ));
        }

        Some(run_dispatch!(None, None))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_full_pipeline_kquant_capture_pre_wo(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
    ) -> Option<Vec<f32>> {
        use ops::full_pipeline::{dispatch_full_pipeline, PipelineIntervention};
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == larql_compute::Activation::GeluTanh)
        {
            &self.ffn.geglu_gelu_tanh_pipeline
        } else {
            &self.ffn.geglu_pipeline
        };
        // Capture geometry must match the target layer.
        let (target_head_dim, target_num_q_heads) = layers
            .get(target_layer)
            .map(|l| (l.head_dim, l.num_q_heads))
            .unwrap_or((head_dim, num_q_heads));
        let intervention = PipelineIntervention {
            target_layer,
            target_head,
            head_dim: target_head_dim,
            num_q_heads: target_num_q_heads,
            replacement_delta: &[], // unused — stop_after_capture returns before hook B
            pre_wo_capture: std::cell::RefCell::new(Vec::new()),
            stop_after_capture: true, // stop after capture, return pre_wo via RefCell
        };
        // dispatch returns empty vec (stop_after_capture=true); ignore it.
        let _ = dispatch_full_pipeline(
            &self.queue,
            &self.bufs,
            &self.q4,
            geglu,
            &self.ffn.geglu_gelu_tanh_pipeline,
            &self.ffn.silu_pipeline,
            &self.ffn.gelu_tanh_pipeline,
            &self.quant.q8_quant_pipeline,
            Some(&self.attention.fused_attn_pipeline),
            &self.quant.q8_matvec_pipeline.state,
            &self.attention.q8_qkv_proj_pipeline.state,
            &self.quant.q4k_matvec_pipeline,
            Some(&self.quant.q4k_matmul_pipeline),
            &self.quant.q6k_matvec_pipeline,
            &self.norms.rms_norm_pipeline,
            &self.norms.residual_add_pipeline,
            &self.norms.rms_norm_q8_pipeline,
            &self.norms.residual_norm_q8_pipeline,
            Some(&self.attention.q4k_qkv_proj_pipeline.state),
            Some(&self.attention.q4kf_qkv_proj_pipeline.state),
            Some(&self.attention.q4k_q6k_qkv_proj_pipeline),
            Some(&self.attention.q4kf_proj_pipeline.state),
            Some(&self.attention.bias_add_pipeline),
            Some(&self.attention.rope_at_pos_pipeline),
            Some(&self.norms.qk_norm_pipeline),
            Some(&self.norms.scale_vector_pipeline),
            Some(&self.ffn.q4k_geglu_silu_down_pipeline),
            Some(&self.ffn.q4k_geglu_gelu_tanh_down_pipeline),
            Some(&self.ffn.q6k_geglu_silu_down_pipeline),
            Some(&self.ffn.q6k_geglu_gelu_tanh_down_pipeline),
            None, // no KV cache
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            use_qk_norm,
            softcap,
            None,                // no MoE
            Some(&intervention), // intervention fires at target_layer then stops
        );
        let captured = intervention.pre_wo_capture.into_inner();
        if captured.is_empty() {
            None
        } else {
            Some(captured)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_prefill_kquant_with_head_replacement(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        use_qk_norm: bool,
        softcap: f32,
        target_layer: usize,
        target_head: usize,
        replacement_delta: &[f32],
    ) -> Option<Vec<f32>> {
        let (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim, rope_base) =
            legacy_l0_geometry(layers);
        let mut cache_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut cache_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );
        let has_moe = layers.iter().any(|l| l.moe.is_some());
        if has_moe {
            // MoE + intervention not yet supported — fall back to non-intervention prefill.
            drop(cache_guard);
            return self.prefill_kquant(layers, x, hidden, inter, seq_len, use_qk_norm, softcap);
        }
        let geglu = if layers
            .first()
            .is_some_and(|l| l.activation == larql_compute::Activation::GeluTanh)
        {
            &self.ffn.geglu_gelu_tanh_pipeline
        } else {
            &self.ffn.geglu_pipeline
        };
        // Intervention geometry must match the target layer.
        let (target_head_dim, target_num_q_heads) = layers
            .get(target_layer)
            .map(|l| (l.head_dim, l.num_q_heads))
            .unwrap_or((head_dim, num_q_heads));
        let intervention = ops::full_pipeline::PipelineIntervention {
            target_layer,
            target_head,
            head_dim: target_head_dim,
            num_q_heads: target_num_q_heads,
            replacement_delta,
            pre_wo_capture: std::cell::RefCell::new(Vec::new()),
            stop_after_capture: false,
        };
        Some(ops::full_pipeline::dispatch_full_pipeline(
            &self.queue,
            &self.bufs,
            &self.q4,
            geglu,
            &self.ffn.geglu_gelu_tanh_pipeline,
            &self.ffn.silu_pipeline,
            &self.ffn.gelu_tanh_pipeline,
            &self.quant.q8_quant_pipeline,
            Some(&self.attention.fused_attn_pipeline),
            &self.quant.q8_matvec_pipeline.state,
            &self.attention.q8_qkv_proj_pipeline.state,
            &self.quant.q4k_matvec_pipeline,
            Some(&self.quant.q4k_matmul_pipeline),
            &self.quant.q6k_matvec_pipeline,
            &self.norms.rms_norm_pipeline,
            &self.norms.residual_add_pipeline,
            &self.norms.rms_norm_q8_pipeline,
            &self.norms.residual_norm_q8_pipeline,
            Some(&self.attention.q4k_qkv_proj_pipeline.state),
            Some(&self.attention.q4kf_qkv_proj_pipeline.state),
            Some(&self.attention.q4k_q6k_qkv_proj_pipeline),
            Some(&self.attention.q4kf_proj_pipeline.state),
            Some(&self.attention.bias_add_pipeline),
            Some(&self.attention.rope_at_pos_pipeline),
            Some(&self.norms.qk_norm_pipeline),
            Some(&self.norms.scale_vector_pipeline),
            Some(&self.ffn.q4k_geglu_silu_down_pipeline),
            Some(&self.ffn.q4k_geglu_gelu_tanh_down_pipeline),
            Some(&self.ffn.q6k_geglu_silu_down_pipeline),
            Some(&self.ffn.q6k_geglu_gelu_tanh_down_pipeline),
            Some(kv),
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            use_qk_norm,
            softcap,
            None,                // no MoE callback
            Some(&intervention), // head replacement
        ))
    }
}
