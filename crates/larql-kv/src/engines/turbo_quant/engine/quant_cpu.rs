//! Q4K decode on the CPU for the TurboQuant engine.

use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
use larql_compute::ComputeBackend;
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};
use larql_vindex::VectorIndex;
use ndarray::Array2;

#[allow(unused_imports)]
use super::*;

impl TurboQuantEngine {
    pub(super) fn prefill_quant_cpu(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let be = Some(backend);
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
        self.layers.clear();

        // Hoist WalkFfn — was rebuilt 34× per prefill.
        let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);

        for layer in 0..num_layers {
            let (h_post_attn, k, v) = run_attention_with_kv_backend(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &h,
                layer,
                be,
                None,
            )?;
            self.layers
                .push(CompressedLayer::compress(&(k, v), &self.tq));

            // Native-quantised FFN; falls back to WalkFfn → dense f32. Both
            // branches return the bare post-FFN hidden —
            // `ffn_decode_step_native` is also `moe_ffn_block_cpu`'s pre-PLE
            // dense slab — so the PLE + layer_scalar tail applies to either.
            let h_post_ffn = larql_inference::vindex::ffn_decode_step_native(
                weights,
                index,
                backend,
                &h_post_attn,
                layer,
            )
            .unwrap_or_else(|| {
                let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h
            });
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
        }

        self.abs_position = token_ids.len();
        Some(last_row(&h))
    }

    pub(super) fn decode_step_quant_cpu(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        use std::time::Instant;
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let timing = self.profiling;
        let t_step = if timing { Some(Instant::now()) } else { None };

        let t_embed = if timing { Some(Instant::now()) } else { None };
        let mut h = embed_tokens_pub(weights, &[token_id]);
        let embed_us = t_embed
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or(0.0);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

        // Hoist WalkFfn — was rebuilt 34× per decode step.
        let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);
        // Codec scratch reused across layers.
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();

        // Per-stage accumulators. For turbo_quant we reuse the existing
        // EngineProfiler slots:
        //   `recompute_hot`  ← codec **decode** (decompress prior K/V)
        //   `recompute_cold` ← codec **encode** (re-encode updated K/V)
        // Semantically these are the per-step codec work that the
        // engine's contract requires; print labels them "recompute_kv
        // (hot/cold)" but for this engine the meaning is decode/encode.
        let mut codec_decode_us = 0.0f64;
        let mut codec_encode_us = 0.0f64;
        let mut attention_us = 0.0f64;
        let mut ffn_us = 0.0f64;

        self.require_prefilled(num_layers).ok()?;
        for layer in 0..num_layers {
            let t_dec = if timing { Some(Instant::now()) } else { None };
            let prior_kv = self.layers[layer].decompress(&self.tq);
            if let Some(t) = t_dec {
                codec_decode_us += t.elapsed().as_secs_f64() * 1e6;
            }

            let t_attn = if timing { Some(Instant::now()) } else { None };
            let (h_post_attn, updated_kv) = larql_inference::vindex::attention_decode_step_native(
                weights,
                index,
                backend,
                &h,
                layer,
                Some(&prior_kv),
                abs_position,
            )
            .or_else(|| {
                run_attention_block_decode_step_backend(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h,
                    layer,
                    Some(&prior_kv),
                    abs_position,
                    Some(backend),
                )
            })?;
            if let Some(t) = t_attn {
                attention_us += t.elapsed().as_secs_f64() * 1e6;
            }

            let t_enc = if timing { Some(Instant::now()) } else { None };
            // Append-only codec path (mirrors `dispatch.rs`'s 2026-05-19
            // fix). The attention call returns the full updated K/V
            // (prior + new); only the LAST row is new, the rest already
            // live in `self.layers[layer].compressed_{k,v}`. Encode just
            // the new row head-by-head and push onto the existing
            // compressed buffer. Per-step compress drops from O(N) to
            // O(head_dim · heads_per_row).
            let layer_slot = &mut self.layers[layer];
            let new_rows = updated_kv.0.shape()[0];
            debug_assert_eq!(new_rows, layer_slot.num_vecs + 1, "decode adds one row");
            let k_last = updated_kv.0.row(new_rows - 1).to_owned();
            let v_last = updated_kv.1.row(new_rows - 1).to_owned();
            layer_slot.append_row(
                k_last.as_slice().expect("k row contig"),
                v_last.as_slice().expect("v row contig"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );
            if let Some(t) = t_enc {
                codec_encode_us += t.elapsed().as_secs_f64() * 1e6;
            }

            let t_ffn = if timing { Some(Instant::now()) } else { None };
            // Both branches return the bare post-FFN hidden —
            // `ffn_decode_step_native` is also `moe_ffn_block_cpu`'s pre-PLE
            // dense slab — so the PLE + layer_scalar tail applies to either.
            let h_post_ffn = larql_inference::vindex::ffn_decode_step_native(
                weights,
                index,
                backend,
                &h_post_attn,
                layer,
            )
            .unwrap_or_else(|| {
                let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h
            });
            let h_out = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
            if let Some(t) = t_ffn {
                ffn_us += t.elapsed().as_secs_f64() * 1e6;
            }
            h = h_out;
        }

        if let Some(t_step) = t_step {
            let p = &mut self.profile;
            p.embed.total_us += embed_us;
            p.embed.count += 1;
            p.recompute_hot.total_us += codec_decode_us;
            p.recompute_hot.count += 1;
            p.attention.total_us += attention_us;
            p.attention.count += 1;
            p.recompute_cold.total_us += codec_encode_us;
            p.recompute_cold.count += 1;
            p.ffn.total_us += ffn_us;
            p.ffn.count += 1;
            p.decode_total.total_us += t_step.elapsed().as_secs_f64() * 1e6;
            p.decode_total.count += 1;
        }

        self.abs_position += 1;
        Some(last_row(&h))
    }
}
