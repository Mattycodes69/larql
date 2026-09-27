//! Metal: prefill_kquant with MoE layers

#[cfg(target_os = "macos")]
mod moe_prefill_integration {
    use larql_compute::backend::DecodeBackend;
    use larql_compute::pipeline::*;
    use larql_compute::MoeLayerWeights;
    use larql_compute_metal::MetalBackend;

    /// Minimal Q4_K weight buffer: one super-block (144 bytes) per row,
    /// all scales = 1.0 (f16 0x3C00), all nibbles = 0.
    fn synth_q4k(rows: usize, cols: usize) -> Vec<u8> {
        let blocks = cols.div_ceil(256);
        let mut v = vec![0u8; rows * blocks * 144];
        for b in 0..rows * blocks {
            v[b * 144 + 1] = 0x3C; // d = f16(1.0) hi byte
        }
        v
    }

    fn layer<'a>(
        q4k: &'a [u8],
        norm: &'a [f32],
        moe: Option<MoeLayerWeights<'a>>,
    ) -> FullPipelineLayer<'a> {
        let q4w = || QuantWeight::new(QuantFormat::Q4_K, q4k, larql_compute::QuantAux::None);
        FullPipelineLayer {
            attn_sinks: None,
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_o_bias: None,
            attn_softcap: 0.0,
            wq: q4w(),
            wk: q4w(),
            wv: q4w(),
            wo: q4w(),
            gate: q4w(),
            up: q4w(),
            down: q4w(),
            input_norm: norm,
            post_attn_norm: norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                64_usize,
                0_usize,
                10000.0_f64,
            ),
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }
    }

    fn null_moe(inter: usize) -> MoeLayerWeights<'static> {
        // num_experts=0 → cpu_moe_forward returns zeros immediately.
        // Sufficient to exercise the callback path without real expert weights.
        MoeLayerWeights {
            expert_scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up: Vec::new(),
            experts_down: Vec::new(),
            routing_policy: larql_compute::MoeRoutingPolicy::gemma4_hybrid(),
            weight_layout: larql_compute::MoeWeightLayout::default(),
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: 0,
            top_k: 1,
            intermediate_size: inter,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: larql_compute::MoeGateRule::Gated(Activation::Silu),
            expert_data_format: larql_compute::QuantFormat::BF16,
        }
    }

    /// `prefill_kquant` on a model with MoE layers returns a vec of the right
    /// length and finite values. Exercises the batched-commit path end-to-end.
    #[test]
    fn prefill_q4_with_moe_returns_correct_shape() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 3usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![
            layer(&q4k, &norm, None),
            layer(&q4k, &norm, Some(null_moe(inter))),
            layer(&q4k, &norm, None),
        ];
        let x = vec![0.0f32; seq_len * hidden];
        let out = metal.prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0);
        let out = out.expect("prefill_kquant must return Some on Metal");
        assert_eq!(
            out.len(),
            seq_len * hidden,
            "output length must be seq_len × hidden"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "output must be finite (no NaN/Inf)"
        );
    }

    /// Variant of [`layer`] with V-norm enabled and learned QK-norm
    /// weights populated — drives the `has_v_norm` + `use_qk_norm`
    /// branches of `dispatch_full_pipeline` (lines 320-381 of
    /// `ops/full_pipeline/dispatch.rs`).
    fn layer_with_qk_v_norms<'a>(
        q4k: &'a [u8],
        norm: &'a [f32],
        head_dim_norm: &'a [f32],
    ) -> FullPipelineLayer<'a> {
        let mut base = layer(q4k, norm, None);
        base.has_v_norm = true;
        base.q_norm_weight = Some(head_dim_norm);
        base.k_norm_weight = Some(head_dim_norm);
        base
    }

    /// `prefill_kquant` with every layer carrying V-norm + learned QK-norm
    /// weights — exercises the prerope QK-norm + parameter-free V-norm
    /// dispatch branches in `ops/full_pipeline/dispatch.rs`.
    #[test]
    fn prefill_q4_with_qk_norm_and_v_norm_branches() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 2usize;
        let head_dim = 64usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let head_norm = vec![1.0f32; head_dim];
        let layers: Vec<_> = (0..3)
            .map(|_| layer_with_qk_v_norms(&q4k, &norm, &head_norm))
            .collect();
        let x = vec![0.01f32; seq_len * hidden];
        // `use_qk_norm = true` drives the `applied_prerope_qk_norm`
        // dispatch branch at `dispatch.rs:353`.
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, true, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Q4_KF QKV format — drives the fused Q4_KF QKV path
    /// (`stages.rs` lines 80-87) and the matching shader dispatch.
    /// Q4_KF is the llama.cpp-port pre-baked-scales format.
    #[test]
    fn prefill_q4_with_q4kf_qkv_format() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        // Build a layer where Q/K/V are Q4_KF but gate/up/down stay Q4_K.
        let q4kf = QuantWeight::new(QuantFormat::Q4_KF, &q4k, larql_compute::QuantAux::None);
        let q4w = || {
            QuantWeight::new(
                QuantFormat::Q4_K,
                q4k.as_slice(),
                larql_compute::QuantAux::None,
            )
        };
        let layers = vec![FullPipelineLayer {
            attn_sinks: None,
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_o_bias: None,
            attn_softcap: 0.0,
            wq: q4kf,
            wk: q4kf,
            wv: q4kf,
            wo: q4w(),
            gate: q4w(),
            up: q4w(),
            down: q4w(),
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                64_usize,
                0_usize,
                10000.0_f64,
            ),
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
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
        }];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Mixed QKV formats (Q4_K Q+K, Q6_K V — the Gemma 4 31B convention)
    /// drives the `all_same_format == false` fallback at
    /// `stages.rs` line 94 and the per-projection encode path
    /// (lines 142-180).
    #[test]
    fn prefill_q4_with_mixed_qkv_formats() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let q4_view = |fmt| QuantWeight::new(fmt, q4k.as_slice(), larql_compute::QuantAux::None);
        let layers = vec![FullPipelineLayer {
            attn_sinks: None,
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_o_bias: None,
            attn_softcap: 0.0,
            wq: q4_view(QuantFormat::Q4_K),
            wk: q4_view(QuantFormat::Q4_K),
            wv: q4_view(QuantFormat::Q6_K),
            wo: q4_view(QuantFormat::Q4_K),
            gate: q4_view(QuantFormat::Q4_K),
            up: q4_view(QuantFormat::Q4_K),
            down: q4_view(QuantFormat::Q4_K),
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                64_usize,
                0_usize,
                10000.0_f64,
            ),
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
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
        }];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("mixed-format prefill returns Some");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Q8_0 QKV format drives the fused-Q8-QKV branch in
    /// `ops/full_pipeline/stages.rs` lines 204-227 + the `q8_qkv_proj`
    /// shader dispatch.  Production Q8 attention path.
    #[test]
    fn prefill_q4_with_q8_0_qkv_drives_fused_q8_qkv_path() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let num_q_heads = 4usize;
        let num_kv_heads = 4usize;
        let head_dim = 64usize;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Q8_0 weights + per-row scales.
        let wq_q8: Vec<u8> = vec![1u8; q_dim * hidden];
        let wq_scales: Vec<f32> = vec![0.01f32; q_dim];
        let wk_q8: Vec<u8> = vec![1u8; kv_dim * hidden];
        let wk_scales: Vec<f32> = vec![0.01f32; kv_dim];
        let wv_q8: Vec<u8> = vec![1u8; kv_dim * hidden];
        let wv_scales: Vec<f32> = vec![0.01f32; kv_dim];

        let q4w =
            |fmt: QuantFormat| QuantWeight::new(fmt, q4k.as_slice(), larql_compute::QuantAux::None);
        let layers = vec![FullPipelineLayer {
            attn_sinks: None,
            attn_q_bias: None,
            attn_k_bias: None,
            attn_v_bias: None,
            attn_o_bias: None,
            attn_softcap: 0.0,
            wq: QuantWeight::new(
                QuantFormat::Q8_0,
                &wq_q8,
                larql_compute::QuantAux::ExternalScales(&wq_scales),
            ),
            wk: QuantWeight::new(
                QuantFormat::Q8_0,
                &wk_q8,
                larql_compute::QuantAux::ExternalScales(&wk_scales),
            ),
            wv: QuantWeight::new(
                QuantFormat::Q8_0,
                &wv_q8,
                larql_compute::QuantAux::ExternalScales(&wv_scales),
            ),
            wo: q4w(QuantFormat::Q4_K),
            gate: q4w(QuantFormat::Q4_K),
            up: q4w(QuantFormat::Q4_K),
            down: q4w(QuantFormat::Q4_K),
            input_norm: &norm,
            post_attn_norm: &norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: 10000.0,
            rotary_dim: 0,
            rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
                64_usize,
                0_usize,
                10000.0_f64,
            ),
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
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
        }];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `LARQL_METAL_DUMP_LAYERS=<dir>` drives the dump helpers in
    /// `ops/full_pipeline/dump.rs` (`dump_h_embed`, `dump_layer0_q_after_stage`,
    /// `dump_layer_snapshots`).
    #[test]
    fn prefill_q4_with_metal_dump_layers_env_drives_dump_helpers() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let tmp = std::env::temp_dir().join("larql-cm-metal-dump-test");
        let _ = std::fs::create_dir_all(&tmp);
        let path = tmp.to_str().unwrap().to_string();
        let path_static: &'static str = Box::leak(path.into_boxed_str());
        let saved = std::env::var_os("LARQL_METAL_DUMP_LAYERS");
        // SAFETY: env vars are process-global; the make-target run is
        // single-threaded for this test's scope.  We restore at end.
        unsafe {
            std::env::set_var("LARQL_METAL_DUMP_LAYERS", path_static);
        }

        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, None)];
        let x = vec![0.01f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);

        unsafe {
            match saved {
                Some(v) => std::env::set_var("LARQL_METAL_DUMP_LAYERS", v),
                None => std::env::remove_var("LARQL_METAL_DUMP_LAYERS"),
            }
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `prefill_kquant_with_head_replacement` exercises the
    /// `PipelineIntervention` hooks in `dispatch_full_pipeline`:
    /// `capture + zero target head` at hook A (dispatch.rs:455-495) and
    /// `replacement_delta` add at hook B (dispatch.rs:541-560).
    #[test]
    fn prefill_q4_with_head_replacement_drives_intervention_hooks() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 2usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, None), layer(&q4k, &norm, None)];
        let x = vec![0.01f32; seq_len * hidden];

        let replacement_delta = vec![0.1f32; seq_len * hidden];
        // Target layer 1, head 0 — exercises both hook A (capture +
        // zero) and hook B (delta add).
        let out = metal
            .prefill_kquant_with_head_replacement(
                &layers,
                &x,
                hidden,
                inter,
                seq_len,
                false,
                0.0,
                /* target_layer */ 1,
                /* target_head */ 0,
                &replacement_delta,
            )
            .expect("prefill_kquant_with_head_replacement returns Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// MoE + head replacement falls back to plain `prefill_kquant` since
    /// the intervention path doesn't support MoE layers (dispatch.rs
    /// `has_moe` early-out at line 473-477).
    #[test]
    fn prefill_q4_with_head_replacement_falls_back_when_moe_present() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 1usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, Some(null_moe(inter)))];
        let x = vec![0.0f32; seq_len * hidden];
        let delta = vec![0.0f32; seq_len * hidden];
        let out = metal
            .prefill_kquant_with_head_replacement(
                &layers, &x, hidden, inter, seq_len, false, 0.0, 0, 0, &delta,
            )
            .expect("MoE fallback still returns Some");
        assert_eq!(out.len(), seq_len * hidden);
    }

    /// `prefill_kquant` on an all-MoE model (every layer has MoE) uses the
    /// per-layer commit path. Result shape and finiteness are the minimum bar;
    /// the benchmark verifies correctness vs. the baseline.
    #[test]
    fn prefill_q4_all_moe_layers_returns_correct_shape() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 4usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers: Vec<_> = (0..4)
            .map(|_| layer(&q4k, &norm, Some(null_moe(inter))))
            .collect();
        let x = vec![0.0f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `prefill_kquant` without MoE (original path) is unaffected by the new
    /// callback infrastructure — same shape and finiteness contract.
    #[test]
    fn prefill_q4_no_moe_unaffected() {
        let metal = MetalBackend::new().expect(
            "Metal backend must build: the shader library failed to compile or no device exists",
        );
        let hidden = 256usize;
        let inter = 256usize;
        let seq_len = 2usize;
        let q4k = synth_q4k(hidden.max(inter), hidden);
        let norm = vec![1.0f32; hidden];
        let layers = vec![layer(&q4k, &norm, None), layer(&q4k, &norm, None)];
        let x = vec![0.0f32; seq_len * hidden];
        let out = metal
            .prefill_kquant(&layers, &x, hidden, inter, seq_len, false, 0.0)
            .expect("prefill_kquant must return Some on Metal");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
