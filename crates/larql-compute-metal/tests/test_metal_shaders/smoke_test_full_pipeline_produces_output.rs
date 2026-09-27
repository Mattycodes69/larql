//! Smoke test: full pipeline produces output

use super::*;

#[test]
fn full_pipeline_seq1_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let inter = 512usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Create synthetic Q4_0 weights for one layer
    let gate_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let up_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let down_data = quantize_q4_0(&vec![0.01f32; hidden * inter]);
    let wq_data = quantize_q4_0(&vec![0.01f32; q_dim * hidden]);
    let wk_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wv_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wo_data = quantize_q4_0(&vec![0.01f32; hidden * q_dim]);

    let norm = vec![1.0f32; hidden];
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // Q4_0 packs its f16 scale inside each 18-byte block, so no external
    // scale array exists. This fixture used to fabricate one out of
    // *input*-quantization scales — the wrong object entirely — which
    // `QuantWeight::new` now refuses.
    let layer = larql_compute::FullPipelineLayer {
        attn_sinks: None,
        attn_q_bias: None,
        attn_k_bias: None,
        attn_v_bias: None,
        attn_o_bias: None,
        attn_softcap: 0.0,
        wq: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &wq_data,
            larql_compute::QuantAux::None,
        ),
        wk: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &wk_data,
            larql_compute::QuantAux::None,
        ),
        wv: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &wv_data,
            larql_compute::QuantAux::None,
        ),
        wo: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &wo_data,
            larql_compute::QuantAux::None,
        ),
        gate: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &gate_data,
            larql_compute::QuantAux::None,
        ),
        up: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &up_data,
            larql_compute::QuantAux::None,
        ),
        down: larql_compute::QuantWeight::new(
            larql_compute::QuantFormat::Q4_0,
            &down_data,
            larql_compute::QuantAux::None,
        ),
        input_norm: &norm,
        post_attn_norm: &norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 1.0,
        has_post_norms: false,
        activation: larql_compute::Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: larql_compute::NormType::RmsNorm,
        ffn_type: larql_compute::FfnType::Gated,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base: 10000.0,
        rotary_dim: 0,
        rope_freq: larql_compute::attention::rope::RopeFreqPlan::unscaled(
            head_dim,
            0_usize,
            10000.0_f64,
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
    };

    let _ = (q_dim, kv_dim, num_q_heads, num_kv_heads, head_dim);
    let result = metal.full_pipeline_q4(
        &[layer],
        &x,
        hidden,
        inter,
        1,     // seq_len
        false, // use_qk_norm
        0.0,   // softcap
    );

    assert!(result.is_some(), "full_pipeline_q4 should return Some");
    let output = result.unwrap();
    assert_eq!(output.len(), hidden);
    // Finiteness is checked separately from magnitude. `v.abs() > 1e-6` is
    // false for NaN, so a NaN-filled result used to fail here with the message
    // "output should be nonzero" — which sent the investigation hunting for a
    // zeroed buffer when in fact every element was NaN. See the construction
    // lock in `backend::MetalBackend::with_options`.
    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "Pipeline output has {non_finite}/{} non-finite values; head={:?}",
        output.len(),
        &output[..output.len().min(8)]
    );
    assert!(
        output.iter().any(|&v| v.abs() > 1e-6),
        "Pipeline output is finite but all-zero; head={:?}",
        &output[..output.len().min(8)]
    );
}
