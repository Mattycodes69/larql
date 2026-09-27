use super::{apply_norm_for_ffn, apply_post_ffn_norm, ResidualCaptureSink};
use larql_models::test_fixtures::{make_gemma3_test_weights, make_test_weights};

// ── ResidualCaptureSink routed capture ──────────────────────────

#[test]
fn sink_push_routing_step_is_noop_without_capture_routing() {
    // The default sink must leave the routed planes empty — the
    // walk-ffn-only capture path stays byte-identical.
    let w = make_gemma3_test_weights();
    let h_capture: Vec<Vec<f32>> = (0..w.num_layers)
        .map(|l| (0..w.hidden_size).map(|i| (l + i) as f32 * 0.1).collect())
        .collect();
    let mut sink = ResidualCaptureSink::default();
    sink.push_routing_step(&w, &h_capture);
    assert!(sink.raw_steps.is_empty());
    assert!(sink.normed_steps.is_empty());
    assert!(sink.routing_steps.is_empty());
}

#[test]
fn sink_with_routing_captures_raw_normed_and_routing_on_moe_fixture() {
    use crate::test_utils::{
        make_test_gemma4_moe_weights, GEMMA4_MOE_NUM_EXPERTS, GEMMA4_MOE_TOP_K,
    };
    let w = make_test_gemma4_moe_weights();
    let arch = &*w.arch;
    let h_capture: Vec<Vec<f32>> = (0..w.num_layers)
        .map(|l| {
            (0..w.hidden_size)
                .map(|i| ((l * 31 + i * 7) % 13) as f32 * 0.2 - 1.0)
                .collect()
        })
        .collect();

    let mut sink = ResidualCaptureSink::with_routing();
    sink.push_routing_step(&w, &h_capture);
    assert_eq!(sink.raw_steps.len(), 1);
    assert_eq!(sink.normed_steps.len(), 1);
    assert_eq!(sink.routing_steps.len(), 1);

    // Raw plane is the untouched post-attention residual.
    assert_eq!(sink.raw_steps[0], h_capture);

    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    for (layer, h_row) in h_capture.iter().enumerate() {
        let normed = &sink.normed_steps[0][layer];
        let pairs = &sink.routing_steps[0][layer];
        assert_eq!(normed.len(), w.hidden_size);
        match crate::vindex::build_moe_router_weights(&w, arch, layer) {
            Some(router) => {
                // Normed row must be exactly route()'s h_norm — same
                // helper, no second norm implementation.
                let (h_norm, indices, weights) = router.route(h_row, norm_offset, eps);
                assert_eq!(normed, &h_norm, "layer {layer} normed row");
                assert_eq!(pairs.len(), GEMMA4_MOE_TOP_K, "layer {layer} pair count");
                for (i, &(e, wt)) in pairs.iter().enumerate() {
                    assert_eq!(e as usize, indices[i]);
                    assert_eq!(wt, weights[i]);
                    assert!((e as usize) < GEMMA4_MOE_NUM_EXPERTS);
                    assert!(wt.is_finite());
                }
            }
            None => {
                assert_eq!(normed, h_row, "non-MoE layer passes raw through");
                assert!(pairs.is_empty(), "non-MoE layer records no pairs");
            }
        }
    }
}

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tol)
}

// ── apply_post_ffn_norm ─────────────────────────────────────────

#[test]
fn apply_post_ffn_norm_passthrough_when_arch_has_no_post_norms() {
    // TinyModel arch reports `has_post_norms() == false`. The
    // helper must short-circuit and return the input verbatim:
    // remote FFN servers running against pre-Gemma-style archs
    // see the FFN output flow straight into the residual add.
    let w = make_test_weights();
    let hidden = w.hidden_size;
    let ffn_out: Vec<f32> = (0..hidden).map(|i| i as f32 - 7.0).collect();
    let out = apply_post_ffn_norm(&w, &ffn_out, 0);
    assert_eq!(out, ffn_out);
}

#[test]
fn apply_post_ffn_norm_applies_named_norm_when_key_present() {
    // Gemma 3 arch supplies `post_feedforward_layernorm_key` for
    // every layer; the helper must route through `apply_norm`.
    // With identity-weight norms the output is RMS-normalised
    // (variance scaled) but distinct from a non-unit-RMS input —
    // "norm never ran" is the exact bug this helper closes for
    // remote FFN paths on post-norm archs.
    let w = make_gemma3_test_weights();
    let hidden = w.hidden_size;
    let ffn_out: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let out = apply_post_ffn_norm(&w, &ffn_out, 0);
    assert_eq!(out.len(), hidden);
    assert!(
        !approx_eq(&out, &ffn_out, 1e-6),
        "post-FFN norm must transform the input on a post-norm arch"
    );
    let rms = (out.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 1e-3,
        "identity-weight post-FFN norm should rescale to unit RMS (got {rms})"
    );
}

#[test]
fn apply_post_ffn_norm_layer_keys_distinct_per_layer() {
    // Per-layer key lookup: both layers must succeed and produce
    // hidden-sized output. Regression surface is a hardcoded
    // layer index in the helper.
    let w = make_gemma3_test_weights();
    let ffn_out: Vec<f32> = vec![0.25; w.hidden_size];
    let out0 = apply_post_ffn_norm(&w, &ffn_out, 0);
    let out1 = apply_post_ffn_norm(&w, &ffn_out, 1);
    assert_eq!(out0.len(), w.hidden_size);
    assert_eq!(out1.len(), w.hidden_size);
}

// ── apply_norm_for_ffn (parallel helper, previously untested) ───

#[test]
fn apply_norm_for_ffn_uses_pre_feedforward_key_on_post_norm_arch() {
    // Post-norm archs (Gemma 3) route the pre-FFN norm through
    // `pre_feedforward_layernorm`, not `post_attention_layernorm`.
    // Local forward and remote-FFN caller must agree.
    let w = make_gemma3_test_weights();
    let h_post_attn: Vec<f32> = (0..w.hidden_size).map(|i| i as f32 * 0.3).collect();
    let out = apply_norm_for_ffn(&w, &h_post_attn, 0);
    assert_eq!(out.len(), w.hidden_size);
    let rms = (out.iter().map(|x| x * x).sum::<f32>() / w.hidden_size as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-3, "expected unit-RMS, got {rms}");
}

#[test]
fn apply_norm_for_ffn_uses_post_attention_key_on_pre_norm_arch() {
    // Pre-norm archs (TinyModel, Llama/Mistral layout) route the
    // pre-FFN norm through `post_attention_layernorm`. The
    // identity-weight fixture means we mainly assert shape and
    // that the call doesn't panic on the key the function selected.
    let w = make_test_weights();
    let h_post_attn: Vec<f32> = (0..w.hidden_size).map(|i| (i as f32 - 5.0) * 0.4).collect();
    let out = apply_norm_for_ffn(&w, &h_post_attn, 0);
    assert_eq!(out.len(), w.hidden_size);
}

// ── prenorm_layers / postnorm_layers composition ────────────────

use super::{postnorm_layers, prenorm_layers};

#[test]
fn prenorm_layers_applies_pre_norm_per_layer() {
    // Multi-layer pre-norm pass over a synthetic h_capture.
    // Output length must match input, and each layer's vector
    // must individually be unit-RMS (identity-weight RMS norm
    // in the Gemma 3 fixture).
    let w = make_gemma3_test_weights();
    let hidden = w.hidden_size;
    let h_capture: Vec<Vec<f32>> = (0..w.num_layers)
        .map(|l| (0..hidden).map(|i| (l + i) as f32 * 0.25 + 1.0).collect())
        .collect();
    let normed = prenorm_layers(&w, &h_capture);
    assert_eq!(normed.len(), w.num_layers);
    for (l, v) in normed.iter().enumerate() {
        assert_eq!(v.len(), hidden);
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-3,
            "layer {l} prenorm should rescale to unit RMS, got {rms}"
        );
    }
}

#[test]
fn postnorm_layers_applies_post_norm_per_layer_on_post_norm_arch() {
    // Multi-layer post-norm pass: raw FFN outputs come back from
    // the remote, helper applies the per-layer post-norm so the
    // residual add sees the same input the local forward path
    // would produce.
    let w = make_gemma3_test_weights();
    let hidden = w.hidden_size;
    let raw: Vec<Vec<f32>> = (0..w.num_layers)
        .map(|l| (0..hidden).map(|i| (l + i) as f32 * 0.5 + 2.0).collect())
        .collect();
    let post = postnorm_layers(&w, raw);
    assert_eq!(post.len(), w.num_layers);
    for (l, v) in post.iter().enumerate() {
        assert_eq!(v.len(), hidden);
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-3,
            "layer {l} postnorm should rescale to unit RMS, got {rms}"
        );
    }
}

#[test]
fn postnorm_layers_is_identity_on_pre_norm_arch() {
    // TinyModel reports `has_post_norms() == false`. The helper
    // must short-circuit each layer to passthrough so the remote
    // FFN's raw output flows straight into the residual add.
    // Regression surface: if the per-layer short-circuit ever
    // gets gated wrong, a pre-norm-arch remote-FFN run silently
    // gets identity-RMS normalised and diverges from local.
    let w = make_test_weights();
    let raw: Vec<Vec<f32>> = (0..w.num_layers)
        .map(|l| (0..w.hidden_size).map(|i| (l + i) as f32).collect())
        .collect();
    let raw_clone = raw.clone();
    let post = postnorm_layers(&w, raw);
    assert_eq!(post, raw_clone);
}

#[test]
fn ffn_norm_round_trip_dispatches_pre_normed_input_and_post_norms_output() {
    // End-to-end shape of the decode-path composition: the
    // dispatch closure (stand-in for the remote FFN) sees a
    // pre-normed buffer, and the caller post-norms its return
    // before the residual add. This is the contract that
    // chrishayuk/larql#114 violated when the f32 fallback
    // branches sent raw residuals.
    let w = make_gemma3_test_weights();
    let hidden = w.hidden_size;
    let h_post_attn: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.3 + 1.5).collect();

    // Mirror exactly what the decode moe_fn does, with an
    // identity-mapping dispatch closure. `seen_input` captures
    // what the "remote" was handed.
    let h_normed = apply_norm_for_ffn(&w, &h_post_attn, 0);
    let seen_input = h_normed.clone();
    let raw_out = h_normed.clone(); // identity dispatch
    let result = apply_post_ffn_norm(&w, &raw_out, 0);

    // Pre-normed input: dispatch sees unit-RMS, not the raw h.
    let in_rms = (seen_input.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
    assert!(
        (in_rms - 1.0).abs() < 1e-3,
        "dispatch must see pre-normed input (unit-RMS), got {in_rms}"
    );

    // Post-normed output: result is unit-RMS too (identity
    // dispatch fed unit-RMS through, post-norm on Gemma 3 keeps
    // it unit-RMS).
    let out_rms = (result.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
    assert!(
        (out_rms - 1.0).abs() < 1e-3,
        "post-normed output should be unit-RMS, got {out_rms}"
    );

    // And the result differs from the raw h_post_attn — without
    // the round-trip norms, an identity-dispatch would return
    // exactly h_post_attn.
    assert_ne!(
        &result, &h_post_attn,
        "decode-path round-trip must transform the residual, not echo it"
    );
}
