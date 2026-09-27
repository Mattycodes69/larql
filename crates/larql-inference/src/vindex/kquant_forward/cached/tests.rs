use super::*;
use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, Q4KTestFixtures};
use larql_compute::CpuBackend;

// ── supports_cached_decode / supports_direct_matvec_decode ──────────

#[test]
fn supports_cached_decode_is_true_for_dense_arch() {
    let weights = make_test_q4k_weights();
    assert!(
        supports_cached_decode(&weights),
        "synthetic Gemma 3-style weights are dense, no KV sharing, no hybrid MoE"
    );
}

#[test]
fn supports_direct_matvec_decode_is_true_for_q4k_synthetic_vindex() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    assert!(
        supports_direct_matvec_decode(&weights, &index),
        "synth Q4_K vindex has Q4_K attn + interleaved data, intermediate divisible by 256"
    );
}

// ── matvec_q4k_or_q6k_q8k dispatcher ────────────────────────────────

// ── predict_kquant_prefill / predict_kquant_decode_step ────────────────────

#[test]
fn predict_kquant_prefill_returns_hidden_with_expected_shape() {
    let mut fx = Q4KTestFixtures::build();
    let token_ids = vec![1u32, 2, 3];
    let (h, cache, _timings) = predict_kquant_prefill(&mut fx.weights, &token_ids, &fx.index);
    assert_eq!(
        h.shape()[0],
        token_ids.len(),
        "prefill returns seq_len rows"
    );
    assert_eq!(h.shape()[1], fx.weights.hidden_size);
    assert!(
        h.iter().all(|v| v.is_finite()),
        "hidden state must be finite"
    );
    assert_eq!(cache.len(), fx.weights.num_layers);
    for entry in &cache {
        assert!(
            entry.is_some(),
            "every layer should have K/V populated after prefill"
        );
    }
}

#[test]
fn predict_kquant_decode_step_appends_kv_and_returns_one_row() {
    let mut fx = Q4KTestFixtures::build();
    let token_ids = vec![1u32, 2, 3];
    let (_, mut cache, _) = predict_kquant_prefill(&mut fx.weights, &token_ids, &fx.index);

    let pre_lens: Vec<usize> = cache
        .iter()
        .map(|c| c.as_ref().map(|(k, _)| k.shape()[0]).unwrap_or(0))
        .collect();

    let (h_new, _step_timings) =
        predict_kquant_decode_step(&fx.weights, 4, &fx.index, &mut cache, token_ids.len())
            .expect("decode step must succeed on a populated cache");

    assert_eq!(h_new.shape(), &[1, fx.weights.hidden_size]);
    assert!(h_new.iter().all(|v| v.is_finite()));

    for (layer, pre) in pre_lens.iter().enumerate() {
        let post = cache[layer]
            .as_ref()
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0);
        assert_eq!(post, pre + 1, "layer {layer} K/V should have grown by 1");
    }
}

#[test]
fn predict_kquant_decode_step_rejects_mismatched_cache_length() {
    let fx = Q4KTestFixtures::build();
    // Cache length doesn't match num_layers — function must return None.
    let mut bad_cache: CpuKvCache = vec![None; fx.weights.num_layers + 1];
    let result = predict_kquant_decode_step(&fx.weights, 1, &fx.index, &mut bad_cache, 0);
    assert!(result.is_none());
}

// ── predict_kquant_decode_step_direct (Q4K × Q8K sdot path) ────────────

/// The direct step must TRACK the staged step, not merely stay finite:
/// same prefill cache, same token, same position → high-cosine hidden
/// agreement. (The q4_common f16 subnormal bug passed the finite-only
/// check below while garbling chained generation on real models —
/// see `examples/ave_direct_step_parity.rs`.)
#[test]
fn predict_kquant_decode_step_direct_tracks_staged_step() {
    let token_ids = vec![1u32, 2, 3];

    let mut fx_a = Q4KTestFixtures::build();
    let (_, mut cache_a, _) = predict_kquant_prefill(&mut fx_a.weights, &token_ids, &fx_a.index);
    let (h_staged, _) = predict_kquant_decode_step(&fx_a.weights, 4, &fx_a.index, &mut cache_a, 3)
        .expect("staged step");

    let mut fx_b = Q4KTestFixtures::build();
    let (_, mut cache_b, _) = predict_kquant_prefill(&mut fx_b.weights, &token_ids, &fx_b.index);
    let backend = CpuBackend;
    let h_direct = predict_kquant_decode_step_direct(
        &mut fx_b.weights,
        4,
        &fx_b.index,
        &backend,
        &mut cache_b,
        3,
    )
    .expect("direct step");

    let a = h_staged.row(0);
    let b = h_direct.row(0);
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    assert!(
        cos > 0.999,
        "direct step diverged from staged step: cosine {cos} (norms {na} vs {nb})"
    );
    let ratio = if na > nb { na / nb } else { nb / na };
    assert!(
        ratio < 1.05,
        "direct step norm drifted from staged: {na} vs {nb}"
    );
}

/// Scaled-RoPE regression: on a Gemma-3 arch with linear
/// `rope_scaling` (position divisor 8 on the global layer), the
/// direct step must still track the staged step. Pre-2026-06-12 the
/// direct path roped Q/K with the UNSCALED `apply_rope_partial_at` —
/// no position divisor, no llama3 scaling — so on any rope-scaled
/// config the global layer's K landed at 8× the position the prefill
/// cache used. The non-scaled fixtures can't see that gap; this one
/// exists to.
#[test]
fn predict_kquant_decode_step_direct_tracks_staged_on_rope_scaled_arch() {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights_rope_scaled};

    let mut weights_a = make_test_q4k_weights_rope_scaled();
    // Guard: the fixture must actually parse into a divisor-8 global
    // layer — otherwise this test silently stops testing anything.
    let scaled_layers: Vec<usize> = (0..weights_a.num_layers)
        .filter(|&l| weights_a.arch.rope_position_divisor_for_layer(l) == 8.0)
        .collect();
    assert!(
        !scaled_layers.is_empty(),
        "fixture drift: no layer carries rope position divisor 8 — \
         the rope_scaling config no longer parses as global-only linear"
    );
    let index = make_test_q4k_vindex(&weights_a);
    assert!(
        supports_direct_matvec_decode(&weights_a, &index),
        "rope-scaled fixture must support the direct-matvec path"
    );

    // Prompt long enough that the scaled position (pos/8) and the
    // unscaled position differ by a large rotary angle.
    let token_ids = vec![1u32, 2, 3, 4, 5];
    let next = 6u32;

    let (_, mut cache_a, _) = predict_kquant_prefill(&mut weights_a, &token_ids, &index);
    let (h_staged, _) =
        predict_kquant_decode_step(&weights_a, next, &index, &mut cache_a, token_ids.len())
            .expect("staged step");

    let mut weights_b = make_test_q4k_weights_rope_scaled();
    let (_, mut cache_b, _) = predict_kquant_prefill(&mut weights_b, &token_ids, &index);
    let backend = CpuBackend;
    let h_direct = predict_kquant_decode_step_direct(
        &mut weights_b,
        next,
        &index,
        &backend,
        &mut cache_b,
        token_ids.len(),
    )
    .expect("direct step");

    // Primary assertion: the K row each path APPENDED to the cache.
    // RoPE is relative — if the direct step ropes both new-Q and
    // new-K at the wrong scale, their geometry to each other is
    // preserved and the hidden state barely moves on a bland random
    // fixture. The appended K row is the object the divisor rotates,
    // and it must match the staged row at every layer (most of all
    // the divisor-8 global layers).
    for layer in 0..weights_a.num_layers {
        let (k_a, _) = cache_a[layer].as_ref().expect("staged cache");
        let (k_b, _) = cache_b[layer].as_ref().expect("direct cache");
        let ra = k_a.row(k_a.nrows() - 1);
        let rb = k_b.row(k_b.nrows() - 1);
        let dot: f32 = ra.iter().zip(rb.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (na * nb);
        assert!(
            cos > 0.999,
            "appended K row diverged at layer {layer} (divisor {}): cosine {cos} — \
             check the rope divisor / llama3 scaling in attention_decode_step_native",
            weights_a.arch.rope_position_divisor_for_layer(layer)
        );
    }

    // Secondary: the hidden state still tracks.
    let a = h_staged.row(0);
    let b = h_direct.row(0);
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    assert!(
        cos > 0.999,
        "direct step hidden diverged from staged on the rope-scaled arch \
         (global layers {scaled_layers:?}): cosine {cos}"
    );
}

#[test]
fn predict_kquant_decode_step_direct_returns_finite_hidden() {
    let mut fx = Q4KTestFixtures::build();
    let token_ids = vec![1u32, 2, 3];
    let (_, mut cache, _) = predict_kquant_prefill(&mut fx.weights, &token_ids, &fx.index);

    let backend = CpuBackend;
    let h_new = predict_kquant_decode_step_direct(
        &mut fx.weights,
        4,
        &fx.index,
        &backend,
        &mut cache,
        token_ids.len(),
    )
    .expect("direct decode step must succeed");

    assert_eq!(h_new.shape(), &[1, fx.weights.hidden_size]);
    assert!(h_new.iter().all(|v| v.is_finite()));
}

#[test]
fn predict_kquant_decode_step_direct_rejects_mismatched_cache_length() {
    let mut fx = Q4KTestFixtures::build();
    let mut bad_cache: CpuKvCache = vec![None; fx.weights.num_layers - 1];
    let backend = CpuBackend;
    let result = predict_kquant_decode_step_direct(
        &mut fx.weights,
        1,
        &fx.index,
        &backend,
        &mut bad_cache,
        0,
    );
    assert!(result.is_none());
}

// ── CachedTimings merge ──────────────────────────────────────────────

#[test]
fn cached_timings_add_accumulates_dequant_ms() {
    let mut t = CachedTimings::default();
    assert_eq!(t.dequant_ms, 0.0);
    t.add(CachedTimings { dequant_ms: 1.5 });
    t.add(CachedTimings { dequant_ms: 2.25 });
    assert_eq!(t.dequant_ms, 3.75);
}

// ── fused_prefill / fused_decode_step ────────────────────────────────
//
// The public fused fast path: dispatches to `backend.prefill_kquant` /
// `backend.decode_token`. **Not Metal-specific** — `CpuBackend` returns
// `supports_quant(Q4_K) == true` (it ships a C Q4 kernel) and may implement either
// method. The functions short-circuit when the vindex lacks the
// interleaved FFN bytes the fused pipeline needs (the case for the
// synthetic test vindex below), regardless of which backend is used.
// The earlier name `metal_fused_*` was a misnomer.

#[test]
fn fused_prefill_returns_none_on_synthetic_vindex() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = CpuBackend;
    let result = fused_prefill(&weights, &index, &[0u32, 1], &backend);
    assert!(
        result.is_none(),
        "synthetic vindex without interleaved fused-pipeline bytes must short-circuit"
    );
}

#[test]
fn fused_decode_step_returns_none_on_synthetic_vindex() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = CpuBackend;
    let result = fused_decode_step(&weights, &index, 0, &backend);
    assert!(
        result.is_none(),
        "synthetic vindex without interleaved fused-pipeline bytes must short-circuit"
    );
}
