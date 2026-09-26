//! Walk-path overflow branches (markov_residual/walk.rs)

use super::*;

#[test]
fn prefill_quant_walk_with_window_populates_cold_kv() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
        .expect("prefill_quant with overflow");
    // window=2 + 4 prompt tokens → cold tier populated → walk.rs
    // lines 67-75 fire.
    assert!(engine.window_tokens() <= 2);
    assert!(engine.cold_bytes() > 0);
}

/// W2 parity: the cached-hot_kv decode path must produce the
/// SAME hidden state as the legacy recompute-from-residuals path,
/// bit-for-bit (or within fp rounding). Drives a few decode steps
/// with caching enabled (default since W2) against a manually
/// hot_kv-cleared store that forces the legacy fallback.
#[test]
fn decode_step_quant_w2_cached_matches_recompute_from_residuals() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;

    // Cached path (W2 default): prefill captures K/V, decode reuses.
    let mut cached = MarkovResidualEngine::new(None);
    cached
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill cached");
    let h_cached_1 = cached
        .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
        .expect("decode cached 1");
    let h_cached_2 = cached
        .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
        .expect("decode cached 2");

    // Recompute path: same engine, but force hot_kv = None after
    // prefill so the fallback recompute fires for every step.
    let mut recompute = MarkovResidualEngine::new(None);
    recompute
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill recompute");
    if let Some(s) = recompute.store.as_mut() {
        s.hot_kv = None;
    }
    let h_recompute_1 = recompute
        .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
        .expect("decode recompute 1");
    if let Some(s) = recompute.store.as_mut() {
        s.hot_kv = None;
    }
    let h_recompute_2 = recompute
        .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
        .expect("decode recompute 2");

    // Bit-equivalence: both paths run the same projection matmuls
    // at the same RoPE positions, so output must match within
    // f32 rounding. (Hidden states aren't normalised here; they
    // come straight from the layer stack.)
    for (a, b) in h_cached_1.iter().zip(h_recompute_1.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "step 1 diverged: cached={a}, recompute={b}"
        );
    }
    for (a, b) in h_cached_2.iter().zip(h_recompute_2.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "step 2 diverged: cached={a}, recompute={b}"
        );
    }
}

/// W2 fast path: both cold_kv AND hot_kv cached. Drives the
/// triple-condition branch in `rs_decode_step_walk` that
/// concatenates a cached cold tier with a cached hot tier
/// (memcpy only, no projection). Achieved by prefilling past
/// the window, then doing several decodes so cold_kv stays
/// populated across steps.
#[test]
fn decode_step_quant_w2_cached_hot_and_cold_steady_state() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    // window=2, 4-token prompt → prefill overflows once,
    // populating cold_kv from the evicted hot_kv slice.
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
        .expect("prefill with overflow");
    let store = engine.store.as_ref().unwrap();
    assert!(store.hot_kv.is_some());
    assert!(store.cold_kv.is_some(), "prefill should populate cold_kv");

    // Multiple decodes — each appends a row to hot_kv (W2 fast
    // path with BOTH caches populated). Subsequent overflows
    // merge into cold_kv via the W2 evicted-K/V flow.
    for tok in 4u32..8 {
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, tok, &*backend)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }
    let store = engine.store.as_ref().unwrap();
    assert!(store.hot_kv.is_some());
    assert!(
        store.cold_kv.is_some(),
        "cold_kv stays populated across steps"
    );
    // Cold grew by ~3 rows (one per decode after the prefill cycle).
    let cold_rows = store.cold_kv.as_ref().unwrap()[0].0.shape()[0];
    assert!(
        cold_rows >= 3,
        "cold_kv should grow with successive overflows, got {cold_rows}"
    );
}

/// Drive the fallback path where `hot_kv` was dropped (legacy
/// recompute-from-residuals). Covers the `if let Some(cold_kv) =
/// &rs.cold_kv` branch with hot_kv=None — the pre-W2 behaviour
/// that's still reachable via the via_executor path.
#[test]
fn decode_step_quant_w2_falls_back_when_hot_kv_dropped() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
        .expect("prefill");
    // Drop hot_kv — forces the recompute path that mirrors pre-W2.
    engine.store.as_mut().unwrap().hot_kv = None;
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
        .expect("decode via fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// W2 cache survives window-overflow: when stored is clipped, the
/// evicted hot_kv rows merge into cold_kv (vs the legacy invalidation
/// that cleared cold_kv and forced recompute on the next step).
#[test]
fn decode_step_quant_w2_overflow_merges_into_cold_kv() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;

    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill within window");
    // After prefill: hot_kv populated (2 rows), no cold_kv.
    assert!(engine.store.as_ref().unwrap().hot_kv.is_some());
    assert!(engine.store.as_ref().unwrap().cold_kv.is_none());
    // Decode a token → no overflow yet (still 2 rows after step
    // since window=2, the new row pushes the oldest out).
    let _ = engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode 1");
    // Overflow fired this step: oldest row evicted from hot_kv,
    // merged into cold_kv.
    let store = engine.store.as_ref().unwrap();
    assert!(
        store.cold_kv.is_some(),
        "post-overflow cold_kv should be populated from evicted hot_kv"
    );
    assert!(store.hot_kv.is_some(), "hot_kv stays alive");
}

/// Drive `rs_decode_step_walk`'s `Some(profiler)` branches — the
/// non-profiled path is covered by `decode_step_q4k_cpu_fallback_*`;
/// the profiled-arm branches are only reached when the engine is
/// built with `with_profiling(true)`. Without this test the
/// `if let (Some(prof), Some(t_step)) = ...` accumulation and the
/// per-stage `if timing { ... }` arms stay uncovered.
#[test]
fn decode_step_q4k_walk_with_profiling_populates_summary() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2)).with_profiling(true);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
        .expect("prefill");
    // First decode: cold_kv branch (hot recompute timing arm).
    engine
        .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
        .expect("decode 1");
    // Second decode: cold_residuals branch (cold recompute timing arm).
    engine
        .decode_step_quant(&weights, &ffn, &index, 5, &*backend)
        .expect("decode 2");
    let summary = engine
        .stage_summary()
        .expect("Q4K walk profiler should populate summary");
    assert_eq!(summary.engine, "markov-rs");
    assert!(summary.steps >= 2);
    // The walk path accumulates into `recompute_*` (one of the two
    // branches will be non-zero depending on which fired); attention
    // and ffn always fire.
    assert!(summary.avg_attention_us > 0.0);
    assert!(summary.avg_ffn_us > 0.0);
    assert!(summary.avg_total_decode_us > 0.0);
}

#[test]
fn decode_step_quant_walk_first_overflow_creates_cold_residuals() {
    // walk.rs lines 305-307: `None => updated_rs.cold_residuals =
    // Some(overflow)`. Fires when prefill didn't overflow (cold = None)
    // but the first decode does (window cap exceeded mid-decode).
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    // window=2, prefill=1 token → no overflow on prefill (cold=None).
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32], &*backend)
        .expect("prefill_quant");
    // Decode until hot exceeds window → first-time cold population.
    engine
        .decode_step_quant(&weights, &ffn, &index, 1, &*backend)
        .expect("decode 1");
    engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode 2 — triggers first-overflow None branch");
    // After overflow, cold tier is populated.
    assert!(engine.cold_bytes() > 0);
}

#[test]
fn decode_step_quant_walk_after_overflow_hits_cold_residuals_branch() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = MarkovResidualEngine::new(Some(2));
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2, 3], &*backend)
        .expect("prefill_quant");
    // First decode: exercises walk.rs cold_kv branch (lines 132-161).
    engine
        .decode_step_quant(&weights, &ffn, &index, 4, &*backend)
        .expect("first decode_step_quant");
    // Second decode: cold_kv was cleared by overflow at the first
    // decode (walk.rs line 309), so this hits the cold_residuals
    // recompute branch (lines 162-187).
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 5, &*backend)
        .expect("second decode_step_quant");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}
