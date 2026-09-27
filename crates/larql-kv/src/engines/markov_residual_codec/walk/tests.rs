use super::*;
use larql_compute::CpuBackend;
use larql_inference::test_utils::{make_test_vindex, make_test_weights};

#[test]
fn prefill_walk_returns_finite_hidden() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let result = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2],
        None,
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
    assert!(result.hidden.iter().all(|v| v.is_finite()));
}

#[test]
fn prefill_walk_with_overflow_populates_cold_tier() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let result = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    assert!(result.store.cold_encoded.is_some());
    assert!(result.store.cold_kv.is_some());
}

#[test]
fn decode_walk_extends_position_and_returns_finite() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1],
        None,
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    assert_eq!(prefill.store.next_position, 2);
    let (h, rs2) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        2,
        prefill.store,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert_eq!(rs2.next_position, 3);
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
fn decode_walk_with_cold_kv_path() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    assert!(prefill.store.cold_kv.is_some());
    let (h, _) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        4,
        prefill.store,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn decode_walk_with_cold_encoded_after_eviction() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    let (_, rs2) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        4,
        prefill.store,
        &CpuBackend,
        None,
    )
    .unwrap();
    // First decode clears cold_kv; second decode exercises the
    // cold_encoded path.
    let (h, _) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        5,
        rs2,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn roundtrip_empty_block() {
    let empty: Array2<f32> = Array2::zeros((0, 8));
    let out = roundtrip(&empty, ColdResidualCodec::Bf16);
    assert_eq!(out.shape(), &[0, 8]);
}

#[test]
fn decode_walk_with_profiler_accumulates_timing_stages() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    let mut prof = EngineProfiler::default();
    let (h, _) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        4,
        prefill.store,
        &CpuBackend,
        Some(&mut prof),
    )
    .unwrap();
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(prof.decode_total.count, 1);
    assert_eq!(prof.attention.count, 1);
    assert_eq!(prof.ffn.count, 1);
    assert_eq!(prof.embed.count, 1);
    // Cold_kv was set by the prefill overflow, so neither recompute branch
    // necessarily fired — but the per-stage counters must always be bumped.
    assert_eq!(prof.recompute_cold.count, 1);
    assert_eq!(prof.recompute_hot.count, 1);
}

#[test]
fn decode_walk_creates_cold_encoded_on_first_overflow() {
    // Prefill *without* overflow (window > prompt_len). After two
    // decode steps the window cap is exceeded for the first time,
    // which exercises the `None`-arm of the `updated_rs.cold_encoded`
    // match (creates fresh `EncodedColdLayer`s).
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1],
        Some(2),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    assert!(prefill.store.cold_encoded.is_none());
    assert!(prefill.store.cold_kv.is_none());

    let (_, rs2) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        2,
        prefill.store,
        &CpuBackend,
        None,
    )
    .unwrap();
    // First overflow hits the None arm and initialises cold_encoded.
    assert!(rs2.cold_encoded.is_some());
    assert_eq!(rs2.cold_encoded.as_ref().unwrap()[0].n_positions, 1);
}

#[test]
fn decode_walk_drops_hot_kv_then_recomputes_from_cold_encoded() {
    // Drive a decode where `hot_kv` is None and only `cold_encoded`
    // is populated. Exercises the `else` arm that decodes the cold
    // payload, concatenates with `h_hot`, and recomputes K/V from
    // the result.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    let mut store = prefill.store;
    // Force the "no caches" path: drop both hot_kv and cold_kv,
    // leaving only the codec-encoded cold tier behind.
    store.hot_kv = None;
    store.cold_kv = None;
    let (h, rs2) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        4,
        store,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
    // hot_kv is re-captured on every decode step.
    assert!(rs2.hot_kv.is_some());
}

// ── Cold-tier retention regressions ─────────────────────────────────

/// Window / prompt sized so the prefill overflows into the encoded
/// cold tier (2 positions) and every decode step overflows again.
const OVERFLOW_WINDOW: usize = 2;
const OVERFLOW_PROMPT: [u32; 4] = [0, 1, 2, 3];
const FIRST_DECODE_TOKEN: u32 = 4;
/// Hidden-state tolerance between the cached-hot-K/V path and the
/// recompute-from-residuals path (same bound as the markov twin's
/// W2 parity test).
const CACHED_VS_RECOMPUTE_TOL: f32 = 1e-4;

fn prefill_overflowed(
    weights: &larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
) -> RsStoreCodec {
    rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(weights),
        index,
        &OVERFLOW_PROMPT,
        Some(OVERFLOW_WINDOW),
        ColdResidualCodec::Bf16,
        &CpuBackend,
    )
    .store
}

fn decode_tok(
    weights: &larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
    tok: u32,
    store: RsStoreCodec,
) -> (Array2<f32>, RsStoreCodec) {
    rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(weights),
        index,
        tok,
        store,
        &CpuBackend,
        None,
    )
    .expect("decode")
}

fn assert_close(a: &Array2<f32>, b: &Array2<f32>, ctx: &str) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < CACHED_VS_RECOMPUTE_TOL,
            "{ctx}: cached={x}, recompute={y}"
        );
    }
}

#[test]
fn decode_walk_reseeds_cold_kv_after_overflow_and_keeps_cold_context() {
    // After the first post-prefill overflow the lossy codec
    // invalidates `cold_kv`. The cached-hot_kv fast path must then
    // re-derive the cold K/V from `cold_encoded` — not attend
    // hot-only and drop the cold context from every later step.
    // Pin by value: steps 2 and 3 on the cached path must match the
    // same steps on a recompute-from-state reference (the branch the
    // dense `decode_step` path uses for the identical state).
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);

    let run = |drop_caches_each_step: bool| -> (Array2<f32>, Array2<f32>, RsStoreCodec) {
        let store = prefill_overflowed(&weights, &index);
        // Step 1 runs the natural cached state in both runs →
        // identical state going into step 2. It overflows once,
        // setting cold_kv = None with cold_encoded non-empty.
        let (_, mut rs2) = decode_tok(&weights, &index, FIRST_DECODE_TOKEN, store);
        assert!(rs2.cold_encoded.is_some());
        assert!(rs2.cold_kv.is_none(), "overflow must invalidate cold_kv");
        if drop_caches_each_step {
            rs2.hot_kv = None;
        }
        let (h2, mut rs3) = decode_tok(&weights, &index, FIRST_DECODE_TOKEN + 1, rs2);
        if drop_caches_each_step {
            rs3.hot_kv = None;
            rs3.cold_kv = None;
        }
        let (h3, rs4) = decode_tok(&weights, &index, FIRST_DECODE_TOKEN + 2, rs3);
        (h2, h3, rs4)
    };

    let (h2_cached, h3_cached, rs_cached) = run(false);
    let (h2_ref, h3_ref, _) = run(true);
    assert_close(
        &h2_cached,
        &h2_ref,
        "step 2 dropped the cold tier from attention",
    );
    assert_close(
        &h3_cached,
        &h3_ref,
        "step 3 dropped the cold tier from attention",
    );
    // Every step here overflows, so the trailing clip re-invalidates
    // cold_kv; what must hold is that the encoded tier kept growing
    // and was never dropped from attention.
    assert_eq!(
        rs_cached.cold_encoded.as_ref().unwrap()[0].n_positions,
        OVERFLOW_PROMPT.len() - OVERFLOW_WINDOW + 3,
        "cold tier must accumulate one evicted row per decode step"
    );
}

#[test]
fn decode_walk_no_cache_step_keeps_hot_kv_capture_aligned() {
    // Cold rows folded into the attention prior while `cold_kv` is
    // None: the post-attention hot_kv capture must slice past them.
    // Caching cold+hot+new as "hot" leaves the next cached step
    // attending the wrong rows (and the next clip keeping the
    // oldest, cold rows while discarding the newest). Pin by value:
    // step 2 cached must match step 2 recomputed. (Historically this
    // state hit the no-cache branch with a from-0 capture; the fixed
    // code re-seeds cold_kv first, and the capture offset counts the
    // cold rows either way.)
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);

    let run = |drop_caches_before_step2: bool| -> Array2<f32> {
        let mut store = prefill_overflowed(&weights, &index);
        store.hot_kv = None;
        store.cold_kv = None;
        // Step 1 runs the same branch in both runs → identical
        // state going into step 2.
        let (_, mut rs2) = decode_tok(&weights, &index, FIRST_DECODE_TOKEN, store);
        if drop_caches_before_step2 {
            rs2.hot_kv = None;
            rs2.cold_kv = None;
        }
        let (h2, _) = decode_tok(&weights, &index, FIRST_DECODE_TOKEN + 1, rs2);
        h2
    };

    let h2_cached = run(false);
    let h2_ref = run(true);
    assert_close(
        &h2_cached,
        &h2_ref,
        "hot_kv captured cold rows as hot after the no-cache step",
    );
}

#[test]
fn decode_walk_with_no_caches_and_empty_cold_encoded() {
    // Same as above but with the cold_encoded payload zeroed out
    // (n_positions == 0). Drives the `_` arm of the `match
    // &rs.cold_encoded` block, which just reuses `h_hot`.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1],
        None,
        ColdResidualCodec::Bf16,
        &CpuBackend,
    );
    let mut store = prefill.store;
    store.hot_kv = None;
    store.cold_kv = None;
    // No cold tier at all → exercises the `_` arm.
    store.cold_encoded = None;
    let (h, _) = rs_decode_step_codec_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        2,
        store,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}
