use super::*;
use larql_compute::CpuBackend;
use larql_inference::test_utils::{make_test_vindex, make_test_weights};

#[test]
fn prefill_walk_returns_finite_hidden_and_full_window_store() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let result = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2],
        None,
        &CpuBackend,
    );
    assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
    assert!(result.hidden.iter().all(|v| v.is_finite()));
    assert!(result.store.cold_residuals.is_none());
    assert!(result.store.cold_kv.is_none());
    assert!(result.store.hot_kv.is_some());
    assert_eq!(result.window_tokens, 3);
    assert!(result.memory_bytes > 0);
}

#[test]
fn prefill_walk_with_overflow_populates_cold_tier_from_evicted_hot_kv() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let result = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        &CpuBackend,
    );
    assert!(result.store.cold_residuals.is_some());
    assert!(result.store.cold_kv.is_some());
    // Window-clipped, but cold tier captured the two evicted rows.
    assert_eq!(result.window_tokens, 2);
    // 2026-05-19 audit fix: cold_residuals[l].shape()[0] is now the
    // doubling capacity, not the logical row count. Use `cold_len`.
    assert_eq!(result.store.cold_len, 2);
}

#[test]
fn decode_walk_extends_position_and_returns_finite() {
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1],
        None,
        &CpuBackend,
    );
    assert_eq!(prefill.store.next_position, 2);
    let (h, rs2) = rs_decode_step_walk(
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
fn decode_walk_with_profiler_accumulates_all_stage_timings() {
    // Window=2 with 4-token prompt → cold_kv populated. The decode
    // step exercises the "hot_kv + cold_kv" fast-path concat, plus
    // the timing accumulators on every per-stage block.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        &CpuBackend,
    );
    let mut prof = EngineProfiler::default();
    let (h, _) = rs_decode_step_walk(
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
    assert_eq!(prof.recompute_cold.count, 1);
    assert_eq!(prof.recompute_hot.count, 1);
}

#[test]
fn decode_walk_recomputes_hot_when_hot_kv_dropped_with_cold_kv_present() {
    // Force the "cached cold_kv only" middle path: drop the hot_kv
    // cache but keep cold_kv. Decode must recompute the hot K/V
    // from h_hot and concat with the cached cold K/V.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        &CpuBackend,
    );
    let mut store = prefill.store;
    store.hot_kv = None;
    let mut prof = EngineProfiler::default();
    let (h, rs2) = rs_decode_step_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        4,
        store,
        &CpuBackend,
        Some(&mut prof),
    )
    .unwrap();
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    // hot_kv is repopulated on every decode step.
    assert!(rs2.hot_kv.is_some());
}

#[test]
fn decode_walk_recomputes_full_when_no_caches_and_cold_residuals_present() {
    // Drop both hot_kv and cold_kv, leaving only the raw
    // cold_residuals behind. Drives the "neither cached" else arm
    // that concatenates cold residuals with h_hot before
    // recomputing the K/V.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1, 2, 3],
        Some(2),
        &CpuBackend,
    );
    let mut store = prefill.store;
    store.hot_kv = None;
    store.cold_kv = None;
    let (h, _) = rs_decode_step_walk(
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
}

// ── Doubling-capacity regression tests (2026-05-19 store migration) ──
//
// `cold_kv` / `cold_residuals` are capacity-padded buffers whose
// logical row count is `cold_len` (`store.rs`). A decode that reads
// `shape()[0]` instead attends zero-padded phantom K/V rows and
// recomputes hot K at shifted RoPE offsets. Each test decodes the
// same logical state twice — once over the padded buffers, once over
// buffers trimmed to exactly `cold_len` (the layout the
// pre-migration code assumed) — and requires bit-identical output.

/// Window / prompt sized so the prefill overflows: `cold_len = 2`
/// while the cold buffers are allocated at the doubling minimum
/// (8 rows), i.e. capacity strictly exceeds the logical length.
const PADDED_WINDOW: usize = 2;
const PADDED_PROMPT: [u32; 4] = [0, 1, 2, 3];
const PADDED_NEXT_TOKEN: u32 = 4;
/// Hidden-state tolerance between the cached-hot-K/V path and the
/// recompute-from-residuals path (same bound as the W2 parity test
/// in engine.rs — the two paths project K/V through different code
/// but must agree to fp rounding).
const CACHED_VS_RECOMPUTE_TOL: f32 = 1e-4;

fn prefill_padded_store(
    weights: &larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
) -> RsStore {
    let store = rs_prefill_walk(
        larql_inference::WeightsView::dense(weights),
        index,
        &PADDED_PROMPT,
        Some(PADDED_WINDOW),
        &CpuBackend,
    )
    .store;
    // Self-check: the fixture must actually be capacity-padded,
    // otherwise these tests stop pinning the bug.
    assert!(store.cold_kv.as_ref().unwrap()[0].0.shape()[0] > store.cold_len);
    assert!(store.cold_residuals.as_ref().unwrap()[0].shape()[0] > store.cold_len);
    store
}

/// Rebuild the cold buffers at exactly `cold_len` rows. Decode over
/// this store is correct under both the old and the fixed code, so
/// it is the parity reference for the capacity-padded twin.
fn with_trimmed_cold(mut store: RsStore) -> RsStore {
    use ndarray::s;
    let c = store.cold_len;
    if let Some(cold) = store.cold_residuals.as_mut() {
        for layer in cold.iter_mut() {
            *layer = layer.slice(s![..c, ..]).to_owned();
        }
    }
    if let Some(kv) = store.cold_kv.as_mut() {
        for (k, v) in kv.iter_mut() {
            *k = k.slice(s![..c, ..]).to_owned();
            *v = v.slice(s![..c, ..]).to_owned();
        }
    }
    store
}

fn decode_once(
    weights: &larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
    store: RsStore,
) -> (Array2<f32>, RsStore) {
    rs_decode_step_walk(
        larql_inference::WeightsView::dense(weights),
        index,
        PADDED_NEXT_TOKEN,
        store,
        &CpuBackend,
        None,
    )
    .expect("decode")
}

fn assert_bits_eq(a: &Array2<f32>, b: &Array2<f32>, ctx: &str) {
    let a_bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let b_bits: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a_bits, b_bits, "{ctx}");
}

#[test]
fn decode_walk_padded_cold_kv_matches_trimmed_reference_cached_path() {
    // Fast path (hot_kv + cold_kv cached): the cold concat must use
    // `cold_len`, not the buffer capacity — otherwise 6 phantom
    // zero-K/V rows join the softmax.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let padded = prefill_padded_store(&weights, &index);
    let trimmed = with_trimmed_cold(prefill_padded_store(&weights, &index));
    let (h_padded, _) = decode_once(&weights, &index, padded);
    let (h_trimmed, _) = decode_once(&weights, &index, trimmed);
    assert_bits_eq(
        &h_padded,
        &h_trimmed,
        "cached path attended capacity slack instead of cold_len rows",
    );
}

#[test]
fn decode_walk_padded_cold_kv_matches_trimmed_reference_hot_dropped() {
    // Middle path (cold_kv only): same cold_len contract when the
    // hot K/V is recomputed.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let mut padded = prefill_padded_store(&weights, &index);
    padded.hot_kv = None;
    let mut trimmed = with_trimmed_cold(prefill_padded_store(&weights, &index));
    trimmed.hot_kv = None;
    let (h_padded, _) = decode_once(&weights, &index, padded);
    let (h_trimmed, _) = decode_once(&weights, &index, trimmed);
    assert_bits_eq(
        &h_padded,
        &h_trimmed,
        "cold_kv-only path attended capacity slack instead of cold_len rows",
    );
}

#[test]
fn decode_walk_padded_cold_residuals_match_trimmed_reference_no_caches() {
    // Slow path (neither cache): reading `shape()[0]` off the padded
    // cold_residuals both injects phantom rows AND shifts the hot
    // rows' RoPE offsets (`full_abs_start + padded_cold_rows`).
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let mut padded = prefill_padded_store(&weights, &index);
    padded.hot_kv = None;
    padded.cold_kv = None;
    let mut trimmed = with_trimmed_cold(prefill_padded_store(&weights, &index));
    trimmed.hot_kv = None;
    trimmed.cold_kv = None;
    let (h_padded, _) = decode_once(&weights, &index, padded);
    let (h_trimmed, _) = decode_once(&weights, &index, trimmed);
    assert_bits_eq(
        &h_padded,
        &h_trimmed,
        "no-cache path recomputed over capacity slack / wrong RoPE offsets",
    );
}

#[test]
fn decode_walk_no_cache_step_leaves_hot_kv_usable_by_next_cached_step() {
    // Path-3 hot_kv capture: with `s_cold` cold rows folded into the
    // attention prior (cold_kv = None), the captured hot K/V must
    // slice past them. Caching cold+hot+new as "hot" desyncs
    // `hot_kv` from `stored`/`hot_len`, so the NEXT step's cached
    // decode attends the wrong rows. Pin by value: step 2 cached
    // must match step 2 recomputed-from-residuals.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);

    let run_step2 = |drop_caches_before_step2: bool| -> Array2<f32> {
        let mut store = prefill_padded_store(&weights, &index);
        store.hot_kv = None;
        store.cold_kv = None;
        // Step 1 takes the no-cache path in both runs → identical
        // state going into step 2.
        let (_, mut rs2) = decode_once(&weights, &index, store);
        if drop_caches_before_step2 {
            rs2.hot_kv = None;
            rs2.cold_kv = None;
        }
        let (h2, _) = rs_decode_step_walk(
            larql_inference::WeightsView::dense(&weights),
            &index,
            PADDED_NEXT_TOKEN + 1,
            rs2,
            &CpuBackend,
            None,
        )
        .expect("step 2");
        h2
    };

    let h2_cached = run_step2(false);
    let h2_recompute = run_step2(true);
    for (a, b) in h2_cached.iter().zip(h2_recompute.iter()) {
        assert!(
            (a - b).abs() < CACHED_VS_RECOMPUTE_TOL,
            "step-2 cached path diverged from recompute reference: \
             cached={a}, recompute={b} — hot_kv captured cold rows as hot"
        );
    }
}

#[test]
fn decode_walk_first_overflow_initializes_cold_residuals() {
    // Prefill without overflow, then decode past the window cap.
    // First overflow exercises the `None` arm of
    // `updated_rs.cold_residuals.as_mut()` — fresh cold tier
    // initialised from the evicted block.
    let weights = make_test_weights();
    let index = make_test_vindex(&weights);
    let prefill = rs_prefill_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        &[0u32, 1],
        Some(2),
        &CpuBackend,
    );
    assert!(prefill.store.cold_residuals.is_none());
    let (_, rs2) = rs_decode_step_walk(
        larql_inference::WeightsView::dense(&weights),
        &index,
        2,
        prefill.store,
        &CpuBackend,
        None,
    )
    .unwrap();
    assert!(rs2.cold_residuals.is_some());
    // 2026-05-19 audit fix: shape()[0] is doubling capacity. Use cold_len.
    assert_eq!(rs2.cold_len, 1);
    assert!(rs2.cold_kv.is_some());
}
