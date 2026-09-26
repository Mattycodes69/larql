use super::*;

#[test]
fn new_engine_is_empty() {
    let eng = WindowedCheckpointEngine::new(512);
    assert_eq!(eng.window_size, 512);
    assert_eq!(eng.archive.len(), 0);
    assert_eq!(eng.checkpoints.len(), 0);
    assert_eq!(eng.current_window_id, 0);
    assert_eq!(eng.memory_bytes(), 0);
}

#[test]
fn engine_info_backend_is_cpu() {
    let eng = WindowedCheckpointEngine::new(256);
    let info = eng.info();
    assert_eq!(info.name, "windowed-checkpoint");
    assert!(
        info.backend.starts_with("cpu"),
        "expected cpu backend, got {:?}",
        info.backend
    );
    assert_eq!(info.config, "window=256");
    assert!(info.summary().contains("windowed-checkpoint"));
    assert!(info.summary().contains("cpu"));
}

#[test]
fn engine_info_config_contains_window_size() {
    let eng = WindowedCheckpointEngine::new(1024);
    assert!(eng.info().config.contains("1024"));
}

#[test]
fn window_tokens_and_cold_bytes_start_zero() {
    let eng = WindowedCheckpointEngine::new(512);
    assert_eq!(eng.window_tokens(), 0);
    assert_eq!(eng.cold_bytes(), 0);
}

// ── prefill / decode cycle ─────────────────────────────────────────────────

#[test]
fn prefill_returns_hidden_state() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(512);
    let h = engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill failed");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(
        h.iter().all(|v| v.is_finite()),
        "hidden state should be finite"
    );
}

#[test]
fn decode_step_returns_hidden_state() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(512);
    engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
    let h = engine.decode_step(&weights, &ffn, 1).expect("decode_step");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
fn window_auto_closes_when_full() {
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let window_size = 3usize;
    let mut engine = WindowedCheckpointEngine::new(window_size);

    // Feed exactly window_size tokens → triggers close
    for tok in 0..window_size as u32 {
        engine
            .process(&weights, &[tok], None)
            .expect("process failed");
    }
    assert_eq!(engine.archive.len(), 1, "one window should be archived");
    assert_eq!(
        engine.current_window_tokens.len(),
        0,
        "current window should be empty"
    );
    assert_eq!(
        engine.checkpoints.len(),
        1,
        "one checkpoint should be saved"
    );
}

#[test]
fn two_full_windows_archives_two() {
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let mut engine = WindowedCheckpointEngine::new(2);

    // 4 tokens = 2 complete windows
    for tok in 0u32..4 {
        engine.process(&weights, &[tok], None).expect("process");
    }
    assert_eq!(engine.archive.len(), 2);
    assert_eq!(engine.checkpoints.len(), 2);
}

#[test]
fn partial_window_after_process() {
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let mut engine = WindowedCheckpointEngine::new(4);

    // 3 tokens < window_size=4 → no close
    engine
        .process(&weights, &[0u32, 1, 2], None)
        .expect("process");
    assert_eq!(engine.archive.len(), 0, "no window closed yet");
    assert_eq!(engine.window_tokens(), 3);
}

#[test]
fn flush_closes_partial_window() {
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let mut engine = WindowedCheckpointEngine::new(4);
    engine.process(&weights, &[0u32, 1], None).expect("process");
    assert_eq!(engine.archive.len(), 0);
    engine.flush();
    assert_eq!(engine.archive.len(), 1, "flush should close partial window");
}

#[test]
fn cold_bytes_grow_after_window_close() {
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let mut engine = WindowedCheckpointEngine::new(2);
    assert_eq!(engine.cold_bytes(), 0);
    engine.process(&weights, &[0u32, 1], None).expect("process"); // closes window
    assert!(
        engine.cold_bytes() > 0,
        "cold tier should grow after window close"
    );
}

#[test]
fn memory_bytes_nonzero_after_prefill() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(512);
    assert_eq!(engine.memory_bytes(), 0);
    engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill");
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn logits_from_windowed_checkpoint_are_finite() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(512);
    let h = engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
    let logits = hidden_to_raw_logits(&weights, &h);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits should be finite"
    );
}

// ── Q4K paths via Q4K fixture ─────────────────────────────────────────
//
// `prefill_quant` first tries `fused_prefill` (Metal fast path); on
// CPU that returns None (no fused decode kernel), so we fall through
// to the dequant + cached-decode path. The Q4K fixture has the attn
// Q4K slices the dequant step needs.

#[test]
fn prefill_quant_cpu_runs_via_dequant_path() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512);
    let h = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill_quant Q4K cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn decode_step_quant_cpu_extends_state() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill_quant");
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode_step_quant Q4K cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// Flags-ON parity gate for the in-place window K/V fast path: an A/B of the
/// in-place steady state vs the owned-concat reference, both driving the
/// resident decode path (`extend_current`) with Q4K-direct attention live.
/// The two must produce bit-identical hidden states every step — the
/// in-place append only changes the window-buffer representation (doubling +
/// views vs fresh owned concat). 13 tokens < window(512), so it stays in one
/// window (no close). Serialised on `Q4K_FLAG_ENV_LOCK`; path selected via
/// the shared `LARQL_MARKOV_INPLACE_KV` thread-local override.
#[test]
fn decode_inplace_matches_owned_concat_flags_on() {
    use crate::engines::markov_residual::compute::set_markov_env_override;
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

    let _q4k = crate::engines::Q4kFlagGuard::set(&[
        (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
        (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
    ]);

    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let ffn = NullFfn;

    let run = |inplace: bool| -> Vec<Vec<u32>> {
        set_markov_env_override(
            "LARQL_MARKOV_INPLACE_KV",
            Some(if inplace { "1" } else { "0" }),
        );
        let mut engine = WindowedCheckpointEngine::new(512);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        let mut hiddens = Vec::new();
        for tok in 3u32..=12 {
            let h = engine
                .decode_step_resident(&weights, &ffn, &index, tok)
                .expect("decode_step_resident");
            assert!(h.iter().all(|v| v.is_finite()));
            hiddens.push(h.iter().map(|v| v.to_bits()).collect());
        }
        hiddens
    };

    let a = run(true);
    let b = run(false);
    assert_eq!(
        a, b,
        "unlimited in-place vs owned-concat hidden states diverged (q4k on)"
    );
}

#[test]
fn decode_step_quant_without_prefill_returns_none() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512);
    // No prefill → decode falls through fast-path checks and returns None
    // (or some empty hidden) without panicking.
    let _ = engine.decode_step_quant(&weights, &ffn, &index, 0, &*backend);
}

// ── Public utility methods (stats, replay_window, summary) ────────────

#[test]
fn engine_stats_summary_includes_archived_and_compression() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(512);
    engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill");
    let stats = engine.stats(&weights);
    assert!(stats.total_tokens >= 3);
    // EngineStats::summary builds a one-line string that includes
    // window count and token count.
    let s = stats.summary();
    assert!(s.contains("windows"));
    assert!(s.contains("tokens"));
}

#[test]
fn engine_stats_with_empty_engine_handles_zero_division() {
    let weights = larql_inference::test_utils::make_test_weights();
    let engine = WindowedCheckpointEngine::new(512);
    let stats = engine.stats(&weights);
    // No prefill → all counters zero, compression ratio short-circuits
    // to 0.0 (no division by zero).
    assert_eq!(stats.total_tokens, 0);
    assert_eq!(stats.archived_windows, 0);
    assert!(
        stats.compression_ratio == 0.0,
        "compression should be 0 when no boundary bytes archived"
    );
    // Summary still produces a string for the empty case.
    let _ = stats.summary();
}

#[test]
fn replay_window_returns_none_for_missing_window() {
    let weights = larql_inference::test_utils::make_test_weights();
    let engine = WindowedCheckpointEngine::new(512);
    // No windows archived → any window_id returns None at the
    // `self.archive.retrieve(window_id)?` line.
    assert!(engine.replay_window(&weights, None, None, 0).is_err());
    assert!(engine.replay_window(&weights, None, None, 99).is_err());
}

#[test]
fn replay_window_succeeds_after_window_overflow() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    // window=2; prefill 4 tokens → archives at least 1 window.
    let mut engine = WindowedCheckpointEngine::new(2);
    engine
        .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
        .expect("prefill 4 tokens");
    let stats = engine.stats(&weights);
    assert!(
        stats.archived_windows >= 1,
        "expected at least 1 archived window after overflow, got {}",
        stats.archived_windows
    );
    // Replay the first archived window — exercises the
    // `rs_extend_from_checkpoint_backend` path (lines 132-138).
    let replay = engine.replay_window(&weights, None, None, 0);
    assert!(replay.is_ok(), "replay_window(0) should succeed");
    let (kv, abs_end) = replay.unwrap();
    assert!(!kv.is_empty(), "replayed K/V cache should be non-empty");
    assert!(
        abs_end < 4,
        "abs_end {abs_end} should be within the prefill"
    );
}

// ── Phase 2: executor-driven path ─────────────────────────────────────

#[test]
fn prefill_quant_via_executor_runs_through_local_walk() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512);
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("executor prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_quant_via_executor_extends_state() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
        .expect("prefill");
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
        .expect("decode");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// Drive `rs_extend_from_checkpoint_quant`'s `Some(profiler)` arms
/// — covers the per-stage `if timing { ... }` blocks and the
/// profiler accumulator at the end of the function.
#[test]
fn process_quant_with_profiling_populates_summary() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512).with_profiling(true);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill");
    engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode");
    let summary = engine
        .stage_summary()
        .expect("windowed_checkpoint profiler should populate summary");
    assert_eq!(summary.engine, "windowed-checkpoint");
    assert!(summary.steps >= 1);
    assert!(summary.avg_attention_us > 0.0);
    assert!(summary.avg_ffn_us > 0.0);
    assert!(summary.avg_total_decode_us > 0.0);
}

/// Counting FFN that records every `forward` call. Proves the executor
/// path actually dispatches through the caller's `FfnBackend` instead
/// of constructing a local `WalkFfn` (the legacy coupling the migration
/// removes).
struct CountingFfn {
    calls: std::sync::atomic::AtomicUsize,
    hidden: usize,
}
impl larql_inference::ffn::FfnBackend for CountingFfn {
    fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        ndarray::Array2::zeros((x.shape()[0], self.hidden))
    }
    fn name(&self) -> &str {
        "counting"
    }
}

#[test]
fn executor_path_honors_ffn_parameter() {
    use larql_inference::layer_executor::LocalWalkExecutor;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);

    let ffn = CountingFfn {
        calls: std::sync::atomic::AtomicUsize::new(0),
        hidden: weights.hidden_size,
    };
    let mut engine = WindowedCheckpointEngine::new(512);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("prefill via executor");

    let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
    // 3 tokens × num_layers — one FFN dispatch per (token, layer)
    // because the engine's per-token loop runs every layer through
    // `run_decode_layer`, which in turn invokes the caller's FFN.
    let expected = 3 * weights.num_layers;
    assert_eq!(
        call_count, expected,
        "executor path should dispatch FFN through the supplied backend \
         once per (token, layer); got {call_count} for {expected} \
         expected — engine is likely constructing its own FFN internally",
    );
}

// ── window_size validation ────────────────────────────────────────────

#[test]
#[should_panic(expected = "window_size must be >= 1")]
fn zero_window_size_is_rejected_at_construction() {
    let _ = WindowedCheckpointEngine::new(0);
}

#[test]
fn window_size_one_is_legal() {
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let mut engine = WindowedCheckpointEngine::new(1);
    engine.process(&weights, &[0u32, 1], None).expect("process");
    assert_eq!(
        engine.archive.len(),
        2,
        "window_size=1 closes one window per token"
    );
}

// ── degenerate close / lost-shadow states ─────────────────────────────

/// A full window with neither a CPU shadow nor a backend handle is
/// unrecoverable K/V-wise, but the close must still archive the
/// tokens and reset the counters — pre-fix it early-returned with
/// the window intact, and `process()` spun forever re-trying the
/// close with zero free slots.
#[test]
fn close_window_without_shadow_or_handle_recovers_bookkeeping() {
    let mut engine = WindowedCheckpointEngine::new(2);
    engine.current_window_tokens = vec![7, 8];
    engine.current_window_kv = None;
    engine.current_window_kv_len = 2;
    engine.close_window();
    assert_eq!(engine.archive.len(), 1, "tokens archived");
    assert!(engine.current_window_tokens.is_empty(), "window reset");
    assert_eq!(engine.abs_offset, 2);
    assert_eq!(engine.current_window_id, 1);
    let (ckpt, abs_end) = engine.checkpoints.load(0).expect("checkpoint entry");
    assert!(
        ckpt.is_empty(),
        "unrecoverable K/V must yield an empty checkpoint, not stale rows"
    );
    assert_eq!(abs_end, 1);
    let (tokens, abs_start) = engine.archive.retrieve(0).expect("archived tokens");
    assert_eq!(tokens, &[7, 8]);
    assert_eq!(abs_start, 0);
}

/// Mid-window decode with the shadow lost must surface an error —
/// pre-fix it silently seeded attention from an empty prior,
/// dropping every in-window token from the context.
#[test]
fn decode_with_lost_window_shadow_errors_instead_of_dropping_context() {
    use larql_inference::ffn::WeightFfn;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = WindowedCheckpointEngine::new(8);
    engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill");
    engine.current_window_kv = None;
    let res = engine.decode_step(&weights, &ffn, 3);
    assert!(
        res.is_err(),
        "mid-window decode without the window shadow must error"
    );
}

/// Same contract on the quant walk path (`extend_current_quant`).
#[test]
fn decode_step_quant_with_lost_window_shadow_errors() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = WindowedCheckpointEngine::new(512);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill");
    // Simulate a failed dispatch step: handle dropped, shadow gone,
    // window tokens still present.
    engine.kv_handle = None;
    engine.current_window_kv = None;
    let res = engine.decode_step_quant(&weights, &ffn, &index, 2, &*backend);
    assert!(
        res.is_err(),
        "quant decode without the window shadow must error"
    );
}

#[test]
fn prefill_quant_via_executor_with_small_window_archives() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    use larql_inference::test_utils::make_test_weights;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    // window=2, 4 tokens → triggers two window-close cycles via
    // `process_via_executor`. Exercises the prior-checkpoint-load
    // branch in `extend_current_via_executor`.
    let mut engine = WindowedCheckpointEngine::new(2);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
        .expect("prefill 4 tokens through executor");
    let stats = engine.stats(&weights);
    assert!(
        stats.archived_windows >= 1,
        "expected at least 1 archived window, got {}",
        stats.archived_windows
    );
}
