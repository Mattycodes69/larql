use super::*;
use larql_inference::ffn::WeightFfn;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::test_utils::make_test_weights;

#[test]
fn prefill_compresses_kv_for_all_layers() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = TurboQuantEngine::new(4);
    assert_eq!(engine.memory_bytes(), 0);
    let h = engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill failed");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(
        engine.layers.len(),
        weights.num_layers,
        "one CompressedLayer per model layer"
    );
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_grows_compressed_cache() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = TurboQuantEngine::new(4);
    engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
    let mem_before = engine.memory_bytes();

    engine.decode_step(&weights, &ffn, 1).expect("decode_step");
    // After decode: K/V cache has one more entry per layer → more compressed bytes
    assert!(
        engine.memory_bytes() > mem_before,
        "compressed cache should grow after each decode step"
    );
}

#[test]
fn logits_finite_after_prefill_and_decode() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = TurboQuantEngine::new(4);
    let h_pre = engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
    assert!(hidden_to_raw_logits(&weights, &h_pre)
        .iter()
        .all(|v| v.is_finite()));
    let h_dec = engine.decode_step(&weights, &ffn, 2).expect("decode");
    assert!(hidden_to_raw_logits(&weights, &h_dec)
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn three_bit_engine_also_works() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = TurboQuantEngine::new(3);
    let h = engine
        .prefill(&weights, &ffn, &[0u32])
        .expect("3-bit prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    // 3-bit uses fewer bytes per compressed vector
    let mem3 = engine.memory_bytes();
    let mut engine4 = TurboQuantEngine::new(4);
    engine4
        .prefill(&weights, &ffn, &[0u32])
        .expect("4-bit prefill");
    assert!(
        mem3 < engine4.memory_bytes(),
        "3-bit should use less memory than 4-bit"
    );
}

// ── Q4K paths via CPU fallback ────────────────────────────────────────
//
// `fused_prefill` / `fused_decode_step` return `None` on a CPU
// backend, so the engine falls through to `prefill_quant_cpu` /
// `decode_step_quant_cpu` against the synthetic VectorIndex. Exercises
// the Q4K branches without needing a real Metal-quantised model.

#[test]
fn prefill_q4k_cpu_fallback_compresses_kv() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    let h = engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
        .expect("prefill_quant cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(
        engine.layers.len(),
        weights.num_layers,
        "one CompressedLayer per model layer after prefill_quant"
    );
    assert!(engine.memory_bytes() > 0);
}

#[test]
fn decode_step_quant_cpu_fallback_grows_compressed_cache() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill_quant");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode_step_quant cpu fallback");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(
        engine.memory_bytes() > mem_before,
        "compressed cache should grow after decode_step_quant"
    );
}

// ── Phase 2: executor-driven path ─────────────────────────────────────

#[test]
fn prefill_quant_via_executor_compresses_kv() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("executor prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(engine.layers.len(), weights.num_layers);
    assert!(engine.memory_bytes() > 0);
}

/// Steps for the append-only regressions. The pre-fix executor path
/// decompressed and re-encoded the whole cache every step, so each
/// stored norm shrank by the codec's reconstruction ratio per step
/// (compounding to ~0.90 of true norm by 20 steps with today's
/// codebooks, ~0.44 with the mis-scaled ones).
const APPEND_ONLY_DECODE_STEPS: usize = 20;
/// Append-only decode never rewrites old rows' bytes, so the first
/// row's decoded norm is bit-stable across steps; tolerance covers
/// f64-accumulation slack only.
const FIRST_ROW_NORM_DRIFT_TOL: f64 = 1e-6;

fn layer0_row0_k_norm(engine: &TurboQuantEngine) -> f64 {
    let (k, _v) = engine.layers[0].decompress(&engine.tq);
    k.row(0)
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Packed bytes of layer 0's first K row (all heads).
fn layer0_row0_k_bytes(engine: &TurboQuantEngine) -> Vec<u8> {
    let l = &engine.layers[0];
    let bytes_per_row = (l.kv_dim / l.head_dim) * engine.tq.bytes_per_vector(l.head_dim);
    l.compressed_k[..bytes_per_row].to_vec()
}

/// Regression for the executor decode path re-encoding the entire
/// cache each step (norm decay ≈ ratio^N on the first cached row).
#[test]
fn executor_decode_is_append_only_first_row_stable() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
        .expect("prefill");
    let norm_after_prefill = layer0_row0_k_norm(&engine);
    let bytes_after_prefill = layer0_row0_k_bytes(&engine);
    assert!(norm_after_prefill > 0.0, "fixture row must be non-zero");

    for step in 0..APPEND_ONLY_DECODE_STEPS {
        engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, (step % 3) as u32)
            .expect("decode step");
    }

    let norm_after = layer0_row0_k_norm(&engine);
    let drift = (norm_after / norm_after_prefill - 1.0).abs();
    assert!(
        drift < FIRST_ROW_NORM_DRIFT_TOL,
        "first cached row's decoded norm drifted by {drift:.6} over \
         {APPEND_ONLY_DECODE_STEPS} executor decode steps \
         ({norm_after_prefill:.6} -> {norm_after:.6}); the append-only \
         invariant is broken"
    );
    assert_eq!(
        layer0_row0_k_bytes(&engine),
        bytes_after_prefill,
        "row-0 packed bytes must never be rewritten by decode"
    );
}

/// Parity of the append-only invariant across the executor and CPU
/// quant decode paths: identical prefill, then N steps down each
/// path — both must leave the first row's packed bytes at the
/// post-prefill snapshot (and therefore equal to each other).
#[test]
fn executor_and_cpu_decode_paths_append_only_parity() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;

    let mut via_executor = TurboQuantEngine::new(4);
    via_executor
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
        .expect("prefill (executor engine)");
    let mut via_cpu = TurboQuantEngine::new(4);
    via_cpu
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
        .expect("prefill (cpu engine)");

    let snapshot = layer0_row0_k_bytes(&via_executor);
    assert_eq!(
        snapshot,
        layer0_row0_k_bytes(&via_cpu),
        "identical prefill must produce identical row-0 bytes"
    );

    for step in 0..APPEND_ONLY_DECODE_STEPS {
        let token = (step % 3) as u32;
        via_executor
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, token)
            .expect("executor decode step");
        via_cpu
            .decode_step_quant(&weights, &ffn, &index, token, &*backend)
            .expect("cpu decode step");
    }

    assert_eq!(
        layer0_row0_k_bytes(&via_executor),
        snapshot,
        "executor path rewrote row-0 bytes"
    );
    assert_eq!(
        layer0_row0_k_bytes(&via_cpu),
        snapshot,
        "cpu path rewrote row-0 bytes"
    );
}

#[test]
fn decode_step_quant_via_executor_grows_cache() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
        .expect("prefill");
    let mem_before = engine.memory_bytes();
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
        .expect("decode");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > mem_before);
}

/// Drive the profiling-on branch of `decode_step_quant_cpu` —
/// covers the `if timing { ... }` arms and the profiler accumulate.
#[test]
fn decode_step_quant_cpu_with_profiling_populates_summary() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4).with_profiling(true);
    engine
        .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
        .expect("prefill");
    engine
        .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
        .expect("decode");
    let summary = engine
        .stage_summary()
        .expect("turbo-quant profiler should populate summary");
    assert_eq!(summary.engine, "turbo-quant");
    assert!(summary.steps >= 1);
    // recompute_hot (codec decode) and recompute_cold (codec encode)
    // both fire per layer per step.
    assert!(summary.avg_recompute_hot_us > 0.0);
    assert!(summary.avg_recompute_cold_us > 0.0);
    assert!(summary.avg_attention_us > 0.0);
    assert!(summary.avg_ffn_us > 0.0);
}

/// Counting FFN — proves the executor path dispatches through the
/// caller-supplied backend instead of constructing a local `WalkFfn`.
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
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = CountingFfn {
        calls: std::sync::atomic::AtomicUsize::new(0),
        hidden: weights.hidden_size,
    };
    let mut engine = TurboQuantEngine::new(4);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("prefill via executor");
    // Prefill runs FFN once per layer (single chunked sequence).
    let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(
        call_count, weights.num_layers,
        "executor path should dispatch FFN through the supplied backend \
         once per layer; got {call_count} for {} layers",
        weights.num_layers
    );
}

/// Minimal `Fused`-kind executor — the engine's executor-routed
/// entry points should detect `dispatch_kind == Fused` and short-
/// circuit to the legacy `prefill_quant` / `decode_step_quant`
/// paths, ignoring the supplied executor's per-layer methods.
struct FusedStubExecutor {
    backend: larql_compute::CpuBackend,
}
impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
    fn backend(&self) -> &dyn larql_compute::ComputeBackend {
        &self.backend
    }
    fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
        larql_inference::layer_executor::ExecutorDispatchKind::Fused
    }
    fn name(&self) -> &str {
        "fused-stub"
    }
}

#[test]
fn fused_executor_short_circuits_prefill_to_legacy_path() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let executor = FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    };
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("fused-stub prefill should route through prefill_quant");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(engine.layers.len(), weights.num_layers);
}

#[test]
fn fused_executor_short_circuits_decode_to_legacy_path() {
    use larql_inference::ffn::NullFfn;
    let weights = make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let executor = FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    };
    let ffn = NullFfn;
    let mut engine = TurboQuantEngine::new(4);
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
        .expect("prefill");
    let h = engine
        .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 3)
        .expect("fused-stub decode should route through decode_step_quant");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn counting_ffn_forward_observed_runs_forward_and_reports_absent() {
    use larql_inference::ffn::FfnBackend;
    let ffn = CountingFfn {
        calls: std::sync::atomic::AtomicUsize::new(0),
        hidden: 8,
    };
    let x = ndarray::Array2::<f32>::zeros((3, 8));
    let (h, obs) = ffn.forward_observed(0, &x);
    assert_eq!(h.shape(), &[3, 8]);
    assert!(
        obs.is_absent(),
        "counting stub must not fabricate activations"
    );
    assert_eq!(ffn.name(), "counting");
    // The default `forward_observed` delegates to `forward`, so
    // exactly one call is recorded.
    assert_eq!(ffn.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}
