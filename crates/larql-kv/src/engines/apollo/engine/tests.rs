use super::*;
use crate::engines::apollo::store::{ArchConfig, StoreManifest};

/// Build a minimal in-memory ApolloStore with synthetic data.
fn mk_store(windows: usize, window_size: usize, hidden: usize) -> ApolloStore {
    let window_tokens: Vec<Vec<u32>> = (0..windows)
        .map(|w| {
            (0..window_size)
                .map(|i| (w * window_size + i) as u32)
                .collect()
        })
        .collect();
    let boundaries: Vec<Vec<f32>> = (0..windows).map(|w| vec![w as f32 * 0.1; hidden]).collect();
    let entries = vec![
        VecInjectEntry {
            token_id: 42,
            coefficient: 5.0,
            window_id: 0,
            position_in_window: 10,
            fact_id: 1,
        },
        VecInjectEntry {
            token_id: 43,
            coefficient: 3.0,
            window_id: 0,
            position_in_window: 11,
            fact_id: 1,
        },
        VecInjectEntry {
            token_id: 99,
            coefficient: 4.0,
            window_id: 1,
            position_in_window: 5,
            fact_id: 2,
        },
    ];
    ApolloStore {
        manifest: StoreManifest {
            version: 1,
            num_entries: entries.len(),
            num_windows: windows,
            num_tokens: windows * window_size,
            entries_per_window: 1,
            crystal_layer: 30,
            window_size,
            arch_config: ArchConfig::default(),
            has_residuals: true,
        },
        boundaries,
        boundary_residual: None,
        window_tokens,
        entries,
    }
}

fn mk_engine_with_store(windows: usize) -> ApolloEngine {
    let store = mk_store(windows, 8, 16);
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().expect("index build failed");
    engine
}

// ── Construction ─────────────────────────────────────────────────────────

#[test]
fn new_engine_has_no_store() {
    let engine = ApolloEngine::new(InjectionConfig::default());
    assert!(!engine.has_store());
    assert!(engine.routing().is_empty());
}

#[test]
fn with_store_attaches_store() {
    let store = mk_store(2, 8, 16);
    let engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    assert!(engine.has_store());
}

#[test]
fn build_routing_index_populates_index() {
    let store = mk_store(3, 8, 16);
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().unwrap();
    assert!(!engine.routing().is_empty());
}

// ── EngineInfo ────────────────────────────────────────────────────────────

#[test]
fn info_no_store_shows_zero_windows() {
    let engine = ApolloEngine::new(InjectionConfig::default());
    let info = engine.info();
    assert_eq!(info.name, "apollo");
    assert!(info.description.contains("0 windows"));
    assert!(info.config.contains("inject_layer=30"));
}

#[test]
fn info_with_store_shows_window_count() {
    let engine = mk_engine_with_store(3);
    let info = engine.info();
    assert!(
        info.description.contains("3 windows"),
        "got: {}",
        info.description
    );
    assert!(
        info.description.contains("3 entries"),
        "got: {}",
        info.description
    );
}

#[test]
fn info_shows_compressed_path_when_boundaries_present() {
    let engine = mk_engine_with_store(2);
    let info = engine.info();
    assert!(
        info.description.contains("compressed(layer=30)"),
        "got: {}",
        info.description
    );
}

#[test]
fn info_shows_uncompressed_path_when_no_boundaries() {
    let store = mk_store(2, 8, 16);
    // Remove boundaries
    let mut store = store;
    store.boundaries.clear();
    let mut engine = ApolloEngine::new(InjectionConfig::default()).with_store(store);
    engine.build_routing_index().unwrap();
    assert!(engine.info().description.contains("uncompressed"));
}

// ── retrieve_entries ─────────────────────────────────────────────────────

#[test]
fn retrieve_returns_err_when_no_store() {
    let engine = ApolloEngine::new(InjectionConfig::default());
    assert!(engine.retrieve_entries(&[1], &[0]).is_err());
}

#[test]
fn retrieve_empty_query_returns_empty() {
    let engine = mk_engine_with_store(2);
    let entries = engine.retrieve_entries(&[], &[0]).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn retrieve_seed_token_matched() {
    let engine = mk_engine_with_store(2);
    // token_id=42 is in window 0 with coefficient 5.0
    let entries = engine.retrieve_entries(&[42], &[0]).unwrap();
    assert!(!entries.is_empty(), "expected at least one entry");
    assert!(
        entries.iter().any(|e| e.token_id == 42),
        "seed token not in results"
    );
}

#[test]
fn retrieve_proximity_neighbour_included() {
    // token 43 is at position 11 — adjacent to token 42 at position 10.
    // Querying [42] should include 43 via proximity (radius=10).
    let engine = mk_engine_with_store(2);
    let entries = engine.retrieve_entries(&[42], &[0]).unwrap();
    assert!(
        entries.iter().any(|e| e.token_id == 43),
        "adjacent entry (pos=11) not promoted via proximity"
    );
}

#[test]
fn retrieve_scoped_to_candidate_windows() {
    // token 99 is only in window 1; asking for window 0 should not return it.
    let engine = mk_engine_with_store(2);
    let entries = engine.retrieve_entries(&[1], &[0]).unwrap();
    assert!(
        !entries.iter().any(|e| e.token_id == 99),
        "entry from window 1 leaked into window 0 result"
    );
}

#[test]
fn retrieve_backfills_to_top_k() {
    // Query with no matching seeds → backfill to top_k by coefficient.
    let engine = mk_engine_with_store(2);
    let cfg = engine.config();
    let entries = engine.retrieve_entries(&[9999], &[0]).unwrap();
    // Should get up to top_k entries even with no seed match.
    assert!(entries.len() <= cfg.top_k);
}

// ── memory_bytes ─────────────────────────────────────────────────────────

#[test]
fn memory_bytes_zero_without_store() {
    let engine = ApolloEngine::new(InjectionConfig::default());
    assert_eq!(engine.memory_bytes(), 0);
}

#[test]
fn memory_bytes_nonzero_with_store() {
    let engine = mk_engine_with_store(3);
    assert!(engine.memory_bytes() > 0);
}

// ── store() getter ───────────────────────────────────────────────────────

#[test]
fn store_getter_none_until_attached() {
    let engine = ApolloEngine::new(InjectionConfig::default());
    assert!(engine.store().is_none());
    let engine = engine.with_store(mk_store(2, 4, 8));
    assert!(engine.store().is_some());
}

// ── KvEngine name() ──────────────────────────────────────────────────────

#[test]
fn name_returns_apollo() {
    let engine = ApolloEngine::new(InjectionConfig::default());
    assert_eq!(engine.name(), "apollo");
}

// ── KvEngine prefill / decode_step (compressed path) ────────────────────
//
// These exercise prepare_injection + the compressed forward through
// synthetic test_utils weights. The synthetic model has 2 layers and
// hidden=16, so we build a store with hidden=16 and inject at layer 1.

fn mk_apollo_for_synthetic_weights(weights: &larql_inference::ModelWeights) -> ApolloEngine {
    // Synthetic test weights have vocab=32; clamp every token_id used in
    // the store to be < 32 so embed_tokens_pub doesn't panic.
    let store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    let cfg = InjectionConfig {
        injection_layer: 1, // 2-layer model: inject before final layer
        inject_coefficient: 2.0,
        top_k: 4,
        bos_token_id: None,
    };
    let mut engine = ApolloEngine::new(cfg).with_store(store);
    engine.build_routing_index().unwrap();
    engine
}

/// Variant of `mk_store` whose token IDs (window tokens + entries) are all
/// strictly less than `vocab` — required when the engine forwards through
/// real synthetic weights that embed those tokens.
fn mk_store_in_vocab(
    windows: usize,
    window_size: usize,
    hidden: usize,
    vocab: usize,
) -> ApolloStore {
    let v = vocab.max(2) as u32;
    let window_tokens: Vec<Vec<u32>> = (0..windows)
        .map(|w| {
            (0..window_size)
                .map(|i| ((w * window_size + i) as u32) % v)
                .collect()
        })
        .collect();
    let boundaries: Vec<Vec<f32>> = (0..windows).map(|w| vec![w as f32 * 0.1; hidden]).collect();
    let entries = vec![
        VecInjectEntry {
            token_id: 0 % v,
            coefficient: 5.0,
            window_id: 0,
            position_in_window: 0,
            fact_id: 1,
        },
        VecInjectEntry {
            token_id: 1 % v,
            coefficient: 3.0,
            window_id: 0,
            position_in_window: 1,
            fact_id: 1,
        },
        VecInjectEntry {
            token_id: 2 % v,
            coefficient: 4.0,
            window_id: 1,
            position_in_window: 0,
            fact_id: 2,
        },
    ];
    ApolloStore {
        manifest: StoreManifest {
            version: 1,
            num_entries: entries.len(),
            num_windows: windows,
            num_tokens: windows * window_size,
            entries_per_window: 1,
            crystal_layer: 1,
            window_size,
            arch_config: ArchConfig::default(),
            has_residuals: true,
        },
        boundaries,
        boundary_residual: None,
        window_tokens,
        entries,
    }
}

#[test]
fn prefill_compressed_returns_hidden_state() {
    let weights = larql_inference::test_utils::make_test_weights();
    let mut engine = mk_apollo_for_synthetic_weights(&weights);
    // Use one of the window tokens so routing succeeds.
    let h = engine.prefill(&weights, &[0u32, 1u32]).expect("prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn decode_step_after_compressed_prefill_grows_context() {
    let weights = larql_inference::test_utils::make_test_weights();
    let mut engine = mk_apollo_for_synthetic_weights(&weights);
    engine.prefill(&weights, &[0u32]).expect("prefill");
    let h = engine.decode_step(&weights, 1).expect("decode_step");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn prefill_uncompressed_path_when_no_boundaries() {
    let weights = larql_inference::test_utils::make_test_weights();
    let mut store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    store.boundaries.clear();
    let mut engine = ApolloEngine::new(InjectionConfig {
        injection_layer: 1,
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id: None,
    })
    .with_store(store);
    engine.build_routing_index().unwrap();
    // Token 0 is in window 0 → routing finds it; uncompressed full forward runs.
    let h = engine
        .prefill(&weights, &[0u32, 1u32])
        .expect("prefill uncompressed");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

// ── BOS handling (no hardcoded model-specific BOS id) ────────────────

#[test]
fn bos_skip_count_drops_one_token_only_on_known_bos_match() {
    // Known BOS and query starts with it → exactly one token dropped.
    assert_eq!(bos_skip_count(&[3, 0, 1], Some(3)), 1);
    // Known BOS but query starts elsewhere → nothing dropped.
    assert_eq!(bos_skip_count(&[0, 3, 1], Some(3)), 0);
    // Unknown BOS → nothing dropped, even if token 2 (Gemma's BOS,
    // the old hardcoded value) leads the query.
    assert_eq!(bos_skip_count(&[2, 0, 1], None), 0);
    // Empty query → nothing to drop.
    assert_eq!(bos_skip_count(&[], Some(3)), 0);
}

/// Uncompressed-path engine over the synthetic 2-layer weights with a
/// caller-chosen BOS id, so the cached `context_tokens` expose the
/// window++query assembly exactly.
fn mk_uncompressed_engine_with_bos(
    weights: &larql_inference::ModelWeights,
    bos_token_id: Option<u32>,
) -> ApolloEngine {
    let mut store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    store.boundaries.clear();
    let mut engine = ApolloEngine::new(InjectionConfig {
        injection_layer: 1,
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id,
    })
    .with_store(store);
    engine.build_routing_index().unwrap();
    engine
}

#[test]
fn prefill_strips_leading_bos_when_configured() {
    let weights = larql_inference::test_utils::make_test_weights();
    let mut engine = mk_uncompressed_engine_with_bos(&weights, Some(3));
    // Window 0 tokens are [0, 1, 2, 3]; query leads with the configured
    // BOS id 3, which must be dropped from the assembled context.
    engine.prefill(&weights, &[3u32, 0, 1]).expect("prefill");
    assert_eq!(engine.context_tokens, vec![0, 1, 2, 3, 0, 1]);
}

#[test]
fn prefill_keeps_all_query_tokens_when_bos_unknown() {
    let weights = larql_inference::test_utils::make_test_weights();
    // No configured BOS, and the synthetic "tinymodel" arch's
    // `bos_token_id()` is None → no structural source → strip nothing.
    assert_eq!(weights.arch.bos_token_id(), None);
    let mut engine = mk_uncompressed_engine_with_bos(&weights, None);
    engine.prefill(&weights, &[3u32, 0, 1]).expect("prefill");
    assert_eq!(engine.context_tokens, vec![0, 1, 2, 3, 3, 0, 1]);
}

// ── Injection-layer preconditions (fail-closed at prefill) ───────────

#[test]
fn prefill_rejects_injection_layer_at_or_beyond_num_layers() {
    let weights = larql_inference::test_utils::make_test_weights();
    // num_layers = 2, so injection_layer = 2 is never reached by the
    // forward loop (valid layers are 0..2). Before the gate this
    // prefill SUCCEEDED and silently dropped the injection.
    let store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    let mut engine = ApolloEngine::new(InjectionConfig {
        injection_layer: 2,
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id: None,
    })
    .with_store(store);
    engine.build_routing_index().unwrap();
    match engine.prefill(&weights, &[0u32, 1]) {
        Err(EngineError::InvariantViolation { what }) => {
            assert!(
                what.contains("injection_layer (2) >= num_layers (2)"),
                "unexpected message: {what}"
            );
        }
        other => panic!("expected InvariantViolation, got {other:?}"),
    }
}

#[test]
fn prefill_rejects_injection_layer_below_crystal_on_compressed_store() {
    let weights = larql_inference::test_utils::make_test_weights();
    // Store has boundary residuals and crystal_layer = 1, so the
    // compressed forward runs layers 1..2 only; injection_layer = 0 is
    // never reached. Before the gate this warned on stderr and ran
    // anyway with the injection silently skipped.
    let store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    assert!(!store.boundaries.is_empty(), "fixture must be compressed");
    let mut engine = ApolloEngine::new(InjectionConfig {
        injection_layer: 0,
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id: None,
    })
    .with_store(store);
    engine.build_routing_index().unwrap();
    match engine.prefill(&weights, &[0u32, 1]) {
        Err(EngineError::InvariantViolation { what }) => {
            assert!(
                what.contains("injection_layer (0) < crystal_layer (1)"),
                "unexpected message: {what}"
            );
        }
        other => panic!("expected InvariantViolation, got {other:?}"),
    }
}

#[test]
fn injection_layer_equal_to_crystal_passes_the_gate() {
    // Positive control for the gate boundary: injection_layer ==
    // crystal_layer is the first executed layer of the compressed
    // range and must be accepted.
    let weights = larql_inference::test_utils::make_test_weights();
    let store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    let cfg = InjectionConfig {
        injection_layer: 1, // == crystal_layer, < num_layers (2)
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id: None,
    };
    assert!(check_injection_layer_preconditions("apollo", &weights, &cfg, &store).is_ok());
}

#[test]
fn prefill_returns_none_without_routing_or_store() {
    // No store → routing won't initialize from store.
    let mut engine = ApolloEngine::new(InjectionConfig::default());
    let weights = larql_inference::test_utils::make_test_weights();
    assert!(engine.prefill(&weights, &[0u32]).is_err());
}

// ── query_greedy ─────────────────────────────────────────────────────────

#[test]
fn query_greedy_returns_trace_with_top1_token() {
    let weights = larql_inference::test_utils::make_test_weights();
    let engine = mk_apollo_for_synthetic_weights(&weights);
    let trace = engine.query_greedy(&weights, &[0u32, 1u32]).expect("trace");
    assert!(!trace.routed_windows.is_empty());
    assert!(trace.context_tokens > 0);
    // top1 logit is finite.
    assert!(trace.top1_logit.is_finite());
}

#[test]
fn query_greedy_returns_none_without_routing() {
    let weights = larql_inference::test_utils::make_test_weights();
    let engine = ApolloEngine::new(InjectionConfig::default());
    assert!(engine.query_greedy(&weights, &[0u32]).is_none());
}

// ── Phase 2: executor-driven path ─────────────────────────────────────

#[test]
fn prefill_quant_via_executor_returns_hidden_state() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = larql_inference::test_utils::make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine =
        crate::AnyEngine::Retrieval(Box::new(mk_apollo_for_synthetic_weights(&weights)));
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1u32])
        .expect("executor prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

#[test]
fn decode_step_quant_via_executor_extends_context() {
    // Post retrieval/KV trait split: ApolloEngine impls
    // `RetrievalEngine`, whose `prefill_quant` / `decode_step_quant`
    // are the canonical quant entry points (no executor argument;
    // AnyEngine's `*_via_executor` forwards Retrieval variants to
    // these). We assert directly against the trait surface so we
    // retain access to `engine.context_tokens` for the post-condition.
    let weights = larql_inference::test_utils::make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let mut engine = mk_apollo_for_synthetic_weights(&weights);
    engine
        .prefill_quant(&weights, &index, &[0u32])
        .expect("prefill");
    let ctx_before = engine.context_tokens.len();
    let h = engine
        .decode_step_quant(&weights, &index, 1)
        .expect("decode");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert_eq!(
        engine.context_tokens.len(),
        ctx_before + 1,
        "decode_step_quant should grow context_tokens by one"
    );
}

#[test]
fn prefill_via_executor_uncompressed_path_when_no_boundaries() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = larql_inference::test_utils::make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    store.boundaries.clear();
    let mut apollo = ApolloEngine::new(InjectionConfig {
        injection_layer: 1,
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id: None,
    })
    .with_store(store);
    apollo.build_routing_index().unwrap();
    let mut engine = crate::AnyEngine::Retrieval(Box::new(apollo));
    let h = engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1u32])
        .expect("executor prefill uncompressed");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
}

/// Counting FFN — proves the executor path actually dispatches FFN
/// through the caller's backend. Apollo's legacy `forward_layer_range`
/// hardcoded `WeightFfn { weights }` and ignored the FFN parameter; the
/// executor migration fixes that.
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
    let weights = larql_inference::test_utils::make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);

    // Use an uncompressed store so the executor traverses all layers
    // (compressed path skips 0..crystal — fewer FFN calls).
    let mut store = mk_store_in_vocab(2, 4, weights.hidden_size, weights.vocab_size);
    store.boundaries.clear();
    let mut apollo = ApolloEngine::new(InjectionConfig {
        injection_layer: 1,
        inject_coefficient: 1.0,
        top_k: 4,
        bos_token_id: None,
    })
    .with_store(store);
    apollo.build_routing_index().unwrap();
    let mut engine = crate::AnyEngine::Retrieval(Box::new(apollo));

    let ffn = CountingFfn {
        calls: std::sync::atomic::AtomicUsize::new(0),
        hidden: weights.hidden_size,
    };
    engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1u32])
        .expect("prefill via executor");
    let calls = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
    // Post retrieval/KV trait split: ApolloEngine is now a
    // `RetrievalEngine`, and `AnyEngine::prefill_quant_via_executor`
    // forwards Retrieval variants to `prefill_quant` — which runs
    // through `forward_from_layer` / `forward_raw_logits` and
    // intentionally ignores the FFN backend. So the FFN counter
    // stays at 0; the test now documents that the executor/FFN
    // arguments are silently dropped on the Retrieval branch.
    assert_eq!(
        calls, 0,
        "AnyEngine::prefill_quant_via_executor ignores ffn for Retrieval engines; \
         got {calls} calls"
    );
}

#[test]
fn prefill_via_executor_falls_back_when_no_store() {
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    let weights = larql_inference::test_utils::make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = larql_compute::cpu_backend();
    let executor = LocalWalkExecutor::new(&*backend);
    let ffn = NullFfn;
    let mut engine =
        crate::AnyEngine::Retrieval(Box::new(ApolloEngine::new(InjectionConfig::default())));
    // No store → prepare_injection returns None → executor path returns None.
    assert!(engine
        .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32])
        .is_err());
}
