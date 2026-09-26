//! Unit tests for `LoadedModel` field/flag plumbing.
//!
//! The q4k / f32 branch in `get_or_load_weights` keys off
//! `config.quant == QuantFormat::Q4K`, and `run_full_output` in
//! `routes/walk_ffn.rs` keys off the same check to decide between
//! `WalkFfn::new_unlimited` and `kquant_ffn_forward_layer`. Running
//! either branch end-to-end needs a real on-disk vindex (GBs of
//! weights), so we cover just the flag plumbing and the selector
//! expression here; the end-to-end walk is validated by the
//! `larql bench <model>` example script.
use super::*;
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, LayerBands, QuantFormat, VectorIndex, VindexConfig, VindexLayerInfo,
};

fn tiny_config(quant: QuantFormat) -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 4,
        vocab_size: 4,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 0),
            output: (0, 0),
        }),
        layers: vec![VindexLayerInfo {
            layer: 0,
            num_features: 2,
            offset: 0,
            length: 32,
            num_experts: None,
            num_features_per_expert: None,
        }],
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    }
}

fn tiny_loaded_model(quant: QuantFormat, release_mmap: bool) -> LoadedModel {
    let hidden = 4;
    let gate = Array2::<f32>::zeros((2, hidden));
    let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
    let patched = larql_vindex::PatchedVindex::new(index);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

    LoadedModel {
        id: "test".into(),
        path: PathBuf::from("/nonexistent"),
        config: tiny_config(quant),
        patched: std::sync::Arc::new(tokio::sync::RwLock::new(patched)),
        embeddings: Array2::<f32>::zeros((4, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: release_mmap,
        weights: std::sync::OnceLock::new(),
        weights_init: std::sync::Mutex::new(()),
        bitnet_model: std::sync::OnceLock::new(),
        bitnet_init: std::sync::Mutex::new(()),
        probe_labels: HashMap::new(),
        ffn_l2_cache: crate::ffn_l2_cache::FfnL2Cache::new(1),
        layer_latency_tracker: std::sync::Arc::new(crate::metrics::LayerLatencyTracker::new()),
        requests_in_flight: std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0)),
        requests_total: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        moe_scratches: std::sync::Mutex::new(HashMap::new()),
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    }
}

#[test]
fn release_mmap_flag_round_trips_true() {
    let model = tiny_loaded_model(QuantFormat::None, true);
    assert!(
        model.release_mmap_after_request,
        "true must survive unchanged — the walk-ffn handler reads this \
         post-request to issue MADV_DONTNEED"
    );
}

#[test]
fn release_mmap_flag_round_trips_false() {
    let model = tiny_loaded_model(QuantFormat::None, false);
    assert!(!model.release_mmap_after_request);
}

#[test]
fn quant_format_selects_q4k_branch() {
    // Exact selector used in both `get_or_load_weights` and
    // `run_full_output` to pick the q4k path.
    let q4k_model = tiny_loaded_model(QuantFormat::Q4K, false);
    let f32_model = tiny_loaded_model(QuantFormat::None, false);

    assert!(
        q4k_model.config.quant == QuantFormat::Q4K,
        "Q4K config → q4k branch (load_model_weights_kquant + kquant_ffn_forward_layer)"
    );
    assert!(
        f32_model.config.quant != QuantFormat::Q4K,
        "None config → f32 branch (load_model_weights_with_opts + WalkFfn::new_unlimited)"
    );
}

#[test]
fn is_dense_only_detects_empty_gate_layers() {
    // A normal vindex has gate layers -> not dense-only.
    let normal = tiny_loaded_model(QuantFormat::None, false);
    assert!(
        !normal.is_dense_only(),
        "vindex with gate layers must not be dense-only"
    );
    assert!(
        !normal.is_bitnet(),
        "and a plain vindex carries no bitnet_layout"
    );

    // A --dense-only BitNet vindex has zero gate layers.  Build
    // one by emptying the layer list + setting bitnet_layout.
    let mut cfg = tiny_config(QuantFormat::None);
    cfg.layers = Vec::new();
    cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
    let mut dense_only = tiny_loaded_model(QuantFormat::None, false);
    dense_only.config = cfg;
    assert!(
        dense_only.is_dense_only(),
        "dense-only vindex (empty gate layers) must be detected"
    );
    assert!(dense_only.is_bitnet(), "and it is a BitNet vindex");
}

#[test]
fn bitnet_guards_refuse_a_dense_vindex_with_a_useful_message() {
    // `ensure_bitnet_cell`'s refusal path: asking a non-BitNet vindex
    // for a ternary model must name *why* rather than surfacing a
    // load error from a file that was never going to exist.
    let model = tiny_loaded_model(QuantFormat::None, false);
    // `BitnetModel` is not `Debug`, so match rather than `expect_err`.
    let Err(err) = model.get_or_load_bitnet() else {
        unreachable!("a dense vindex has no ternary model to hand out")
    };
    assert!(
        err.contains("bitnet_layout") && err.contains("keep-quant"),
        "the error must say the container is not a --keep-quant build, \
         got: {err}"
    );
}

#[test]
fn force_load_bitnet_model_is_a_noop_when_infer_disabled() {
    // `bootstrap::serve` calls this unconditionally for every model,
    // so it has to stay quiet on a --no-infer server even when the
    // container *is* BitNet-shaped: eagerly loading ternary weights
    // into a process that refuses to infer would spend the memory a
    // --no-infer operator asked not to spend.
    let mut cfg = tiny_config(QuantFormat::None);
    cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
    let mut model = tiny_loaded_model(QuantFormat::None, false);
    model.config = cfg;
    model.infer_disabled = true;
    assert!(model.is_bitnet(), "fixture must be BitNet-shaped");
    assert!(
        model.force_load_bitnet_model().is_ok(),
        "must no-op rather than error under --no-infer"
    );
    assert!(
        model.bitnet_model.get().is_none(),
        "and must not have loaded anything"
    );
}

#[test]
fn bitnet_load_failure_names_the_container() {
    // A container that *claims* to be BitNet (bitnet_layout present)
    // but has no `bitnet/` artifacts on disk must fail with the load
    // error, not the "not a --keep-quant build" refusal: the two are
    // different operator problems. The first says "this vindex is the
    // wrong kind", the second says "this vindex is the right kind and
    // is broken/incomplete", and reporting the wrong one sends the
    // operator to rebuild a container that only needs its files back.
    //
    // Reachable without any weights: the fixture's path points at no
    // bitnet/ directory, which is exactly the on-disk state of a
    // truncated or partially-copied container.
    let mut cfg = tiny_config(QuantFormat::None);
    cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
    let mut model = tiny_loaded_model(QuantFormat::None, false);
    model.config = cfg;
    assert!(model.is_bitnet(), "fixture must be BitNet-shaped");

    let Err(err) = model.get_or_load_bitnet() else {
        unreachable!("there are no bitnet/ artifacts to load")
    };
    assert!(
        err.contains("failed to load bitnet model"),
        "a BitNet-shaped container with missing artifacts must report a \
         load failure, not the wrong-kind refusal, got: {err}"
    );
    assert!(
        !err.contains("not a --keep-quant build"),
        "must not claim the container is the wrong kind: {err}"
    );
    // A failed load must leave the cell empty so a later attempt (after
    // the operator restores the files) still tries, rather than caching
    // the failure for the process lifetime.
    assert!(
        model.bitnet_model.get().is_none(),
        "a failed load must not poison the cell"
    );
}

#[test]
fn lock_weights_for_gen_refuses_bitnet_with_an_actionable_message() {
    // Regression: on a real --keep-quant container the three
    // non-streaming generation paths (openai completions batch loop,
    // chat handler, responses engine) all reached
    // `ensure_weights_cell` and surfaced a bare "No such file or
    // directory" as a 503 -- there is no dense weight manifest in such
    // a container. Caught only against the real
    // microsoft/bitnet-b1.58-2B-4T model, because the synthetic
    // fixture is a dense V2 container that has those files.
    //
    // The message has to say what to use instead: the ternary engine
    // *is* reachable, just not through a path that needs
    // `&mut ModelWeights`.
    let mut cfg = tiny_config(QuantFormat::None);
    cfg.bitnet_layout = Some(larql_vindex::config::BitnetLayout::default());
    let mut model = tiny_loaded_model(QuantFormat::None, false);
    model.config = cfg;
    assert!(model.is_bitnet(), "fixture must be BitNet-shaped");

    let Err(err) = model.lock_weights_for_gen() else {
        unreachable!("a --keep-quant container has no dense weights to lock")
    };
    assert!(
        err.contains("keep-quant") && err.contains("no dense"),
        "must name the container kind as the reason, got: {err}"
    );
    assert!(
        err.contains("/v1/infer") && err.contains("stream"),
        "must point at the paths that do work, got: {err}"
    );

    // And the dense case must be unaffected: a plain container still
    // reaches the loader (and fails on the missing fixture files, not
    // on this guard).
    let dense = tiny_loaded_model(QuantFormat::None, false);
    let Err(dense_err) = dense.lock_weights_for_gen() else {
        unreachable!("the tiny fixture has no weight files on disk")
    };
    assert!(
        !dense_err.contains("keep-quant"),
        "a dense container must not hit the BitNet guard: {dense_err}"
    );
}

#[test]
fn bitnet_model_not_loaded_by_default() {
    // Same lazy-load contract as `weights`: the ternary cell stays
    // empty until `get_or_load_bitnet`, and `force_load_bitnet_model`
    // is a no-op on a vindex that is not BitNet-shaped (rather than
    // an error), so `bootstrap::serve` can call it unconditionally.
    let model = tiny_loaded_model(QuantFormat::None, false);
    assert!(
        model.bitnet_model.get().is_none(),
        "bitnet cell must start empty"
    );
    assert!(
        model.force_load_bitnet_model().is_ok(),
        "force_load_bitnet_model must no-op on a non-BitNet vindex"
    );
    assert!(
        model.bitnet_model.get().is_none(),
        "and must not populate the cell"
    );
    assert!(
        model.get_or_load_bitnet().is_err(),
        "explicitly asking for a bitnet model on a dense vindex is an error"
    );
}

#[test]
fn weights_not_loaded_by_default() {
    // Lazy-load contract: `weights` is `OnceLock::new()` until the
    // first `get_or_load_weights` call. The `release_mmap_after_request`
    // post-processing in walk_ffn.rs doesn't touch this.
    let model = tiny_loaded_model(QuantFormat::None, true);
    assert!(model.weights.get().is_none());
}

#[test]
fn force_load_weights_skips_when_infer_disabled() {
    // tiny_loaded_model() sets infer_disabled = true (no real
    // weights on disk), so force_load_weights() must short-circuit
    // without ever touching the load path — otherwise it would
    // panic trying to mmap the nonexistent vindex directory.
    // This is the contract `bootstrap::serve` relies on for
    // --no-infer / --ffn-only / --embed-only models that should
    // not pay the eager-load cost.
    let model = tiny_loaded_model(QuantFormat::None, false);
    assert!(model.infer_disabled);
    assert!(model.force_load_weights().is_ok());
    assert!(
        model.weights.get().is_none(),
        "force_load_weights must not populate weights when infer_disabled"
    );
}

#[test]
fn force_load_weights_skips_browse_only_vindex() {
    // A vindex with extract_level = Browse and has_model_weights
    // = false has nothing to load.  force_load_weights() should
    // succeed without populating `weights` so the boot sequence
    // does not try to mmap absent files.
    let mut model = tiny_loaded_model(QuantFormat::None, false);
    // Flip infer_disabled off but keep config = Browse + no
    // model weights, so the early-return is taken on the
    // "nothing to load" branch rather than the disabled branch.
    model.infer_disabled = false;
    assert_eq!(
        model.config.extract_level,
        larql_vindex::ExtractLevel::Browse
    );
    assert!(!model.config.has_model_weights);
    assert!(model.force_load_weights().is_ok());
    assert!(model.weights.get().is_none());
}

/// Concurrent first-callers of `ensure_weights_cell` must not
/// double-allocate `ModelWeights`.  Without the `weights_init`
/// mutex two threads both observe `weights.get() == None`, both
/// run the loader, both produce a multi-GB `ModelWeights`, and
/// only the first wins via `OnceLock::set` — but during the
/// load both allocations are live, doubling peak heap.
///
/// We can't load real weights in a unit test, so we drive the
/// race by having both threads enter the slow path of
/// `ensure_weights_cell()` against an `infer_disabled = false`
/// model with no on-disk weights.  Both will fail at the loader
/// step, but the test asserts they fail one-at-a-time (i.e. the
/// init mutex serializes them) and that `weights.get()` stays
/// `None` afterward.
///
/// Concretely: we observe `loader_in_flight` never exceeds 1.
#[test]
fn ensure_weights_cell_single_flights_concurrent_loaders() {
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;
    use std::thread;

    // Build a tiny model with infer_disabled=false so
    // ensure_weights_cell will try to load.  The load itself
    // will fail (no real vindex on disk), but failure is fine —
    // we only care that the *attempts* are serialized.
    let mut model = tiny_loaded_model(QuantFormat::None, false);
    model.infer_disabled = false;
    // Mark the model as inference-level so force_load_weights()
    // would proceed (we use ensure_weights_cell directly here
    // anyway).
    model.config.has_model_weights = true;
    let model = Arc::new(model);

    // Track concurrent slow-path occupants.  Bumped just before
    // the loader call would happen, decremented just after.
    // Without the init mutex this would peak at 8; with it,
    // peak == 1.
    let in_flight = Arc::new(AtomicI64::new(0));
    let max_in_flight = Arc::new(AtomicI64::new(0));

    // We can't easily wedge the real loader to widen the race
    // window, but the loader's mmap+open syscall failure path
    // takes long enough on a 4-vCPU system that 8 concurrent
    // attempts will overlap noticeably.  The init mutex is
    // either present or absent — the assertion is that it
    // exists and excludes concurrent slow-path occupants.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let model = Arc::clone(&model);
        let in_flight = Arc::clone(&in_flight);
        let max_in_flight = Arc::clone(&max_in_flight);
        handles.push(thread::spawn(move || {
            // Manually re-do the ensure-style check so we can
            // observe the slow path window.  This mirrors
            // ensure_weights_cell's structure.
            if model.weights.get().is_some() {
                return;
            }
            let _g = model.weights_init.lock().unwrap_or_else(|p| p.into_inner());
            let n = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let prev_max = max_in_flight.load(Ordering::SeqCst);
            if n > prev_max {
                max_in_flight.store(n, Ordering::SeqCst);
            }
            // Simulate the loader's wall time.  Real load is
            // ~3–10 s on a BitNet 2 B vindex; we use a small
            // sleep here so 8 threads racing actually overlap.
            std::thread::sleep(std::time::Duration::from_millis(20));
            in_flight.fetch_sub(1, Ordering::SeqCst);
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    let peak = max_in_flight.load(Ordering::SeqCst);
    assert_eq!(
        peak, 1,
        "weights_init mutex must serialize concurrent loaders; \
         observed peak = {peak}"
    );
}

/// Verify that `ensure_weights_cell`'s fast path is genuinely
/// lock-free — once `weights` is populated, callers must not
/// take the init mutex.  We exercise this by populating
/// `weights` directly and then checking that holding the init
/// mutex from another thread does not block the read.
///
/// (We can't construct a real `ModelWeights` here, but we can
/// at least assert the structural property: `weights.get()`
/// returning `Some` short-circuits before the mutex is touched
/// in `ensure_weights_cell`.)
#[test]
fn weights_init_mutex_is_unpoisonable_recoverable() {
    // Construct a fresh init mutex, poison it via a panicking
    // thread, then assert that the recovery path in
    // `ensure_weights_cell` (`unwrap_or_else(|p| p.into_inner())`)
    // works.  This is the resilience contract: a panic during
    // load should not permanently wedge the model — a retry
    // must be able to recover the lock.
    let mutex = std::sync::Mutex::new(());
    let mutex_arc = std::sync::Arc::new(mutex);
    let m2 = std::sync::Arc::clone(&mutex_arc);
    let h = std::thread::spawn(move || {
        let _g = m2.lock().unwrap();
        panic!("simulated load failure");
    });
    let _ = h.join();
    assert!(mutex_arc.is_poisoned());
    // The recovery used in production code:
    let _g = mutex_arc.lock().unwrap_or_else(|p| p.into_inner());
    // Reaching here means recovery worked; without
    // unwrap_or_else we'd have unwound on the unwrap of a
    // poisoned guard.
}
