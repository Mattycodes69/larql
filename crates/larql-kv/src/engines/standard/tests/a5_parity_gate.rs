//! A5 parity gate

use super::*;

#[cfg(not(windows))]
#[test]
fn async_parity_standard_unbounded_matches_sync_engine() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = &[2u32, 3, 5, 7];
    let max = 6;
    let sync = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
    let asynch = run_engine_async(&weights, &tokenizer, &ffn, prompt, max, None);
    assert_eq!(
        sync, asynch,
        "with_async_backend must produce identical tokens to with_backend (CpuBackend, window=None)"
    );
}

#[test]
fn async_parity_standard_windowed_matches_sync_engine() {
    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt = &[1u32, 2, 3, 4, 5];
    let max = 5;
    let window = Some(3);
    let sync = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
    let asynch = run_engine_async(&weights, &tokenizer, &ffn, prompt, max, window);
    assert_eq!(
        sync, asynch,
        "with_async_backend must produce identical tokens to with_backend (CpuBackend, sliding window)"
    );
}

#[test]
fn async_engine_reports_backend_name() {
    let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
    let engine = StandardEngine::with_async_backend(None, backend);
    // info() reports the underlying ComputeBackend::name() regardless
    // of which slot variant the engine holds. CpuBackend returns
    // "cpu (BLAS + C Q4 kernel)" or similar — just assert the prefix.
    assert!(
        engine.info().backend.starts_with("cpu"),
        "expected backend name to start with \"cpu\", got {:?}",
        engine.info().backend
    );
}

/// Multi-step parity proof: 64 decode steps through both sync and
/// async dispatch, asserting that *every* intermediate hidden state
/// is bit-identical. Catches subtle drift that the short-run tests
/// above would miss — e.g. a one-time K/V append difference, a
/// per-step accumulating error, a divergence that only surfaces
/// after many steps.
///
/// This is the accuracy proof for A5: with `Ready*`-wrapped CPU
/// async, the two paths must produce identical output over a long
/// generation, not just a 4-token sample.
#[cfg(not(windows))]
#[test]
fn async_parity_long_run_no_drift() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..16).collect();
    let max_steps = 64;

    let mut sync_engine = StandardEngine::new(None);
    let sync_h0 = sync_engine
        .prefill(&weights, &ffn, &prompt)
        .expect("sync prefill");

    let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
    let mut async_engine = StandardEngine::with_async_backend(None, backend);
    let async_h0 = async_engine
        .prefill(&weights, &ffn, &prompt)
        .expect("async prefill");

    assert_eq!(
        sync_h0, async_h0,
        "prefill hidden must match bit-for-bit between sync and async dispatch"
    );

    let mut token = 1u32;
    for step in 0..max_steps {
        let sync_h = sync_engine
            .decode_step(&weights, &ffn, token)
            .expect("sync decode_step");
        let async_h = async_engine
            .decode_step(&weights, &ffn, token)
            .expect("async decode_step");
        assert_eq!(
            sync_h, async_h,
            "hidden mismatch at decode step {step} (token={token})"
        );
        let logits = larql_inference::forward::hidden_to_raw_logits(&weights, &sync_h);
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    assert_eq!(
        sync_engine.window_tokens(),
        async_engine.window_tokens(),
        "post-run cache size must match"
    );
    assert_eq!(
        sync_engine.memory_bytes(),
        async_engine.memory_bytes(),
        "post-run cache memory must match"
    );
}

/// Sliding-window variant of the long-run parity test. Different
/// code path through `clip_kv` per step; same accuracy contract.
#[cfg(not(windows))]
#[test]
fn async_parity_long_run_windowed_no_drift() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();
    let max_steps = 64;
    let window = Some(4);

    let mut sync_engine = StandardEngine::new(window);
    sync_engine.prefill(&weights, &ffn, &prompt).unwrap();

    let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
    let mut async_engine = StandardEngine::with_async_backend(window, backend);
    async_engine.prefill(&weights, &ffn, &prompt).unwrap();

    let mut token = 1u32;
    for step in 0..max_steps {
        let sync_h = sync_engine.decode_step(&weights, &ffn, token).unwrap();
        let async_h = async_engine.decode_step(&weights, &ffn, token).unwrap();
        assert_eq!(
            sync_h, async_h,
            "windowed hidden mismatch at decode step {step}"
        );
        let logits = larql_inference::forward::hidden_to_raw_logits(&weights, &sync_h);
        token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    // Sliding window clips at the helper level — both should end at
    // the same window size.
    assert_eq!(sync_engine.window_tokens(), async_engine.window_tokens());
}
