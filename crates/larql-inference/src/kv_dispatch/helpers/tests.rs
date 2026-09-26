//! Sync vs async dispatch parity, plus dispatch edge cases.
//!
//! Parity against the legacy `kv_prefill_run` / `kv_decode_step_run`
//! reference lives in `larql-kv/tests/dispatch_parity.rs` — moved
//! out of this module so it can import both crates without forcing
//! a dev-dep cycle that compiles `larql-inference` twice.

use super::super::KvDispatch;
use super::*;
use crate::ffn::WeightFfn;
use crate::test_utils::make_test_weights;
use larql_compute::CpuBackend;

#[test]
fn multi_step_decode_via_dispatch_keeps_handles_finite() {
    // Three decode steps in sequence — verifies the handle state
    // carries forward correctly across calls (same shape as
    // bit-parity test in larql-kv/tests/dispatch_parity.rs, but
    // self-contained: no legacy reference, just the dispatch path
    // and a finite-ness invariant).
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1];

    let (_, mut handles) = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    for step in 0..3 {
        let token = (2 + step) as u32;
        let abs_position = prompt.len() + step;
        let h_trait = kv_decode_step_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &mut handles,
            token,
            abs_position,
            None,
            None,
        )
        .expect("decode trait")
        .expect("dispatch produced a result");
        assert!(
            h_trait.iter().all(|v| v.is_finite()),
            "step {step} produced non-finite hidden state"
        );
    }
}

#[test]
fn prefill_empty_prompt_is_not_applicable_not_a_refusal() {
    // An empty prompt is nothing to do, which is `Ok(None)`. Pinned as a
    // shape rather than "is not Ok(Some)": collapsing it into `Err` would
    // make the engine report a refusal for a caller-side input condition.
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let result = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &[],
        None,
        None,
    );
    assert!(matches!(result, Ok(None)));
}

// ── Async helper parity ─────────────────────────────────────────

#[test]
fn prefill_async_matches_sync_dispatch() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3];

    let (h_sync, handles_sync) = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");
    let (h_async, handles_async) = kv_prefill_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    assert_eq!(h_sync, h_async, "async prefill hidden must match sync");
    assert_eq!(handles_sync.len(), handles_async.len());
    for (i, (s, a)) in handles_sync.iter().zip(handles_async.iter()).enumerate() {
        let (k_s, v_s) = backend.read_kv_to_host(s).unwrap();
        let (k_a, v_a) = backend.read_kv_to_host(a).unwrap();
        assert_eq!(k_s, k_a, "K mismatch at layer {i}");
        assert_eq!(v_s, v_a, "V mismatch at layer {i}");
    }
}

#[test]
fn prefill_async_windowed_matches_sync_dispatch() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2, 3, 4];
    let window = Some(2);

    let (h_sync, _) = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        window,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");
    let (h_async, _) = kv_prefill_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        window,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    assert_eq!(h_sync, h_async, "windowed async prefill must match sync");
}

#[test]
fn decode_step_async_matches_sync_dispatch() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1, 2];

    let (_, mut handles_sync) = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");
    let (_, mut handles_async) = kv_prefill_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    let next_token = 3u32;
    let abs_position = prompt.len();

    let h_sync = kv_decode_step_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &mut handles_sync,
        next_token,
        abs_position,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");
    let h_async = kv_decode_step_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &mut handles_async,
        next_token,
        abs_position,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    assert_eq!(h_sync, h_async, "async decode_step hidden must match sync");
}

#[test]
fn multi_step_decode_async_matches_sync_dispatch() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let prompt = vec![0u32, 1];

    let (_, mut handles_sync) = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");
    let (_, mut handles_async) = kv_prefill_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &prompt,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    for step in 0..3 {
        let token = (2 + step) as u32;
        let abs_position = prompt.len() + step;
        let h_sync = kv_decode_step_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &mut handles_sync,
            token,
            abs_position,
            None,
            None,
        )
        .unwrap()
        .expect("dispatch produced a result");
        let h_async = kv_decode_step_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &mut handles_async,
            token,
            abs_position,
            None,
            None,
        )
        .unwrap()
        .expect("dispatch produced a result");
        assert_eq!(h_sync, h_async, "step {step} async vs sync must match");
    }
}

#[test]
fn prefill_async_empty_prompt_is_not_applicable_not_a_refusal() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let result = kv_prefill_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &[],
        None,
        None,
    );
    assert!(matches!(result, Ok(None)));
}

// ─── Phase 1d.3a: embed-hoist bit-identity (sync + async) ───────────────
//
// These pin the refactor that landed `kv_prefill_from_hidden_via_dispatch`
// as the new MM-aware entry point. The old `kv_prefill_via_dispatch` is
// now a two-line wrapper that runs `embed_tokens_pub` then delegates.
// Bit-identity of the wrapper-vs-direct path is the contract that makes
// the engine seam (per ADR-0023) safe to land — and we have to verify
// sync and async separately because they diverge on real (non-CPU)
// backends, and on CPU the async path additionally calls flush() (a
// no-op on CpuBackend but a real ordering primitive elsewhere).

#[test]
fn prefill_via_dispatch_bit_identical_to_from_hidden_sync() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let tokens = vec![0u32, 1, 2, 3];

    let (h_text, handles_text) = kv_prefill_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &tokens,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    let initial_hidden = embed_tokens_pub(&weights, &tokens);
    let (h_hidden, handles_hidden) = kv_prefill_from_hidden_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &initial_hidden,
        Some(&tokens),
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    assert_eq!(
        h_text, h_hidden,
        "text and from-hidden paths must produce bit-identical last-row hidden"
    );
    assert_eq!(
        handles_text.len(),
        handles_hidden.len(),
        "handle count (= num_layers) must match across paths"
    );
}

#[test]
fn prefill_via_dispatch_bit_identical_to_from_hidden_async() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let tokens = vec![0u32, 1, 2, 3];

    let (h_text, handles_text) = kv_prefill_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &tokens,
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    let initial_hidden = embed_tokens_pub(&weights, &tokens);
    let (h_hidden, handles_hidden) = kv_prefill_from_hidden_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &initial_hidden,
        Some(&tokens),
        None,
        None,
    )
    .unwrap()
    .expect("dispatch produced a result");

    assert_eq!(
        h_text, h_hidden,
        "async text and from-hidden paths must produce bit-identical last-row hidden"
    );
    assert_eq!(handles_text.len(), handles_hidden.len());
}

#[test]
fn prefill_from_hidden_is_not_applicable_on_empty_input() {
    let weights = make_test_weights();
    let backend = CpuBackend;
    let ffn = WeightFfn { weights: &weights };
    let empty_hidden = Array2::<f32>::zeros((0, weights.hidden_size));
    let result = kv_prefill_from_hidden_via_dispatch(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &empty_hidden,
        None,
        None,
        None,
    );
    assert!(
        matches!(result, Ok(None)),
        "zero-row hidden is not applicable, not a refusal"
    );

    let result_async = kv_prefill_from_hidden_via_dispatch_async(
        &backend,
        larql_models::WeightsView::dense(&weights),
        &ffn,
        &empty_hidden,
        None,
        None,
        None,
    );
    assert!(
        matches!(result_async, Ok(None)),
        "async zero-row hidden is not applicable, not a refusal"
    );
}
