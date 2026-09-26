//! The engine's handling of every dispatch outcome on both backend
//! slots: hidden-row decode on sync and async, a backend that declines
//! (the bare `None` arm), and the bookkeeping guards around the
//! position map.

use super::*;
use larql_compute::kv_dispatch::{KvDispatch, KvHandle};
use larql_compute::{ComputeBackend, DecodeBackend, MatMul, QuantMatVec};
use ndarray::{Array2, ArrayView2};

const PROMPT: [u32; 3] = [0, 1, 2];

/// Runs prefill on the CPU but declines every decode step, the way a
/// backend without a step kernel answers. Everything a declined step
/// then needs (truncate for the rewind) is delegated.
struct DecliningStepBackend;

impl MatMul for DecliningStepBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        CpuBackend.matmul(a, b)
    }
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        CpuBackend.matmul_transb(a, b)
    }
}
impl QuantMatVec for DecliningStepBackend {}
impl DecodeBackend for DecliningStepBackend {}
impl ComputeBackend for DecliningStepBackend {
    fn name(&self) -> &str {
        "declining-step"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl KvDispatch for DecliningStepBackend {
    fn attention_prefill(
        &self,
        weights: larql_inference::WeightsView,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&dyn larql_compute::KvIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CpuBackend.attention_prefill(weights, tokens_embedded, layer, window, index)
    }
    fn truncate_kv(&self, handle: &mut KvHandle, len: usize) -> bool {
        CpuBackend.truncate_kv(handle, len)
    }
}

fn async_engine() -> StandardEngine {
    StandardEngine::with_async_backend(None, Box::new(CpuBackend))
}

fn embedded(weights: &larql_inference::ModelWeights, ids: &[u32]) -> Array2<f32> {
    larql_inference::forward::embed_tokens_pub(weights, ids)
}

#[test]
fn hidden_row_decode_matches_token_decode_on_the_sync_slot() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let next = PROMPT.len() as u32;

    let mut by_token = StandardEngine::new(None);
    by_token.prefill(&weights, &ffn, &PROMPT).unwrap();
    let want = by_token.decode_step(&weights, &ffn, next).unwrap();

    let mut by_hidden = StandardEngine::new(None);
    by_hidden.prefill(&weights, &ffn, &PROMPT).unwrap();
    let got = by_hidden
        .decode_step_from_hidden(&weights, &ffn, &embedded(&weights, &[next]))
        .unwrap();
    assert_eq!(got, want, "tinymodel has no PLE, so the two inputs agree");
    assert_eq!(by_hidden.abs_position, PROMPT.len() + 1);
}

#[test]
fn hidden_row_decode_matches_token_decode_on_the_async_slot() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let next = PROMPT.len() as u32;

    let mut by_token = async_engine();
    by_token.prefill(&weights, &ffn, &PROMPT).unwrap();
    let want = by_token.decode_step(&weights, &ffn, next).unwrap();

    let mut by_hidden = async_engine();
    by_hidden.prefill(&weights, &ffn, &PROMPT).unwrap();
    let got = by_hidden
        .decode_step_from_hidden(&weights, &ffn, &embedded(&weights, &[next]))
        .unwrap();
    assert_eq!(got, want);
    assert_eq!(by_hidden.abs_position, PROMPT.len() + 1);
}

#[test]
fn a_declined_decode_step_is_a_backend_failure_and_rewinds_the_cache() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::with_backend(None, Box::new(DecliningStepBackend));
    engine.prefill(&weights, &ffn, &PROMPT).unwrap();
    let rows_before = engine.window_tokens();

    let err = engine.decode_step(&weights, &ffn, 0).unwrap_err();
    assert!(
        matches!(err, EngineError::BackendFailure { .. }),
        "a declining backend is a failure, not a refusal: {err:?}"
    );
    assert_eq!(engine.abs_position, PROMPT.len(), "position must not move");
    assert_eq!(engine.window_tokens(), rows_before, "cache rewound");
    assert!(
        engine.invalidated.is_none(),
        "an exact rewind keeps it live"
    );
}

#[test]
fn a_backend_that_declines_prefill_is_a_backend_failure() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    // A default `KvDispatch` answers no intent at all.
    struct SilentBackend;
    impl MatMul for SilentBackend {
        fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            CpuBackend.matmul(a, b)
        }
        fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
            CpuBackend.matmul_transb(a, b)
        }
    }
    impl QuantMatVec for SilentBackend {}
    impl DecodeBackend for SilentBackend {}
    impl ComputeBackend for SilentBackend {
        fn name(&self) -> &str {
            "silent"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    impl KvDispatch for SilentBackend {}

    let mut engine = StandardEngine::with_backend(None, Box::new(SilentBackend));
    let err = engine.prefill(&weights, &ffn, &PROMPT).unwrap_err();
    assert!(matches!(err, EngineError::BackendFailure { .. }), "{err:?}");
    assert!(engine.handles.is_none(), "no cache from a declined prefill");
}

#[test]
fn empty_input_reaching_the_async_prefill_body_is_a_backend_failure() {
    // `prefill` refuses an empty prompt up front; the shared body keeps
    // its own answer for the case, on both slots.
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let err = async_engine()
        .do_prefill(&weights, &ffn, &[], None)
        .unwrap_err();
    assert!(matches!(err, EngineError::BackendFailure { .. }), "{err:?}");
}

#[test]
fn a_zero_row_hidden_prefill_is_a_backend_failure() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let empty = Array2::<f32>::zeros((0, weights.hidden_size));
    let err = StandardEngine::new(None)
        .prefill_from_hidden(&weights, &ffn, &empty)
        .unwrap_err();
    assert!(matches!(err, EngineError::BackendFailure { .. }), "{err:?}");
}

#[test]
fn decode_append_bookkeeping_is_a_noop_without_a_per_layer_cache() {
    // Before any prefill there is no per-layer map to extend.
    let mut fresh = StandardEngine::new(None);
    fresh.record_decode_append(0);
    assert!(fresh.row_positions.max_position().is_none());

    // Per-layer mode recorded but the handles gone: nothing to mirror.
    let mut handleless = StandardEngine::new(None);
    handleless.prefill_mode = Some(PrefillDispatchMode::PerLayer);
    handleless.record_decode_append(0);
    assert!(handleless.row_positions.max_position().is_none());
}

#[test]
fn decode_append_skips_layers_the_map_does_not_describe() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    engine.prefill(&weights, &ffn, &PROMPT).unwrap();
    engine.row_positions = Default::default();
    engine.record_decode_append(PROMPT.len() as u64);
    assert!(
        engine.row_positions.max_position().is_none(),
        "no layer entry is invented for an unmapped layer"
    );
}

#[test]
fn coarse_mode_exposes_no_per_layer_handles() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    engine.prefill(&weights, &ffn, &PROMPT).unwrap();
    assert!(engine.layer_handles().is_some());
    engine.prefill_mode = Some(PrefillDispatchMode::Coarse);
    assert!(engine.layer_handles().is_none());
}
