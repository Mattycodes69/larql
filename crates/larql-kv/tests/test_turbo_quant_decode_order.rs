//! Decoding before prefill is an engine-usage error the engine refuses,
//! not an out-of-bounds index into its layer caches.

use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_kv::{AnyEngine, EngineKind};

#[test]
fn turbo_quant_decode_before_prefill_is_refused_not_a_panic() {
    let weights = make_test_weights();
    let AnyEngine::Kv(mut engine) = EngineKind::from_name("turbo-quant")
        .unwrap()
        .build(larql_inference::cpu_engine_backend())
    else {
        panic!("turbo-quant is a KV engine");
    };
    let ffn = WeightFfn { weights: &weights };
    let err = engine
        .decode_step(&weights, &ffn, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("before prefill"), "{err}");
}
