//! `boundary-per-layer` with no `layers=N` takes the served model's depth;
//! no model family's layer count is assumed.

use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_kv::{AnyEngine, EngineKind};

#[test]
fn an_undeclared_depth_is_the_served_models() {
    let weights = make_test_weights();
    let kind = EngineKind::from_name("boundary-per-layer").expect("parses");
    let AnyEngine::Kv(mut engine) = kind.build(larql_inference::cpu_engine_backend()) else {
        panic!("boundary-per-layer is a KV engine");
    };
    let ffn = WeightFfn { weights: &weights };
    engine.prefill(&weights, &ffn, &[0, 1]).expect("prefill");
    let config = engine.info().config;
    assert!(
        config.ends_with(&format!("layers={}", weights.num_layers)),
        "policy not sized to the model: {config}"
    );
}
