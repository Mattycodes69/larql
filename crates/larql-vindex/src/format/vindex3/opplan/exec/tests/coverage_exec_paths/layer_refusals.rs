//! Every stateful operator refuses, inside the layer, to run without a
//! continuation provider — the traversal's pre-flight refuses first, and
//! these pin the second line for when a caller reaches a layer directly.

use super::super::super::hyper_connection::Mutation;
use super::super::super::kv::ContinuationProvider;
use super::super::super::layer_exec::execute_layer;
use super::super::super::trace::Plane;
use super::super::super::PlaneEvent;
use super::*;
use crate::error::VindexError;
use crate::format::vindex3::opplan::tests::conv_qkv::miniature_hybrid;
use crate::format::vindex3::opplan::tests::kda_mla_exec::miniature_kimi;
use crate::format::vindex3::opplan::tests::mamba2::miniature_mamba2;
use crate::format::vindex3::opplan::LayerAttention;

/// Positions in the batch each layer is run over.
const POSITIONS: usize = 2;

/// Runs `layer` of `fixture` over a zero row plane with NO provider.
fn run_without_provider(fixture: &Prepared, layer: usize) -> Result<(), VindexError> {
    let hidden = fixture.ops.hidden();
    let mut h = Plane::Rows(vec![vec![0.0; hidden]; POSITIONS]);
    let mut sink = |_: PlaneEvent| Ok(());
    // The provider type the traversal itself instantiates the layer with,
    // so this exercises the same monomorphisation production runs.
    execute_layer::<_, dyn ContinuationProvider>(
        &fixture.plan.layers[layer],
        &fixture.ops.layers()[layer],
        &mut h,
        hidden,
        &ReferenceBackend,
        None,
        None,
        None,
        layer,
        &mut sink,
        Mutation::None,
    )
    .map(|_| ())
}

/// The first layer of `fixture` whose attention `is` the operator.
fn first_layer(fixture: &Prepared, is: fn(&LayerAttention) -> bool) -> usize {
    fixture
        .plan
        .layers
        .iter()
        .position(|l| is(&l.attention))
        .expect("the fixture carries the operator")
}

fn assert_refused(fixture: &Prepared, layer: usize, needle: &str) {
    let err = run_without_provider(fixture, layer)
        .expect_err("a stateful layer must not run without a provider")
        .to_string();
    assert!(err.contains(needle), "layer {layer}: {err}");
}

const NO_RECURRENCE_PROVIDER: &str = "needs durable continuation state";

#[test]
fn a_mamba2_mixer_refuses_to_run_without_a_provider() {
    let fixture = prepared(|d| miniature_mamba2(d, None));
    let layer = first_layer(&fixture, |a| matches!(a, LayerAttention::Mamba2(_)));
    assert_refused(&fixture, layer, NO_RECURRENCE_PROVIDER);
}

#[test]
fn kda_and_mla_layers_refuse_to_run_without_a_provider() {
    let fixture = prepared(miniature_kimi);
    let kda = first_layer(&fixture, |a| matches!(a, LayerAttention::Kda(_)));
    assert_refused(&fixture, kda, NO_RECURRENCE_PROVIDER);
    let mla = first_layer(&fixture, |a| matches!(a, LayerAttention::Mla(_)));
    assert_refused(&fixture, mla, "keeps a per-position latent cache");
}

#[test]
fn a_conv_qkv_layer_refuses_to_run_without_a_provider() {
    let fixture = prepared(miniature_hybrid);
    let conv = first_layer(&fixture, |a| matches!(a, LayerAttention::ConvQkv(_)));
    assert_refused(&fixture, conv, "keeps a KV cache and a conv history");
}
