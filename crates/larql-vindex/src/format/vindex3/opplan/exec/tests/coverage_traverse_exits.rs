//! The batch traversal's exits for the two multi-stream topologies, under
//! every combination of exit parts an image can hold.
//!
//! Preparation hands a whole stack all of its parts, so the combinations
//! it never produces — an exit reduction with no final norm after it, a
//! whole stack missing its reduction — are staged by removing a part
//! from a prepared image through the prepared module's test seams. The
//! first must still reduce; the second must refuse rather than leave
//! through the wrong operation.

use super::super::hyper_connection::Mutation;
use super::super::prepared::tests::attention_residual_sites_for_test;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::reference::ReferenceBackend;
use super::super::{
    execute_prepared_streaming_mutated, FinalOutput, FinalState, Plane, ResumePoint,
};
use super::attn_res_2b_batch::TOKENS;
use super::attn_res_substrate::{prepare as prepare_attn_res, substrate, Substrate, LAYERS};
use super::wave19_hc_decode::prepare as prepare_hc;
use super::wave19_hc_substrate::{self as hc, Oracle, Variant, POSITIONS as HC_POSITIONS};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::ComponentOpPlan;

fn run(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    resume: Option<ResumePoint>,
    mutation: Mutation,
) -> Result<FinalOutput, VindexError> {
    execute_prepared_streaming_mutated(
        plan,
        ops,
        tokens,
        &ReferenceBackend::new(),
        resume,
        &mut |_| Ok(()),
        mutation,
        None,
    )
}

fn attn_res_image(sub: &Substrate, slice: ExecutionSlice) -> PreparedOperands {
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    PreparedOperands::load(&sub.plan, &store, &ReferenceBackend::new(), slice)
        .expect("the attention-residual substrate prepares")
}

/// With no final norm after it, the exit reduction's output IS the final
/// hidden state — and skipping the reduction leaves the bare prefix. The
/// two differ, so the exit is still load-bearing without the norm.
#[test]
fn an_attention_residual_exit_without_a_final_norm_leaves_its_reduction() {
    let sub = substrate();
    let (_store, ops) = prepare_attn_res(&sub);
    let ops = ops.without_final_norm_for_test();
    let reduced = run(&sub.plan, &ops, &TOKENS, None, Mutation::None).unwrap();
    let skipped = run(&sub.plan, &ops, &TOKENS, None, Mutation::AttnResExitSkipped).unwrap();
    assert!(reduced.logits.is_some() && skipped.logits.is_some());
    assert_ne!(reduced.exit, skipped.exit);
}

/// A whole attention-residual stack that has lost its exit reduction
/// refuses at the exit instead of normalising an unreduced history.
#[test]
fn a_whole_attention_residual_stack_without_its_exit_refuses() {
    let sub = substrate();
    let (_store, ops) = prepare_attn_res(&sub);
    let ops = ops.without_attention_residual_exit_for_test();
    let err = run(&sub.plan, &ops, &TOKENS, None, Mutation::None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no exit reduction"), "{err}");
}

/// An attention-residual image with no exit, no final norm and no head
/// hands on its last position's history — the output a stack with
/// nothing after its layers has — and a whole image accounts for its
/// site glue.
#[test]
fn an_attention_residual_image_with_no_exit_parts_hands_on_its_history() {
    let sub = substrate();
    let full = attn_res_image(&sub, ExecutionSlice::Full);
    assert!(full.residency_census().glue.widened_f32 > 0);
    let bare = full
        .without_attention_residual_exit_for_test()
        .without_final_norm_for_test()
        .without_output_for_test();
    let out = run(&sub.plan, &bare, &TOKENS, None, Mutation::None).unwrap();
    assert!(out.logits.is_none());
    assert!(matches!(out.exit, FinalState::History(_)));
    // A layer-range shard of the same stack has no embedding, so it
    // cannot run from token ids at all.
    let shard = attn_res_image(
        &sub,
        ExecutionSlice::LayerRange {
            start: 0,
            end: LAYERS,
        },
    );
    let err = run(&sub.plan, &shard, &TOKENS, None, Mutation::None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no embedding table"), "{err}");
}

/// Every layer of the substrate carries its sites under the declared
/// topology; a layer without them, or with them under an undeclared one,
/// is refused by name.
#[test]
fn attention_residual_sites_must_agree_with_the_declared_topology() {
    let sub = substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let hidden = sub.plan.embedding.as_ref().unwrap().table.shape[1];
    let layer = &sub.plan.layers[0];
    assert!(attention_residual_sites_for_test(layer, true, hidden, (&store).into()).unwrap());

    let err = attention_residual_sites_for_test(layer, false, hidden, (&store).into())
        .unwrap_err()
        .to_string();
    assert!(err.contains("declares no block size"), "{err}");

    let mut bare = layer.clone();
    bare.attention_residual = None;
    let err = attention_residual_sites_for_test(&bare, true, hidden, (&store).into())
        .unwrap_err()
        .to_string();
    assert!(err.contains("carries no attention-residual sites"), "{err}");
}

fn hc_resume() -> ResumePoint {
    let oracle = Oracle::load();
    ResumePoint {
        next_layer: 0,
        hidden: Plane::Bundles((0..HC_POSITIONS).map(|p| oracle.input(p)).collect()),
    }
}

/// With no final norm, the head's reduction is the final hidden state;
/// without the head, a whole stack refuses rather than leaving as a
/// bundle a head was declared to reduce.
#[test]
fn a_hyper_connected_exit_reduces_without_a_norm_and_refuses_without_its_head() {
    let sub = hc::build(Variant::HeadBearing);
    let tokens = [0u32; HC_POSITIONS];
    let ops = prepare_hc(&sub, ExecutionSlice::Full)
        .unwrap()
        .without_final_norm_for_test();
    let out = run(&sub.plan, &ops, &tokens, Some(hc_resume()), Mutation::None).unwrap();
    assert!(out.logits.is_some());
    assert!(matches!(out.exit, FinalState::Hidden(_)));

    let headless = prepare_hc(&sub, ExecutionSlice::Full)
        .unwrap()
        .without_hyper_connection_head_for_test();
    let err = run(
        &sub.plan,
        &headless,
        &tokens,
        Some(hc_resume()),
        Mutation::None,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("no head reduction"), "{err}");
}
