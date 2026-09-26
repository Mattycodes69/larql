//! The positive witness
//! The controls

use super::*;

#[test]
fn the_decode_traversal_runs_the_bundle_and_the_witness_holds() {
    assert_all_hold(&witness_headless(Mutation::None));
}

#[test]
fn the_hybrid_ffn_receives_the_reduced_vector() {
    assert_all_hold(&witness_hybrid(Mutation::None));
}

/// The embedding enters replicated: at layer 0's attention site the
/// reduced vector is `(Σ pre) · e` for the row the token embeds to.
#[test]
fn the_embedding_enters_every_stream_and_the_head_reduces_the_exit() {
    let sub = substrate::build(Variant::HeadBearing);
    let oracle = Oracle::load();
    let tokens = [2u32, 5, 1];
    let run = run_from_tokens(&sub, &tokens, Mutation::None);
    for (position, &token) in tokens.iter().enumerate() {
        let record = run.witness.record(0, HcSite::Attention, position).unwrap();
        let e = substrate::llama_embedding_row(token);
        let weight: f32 = record.split.pre.iter().sum();
        let expected: Vec<f32> = e.iter().map(|v| v * weight).collect();
        close(&record.reduced, &expected, "replicated embedding").unwrap();
    }
    // The exit: the bundle after the last layer reduces through the
    // head's own operation (the oracle's head weights), then the final
    // norm and the output head run as they always have.
    let head = oracle.head();
    let hc = oracle.topology();
    for step in &run.steps {
        let bundle = step.bundle.as_ref().expect("a bundle left the stack");
        let expected = head_reduce(
            bundle.as_flat(),
            STREAMS,
            HIDDEN,
            &head.weights(),
            NORM_EPS,
            hc.sinkhorn_eps,
        );
        close(step.exit.as_deref().unwrap(), &expected, "head reduction").unwrap();
        assert_eq!(step.logits.as_ref().unwrap().len(), VOCAB);
    }
    assert!(run
        .witness
        .events
        .iter()
        .any(|e| matches!(e, StepEvent::Logits { vocab } if *vocab == VOCAB)));
}

/// Observation cannot fork the semantics: the unobserved step and the
/// witnessed step produce the same logits, bit for bit.
#[test]
fn the_witnessed_step_is_the_unobserved_step() {
    let sub = substrate::build(Variant::HeadBearing);
    let ops = prepare(&sub, ExecutionSlice::Full).unwrap();
    let backend = ReferenceBackend::new();
    let tokens = [3u32, 0, 6];
    let mut kv = RowKvState::default();
    let mut plain = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let unobserved: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| {
            plain
                .step_observed(t, &mut NoopObserver)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect();
    let mut kv = RowKvState::default();
    let mut seen = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let observed: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| seen.step_observed(t, &mut witness).unwrap().logits.unwrap())
        .collect();
    assert_eq!(unobserved, observed);
    assert_eq!(witness.records.len(), LAYERS * SITES * tokens.len());
}

#[test]
fn mutant_a_bypassing_the_composition_is_caught_by_a1_and_a5() {
    assert_caught_by(
        &witness_headless(Mutation::BypassComposition),
        &["A1", "A5"],
    );
}

#[test]
fn mutant_b_one_sinkhorn_iteration_is_caught_by_a1() {
    let results = witness_headless(Mutation::SingleIteration);
    assert_caught_by(&results, &["A1"]);
    // The defect is IN the split, so the expansion is consistent with
    // the (wrong) split it reports: A2 must not be the one that catches
    // it, or the assertions are not measuring what they claim.
    assert_all_hold(&results[1..2]);
}

#[test]
fn mutant_b_a_uniform_reduction_is_caught_by_a1() {
    assert_caught_by(&witness_headless(Mutation::UniformReduction), &["A1"]);
}

#[test]
fn mutant_b_a_transposed_combination_is_caught_by_a2_and_not_a1() {
    let results = witness_headless(Mutation::TransposedCombination);
    assert_caught_by(&results, &["A2"]);
    // The split reported is the correct one — only the expansion lies.
    assert_all_hold(&results[..1]);
}

#[test]
fn mutant_c_pre_norm_on_a_stream_is_caught_by_a3_alone() {
    let results = witness_headless(Mutation::PreNormOnStreamZero);
    assert_caught_by(&results, &["A3"]);
    // The split, the reduction and the expansion are all correct; only
    // what the operator SAW is wrong, and only A3 looks there.
    assert_all_hold(&results[..1]);
}

#[test]
fn mutant_c_pre_norm_on_the_stream_mean_is_caught_by_a3() {
    assert_caught_by(&witness_headless(Mutation::PreNormOnStreamMean), &["A3"]);
}

#[test]
fn mutant_d_hybrid_residual_from_a_stream_is_caught_by_a4() {
    let results = witness_hybrid(Mutation::HybridResidualFromStreamZero);
    assert_caught_by(&results, &["A4"]);
    // The attention site is untouched by an FFN-site defect.
    assert_all_hold(&results[..1]);
}

#[test]
fn mutant_d_hybrid_residual_from_the_stream_mean_is_caught_by_a4() {
    assert_caught_by(
        &witness_hybrid(Mutation::HybridResidualFromStreamMean),
        &["A4"],
    );
}

/// Mutant (d) has nothing to bite on a dense FFN: the dense path reads
/// only its normed input, so the residual argument is inert there. The
/// hybrid estate is what makes the defect observable — recorded so the
/// choice of substrate is a measured fact, not an assumption.
#[test]
fn mutant_d_is_invisible_on_the_dense_estate() {
    assert_all_hold(&witness_headless(Mutation::HybridResidualFromStreamZero));
}
