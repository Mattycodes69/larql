//! mechanism subjects (fixtures)

use super::*;

#[test]
fn mechanism_dense_softmax() {
    let _serial = serial();
    let subject = subjects::fixture(dense_f32_model, "mem1-dense");
    let journey = Journey {
        prefill: (1..9).collect(),
        resume: vec![20, 21, 22, 23],
        decode: vec![30, 31, 32, 33],
    };
    measure_subject(
        &subject,
        &ReferenceBackend::new(),
        &journey,
        false,
        Value::Null,
    );
}

#[test]
fn mechanism_sliding_window() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9],
        decode: vec![1, 2, 3, 4],
    };
    let backend = ReferenceBackend::new();
    let mut prefix = G_TOKENS.to_vec();
    prefix.extend([5, 9]);
    let witness = output_dependence(&subject, &backend, &prefix, 1);
    assert!(
        prefix.len() > G_WINDOW,
        "the witness state must cross the window"
    );
    measure_subject(
        &subject,
        &backend,
        &journey,
        false,
        json!({ "I2_output_dependence": witness }),
    );
}

#[test]
fn mechanism_kda_mla_hybrid() {
    let _serial = serial();
    let subject = subjects::fixture(hybrid_kda_mla_f32_model, "mem1-kda-mla");
    let journey = Journey {
        prefill: vec![1, 2, 3, 4, 5, 6],
        resume: vec![7, 8, 9],
        decode: vec![10, 11, 12],
    };
    let mla = mla_projection_counts(&subject, &journey);
    measure_subject(
        &subject,
        &ReferenceBackend::new(),
        &journey,
        true,
        json!({ "I5_mla": mla }),
    );
}

/// Gated DeltaNet (D6): the fourth recurrent operator, for M8's
/// per-call copy clause on the serial reference backend.
#[test]
fn mechanism_gated_delta_hybrid() {
    let _serial = serial();
    let subject = subjects::fixture(hybrid_lllf_f32_model, "mem1-gated-delta");
    let journey = Journey {
        prefill: vec![1, 2, 3, 4, 5, 6],
        resume: vec![7, 8, 9],
        decode: vec![10, 11, 12],
    };
    measure_subject(
        &subject,
        &ReferenceBackend::new(),
        &journey,
        true,
        Value::Null,
    );
}
