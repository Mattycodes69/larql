//! VIEW-1 V4: B (bounded retention) and S3 end to end

use super::*;

#[test]
fn b_exact_retention_on_the_sliding_fixture() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9],
        decode: vec![1, 2, 3, 4],
    };
    let record = retention_proof(&subject, &ReferenceBackend::new(), &journey);
    let sliding = &record["resident"][0];
    assert_eq!(sliding["window"], G_WINDOW);
    assert_eq!(
        sliding["resident_rows"], G_WINDOW,
        "a sliding layer holds its window"
    );
    assert_eq!(
        record["resident"][1]["resident_rows"], record["resident"][1]["end"],
        "a full layer holds everything"
    );
}

/// S3 end to end: a provider that drops a row the plan still requires is
/// refused by name at the caller of the step — and no attention kernel runs.
#[test]
fn s3_a_retention_bug_is_refused_before_any_kernel() {
    use larql_vindex::format::vindex3::opplan::exec::prefill_prepared;
    use std::sync::atomic::Ordering;
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let backend = counting::Counting::new(ReferenceBackend::new());
    let ops = subject.prepare(&backend);
    let mut kv = retaining::ExactRetention::default();
    prefill_prepared(&subject.plan, &ops, &G_TOKENS, &backend, &mut kv).unwrap();
    // A healthy step, then the seeded bug on the next step's layer-0 append.
    DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv)
        .unwrap()
        .step(1)
        .unwrap();
    kv.arm_violation();
    DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv)
        .unwrap()
        .step(2)
        .unwrap();
    assert!(
        kv.checks.iter().any(|c| !c.holds()),
        "the seed must have dropped a required row"
    );

    // Decode: the refusal reaches the caller; no kernel ran.
    let before = backend.attention_steps.load(Ordering::SeqCst);
    let err = DecodeSession::over_prepared(&subject.plan, &ops, &backend, &mut kv)
        .unwrap()
        .step(3)
        .err()
        .expect("a step lacking a required row must be refused");
    assert_eq!(
        backend.attention_steps.load(Ordering::SeqCst),
        before,
        "no attention kernel may run"
    );
    let message = err.to_string();
    assert!(
        message.contains("the plan requires"),
        "the refusal must be named: {message}"
    );

    // Resumed prefill takes the same door.
    let before = backend.attention_steps.load(Ordering::SeqCst);
    let err = prefill_prepared(&subject.plan, &ops, &[7, 8], &backend, &mut kv)
        .expect_err("a resumed prefill lacking a required row must be refused");
    assert_eq!(
        backend.attention_steps.load(Ordering::SeqCst),
        before,
        "no attention kernel may run"
    );
    assert!(err.to_string().contains("the plan requires"));
}

#[test]
#[ignore = "real container: LARQL_MEM1_GEMMA (B past the sliding window)"]
fn real_retention_gemma3_4b() {
    let _serial = serial();
    let dir = std::env::var_os("LARQL_MEM1_GEMMA").expect("set LARQL_MEM1_GEMMA");
    let subject = subjects::open(std::path::Path::new(&dir), "gemma3-4b-it");
    let journey = Journey {
        prefill: tokens(1000, 1100),
        resume: tokens(3000, 16),
        decode: tokens(4000, 8),
    };
    let record = retention_proof(&subject, &ProductionBackend::new(), &journey);
    for layer in record["resident"].as_array().unwrap() {
        if let Some(w) = layer["window"].as_u64() {
            assert_eq!(layer["resident_rows"].as_u64(), Some(w), "{layer}");
        }
    }
}
