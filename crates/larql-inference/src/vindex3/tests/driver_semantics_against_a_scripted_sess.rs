//! Driver semantics against a scripted session
//! the masked driver (N0.6)
//! Seam parity against the direct DecodeSession harness
//! Continuation state behind the KvState seam (VI3-INF-2)

use super::*;

#[test]
fn the_driver_emits_greedy_ids_in_callback_order() {
    let mut session = ScriptedSession::new(vec![
        row_peaking_at(2),
        row_peaking_at(0),
        row_peaking_at(3),
    ]);
    let mut streamed = Vec::new();
    let result = generate_session(
        &mut session,
        &[7, 8],
        3,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |id| streamed.push(id),
    )
    .unwrap();
    assert_eq!(result.tokens, vec![2, 0, 3]);
    assert_eq!(streamed, result.tokens);
    assert_eq!(result.prompt_len, 2);
    // Prompt (2) + steps for all but the last emitted token (2).
    assert_eq!(session.position(), 4);
}

#[test]
fn a_stop_token_ends_generation_before_being_emitted() {
    let mut session = ScriptedSession::new(vec![row_peaking_at(1), row_peaking_at(3)]);
    let result = generate_session(
        &mut session,
        &[7],
        4,
        SamplingConfig::greedy(),
        &EosConfig::empty().with_eos_id(3),
        |_| {},
    )
    .unwrap();
    // Token 1 emitted; the stop token 3 ended the run without appearing.
    assert_eq!(result.tokens, vec![1]);
}

#[test]
fn the_masked_driver_obeys_the_mask_over_greedy_preference() {
    // Every scripted row peaks at id 2, but the mask only ever admits
    // id 1 — the mask must win, and it must see the generated-so-far
    // history grow.
    let mut session = ScriptedSession::new(vec![row_peaking_at(2), row_peaking_at(2)]);
    let logits = session.prefill(&[7]).unwrap();
    let mut histories = Vec::new();
    let mut mask = |generated: &[u32], logits: &mut Vec<f32>| {
        histories.push(generated.to_vec());
        for (id, logit) in logits.iter_mut().enumerate() {
            if id != 1 {
                *logit = f32::NEG_INFINITY;
            }
        }
    };
    let result = continue_session_masked(
        &mut session,
        logits,
        2,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        &mut mask,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.tokens, vec![1, 1]);
    assert_eq!(histories, vec![vec![], vec![1]]);
}

#[test]
fn mask_exhaustion_before_first_emission_is_an_error() {
    let mut session = ScriptedSession::new(vec![row_peaking_at(0)]);
    let logits = session.prefill(&[7]).unwrap();
    let mut mask =
        |_: &[u32], logits: &mut Vec<f32>| logits.iter_mut().for_each(|l| *l = f32::NEG_INFINITY);
    let err = continue_session_masked(
        &mut session,
        logits,
        2,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        &mut mask,
        |_| {},
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("sampler produced no token"),
        "{err}"
    );
}

#[test]
fn mask_exhaustion_after_emission_is_a_natural_stop() {
    // The grammar completing (nothing admissible any more) mirrors the
    // V2 constrained driver: a clean stop, not an error.
    let mut session = ScriptedSession::new(vec![row_peaking_at(0), row_peaking_at(0)]);
    let logits = session.prefill(&[7]).unwrap();
    let mut calls = 0usize;
    let mut mask = |_: &[u32], logits: &mut Vec<f32>| {
        calls += 1;
        if calls > 1 {
            logits.iter_mut().for_each(|l| *l = f32::NEG_INFINITY);
        }
    };
    let result = continue_session_masked(
        &mut session,
        logits,
        4,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        &mut mask,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.tokens, vec![0], "one emission, then a clean stop");
}

#[test]
fn a_zero_token_budget_prefills_but_emits_nothing() {
    let mut session = ScriptedSession::new(vec![row_peaking_at(1)]);
    let mut fired = false;
    let result = generate_session(
        &mut session,
        &[7, 8, 9],
        0,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| fired = true,
    )
    .unwrap();
    assert!(result.tokens.is_empty());
    assert!(!fired);
    // The prompt was still consumed — a follow-up call can continue.
    assert_eq!(session.position(), 3);
}

#[test]
fn non_finite_logits_surface_as_an_error_not_a_token() {
    let mut session = ScriptedSession::new(vec![vec![f32::NAN, f32::NAN]]);
    let err = generate_session(
        &mut session,
        &[7],
        2,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| {},
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("sampler produced no token"),
        "{err}"
    );
}

#[test]
fn reference_runtime_matches_the_decode_harness_bit_for_bit() {
    assert_seam_parity(&ReferenceBackend::new(), ReferenceBackend::new());
}

#[test]
fn production_runtime_matches_the_decode_harness_bit_for_bit() {
    assert_seam_parity(&ProductionBackend::new(), ProductionBackend::new());
}

/// Control: the instrument must fail on known-different input. A
/// reversed prompt is a different program input, so its prefill logits
/// must differ from the harness's — otherwise the bit-equality above
/// is vacuous.
#[test]
fn the_parity_instrument_detects_a_diverged_prompt() {
    let container = container_with(miniature_glimmer);
    let backend = ReferenceBackend::new();
    let (_, harness_rows) = harness_decode(container.path(), &backend, &G_TOKENS, 1);

    let runtime = Vindex3Runtime::open(container.path(), COMPONENT, backend).unwrap();
    let mut session = runtime.session(&row(runtime.plan())).unwrap();
    let mut reversed = G_TOKENS.to_vec();
    reversed.reverse();
    assert_ne!(reversed, G_TOKENS.to_vec());
    let prefill = session.prefill(&reversed).unwrap();
    assert_ne!(
        prefill, harness_rows[0],
        "instrument cannot distinguish different prompts"
    );
}

/// The dense Llama-shaped anatomy through the same seam — a second,
/// independent plan shape (two norms, RoPE everywhere, no gates).
#[test]
fn dense_runtime_matches_the_decode_harness_bit_for_bit() {
    let container = container_with(dense_f32_model);
    let backend = ReferenceBackend::new();
    let prompt = [5u32, 99, 42];
    let (harness_ids, _) = harness_decode(container.path(), &backend, &prompt, NEW_TOKENS);

    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();
    assert_eq!(runtime.plan().component, COMPONENT);
    assert!(!runtime.backend().name().is_empty());
    let mut session = runtime.session(&row(runtime.plan())).unwrap();
    let result = generate_session(
        &mut session,
        &prompt,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| {},
    )
    .unwrap();
    assert_eq!(result.tokens, harness_ids);
}

/// Caller-owned continuation state must change nothing about the
/// generated ids, and must still hold every row after the session is
/// gone — with per-layer geometry matching the plan's own statement
/// ([`plan_kv_geometry`]), which is the whole point of the seam: KV
/// policy reads the program, not a family registry.
#[test]
fn a_caller_owned_kv_state_generates_identically_and_outlives_the_session() {
    let container = container_with(miniature_glimmer);
    let runtime =
        Vindex3Runtime::open(container.path(), COMPONENT, ReferenceBackend::new()).unwrap();

    let mut default_session = runtime.session(&row(runtime.plan())).unwrap();
    let baseline = generate_session(
        &mut default_session,
        &G_TOKENS,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_| {},
    )
    .unwrap();

    let mut kv = RowKvState::default();
    {
        let mut session = runtime.session_with_kv(&mut kv).unwrap();
        let provided = generate_session(
            &mut session,
            &G_TOKENS,
            NEW_TOKENS,
            SamplingConfig::greedy(),
            &EosConfig::empty(),
            |_| {},
        )
        .unwrap();
        assert_eq!(provided, baseline, "caller-owned KV diverged the ids");
    }

    // The session is gone; the caller still holds the conversation's
    // continuation state, one row pair per consumed position, at the
    // width the plan declares per layer.
    let geometry = plan_kv_geometry(runtime.plan());
    let consumed = G_TOKENS.len() + NEW_TOKENS - 1;
    for (layer, geo) in geometry.iter().enumerate() {
        assert_eq!(kv.rows(layer).to_owned_rows().0.len(), consumed);
        assert_eq!(kv.rows(layer).to_owned_rows().1.len(), consumed);
        assert!(kv
            .rows(layer)
            .to_owned_rows()
            .0
            .iter()
            .all(|row| row.len() == geo.kv_dim));
    }
    // …and the miniature's sliding+full split is explicit in that
    // geometry — no ModelArchitecture consulted anywhere in this test.
    assert_eq!(geometry[0].window, Some(3));
    assert_eq!(geometry[1].window, None);
}

#[test]
fn reference_batch_prefill_resumes_to_the_harness_ids() {
    assert_prefill_resume_matches_harness(&ReferenceBackend::new(), ReferenceBackend::new());
}

#[test]
fn production_batch_prefill_resumes_to_the_harness_ids() {
    assert_prefill_resume_matches_harness(&ProductionBackend::new(), ProductionBackend::new());
}
