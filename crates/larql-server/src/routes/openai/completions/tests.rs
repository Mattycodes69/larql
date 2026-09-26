use super::*;

#[test]
fn deserialize_single_string_prompt() {
    let json = serde_json::json!({"prompt": "hello"});
    let req: CompletionsRequest = serde_json::from_value(json).unwrap();
    match req.prompt {
        CompletionPrompt::Single(s) => assert_eq!(s, "hello"),
        _ => panic!(),
    }
}

#[test]
fn deserialize_string_array_prompt() {
    let json = serde_json::json!({"prompt": ["a", "b"]});
    let req: CompletionsRequest = serde_json::from_value(json).unwrap();
    match req.prompt {
        CompletionPrompt::Batch(v) => assert_eq!(v, vec!["a", "b"]),
        _ => panic!(),
    }
}

fn toks(parts: &[&str]) -> Vec<(String, f64)> {
    parts.iter().map(|s| (s.to_string(), 1.0)).collect()
}

#[test]
fn build_text_completion_chunk_shapes_token_and_final_events() {
    // Per-token chunk: text present, finish_reason null.
    let mid = build_text_completion_chunk("cmpl-1", "synthetic", Some(" Paris"), None);
    let v: serde_json::Value = serde_json::from_str(&mid).unwrap();
    assert_eq!(v["object"], TEXT_COMPLETION_OBJECT);
    assert_eq!(v["model"], "synthetic");
    assert_eq!(v["choices"][0]["text"], " Paris");
    assert!(v["choices"][0]["finish_reason"].is_null());
    assert!(v["choices"][0]["logprobs"].is_null());

    // Final chunk: no text (defaults to ""), finish_reason set.
    let last = build_text_completion_chunk("cmpl-1", "synthetic", None, Some(FINISH_REASON_STOP));
    let v: serde_json::Value = serde_json::from_str(&last).unwrap();
    assert_eq!(v["choices"][0]["text"], "");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[test]
fn finalize_completion_plain_run_is_length_and_keeps_all_tokens() {
    let (text, kept, reason) = finalize_completion(&toks(&["Par", "is"]), &[]);
    assert_eq!(text, "Paris");
    assert_eq!(kept.len(), 2);
    assert_eq!(reason, "length");
}

#[test]
fn finalize_completion_stops_and_truncates_on_end_of_turn_token() {
    // The `<eos>` marker ends the turn: it's included, anything after is
    // dropped, finish_reason flips to "stop".
    let (text, kept, reason) = finalize_completion(&toks(&["hi", "<eos>", "ignored"]), &[]);
    assert_eq!(text, "hi<eos>");
    assert_eq!(kept.len(), 2);
    assert_eq!(reason, "stop");
}

#[test]
fn finalize_completion_trims_at_stop_string_and_realigns_tokens() {
    // "STOP" is a stop string; text is trimmed at it and the token list
    // is truncated to stay byte-aligned with the trimmed text.
    let (text, kept, reason) =
        finalize_completion(&toks(&["foo", "STOP", "bar"]), &["STOP".to_string()]);
    assert_eq!(text, "foo");
    assert_eq!(reason, "stop");
    // Only the "foo" token survives the byte-boundary retention.
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].0, "foo");
}

#[test]
fn finalize_completion_no_stop_match_is_unchanged() {
    let (text, kept, reason) = finalize_completion(&toks(&["a", "b"]), &["zzz".to_string()]);
    assert_eq!(text, "ab");
    assert_eq!(kept.len(), 2);
    assert_eq!(reason, "length");
}

#[test]
fn build_completion_logprobs_aligns_offsets_and_arrays() {
    let toks = vec![("Paris".to_string(), 1.0), (" is".to_string(), 1.0)];
    let lp = build_completion_logprobs(&toks);
    assert_eq!(lp.tokens, vec!["Paris".to_string(), " is".to_string()]);
    assert_eq!(lp.token_logprobs.len(), 2);
    assert_eq!(lp.text_offset, vec![0, 5]);
    assert_eq!(lp.top_logprobs.len(), 2);
    // prob=1.0 → logprob=0.0.
    assert!((lp.token_logprobs[0] - 0.0).abs() < 1e-6);
    // top_logprobs[i] currently contains just the picked token.
    assert!(lp.top_logprobs[0].contains_key("Paris"));
}
