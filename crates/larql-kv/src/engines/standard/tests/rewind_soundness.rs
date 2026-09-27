//! Rewind soundness
//! Phase 1d.3a: StandardEngine entry-point agreement

use super::*;

#[test]
fn an_unbounded_cache_is_always_rewindable() {
    // Unbounded caches only append, so truncating to the recorded length
    // restores the exact prior state whatever the lengths were.
    assert!(StandardEngine::rewind_is_sound(None, &[0, 7, 4096]));
    assert!(StandardEngine::rewind_is_sound(None, &[]));
}

#[test]
fn a_windowed_cache_is_rewindable_only_with_room_to_spare() {
    const W: usize = 4;
    // Every layer strictly below the window: the step's own append fits,
    // so no eviction fires and the truncate is exact.
    assert!(StandardEngine::rewind_is_sound(Some(W), &[0, 1, 3]));
    // A layer *at* the window evicts to make room, and the evicted row is
    // gone — length would come back while the contents had shifted.
    assert!(!StandardEngine::rewind_is_sound(Some(W), &[3, 4]));
    assert!(!StandardEngine::rewind_is_sound(Some(W), &[W]));
}

#[test]
fn one_unrewindable_layer_condemns_the_whole_step() {
    // All-or-nothing: the layers are only meaningful together, so a single
    // layer at the window makes the cache untrustworthy even if every
    // other layer had room.
    const W: usize = 8;
    assert!(!StandardEngine::rewind_is_sound(Some(W), &[0, 0, 0, W, 0]));
}

#[test]
fn engine_name() {
    assert_eq!(StandardEngine::new(None).name(), "standard");
}

#[test]
fn engine_info_unbounded() {
    let info = StandardEngine::new(None).info();
    assert!(info.config.contains("full"));
}

#[test]
fn engine_info_windowed() {
    let info = StandardEngine::new(Some(128)).info();
    assert!(info.config.contains("128"));
}

#[test]
fn memory_zero_before_prefill() {
    let eng = StandardEngine::new(None);
    assert_eq!(eng.memory_bytes(), 0);
    assert_eq!(eng.window_tokens(), 0);
    assert_eq!(eng.cold_bytes(), 0);
}

#[test]
fn prefill_populates_cache_and_returns_hidden() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    let h = engine
        .prefill(&weights, &ffn, &[0u32, 1, 2])
        .expect("prefill");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(engine.memory_bytes() > 0, "cache should be populated");
    assert!(engine.window_tokens() >= 3);
}

#[test]
fn standard_supports_multimodal() {
    let engine = StandardEngine::new(None);
    assert!(
        engine.supports_multimodal(),
        "StandardEngine is the Phase 1d MM-capable engine"
    );
}

#[test]
fn prefill_and_prefill_from_hidden_agree_on_hidden_and_abs_position() {
    use larql_inference::forward::embed_tokens_pub;
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let tokens = [0u32, 1, 2, 3];

    let mut engine_text = StandardEngine::new(None);
    let h_text = engine_text
        .prefill(&weights, &ffn, &tokens)
        .expect("prefill text");
    let abs_text = engine_text.abs_position;

    let mut engine_hidden = StandardEngine::new(None);
    let initial_hidden = embed_tokens_pub(&weights, &tokens);
    let h_hidden = engine_hidden
        .prefill_from_hidden(&weights, &ffn, &initial_hidden)
        .expect("prefill_from_hidden");
    let abs_hidden = engine_hidden.abs_position;

    // (a) hidden state must match bit-identically — same dispatch
    // path, just with the embed hoisted out of the engine.
    assert_eq!(
        h_text, h_hidden,
        "prefill(tokens) and prefill_from_hidden(embed_tokens_pub(tokens)) \
         must produce identical hidden state"
    );
    // (b) `abs_position` must be set from the hidden's row count.
    // For a text-only input where hidden.nrows() == tokens.len(),
    // both paths land on the same value. Phase 1d MM (where vision
    // rows expand the hidden) WILL diverge from token count —
    // that's the whole point of deriving it from `initial_hidden.nrows()`.
    assert_eq!(
        abs_text, abs_hidden,
        "abs_position must agree between text and from-hidden paths \
         (text=tokens.len(), hidden=nrows; for text-only input they coincide)"
    );
    assert_eq!(
        abs_hidden,
        tokens.len(),
        "abs_position after from-hidden prefill must equal input row count"
    );
}

#[test]
fn prefill_from_hidden_abs_position_derives_from_nrows_not_tokens() {
    // Specifically pin the contract that `abs_position` is set from
    // the hidden's row count. For MM, the host's hidden will have
    // more rows than the text token count (image marker + 256 vision
    // rows + text). Synthesize that shape and verify the engine
    // records the full row count.
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };

    // 7 rows that are NOT pure tokens — emulate "text + 3 vision +
    // text". Just any Array2 with finite values that the layer
    // graph can run through.
    let mm_rows = 7usize;
    let mut hidden = Array2::<f32>::zeros((mm_rows, weights.hidden_size));
    for r in 0..mm_rows {
        for c in 0..weights.hidden_size {
            hidden[[r, c]] = ((r * 13 + c * 7) % 17) as f32 * 0.01 - 0.08;
        }
    }
    let mut engine = StandardEngine::new(None);
    let _ = engine.prefill_from_hidden(&weights, &ffn, &hidden);
    assert_eq!(
        engine.abs_position, mm_rows,
        "abs_position must = initial_hidden.nrows(), not any token count"
    );
}

#[test]
fn decode_step_produces_finite_logits() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
    let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
    assert_eq!(h.shape(), &[1, weights.hidden_size]);
    assert!(hidden_to_raw_logits(&weights, &h)
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn cache_grows_with_decode_steps() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
    let after_prefill = engine.memory_bytes();
    engine.decode_step(&weights, &ffn, 1).expect("decode 1");
    let after_one = engine.memory_bytes();
    engine.decode_step(&weights, &ffn, 2).expect("decode 2");
    let after_two = engine.memory_bytes();
    assert!(after_one > after_prefill);
    assert!(after_two > after_one);
}

#[test]
fn sliding_window_clips_cache() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let window = 2usize;
    let mut engine = StandardEngine::new(Some(window));
    // Prefill with 4 tokens — cache should clip to last `window` per layer.
    engine
        .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
        .expect("prefill 4 tokens");
    assert!(
        engine.window_tokens() <= window,
        "expected window_tokens ≤ {window}, got {}",
        engine.window_tokens()
    );
}

#[test]
fn decode_step_without_prefill_returns_none() {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = StandardEngine::new(None);
    assert!(engine.decode_step(&weights, &ffn, 0).is_err());
}
