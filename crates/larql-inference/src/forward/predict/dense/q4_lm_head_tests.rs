use super::*;
use crate::test_utils::Q4KTestFixtures;

/// Round-trip the Q4_K lm_head matmul path: build a synthetic
/// hidden vector, run it through `logits_to_predictions_q4_lm_head`
/// against the synth Q4_K lm_head bytes, and confirm the
/// predictions are well-formed.
#[test]
fn logits_to_predictions_q4_lm_head_produces_valid_top_k() {
    let fx = Q4KTestFixtures::build();
    let h = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
        ((j as f32) * 0.013).sin() * 0.5
    });
    let q4_bytes = fx
        .index
        .storage
        .lm_head_kquant_view()
        .expect("synth Q4 lm_head present")
        .as_ref()
        .to_vec();
    let backend = larql_compute::CpuBackend;

    let result = logits_to_predictions_q4_lm_head(
        &fx.weights,
        &h,
        &q4_bytes,
        fx.weights.vocab_size,
        &backend,
        &fx.tokenizer,
        5,
        1.0,
    );

    // Top-5 with vocab=256 → returns 5 entries.
    assert_eq!(result.token_ids.len(), 5);
    assert_eq!(result.predictions.len(), 5);
    // Probabilities should be in [0, 1] and descending.
    for (i, (_, p)) in result.predictions.iter().enumerate() {
        assert!(
            *p >= 0.0 && *p <= 1.0,
            "prob[{i}] = {p} should be in [0, 1]"
        );
        if i > 0 {
            let prev = result.predictions[i - 1].1;
            assert!(prev >= *p, "prob should be descending: prev={prev} cur={p}");
        }
    }
    // Top token is a valid vocab ID.
    assert!((result.token_ids[0] as usize) < fx.weights.vocab_size);
}

/// Multi-position hidden state: the function takes the LAST row.
#[test]
fn logits_to_predictions_q4_lm_head_uses_last_row() {
    let fx = Q4KTestFixtures::build();
    // Two-row hidden: row 0 deliberately set to a value that
    // would yield different argmax than row 1 — confirms we read
    // the last row, not the first.
    let mut h = Array2::<f32>::zeros((2, fx.weights.hidden_size));
    for j in 0..fx.weights.hidden_size {
        h[[0, j]] = -1.0; // row 0 (ignored)
        h[[1, j]] = ((j as f32) * 0.013).sin() * 0.5; // row 1 (used)
    }
    // Single-row reference for comparison.
    let h1 = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
        ((j as f32) * 0.013).sin() * 0.5
    });
    let q4_bytes = fx
        .index
        .storage
        .lm_head_kquant_view()
        .expect("Q4 lm_head present")
        .as_ref()
        .to_vec();
    let backend = larql_compute::CpuBackend;

    let r2 = logits_to_predictions_q4_lm_head(
        &fx.weights,
        &h,
        &q4_bytes,
        fx.weights.vocab_size,
        &backend,
        &fx.tokenizer,
        1,
        1.0,
    );
    let r1 = logits_to_predictions_q4_lm_head(
        &fx.weights,
        &h1,
        &q4_bytes,
        fx.weights.vocab_size,
        &backend,
        &fx.tokenizer,
        1,
        1.0,
    );
    assert_eq!(
        r2.token_ids[0], r1.token_ids[0],
        "two-row input's last row should produce the same top-1 as the single-row input"
    );
}

/// Temperature scales the softmax — high T flattens the
/// distribution, low T sharpens it. Confirm the top-1 probability
/// is non-increasing as T grows.
#[test]
fn logits_to_predictions_q4_lm_head_high_temp_flattens() {
    let fx = Q4KTestFixtures::build();
    let h = Array2::from_shape_fn((1, fx.weights.hidden_size), |(_, j)| {
        ((j as f32) * 0.011).cos() * 0.4
    });
    let q4_bytes = fx
        .index
        .storage
        .lm_head_kquant_view()
        .expect("Q4 lm_head present")
        .as_ref()
        .to_vec();
    let backend = larql_compute::CpuBackend;

    let cold = logits_to_predictions_q4_lm_head(
        &fx.weights,
        &h,
        &q4_bytes,
        fx.weights.vocab_size,
        &backend,
        &fx.tokenizer,
        1,
        1.0,
    );
    let hot = logits_to_predictions_q4_lm_head(
        &fx.weights,
        &h,
        &q4_bytes,
        fx.weights.vocab_size,
        &backend,
        &fx.tokenizer,
        1,
        10.0,
    );
    assert!(
        cold.predictions[0].1 >= hot.predictions[0].1 - 1e-6,
        "T=10 should not produce a sharper top-1 than T=1: cold={} hot={}",
        cold.predictions[0].1,
        hot.predictions[0].1
    );
}
