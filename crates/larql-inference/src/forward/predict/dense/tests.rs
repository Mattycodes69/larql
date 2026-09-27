use super::*;
use crate::test_utils::TestFixtures;
use ndarray::Array2;

#[test]
fn cmp_desc_nan_last_orders_descending_with_nan_last() {
    let mut v = [(0, 1.0f32), (1, 3.0), (2, f32::NAN), (3, 2.0)];
    v.sort_by(cmp_desc_nan_last);
    // 3.0 first (highest), 2.0, 1.0, NaN last.
    let order: Vec<usize> = v.iter().map(|(i, _)| *i).collect();
    assert_eq!(order, vec![1, 3, 0, 2]);
}

#[test]
fn predict_returns_top_k_predictions() {
    let fx = TestFixtures::build();
    let r = predict(&fx.weights, &fx.tokenizer, &[0u32, 1], 5);
    assert!(r.predictions.len() <= 5);
}

#[test]
fn predict_with_temperature_high_temp_smooths_distribution() {
    // Higher temperature → flatter distribution. Top probability at
    // T=10 should be lower than top at T=1 (synthetic weights make
    // the actual predictions chaotic but the relationship holds).
    let fx = TestFixtures::build();
    let cold = predict_with_temperature(&fx.weights, &fx.tokenizer, &[0u32, 1], 10, 1.0);
    let hot = predict_with_temperature(&fx.weights, &fx.tokenizer, &[0u32, 1], 10, 10.0);
    assert_eq!(cold.predictions.len(), hot.predictions.len());
    if !cold.predictions.is_empty() && !hot.predictions.is_empty() {
        assert!(
            cold.predictions[0].1 >= hot.predictions[0].1 - 1e-6,
            "high T should not produce a sharper top-1 than low T"
        );
    }
}

#[test]
fn logit_lens_top1_returns_some_for_correct_shape() {
    let fx = TestFixtures::build();
    let residual = vec![0.5f32; fx.weights.hidden_size];
    let result = logit_lens_top1(&fx.weights, &fx.tokenizer, &residual);
    assert!(result.is_some());
}

#[test]
fn logit_lens_top1_returns_none_for_wrong_shape() {
    let fx = TestFixtures::build();
    let residual = vec![0.5f32; fx.weights.hidden_size + 1];
    assert!(logit_lens_top1(&fx.weights, &fx.tokenizer, &residual).is_none());
}

#[test]
fn predict_from_hidden_resumes_at_start_layer() {
    let fx = TestFixtures::build();
    let h = Array2::<f32>::zeros((2, fx.weights.hidden_size));
    let r = predict_from_hidden(&fx.weights, &fx.tokenizer, &h, 0, 3);
    assert!(r.predictions.len() <= 3);
}

#[test]
fn predict_from_hidden_with_ffn_handles_empty_token_ids() {
    // Empty token_ids → ple_inputs stays empty; predict still runs.
    let fx = TestFixtures::build();
    let h = Array2::<f32>::ones((1, fx.weights.hidden_size));
    let ffn = crate::ffn::WeightFfn {
        weights: &fx.weights,
    };
    let r = predict_from_hidden_with_ffn(&fx.weights, &fx.tokenizer, &h, 0, 5, &ffn, &[]);
    assert!(r.predictions.len() <= 5);
}

#[test]
fn predict_with_ffn_trace_returns_per_layer_residuals() {
    let fx = TestFixtures::build();
    let ffn = crate::ffn::WeightFfn {
        weights: &fx.weights,
    };
    let r = predict_with_ffn_trace(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &ffn);
    assert_eq!(r.residuals.len(), fx.weights.num_layers);
    for residual in &r.residuals {
        assert_eq!(residual.len(), fx.weights.hidden_size);
        assert!(residual.iter().all(|v| v.is_finite()));
    }
    assert!(r.predictions.len() <= 3);
}

#[test]
fn logits_to_predictions_pub_matches_internal() {
    let fx = TestFixtures::build();
    let h = Array2::<f32>::from_elem((2, fx.weights.hidden_size), 0.1f32);
    let r_pub = logits_to_predictions_pub(&fx.weights, &h, &fx.tokenizer, 5, 1.0);
    let r_priv = logits_to_predictions(&fx.weights, &h, &fx.tokenizer, 5, 1.0);
    assert_eq!(r_pub.predictions.len(), r_priv.predictions.len());
    for ((a_t, a_p), (b_t, b_p)) in r_pub.predictions.iter().zip(r_priv.predictions.iter()) {
        assert_eq!(a_t, b_t);
        assert!((a_p - b_p).abs() < 1e-6);
    }
}
