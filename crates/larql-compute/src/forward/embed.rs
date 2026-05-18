//! Token embedding — lookup + architecture-specific scaling.

use larql_models::ModelWeights;
use ndarray::Array2;

/// Embed token IDs with architecture-specific scaling.
///
/// Looks up one row per token in `weights.embed`, multiplies by
/// `arch.embed_scale()`. The scale factor handles models that store
/// pre-scaled embeddings (e.g. Gemma) vs. those that don't.
pub fn embed_tokens_pub(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::test_fixtures::make_test_weights;

    #[test]
    fn embed_tokens_shape() {
        let weights = make_test_weights();
        let ids = [0u32, 1, 5];
        let out = embed_tokens_pub(&weights, &ids);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
    }

    #[test]
    fn embed_tokens_single() {
        let weights = make_test_weights();
        let out = embed_tokens_pub(&weights, &[0u32]);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embed_different_tokens_differ() {
        let weights = make_test_weights();
        let e0 = embed_tokens_pub(&weights, &[0u32]);
        let e1 = embed_tokens_pub(&weights, &[1u32]);
        let differ = e0.iter().zip(e1.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "different token ids should produce different embeddings"
        );
    }

    #[test]
    fn embed_same_token_is_deterministic() {
        let weights = make_test_weights();
        let a = embed_tokens_pub(&weights, &[3u32]);
        let b = embed_tokens_pub(&weights, &[3u32]);
        assert_eq!(a, b, "embedding should be deterministic");
    }

    #[test]
    fn embed_empty_token_list_returns_zero_rows() {
        let weights = make_test_weights();
        let out = embed_tokens_pub(&weights, &[]);
        assert_eq!(out.shape(), &[0, weights.hidden_size]);
    }

    #[test]
    fn embed_scaled_by_arch_embed_scale() {
        // TinyModel reports embed_scale = 1.0, so embedded row equals
        // the raw embed table row. Pin that contract for future
        // architectures that override the scale.
        let weights = make_test_weights();
        let out = embed_tokens_pub(&weights, &[2u32]);
        let scale = weights.arch.embed_scale();
        let raw = weights.embed.row(2);
        for (j, v) in out.row(0).iter().enumerate() {
            assert!(
                (v - raw[j] * scale).abs() < 1e-6,
                "embed_tokens_pub did not apply arch.embed_scale() at col {j}"
            );
        }
    }
}
