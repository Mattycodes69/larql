//! Token embedding — re-exported from `larql_compute::forward::embed`.
//!
//! The arch-aware scaling logic moved to `larql-compute` (ADR-0022
//! Step 2b). This module preserves the `pub(super) embed_tokens`
//! private convenience used by sibling `forward/*` modules and the
//! `pub use` chain for `crate::forward::embed_tokens_pub`.

use crate::model::ModelWeights;
use ndarray::Array2;

pub use larql_compute::forward::embed_tokens_pub;

/// Private convenience used by sibling `forward/*` modules
/// (`trace.rs`, `predict/raw.rs`, `predict/ffn.rs`, `predict/dense.rs`).
/// Thin delegate to [`embed_tokens_pub`]; kept here so those modules
/// don't need to import a path outside the forward module tree.
pub(super) fn embed_tokens(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    embed_tokens_pub(weights, token_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;

    #[test]
    fn embed_tokens_super_delegates_to_pub() {
        // Pin that the pub(super) shim returns bit-identical output to
        // the underlying `embed_tokens_pub`. Future refactors might be
        // tempted to specialise the super variant — this test catches
        // any drift.
        let weights = make_test_weights();
        let ids = [1u32, 2, 3];
        let via_super = embed_tokens(&weights, &ids);
        let via_pub = embed_tokens_pub(&weights, &ids);
        assert_eq!(via_super, via_pub);
    }
}
