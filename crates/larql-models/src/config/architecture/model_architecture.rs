//! The [`ModelArchitecture`] trait: the union of its concept supertraits, plus
//! the predicates that need more than one of them.

use super::Embeddings;
use crate::config::NormType;
use crate::validation::ConfigValidationResult;

/// Architecture-specific behavior. Describes how a model is structured
/// without performing any computation.
pub trait ModelArchitecture: Embeddings {
    /// Whether per-layer K/V is a pure function of `(pre-attention
    /// residual, layer weights, absolute position)` — the structural
    /// precondition for engines that persist the residual stream and
    /// rebuild K/V on demand (the markov-residual family; see
    /// `crates/larql-inference/docs/specs/markov-residual-engine.md` §4).
    ///
    /// Derived from structural metadata, not an architecture allowlist:
    /// - the norm feeding the K/V projections must be stateless at
    ///   inference time — both current [`NormType`] variants are, and a
    ///   future stateful variant falls out of the `matches!` and fails
    ///   closed;
    /// - position must enter solely through RoPE re-applied at the
    ///   row's absolute index; learned absolute position tables
    ///   ([`Self::position_embed_key`]) sit outside the residual
    ///   recompute path;
    /// - K/V must be a direct projection of the normed residual — MLA
    ///   ([`Self::uses_mla`]) routes K/V through a decode-time latent
    ///   the recompute path cannot rebuild.
    fn kv_recomputable_from_residuals(&self) -> bool {
        let norm_stateless = matches!(self.norm_type(), NormType::RmsNorm | NormType::LayerNorm);
        norm_stateless && self.position_embed_key().is_none() && !self.uses_mla()
    }

    /// Validate parsed architecture dimensions and cross-field invariants.
    ///
    /// Detection is intentionally permissive so callers can inspect partially
    /// specified configs. Call this before inference or extraction to fail
    /// early on inconsistent head geometry, RoPE settings, per-layer metadata,
    /// MoE routing, or other values that would otherwise surface later.
    fn validate(&self) -> ConfigValidationResult {
        crate::validation::validate_architecture(self)
    }
}
