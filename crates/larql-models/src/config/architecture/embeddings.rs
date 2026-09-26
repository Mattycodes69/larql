//! The [`Embeddings`] slice of [`super::ModelArchitecture`].

use super::{LatentAttention, IDENTITY_SCALE};

/// Embedding and output scaling: PLE, embedding/logit scales, residual
/// multipliers and final softcaps.
pub trait Embeddings: LatentAttention {
    /// The embedding-scale *operation*, when the model has one.
    /// Gemma: sqrt(hidden_size), Granite: `embedding_multiplier`.
    ///
    /// `None` means the model declares no embedding-scale operation — a
    /// different claim from `Some(1.0)`, which means the source semantics
    /// specify a multiply by one. Collapsing the two here would destroy
    /// the distinction before inventory or vindex ever sees it, so the
    /// front end preserves absence as absence and only
    /// [`embed_scale_multiplier`](Self::embed_scale_multiplier) may turn
    /// it into a number.
    fn embed_scale(&self) -> Option<f32> {
        self.config().embedding_multiplier.map(|v| v as f32)
    }

    /// The factor the forward path multiplies embeddings by.
    ///
    /// Executing "no embedding-scale operation" and "multiply by one"
    /// coincide numerically, and execution must eventually produce a
    /// number. This is the single, named place that coincidence is
    /// allowed — a judgment about lowering, not a fallback that hides a
    /// missing fact. Semantic consumers must read
    /// [`embed_scale`](Self::embed_scale) instead.
    fn embed_scale_multiplier(&self) -> f32 {
        self.embed_scale().unwrap_or(IDENTITY_SCALE)
    }

    /// Whether this model uses per-layer embeddings (PLE).
    /// When true, each layer adds a gated embedding lookup to the hidden state.
    fn has_per_layer_embeddings(&self) -> bool {
        self.config().per_layer_embed_dim.unwrap_or(0) > 0
    }

    /// Per-layer embedding dimension. 0 if PLE is not used.
    fn per_layer_embed_dim(&self) -> usize {
        self.config().per_layer_embed_dim.unwrap_or(0)
    }

    /// Key for the shared per-layer embedding matrix [vocab, num_layers * ple_dim].
    fn per_layer_embed_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("embed_tokens_per_layer.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the model-level PLE projection [num_layers * ple_dim, hidden].
    fn per_layer_model_projection_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("per_layer_model_projection.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the shared PLE projection RMSNorm weight.
    fn per_layer_projection_norm_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("per_layer_projection_norm.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the per-layer input gate projection [ple_dim, hidden].
    fn per_layer_input_gate_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}per_layer_input_gate.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Key for the per-layer output projection [hidden, ple_dim].
    fn per_layer_projection_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}per_layer_projection.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Key for the post-PLE norm weight.
    fn post_per_layer_input_norm_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}post_per_layer_input_norm.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Multiplier on the final hidden state before the vocabulary
    /// projection. `None` = the model declares no output-multiplier
    /// operation, which is a different claim from `Some(1.0)`.
    ///
    /// Canonical name, same reasoning as [`qk_scale_factor`](Self::qk_scale_factor):
    /// **The multiplicative factor**, already resolved. Callers multiply by
    /// it; nobody downstream needs to know which spelling produced it.
    ///
    /// The two spellings are not interchangeable, which is the trap this
    /// method exists to close:
    ///
    /// ```text
    /// output_multiplier   x  ->  logits * x        a multiplier
    /// logits_scaling      d  ->  logits / d        a DIVISOR
    /// ```
    ///
    /// Scaling does commute through the linear head, so "before the vocab
    /// projection" and "on the logits" are the same number — but only after
    /// the divisor has been inverted. Passing Granite's `logits_scaling`
    /// through as a multiplier put its logits out by `d²` (a factor of 100
    /// at `logits_scaling = 10`), which saturates every softmax built on
    /// them.
    ///
    /// That error is invisible to greedy decoding: a positive scalar cannot
    /// reorder logits, so argmax, generated ids and every oracle built on
    /// them agree exactly while the distribution is wrong. Only a
    /// probability-space measurement — KL, NLL, top-p, temperature — can
    /// see it, which is how it survived.
    ///
    /// A non-finite or zero `logits_scaling` yields `None` rather than an
    /// infinity that would silently annihilate the head.
    fn logit_scale(&self) -> Option<f64> {
        if let Some(m) = self.config().output_multiplier {
            return Some(m);
        }
        match self.config().logits_scaling {
            Some(d) if d.is_finite() && d != 0.0 => Some(1.0 / d),
            _ => None,
        }
    }

    /// Residual-stream scaling: the sublayer's own output (attention or
    /// FFN) is multiplied by this before being added into the residual
    /// stream, at both sites, with the same value (Granite's
    /// `residual_multiplier`). `None` = no such operation, distinct from
    /// `Some(1.0)`. No other family in this registry scales the residual
    /// stream, so there is no second spelling to resolve here yet.
    fn residual_scale(&self) -> Option<f32> {
        self.config().residual_multiplier.map(|v| v as f32)
    }

    /// Final logit softcapping value (None = disabled).
    /// Applied to output logits: logits = tanh(logits / cap) * cap
    fn final_logit_softcapping(&self) -> Option<f32> {
        self.config().final_logit_softcapping.map(|v| v as f32)
    }

    /// Residual stream scaling factor applied after attention and FFN additions.
    fn residual_multiplier(&self) -> f32 {
        self.config()
            .residual_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    /// Logits scaling factor applied to final logits before softmax.
    fn logits_scaling(&self) -> f32 {
        self.config()
            .logits_scaling
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }
}
