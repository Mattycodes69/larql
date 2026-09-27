//! The output-projection policy every loader shares.

use crate::detect::ModelError;
use crate::{ModelArchitecture, WeightArray};

/// Output-projection tensor key. Not architecture-derived: every family in
/// the workspace names its text head this way once keys are normalised.
pub(crate) const LM_HEAD_KEY: &str = "lm_head.weight";

/// Resolve the output projection from the tensor the loader `found`, if any.
///
/// Absent means the model ties it to the embedding matrix — the
/// near-universal convention — but *only* when the config agrees. A
/// checkpoint that declares `tie_word_embeddings: false` and then fails to
/// produce the tensor has lost it somewhere (a key the architecture does not
/// name, a skip filter, a truncated shard), and tying anyway would serve a
/// wrong output projection that still produces fluent text. GPT-OSS and
/// OLMoE both declare `false`.
pub(crate) fn resolve_lm_head(
    found: Option<&WeightArray>,
    embed: &WeightArray,
    arch: &dyn ModelArchitecture,
) -> Result<WeightArray, ModelError> {
    match found {
        Some(t) => Ok(t.clone()),
        // No text head exists in this architecture: the embedding is stored
        // as a placeholder so `ModelWeights` keeps its shape, and must never
        // be sampled from. The untied-but-missing error is a text-LM
        // invariant and does not apply.
        None if !arch.has_lm_head() => Ok(embed.clone()),
        None if arch.config().tie_word_embeddings == Some(false) => {
            Err(ModelError::MissingTensor(format!(
                "{LM_HEAD_KEY} (config declares tie_word_embeddings: false, so it \
                 must not be tied to {})",
                arch.embed_key()
            )))
        }
        None => Ok(embed.clone()),
    }
}
