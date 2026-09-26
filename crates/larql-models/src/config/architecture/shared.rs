//! Constants and derivations shared by the architecture supertraits.

/// The multiplier that leaves a value unchanged.
///
/// Named so that lowering an absent operation to "multiply by one" reads
/// as the deliberate identity it is, and cannot be mistaken for a magic
/// constant standing in for an unread config fact.
pub(super) const IDENTITY_SCALE: f32 = 1.0;

/// SiTU-GLU's gate softcap when the checkpoint declares none — the
/// reference's `beta or 1.0` (`modeling_kimi_linear.py` L91). Named
/// because `1.0` here is a transcribed fallback from one line of one
/// file, not a neutral scale that happens to be one.
pub(super) const SITU_DEFAULT_BETA: f32 = 1.0;

/// The position divisor of an unscaled rotary: positions enter the
/// rotation as declared. What `rope_position_divisor_for_layer` answers
/// for every layer of a checkpoint without linear scaling, and for the
/// layers a family's override leaves plain (Gemma 3's sliding layers).
pub const UNSCALED_POSITION_DIVISOR: f64 = 1.0;

/// Attention score scale from a declared `query_pre_attn_scalar` (or any
/// other scalar the score is `1/sqrt` of, e.g. `head_dim`).
///
/// The one definition of this derivation — [`ModelArchitecture::attention_scale`]
/// and [`ModelArchitecture::attention_scale_for_layer`] both call it, and so
/// does the VINDEX3 plan's carriage check (`larql-vindex`'s
/// `plan::carriage::canonical_declared`), which needs to know that a
/// declared `256` and a carried `0.0625` are the same fact seen on either
/// side of this formula, not a dropped one. A second copy anywhere would be
/// exactly the kind of formula duplication `Activation::uses_gelu_tanh_gate_up`
/// warns about.
pub fn score_scale_from_query_pre_attn_scalar(scalar: f64) -> f64 {
    scalar.powf(-0.5)
}
