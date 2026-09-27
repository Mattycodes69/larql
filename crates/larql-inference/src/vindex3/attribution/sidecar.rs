//! Attention source-evidence sidecars, post-attention transforms and prepared projections.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

/// Lossless source evidence bound to the exact canonical observation that
/// produced it.
///
/// This is a sidecar rather than another execution surface: callers populate
/// [`AttentionSiteEvidence`] from the borrowed HEAD tap in the same run, then
/// seal it here with the site coordinate, prepared image, provenance, receipt
/// and event sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSourceEvidenceSidecar {
    pub(super) schema: String,
    pub support_coordinate_id: String,
    pub support_coordinate: SupportCoordinate,
    pub support_observation_id: String,
    pub support_observation: SupportObservationIdentity,
    pub evidence: AttentionSiteEvidence,
}

impl AttentionSourceEvidenceSidecar {
    /// Seal source evidence from one exact observed site.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: SupportCoordinateContext,
        execution_identity: impl Into<String>,
        provenance_fingerprint: impl Into<String>,
        prepared_image_fingerprint: impl Into<String>,
        observation_receipt_digest: impl Into<String>,
        event_sequence: u64,
        evidence: AttentionSiteEvidence,
    ) -> Result<Self, AttributionIdentityError> {
        let support_coordinate = SupportCoordinate::site(context)?;
        let support_coordinate_id = support_coordinate.identity()?;
        let support_observation = SupportObservationIdentity::new(
            &support_coordinate,
            execution_identity,
            provenance_fingerprint,
            prepared_image_fingerprint,
            observation_receipt_digest,
            event_sequence,
        )?;
        let support_observation_id = support_observation.identity()?;
        let sidecar = Self {
            schema: SOURCE_EVIDENCE_SCHEMA.to_string(),
            support_coordinate_id,
            support_coordinate,
            support_observation_id,
            support_observation,
            evidence,
        };
        sidecar.validate()?;
        Ok(sidecar)
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate all identity joins without trusting persisted digest fields.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(SOURCE_EVIDENCE_SCHEMA, &self.schema)?;
        if !matches!(&self.support_coordinate.address, SupportAddress::Site) {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_coordinate_kind",
            });
        }
        if self.support_coordinate.identity()? != self.support_coordinate_id {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_coordinate_id",
            });
        }
        if self.support_observation.support_coordinate_id != self.support_coordinate_id {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_observation.support_coordinate_id",
            });
        }
        if self.support_observation.identity()? != self.support_observation_id {
            return Err(AttributionIdentityError::IdentityBindingMismatch {
                field: "support_observation_id",
            });
        }
        if self.evidence.visible_source_range.start >= self.evidence.visible_source_range.end {
            return Err(AttributionIdentityError::InvalidSpan {
                start: self.evidence.visible_source_range.start,
                end: self.evidence.visible_source_range.end,
            });
        }
        if self.support_coordinate.context.position.checked_add(1)
            != Some(self.evidence.visible_source_range.end)
        {
            return Err(AttributionIdentityError::SourceRangePositionMismatch {
                start: self.evidence.visible_source_range.start,
                end: self.evidence.visible_source_range.end,
                position: self.support_coordinate.context.position,
            });
        }
        Ok(())
    }

    /// Deterministic bytes for replay comparison and content addressing.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AttributionIdentityError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    /// Content identity of this sealed evidence object.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        Ok(hash_bytes(&self.canonical_bytes()?))
    }
}

/// The linear transform applied to raw output-projection children before the
/// residual write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PostAttentionTransform {
    Identity {
        residual_scale: f32,
    },
    RmsNorm {
        epsilon: f64,
        weight_offset: f32,
        weights: Vec<f32>,
        residual_scale: f32,
    },
}

/// Adapter onto the exact prepared image that produced an observation.
///
/// An implementation must project through that image's effective `W_O` head
/// slice.  Re-loading or widening checkpoint weights is outside this seam.
pub trait PreparedAttentionProjection {
    fn hidden_size(&self) -> usize;

    fn project_head(&self, query_head: usize, values: &[f32]) -> Result<Vec<f32>, String>;

    fn output_bias(&self) -> Result<Option<Vec<f32>>, String>;

    fn post_attention_transform(&self) -> Result<PostAttentionTransform, String>;
}

/// One fixed reader from a basis frozen before observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionReader {
    pub identity: String,
    pub values: Vec<f32>,
}

impl AttributionReader {
    pub fn new(
        identity: impl Into<String>,
        values: Vec<f32>,
    ) -> Result<Self, AttributionIdentityError> {
        let reader = Self {
            identity: identity.into(),
            values,
        };
        require_nonempty("reader_identity", &reader.identity)?;
        Ok(reader)
    }
}

/// A fixed-reader projection with both signed and absolute-mass shares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionMeasurement {
    pub reader_identity: String,
    pub value: f64,
    /// Absent exactly when the frozen signed denominator is unstable.
    pub signed_share: Option<f64>,
    /// Absent exactly when the frozen nonnegative mass denominator is zero.
    pub absolute_mass_share: Option<f64>,
}

/// Descriptive support for one exact source token under one query head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceContribution {
    pub position: usize,
    pub source_role: SourceRole,
    pub attention_probability: f32,
    pub contribution: Vec<f32>,
    pub contribution_norm: f64,
    pub selected_token_projection: Vec<ProjectionMeasurement>,
}

/// A source-role sum.  This is a measurement over exact source coordinates,
/// never a replacement coordinate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRoleContribution {
    pub source_role: SourceRole,
    pub selected_token_projection: Vec<ProjectionMeasurement>,
}

/// Descriptive support for one query head.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadContribution {
    pub query_head: usize,
    pub kv_head: usize,
    pub sink: f32,
    pub attention_probability_error: f64,
    pub contribution: Vec<f32>,
    pub contribution_norm: f64,
    pub selected_token_projection: Vec<ProjectionMeasurement>,
    pub sources: Vec<SourceContribution>,
    pub source_role_projection: Vec<SourceRoleContribution>,
    pub source_reconstruction_relative_l2: f64,
}

/// Per-reader check that site projection equals bias plus all head
/// projections under the recorded residual write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionReconstruction {
    pub reader_identity: String,
    pub absolute_error: f64,
    pub permitted_error: f64,
}

/// The complete deterministic result for one attention write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionDescriptiveSupport {
    pub attr1d_contract_identity: String,
    pub site_contribution: Vec<f32>,
    pub bias_contribution: Vec<f32>,
    pub reconstructed_delta: Vec<f32>,
    pub head_reconstruction_relative_l2: f64,
    pub site_projection: Vec<ProjectionMeasurement>,
    pub bias_projection: Vec<ProjectionMeasurement>,
    pub heads: Vec<HeadContribution>,
    pub projection_reconstruction: Vec<ProjectionReconstruction>,
}

impl AttentionDescriptiveSupport {
    /// All frozen numerical gates, excluding replay/backend checks which need
    /// two separately produced results.
    pub fn passes_numerical_gates(&self) -> bool {
        self.head_reconstruction_relative_l2 <= HEAD_SUM_MAX_RELATIVE_L2
            && self.heads.iter().all(|head| {
                head.source_reconstruction_relative_l2 <= SOURCE_SPLIT_MAX_RELATIVE_L2
                    && head.attention_probability_error <= ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR
            })
            && self
                .projection_reconstruction
                .iter()
                .all(|check| check.absolute_error <= check.permitted_error)
    }
}
