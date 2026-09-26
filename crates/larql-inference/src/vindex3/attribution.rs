//! ATTR-1D descriptive support identities and prompt-role mapping.
//!
//! This module implements the identity boundary frozen in
//! `docs/v3-attr-1d-descriptive-support.md`:
//!
//! - [`SupportCoordinate`] names a stable physical location and contains no
//!   execution, backend, measurement, semantic-transition, or causal fields;
//! - [`SupportObservationIdentity`] binds that coordinate to one exact
//!   observed execution; and
//! - [`PromptRoleMap`] assigns source roles from construction-time token spans
//!   only, refusing ambiguous or out-of-range inputs.
//!
//! The numerical contribution transform consumes these identities.  It does
//! not get to redefine them after observing a contribution or intervention.

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod describe;
mod roles;
mod sidecar;
pub use describe::*;
use roles::*;
pub use sidecar::*;

/// Frozen schema for a stable physical support coordinate.
pub const SUPPORT_COORDINATE_SCHEMA: &str = "larql.attr1d.support-coordinate.v1";
/// Frozen schema for an execution-specific observation binding.
pub const SUPPORT_OBSERVATION_SCHEMA: &str = "larql.attr1d.support-observation.v1";
/// Schema for the construction-time source-role manifest.
pub const PROMPT_ROLE_MAP_SCHEMA: &str = "larql.attr1d.prompt-role-map.v1";
/// Schema for the lossless same-run source evidence consumed by ATTR-1D.
pub const SOURCE_EVIDENCE_SCHEMA: &str = "larql.attr1d.source-evidence.v1";
/// Frozen ATTR-1D contract identity consumed by descriptive records.
pub const ATTR1D_CONTRACT_IDENTITY: &str =
    "sha256:4f48973807db0695d5f30b83a00c73637afbd0dc77d116b5a1f9eb6ec913431c";

/// Frozen numerical acceptance thresholds.
pub const HEAD_SUM_MAX_RELATIVE_L2: f64 = 1e-4;
pub const SOURCE_SPLIT_MAX_RELATIVE_L2: f64 = 1e-5;
pub const PROJECTION_RECONSTRUCTION_SCALE: f64 = 1e-6;
pub const ATTENTION_PROBABILITY_MAX_ABSOLUTE_ERROR: f64 = 1e-6;
pub const VECTOR_DENOMINATOR_FLOOR: f64 = 1e-12;

/// Errors are refusals: callers must not repair an invalid identity or role
/// map using observed model behaviour.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AttributionIdentityError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("expected schema {expected}, found {actual}")]
    WrongSchema {
        expected: &'static str,
        actual: String,
    },
    #[error("token span [{start}, {end}) is empty or reversed")]
    InvalidSpan { start: usize, end: usize },
    #[error("token span [{start}, {end}) exceeds prompt length {prompt_len}")]
    SpanOutsidePrompt {
        start: usize,
        end: usize,
        prompt_len: usize,
    },
    #[error(
        "declared {left_role:?} span [{left_start}, {left_end}) overlaps {right_role:?} span [{right_start}, {right_end})"
    )]
    OverlappingRoleSpans {
        left_role: SourceRole,
        left_start: usize,
        left_end: usize,
        right_role: SourceRole,
        right_start: usize,
        right_end: usize,
    },
    #[error("final prompt position {position} is outside prompt length {prompt_len}")]
    FinalPromptPositionOutsidePrompt { position: usize, prompt_len: usize },
    #[error("source position {position} is outside visible range [{start}, {end})")]
    SourcePositionOutsideVisibleRange {
        position: usize,
        start: usize,
        end: usize,
    },
    #[error("source coordinates must be one-token spans, found [{start}, {end})")]
    SourceCoordinateNotOneToken { start: usize, end: usize },
    #[error("{field} does not match its recomputed identity")]
    IdentityBindingMismatch { field: &'static str },
    #[error(
        "visible source range [{start}, {end}) does not end after observed position {position}"
    )]
    SourceRangePositionMismatch {
        start: usize,
        end: usize,
        position: usize,
    },
    #[error("identity JSON serialization failed: {0}")]
    Serialization(String),
}

/// Refusals raised by the deterministic numerical transform.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DescriptiveTransformError {
    #[error(transparent)]
    Identity(#[from] AttributionIdentityError),
    #[error("{field} width {actual} does not match expected width {expected}")]
    Width {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("expected {expected} query heads, observed {actual}")]
    HeadCoverage { expected: usize, actual: usize },
    #[error("query head {head} was observed more than once")]
    DuplicateHead { head: usize },
    #[error("query-head coverage is not the exact range 0..{expected}: {observed:?}")]
    NonCanonicalHeadCoverage {
        expected: usize,
        observed: Vec<usize>,
    },
    #[error("head {head} source position {position} was observed more than once")]
    DuplicateSource { head: usize, position: usize },
    #[error(
        "head {head} source coverage does not match visible range [{start}, {end}): {observed:?}"
    )]
    SourceCoverage {
        head: usize,
        start: usize,
        end: usize,
        observed: Vec<usize>,
    },
    #[error("{field} contains a non-finite scalar")]
    NonFinite { field: &'static str },
    #[error("post-attention RMS epsilon must be finite and positive, found {epsilon}")]
    InvalidRmsEpsilon { epsilon: f64 },
    #[error("prepared head projection refused: {0}")]
    Projection(String),
}

/// The physical context shared by site, bias, head and source coordinates.
///
/// These identities describe the logical prepared program and its input
/// layout.  In particular, they do not name a backend realization or run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportCoordinateContext {
    pub model_system_identity: String,
    pub logical_plan_identity: String,
    pub component_identity: String,
    pub input_layout_identity: String,
    pub position: usize,
    pub layer: usize,
}

impl SupportCoordinateContext {
    /// Construct and validate the backend-neutral coordinate context.
    pub fn new(
        model_system_identity: impl Into<String>,
        logical_plan_identity: impl Into<String>,
        component_identity: impl Into<String>,
        input_layout_identity: impl Into<String>,
        position: usize,
        layer: usize,
    ) -> Result<Self, AttributionIdentityError> {
        let context = Self {
            model_system_identity: model_system_identity.into(),
            logical_plan_identity: logical_plan_identity.into(),
            component_identity: component_identity.into(),
            input_layout_identity: input_layout_identity.into(),
            position,
            layer,
        };
        context.validate()?;
        Ok(context)
    }

    fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_nonempty("model_system_identity", &self.model_system_identity)?;
        require_nonempty("logical_plan_identity", &self.logical_plan_identity)?;
        require_nonempty("component_identity", &self.component_identity)?;
        require_nonempty("input_layout_identity", &self.input_layout_identity)
    }
}

/// ATTR-1D v1 observes attention support.  FFN support gets its own frozen
/// algebra before this enum is extended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportOperator {
    Attention,
}

/// Physical observation site.  Kept separate from [`SupportOperator`] because
/// later operator contracts may admit more than one site per operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportSite {
    Attention,
}

/// Source roles frozen from prompt construction, never inferred from decoded
/// text or from the model's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SourceRole {
    Bos,
    Relation,
    Entity,
    Last,
    Other,
}

/// The typed physical address below a coordinate context.
///
/// The internally tagged representation serializes the frozen flat
/// `coordinate_kind` shape rather than an implementation-specific nested
/// object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "coordinate_kind", rename_all = "snake_case")]
pub enum SupportAddress {
    Site,
    Bias,
    Head {
        query_head: usize,
        kv_head: usize,
    },
    Source {
        query_head: usize,
        kv_head: usize,
        source_start: usize,
        source_end: usize,
        source_role: SourceRole,
        role_map_identity: String,
    },
}

impl SupportAddress {
    fn validate(&self) -> Result<(), AttributionIdentityError> {
        if let Self::Source {
            source_start,
            source_end,
            role_map_identity,
            ..
        } = self
        {
            if source_start.checked_add(1) != Some(*source_end) {
                return Err(AttributionIdentityError::SourceCoordinateNotOneToken {
                    start: *source_start,
                    end: *source_end,
                });
            }
            require_nonempty("role_map_identity", role_map_identity)?;
        }
        Ok(())
    }
}

/// Stable, backend- and execution-neutral physical support identity.
///
/// Measurements and observation provenance intentionally cannot be expressed
/// by this type.  Deserialised coordinates are validated before hashing, so a
/// wrong schema or malformed source span cannot acquire a lawful ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportCoordinate {
    schema: String,
    #[serde(flatten)]
    pub context: SupportCoordinateContext,
    pub operator: SupportOperator,
    pub site: SupportSite,
    #[serde(flatten)]
    pub address: SupportAddress,
}

impl SupportCoordinate {
    fn new(
        context: SupportCoordinateContext,
        address: SupportAddress,
    ) -> Result<Self, AttributionIdentityError> {
        let coordinate = Self {
            schema: SUPPORT_COORDINATE_SCHEMA.to_string(),
            context,
            operator: SupportOperator::Attention,
            site: SupportSite::Attention,
            address,
        };
        coordinate.validate()?;
        Ok(coordinate)
    }

    /// Construct the whole attention-site coordinate.
    pub fn site(context: SupportCoordinateContext) -> Result<Self, AttributionIdentityError> {
        Self::new(context, SupportAddress::Site)
    }

    /// Construct the once-only output-bias coordinate.
    pub fn bias(context: SupportCoordinateContext) -> Result<Self, AttributionIdentityError> {
        Self::new(context, SupportAddress::Bias)
    }

    /// Construct one query-head coordinate.
    pub fn head(
        context: SupportCoordinateContext,
        query_head: usize,
        kv_head: usize,
    ) -> Result<Self, AttributionIdentityError> {
        Self::new(
            context,
            SupportAddress::Head {
                query_head,
                kv_head,
            },
        )
    }

    /// Construct an exact one-token source coordinate.  The role and role-map
    /// identity are derived here so callers cannot hand-label a source after
    /// looking at its contribution.
    pub fn source(
        context: SupportCoordinateContext,
        query_head: usize,
        kv_head: usize,
        source_position: usize,
        role_map: &PromptRoleMap,
        visible_source_range: TokenSpan,
    ) -> Result<Self, AttributionIdentityError> {
        let source_end = source_position.checked_add(1).ok_or(
            AttributionIdentityError::SourceCoordinateNotOneToken {
                start: source_position,
                end: source_position,
            },
        )?;
        Self::new(
            context,
            SupportAddress::Source {
                query_head,
                kv_head,
                source_start: source_position,
                source_end,
                source_role: role_map.role_at(source_position, visible_source_range)?,
                role_map_identity: role_map.identity()?,
            },
        )
    }

    /// Frozen schema named by this value.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate invariants that serde alone cannot express.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(SUPPORT_COORDINATE_SCHEMA, &self.schema)?;
        self.context.validate()?;
        self.address.validate()
    }

    /// Canonical JSON bytes used by the frozen coordinate hash.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AttributionIdentityError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    /// `sha256:` identity over sorted-key, whitespace-free canonical JSON.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        Ok(hash_bytes(&self.canonical_bytes()?))
    }
}

/// One exact observation of a stable support coordinate.
///
/// Reference and production backends intentionally produce different values
/// of this type while sharing the same [`SupportCoordinate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportObservationIdentity {
    schema: String,
    pub support_coordinate_id: String,
    pub execution_identity: String,
    pub provenance_fingerprint: String,
    pub prepared_image_fingerprint: String,
    pub observation_receipt_digest: String,
    pub event_sequence: u64,
}

impl SupportObservationIdentity {
    /// Bind a coordinate to one execution and one sequenced observation.
    pub fn new(
        coordinate: &SupportCoordinate,
        execution_identity: impl Into<String>,
        provenance_fingerprint: impl Into<String>,
        prepared_image_fingerprint: impl Into<String>,
        observation_receipt_digest: impl Into<String>,
        event_sequence: u64,
    ) -> Result<Self, AttributionIdentityError> {
        let observation = Self {
            schema: SUPPORT_OBSERVATION_SCHEMA.to_string(),
            support_coordinate_id: coordinate.identity()?,
            execution_identity: execution_identity.into(),
            provenance_fingerprint: provenance_fingerprint.into(),
            prepared_image_fingerprint: prepared_image_fingerprint.into(),
            observation_receipt_digest: observation_receipt_digest.into(),
            event_sequence,
        };
        observation.validate()?;
        Ok(observation)
    }

    /// Frozen schema named by this value.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate the execution binding before hashing or persistence.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(SUPPORT_OBSERVATION_SCHEMA, &self.schema)?;
        require_nonempty("support_coordinate_id", &self.support_coordinate_id)?;
        require_nonempty("execution_identity", &self.execution_identity)?;
        require_nonempty("provenance_fingerprint", &self.provenance_fingerprint)?;
        require_nonempty(
            "prepared_image_fingerprint",
            &self.prepared_image_fingerprint,
        )?;
        require_nonempty(
            "observation_receipt_digest",
            &self.observation_receipt_digest,
        )
    }

    /// Canonical JSON bytes used by the frozen observation hash.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AttributionIdentityError> {
        self.validate()?;
        canonical_json_bytes(self)
    }

    /// `sha256:` identity over this exact execution binding.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        Ok(hash_bytes(&self.canonical_bytes()?))
    }
}

/// One source row borrowed from the canonical attention tap and copied into
/// the lossless ATTR-1D input sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSourceEvidence {
    pub position: usize,
    pub attention_probability: f32,
    pub values: Vec<f32>,
}

/// Complete evidence for one query head at one attention write.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionHeadEvidence {
    pub query_head: usize,
    pub kv_head: usize,
    pub sink: f32,
    /// The kernel's `Σ_t a[h,t] · v[g(h),t]`, before its optional gate.
    pub mixed_values: Vec<f32>,
    /// Activated output-gate values for this head, when declared by the plan.
    pub gate: Option<Vec<f32>>,
    pub sources: Vec<AttentionSourceEvidence>,
}

/// Lossless observed inputs for one attention support transform.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSiteEvidence {
    pub expected_query_heads: usize,
    pub visible_source_range: TokenSpan,
    pub recorded_delta: Vec<f32>,
    pub heads: Vec<AttentionHeadEvidence>,
}

#[cfg(test)]
mod tests;
