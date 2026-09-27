//! Prompt role maps, source roles and evidence validation.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[allow(unused_imports)]
use super::*;

impl PromptRoleMap {
    /// Construct a role map, refusing malformed, out-of-prompt, or overlapping
    /// labelled spans.
    pub fn new(
        prompt_token_count: usize,
        bos_spans: Vec<TokenSpan>,
        relation_spans: Vec<TokenSpan>,
        entity_spans: Vec<TokenSpan>,
        final_prompt_position: usize,
    ) -> Result<Self, AttributionIdentityError> {
        let role_map = Self {
            schema: PROMPT_ROLE_MAP_SCHEMA.to_string(),
            prompt_token_count,
            bos_spans,
            relation_spans,
            entity_spans,
            final_prompt_position,
        };
        role_map.validate()?;
        Ok(role_map)
    }

    /// Frozen schema named by this value.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Validate construction-time span authority.
    pub fn validate(&self) -> Result<(), AttributionIdentityError> {
        require_schema(PROMPT_ROLE_MAP_SCHEMA, &self.schema)?;
        if self.final_prompt_position >= self.prompt_token_count {
            return Err(AttributionIdentityError::FinalPromptPositionOutsidePrompt {
                position: self.final_prompt_position,
                prompt_len: self.prompt_token_count,
            });
        }

        let mut labelled = Vec::new();
        for (role, spans) in [
            (SourceRole::Bos, self.bos_spans.as_slice()),
            (SourceRole::Relation, self.relation_spans.as_slice()),
            (SourceRole::Entity, self.entity_spans.as_slice()),
        ] {
            for span in spans {
                if span.start >= span.end {
                    return Err(AttributionIdentityError::InvalidSpan {
                        start: span.start,
                        end: span.end,
                    });
                }
                if span.end > self.prompt_token_count {
                    return Err(AttributionIdentityError::SpanOutsidePrompt {
                        start: span.start,
                        end: span.end,
                        prompt_len: self.prompt_token_count,
                    });
                }
                labelled.push((role, *span));
            }
        }

        for left_index in 0..labelled.len() {
            let (left_role, left) = labelled[left_index];
            for &(right_role, right) in &labelled[left_index + 1..] {
                if left.overlaps(right) {
                    return Err(AttributionIdentityError::OverlappingRoleSpans {
                        left_role,
                        left_start: left.start,
                        left_end: left.end,
                        right_role,
                        right_start: right.start,
                        right_end: right.end,
                    });
                }
            }
        }
        Ok(())
    }

    /// Stable identity of the prompt-structure declaration.
    pub fn identity(&self) -> Result<String, AttributionIdentityError> {
        self.validate()?;
        Ok(hash_bytes(&canonical_json_bytes(self)?))
    }

    /// Assign a role under the frozen precedence rule.
    ///
    /// `visible_source_range` comes from the exact observation.  A position
    /// outside it is refused rather than clipped or silently mapped to
    /// [`SourceRole::Other`].  Generated positions inside the visible range
    /// but beyond the prompt are `OTHER`.
    pub fn role_at(
        &self,
        position: usize,
        visible_source_range: TokenSpan,
    ) -> Result<SourceRole, AttributionIdentityError> {
        self.validate()?;
        if visible_source_range.start >= visible_source_range.end {
            return Err(AttributionIdentityError::InvalidSpan {
                start: visible_source_range.start,
                end: visible_source_range.end,
            });
        }
        if !visible_source_range.contains(position) {
            return Err(
                AttributionIdentityError::SourcePositionOutsideVisibleRange {
                    position,
                    start: visible_source_range.start,
                    end: visible_source_range.end,
                },
            );
        }

        if contains_position(&self.bos_spans, position) {
            return Ok(SourceRole::Bos);
        }
        if contains_position(&self.relation_spans, position) {
            return Ok(SourceRole::Relation);
        }
        if contains_position(&self.entity_spans, position) {
            return Ok(SourceRole::Entity);
        }
        if position == self.final_prompt_position {
            return Ok(SourceRole::Last);
        }
        Ok(SourceRole::Other)
    }
}

impl SourceRole {
    pub(super) fn ordered() -> &'static [Self; 5] {
        &[
            Self::Bos,
            Self::Relation,
            Self::Entity,
            Self::Last,
            Self::Other,
        ]
    }
}

pub(super) fn validate_site_evidence(
    evidence: &AttentionSiteEvidence,
    hidden: usize,
    readers: &[AttributionReader],
) -> Result<(), DescriptiveTransformError> {
    if evidence.visible_source_range.start >= evidence.visible_source_range.end {
        return Err(AttributionIdentityError::InvalidSpan {
            start: evidence.visible_source_range.start,
            end: evidence.visible_source_range.end,
        }
        .into());
    }
    require_width("recorded_delta", evidence.recorded_delta.len(), hidden)?;
    require_finite("recorded_delta", &evidence.recorded_delta)?;
    if evidence.heads.len() != evidence.expected_query_heads {
        return Err(DescriptiveTransformError::HeadCoverage {
            expected: evidence.expected_query_heads,
            actual: evidence.heads.len(),
        });
    }
    let mut seen_heads = BTreeSet::new();
    let mut observed_heads = Vec::with_capacity(evidence.heads.len());
    for head in &evidence.heads {
        if !seen_heads.insert(head.query_head) {
            return Err(DescriptiveTransformError::DuplicateHead {
                head: head.query_head,
            });
        }
        observed_heads.push(head.query_head);
        require_finite("mixed_values", &head.mixed_values)?;
        if !head.sink.is_finite() {
            return Err(DescriptiveTransformError::NonFinite { field: "sink" });
        }
        if let Some(gate) = &head.gate {
            require_width("head_gate", gate.len(), head.mixed_values.len())?;
            require_finite("head_gate", gate)?;
        }
        let mut seen_positions = BTreeSet::new();
        let mut positions = Vec::with_capacity(head.sources.len());
        for source in &head.sources {
            if !seen_positions.insert(source.position) {
                return Err(DescriptiveTransformError::DuplicateSource {
                    head: head.query_head,
                    position: source.position,
                });
            }
            positions.push(source.position);
            require_width(
                "source_values",
                source.values.len(),
                head.mixed_values.len(),
            )?;
            require_finite("source_values", &source.values)?;
            if !source.attention_probability.is_finite() {
                return Err(DescriptiveTransformError::NonFinite {
                    field: "attention_probability",
                });
            }
        }
        let expected = (evidence.visible_source_range.start..evidence.visible_source_range.end)
            .collect::<Vec<_>>();
        if positions != expected {
            return Err(DescriptiveTransformError::SourceCoverage {
                head: head.query_head,
                start: evidence.visible_source_range.start,
                end: evidence.visible_source_range.end,
                observed: positions,
            });
        }
    }
    let expected = (0..evidence.expected_query_heads).collect::<Vec<_>>();
    if observed_heads != expected {
        return Err(DescriptiveTransformError::NonCanonicalHeadCoverage {
            expected: evidence.expected_query_heads,
            observed: observed_heads,
        });
    }
    for reader in readers {
        require_nonempty("reader_identity", &reader.identity)?;
        require_width("reader", reader.values.len(), hidden)?;
        require_finite("reader", &reader.values)?;
    }
    Ok(())
}

pub(super) fn validate_post_transform(
    transform: &PostAttentionTransform,
    hidden: usize,
) -> Result<(), DescriptiveTransformError> {
    match transform {
        PostAttentionTransform::Identity { residual_scale } => {
            if !residual_scale.is_finite() {
                return Err(DescriptiveTransformError::NonFinite {
                    field: "residual_scale",
                });
            }
        }
        PostAttentionTransform::RmsNorm {
            epsilon,
            weight_offset,
            weights,
            residual_scale,
        } => {
            if !epsilon.is_finite() || *epsilon <= 0.0 {
                return Err(DescriptiveTransformError::InvalidRmsEpsilon { epsilon: *epsilon });
            }
            if !weight_offset.is_finite() || !residual_scale.is_finite() {
                return Err(DescriptiveTransformError::NonFinite {
                    field: "post_attention_transform",
                });
            }
            require_width("post_attention_norm_weights", weights.len(), hidden)?;
            require_finite("post_attention_norm_weights", weights)?;
        }
    }
    Ok(())
}

pub(super) fn gate_values(values: &[f32], gate: Option<&[f32]>) -> Vec<f32> {
    match gate {
        Some(gate) => values
            .iter()
            .zip(gate)
            .map(|(value, gate)| value * gate)
            .collect(),
        None => values.to_vec(),
    }
}

pub(super) fn apply_post_transform(
    values: &[f32],
    raw_output: &[f64],
    transform: &PostAttentionTransform,
) -> Vec<f32> {
    match transform {
        PostAttentionTransform::Identity { residual_scale } => {
            values.iter().map(|value| value * residual_scale).collect()
        }
        PostAttentionTransform::RmsNorm {
            epsilon,
            weight_offset,
            weights,
            residual_scale,
        } => {
            let mean_square =
                raw_output.iter().map(|value| value * value).sum::<f64>() / raw_output.len() as f64;
            let alpha = 1.0 / (mean_square + epsilon).sqrt();
            values
                .iter()
                .zip(weights)
                .map(|(value, weight)| {
                    (f64::from(*value)
                        * alpha
                        * f64::from(*weight_offset + *weight)
                        * f64::from(*residual_scale)) as f32
                })
                .collect()
        }
    }
}

pub(super) fn project_readers(readers: &[AttributionReader], values: &[f32]) -> Vec<f64> {
    readers
        .iter()
        .map(|reader| {
            reader
                .values
                .iter()
                .zip(values)
                .map(|(left, right)| f64::from(*left) * f64::from(*right))
                .sum()
        })
        .collect()
}

pub(super) fn relative_l2(reconstructed: &[f64], recorded: &[f32]) -> f64 {
    let error = reconstructed
        .iter()
        .zip(recorded)
        .map(|(left, right)| (*left - f64::from(*right)).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = recorded
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    error / norm.max(VECTOR_DENOMINATOR_FLOOR)
}

pub(super) fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(super) fn stable_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > VECTOR_DENOMINATOR_FLOOR).then_some(numerator / denominator)
}

pub(super) fn require_width(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), DescriptiveTransformError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DescriptiveTransformError::Width {
            field,
            actual,
            expected,
        })
    }
}

pub(super) fn require_finite(
    field: &'static str,
    values: &[f32],
) -> Result<(), DescriptiveTransformError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(DescriptiveTransformError::NonFinite { field })
    }
}

pub(super) fn contains_position(spans: &[TokenSpan], position: usize) -> bool {
    spans.iter().any(|span| span.contains(position))
}

pub(super) fn require_nonempty(
    field: &'static str,
    value: &str,
) -> Result<(), AttributionIdentityError> {
    if value.trim().is_empty() {
        Err(AttributionIdentityError::EmptyField { field })
    } else {
        Ok(())
    }
}

pub(super) fn require_schema(
    expected: &'static str,
    actual: &str,
) -> Result<(), AttributionIdentityError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AttributionIdentityError::WrongSchema {
            expected,
            actual: actual.to_string(),
        })
    }
}

pub(super) fn canonical_json_bytes<T: Serialize>(
    value: &T,
) -> Result<Vec<u8>, AttributionIdentityError> {
    let value = serde_json::to_value(value)
        .map_err(|error| AttributionIdentityError::Serialization(error.to_string()))?;
    let mut out = String::new();
    write_canonical_json(&value, &mut out);
    Ok(out.into_bytes())
}

pub(super) fn write_canonical_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical_json(&map[key], out);
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        leaf => out.push_str(&leaf.to_string()),
    }
}

pub(super) fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}
