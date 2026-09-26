//! Describing attention support over token spans.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

/// Apply the frozen ATTR-1D attention algebra.
///
/// Event coverage and dimensions are refused before arithmetic.  Gate values
/// multiply both the independently observed mixed head and every source row;
/// source attention probability is never treated as contribution by itself.
pub fn describe_attention_support(
    evidence: &AttentionSiteEvidence,
    prepared: &dyn PreparedAttentionProjection,
    readers: &[AttributionReader],
    role_map: &PromptRoleMap,
) -> Result<AttentionDescriptiveSupport, DescriptiveTransformError> {
    role_map.validate()?;
    validate_site_evidence(evidence, prepared.hidden_size(), readers)?;

    let hidden = prepared.hidden_size();
    let bias = prepared
        .output_bias()
        .map_err(DescriptiveTransformError::Projection)?
        .unwrap_or_else(|| vec![0.0; hidden]);
    require_width("output_bias", bias.len(), hidden)?;
    require_finite("output_bias", &bias)?;
    let post = prepared
        .post_attention_transform()
        .map_err(DescriptiveTransformError::Projection)?;
    validate_post_transform(&post, hidden)?;

    struct RawHead {
        query_head: usize,
        kv_head: usize,
        sink: f32,
        probability_error: f64,
        contribution: Vec<f32>,
        sources: Vec<(usize, SourceRole, f32, Vec<f32>)>,
    }

    let mut raw_heads = Vec::with_capacity(evidence.heads.len());
    for head in &evidence.heads {
        let gated_mixed = gate_values(&head.mixed_values, head.gate.as_deref());
        let contribution = prepared
            .project_head(head.query_head, &gated_mixed)
            .map_err(DescriptiveTransformError::Projection)?;
        require_width("head_projection", contribution.len(), hidden)?;
        require_finite("head_projection", &contribution)?;

        let mut sources = Vec::with_capacity(head.sources.len());
        for source in &head.sources {
            let weighted: Vec<f32> = source
                .values
                .iter()
                .map(|value| source.attention_probability * value)
                .collect();
            let gated = gate_values(&weighted, head.gate.as_deref());
            let projected = prepared
                .project_head(head.query_head, &gated)
                .map_err(DescriptiveTransformError::Projection)?;
            require_width("source_projection", projected.len(), hidden)?;
            require_finite("source_projection", &projected)?;
            let role = role_map.role_at(source.position, evidence.visible_source_range)?;
            sources.push((
                source.position,
                role,
                source.attention_probability,
                projected,
            ));
        }
        let probability_sum = head
            .sources
            .iter()
            .map(|source| f64::from(source.attention_probability))
            .sum::<f64>()
            + f64::from(head.sink);
        raw_heads.push(RawHead {
            query_head: head.query_head,
            kv_head: head.kv_head,
            sink: head.sink,
            probability_error: (probability_sum - 1.0).abs(),
            contribution,
            sources,
        });
    }

    // The RMS scalar is computed over the independently observed head mixtures
    // plus the once-only bias, exactly as the executor's head reader does.
    let mut raw_output = bias
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    for head in &raw_heads {
        for (sum, value) in raw_output.iter_mut().zip(&head.contribution) {
            *sum += f64::from(*value);
        }
    }
    let applied_bias = apply_post_transform(&bias, &raw_output, &post);

    struct AppliedHead {
        query_head: usize,
        kv_head: usize,
        sink: f32,
        probability_error: f64,
        contribution: Vec<f32>,
        sources: Vec<(usize, SourceRole, f32, Vec<f32>)>,
    }
    let applied_heads = raw_heads
        .into_iter()
        .map(|head| AppliedHead {
            query_head: head.query_head,
            kv_head: head.kv_head,
            sink: head.sink,
            probability_error: head.probability_error,
            contribution: apply_post_transform(&head.contribution, &raw_output, &post),
            sources: head
                .sources
                .into_iter()
                .map(|(position, role, probability, contribution)| {
                    (
                        position,
                        role,
                        probability,
                        apply_post_transform(&contribution, &raw_output, &post),
                    )
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let mut reconstructed = applied_bias
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    for head in &applied_heads {
        for (sum, value) in reconstructed.iter_mut().zip(&head.contribution) {
            *sum += f64::from(*value);
        }
    }
    let reconstructed_delta = reconstructed
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let head_reconstruction_relative_l2 = relative_l2(&reconstructed, &evidence.recorded_delta);

    let site_values = project_readers(readers, &evidence.recorded_delta);
    let bias_values = project_readers(readers, &applied_bias);
    let head_values = applied_heads
        .iter()
        .map(|head| project_readers(readers, &head.contribution))
        .collect::<Vec<_>>();
    let absolute_head_denominators = readers
        .iter()
        .enumerate()
        .map(|(reader, _)| {
            bias_values[reader].abs()
                + head_values
                    .iter()
                    .map(|values| values[reader].abs())
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();

    let mut heads = Vec::with_capacity(applied_heads.len());
    for (head_index, head) in applied_heads.iter().enumerate() {
        let source_values = head
            .sources
            .iter()
            .map(|(_, _, _, contribution)| project_readers(readers, contribution))
            .collect::<Vec<_>>();
        let absolute_source_denominators = readers
            .iter()
            .enumerate()
            .map(|(reader, _)| {
                source_values
                    .iter()
                    .map(|values| values[reader].abs())
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();

        let sources = head
            .sources
            .iter()
            .zip(&source_values)
            .map(
                |((position, role, probability, contribution), values)| SourceContribution {
                    position: *position,
                    source_role: *role,
                    attention_probability: *probability,
                    contribution: contribution.clone(),
                    contribution_norm: l2_norm(contribution),
                    selected_token_projection: readers
                        .iter()
                        .enumerate()
                        .map(|(reader_index, reader)| ProjectionMeasurement {
                            reader_identity: reader.identity.clone(),
                            value: values[reader_index],
                            signed_share: stable_ratio(
                                values[reader_index],
                                head_values[head_index][reader_index],
                            ),
                            absolute_mass_share: stable_ratio(
                                values[reader_index].abs(),
                                absolute_source_denominators[reader_index],
                            ),
                        })
                        .collect(),
                },
            )
            .collect::<Vec<_>>();

        let source_role_projection = SourceRole::ordered()
            .iter()
            .filter_map(|role| {
                let indices = sources
                    .iter()
                    .enumerate()
                    .filter_map(|(index, source)| (source.source_role == *role).then_some(index))
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    return None;
                }
                Some(SourceRoleContribution {
                    source_role: *role,
                    selected_token_projection: readers
                        .iter()
                        .enumerate()
                        .map(|(reader_index, reader)| {
                            let value = indices
                                .iter()
                                .map(|index| source_values[*index][reader_index])
                                .sum::<f64>();
                            ProjectionMeasurement {
                                reader_identity: reader.identity.clone(),
                                value,
                                signed_share: stable_ratio(
                                    value,
                                    head_values[head_index][reader_index],
                                ),
                                absolute_mass_share: stable_ratio(
                                    value.abs(),
                                    absolute_source_denominators[reader_index],
                                ),
                            }
                        })
                        .collect(),
                })
            })
            .collect();

        let mut source_sum = vec![0.0f64; hidden];
        for (_, _, _, contribution) in &head.sources {
            for (sum, value) in source_sum.iter_mut().zip(contribution) {
                *sum += f64::from(*value);
            }
        }
        heads.push(HeadContribution {
            query_head: head.query_head,
            kv_head: head.kv_head,
            sink: head.sink,
            attention_probability_error: head.probability_error,
            contribution: head.contribution.clone(),
            contribution_norm: l2_norm(&head.contribution),
            selected_token_projection: readers
                .iter()
                .enumerate()
                .map(|(reader_index, reader)| ProjectionMeasurement {
                    reader_identity: reader.identity.clone(),
                    value: head_values[head_index][reader_index],
                    signed_share: stable_ratio(
                        head_values[head_index][reader_index],
                        site_values[reader_index],
                    ),
                    absolute_mass_share: stable_ratio(
                        head_values[head_index][reader_index].abs(),
                        absolute_head_denominators[reader_index],
                    ),
                })
                .collect(),
            sources,
            source_role_projection,
            source_reconstruction_relative_l2: relative_l2(&source_sum, &head.contribution),
        });
    }

    let projection_reconstruction = readers
        .iter()
        .enumerate()
        .map(|(reader_index, reader)| {
            let reconstructed = bias_values[reader_index]
                + head_values
                    .iter()
                    .map(|values| values[reader_index])
                    .sum::<f64>();
            let absolute_terms = bias_values[reader_index].abs()
                + head_values
                    .iter()
                    .map(|values| values[reader_index].abs())
                    .sum::<f64>();
            ProjectionReconstruction {
                reader_identity: reader.identity.clone(),
                absolute_error: (reconstructed - site_values[reader_index]).abs(),
                permitted_error: PROJECTION_RECONSTRUCTION_SCALE * absolute_terms.max(1.0),
            }
        })
        .collect();

    Ok(AttentionDescriptiveSupport {
        attr1d_contract_identity: ATTR1D_CONTRACT_IDENTITY.to_string(),
        site_contribution: evidence.recorded_delta.clone(),
        bias_contribution: applied_bias,
        reconstructed_delta,
        head_reconstruction_relative_l2,
        site_projection: readers
            .iter()
            .enumerate()
            .map(|(reader_index, reader)| ProjectionMeasurement {
                reader_identity: reader.identity.clone(),
                value: site_values[reader_index],
                signed_share: stable_ratio(site_values[reader_index], site_values[reader_index]),
                absolute_mass_share: None,
            })
            .collect(),
        bias_projection: readers
            .iter()
            .enumerate()
            .map(|(reader_index, reader)| ProjectionMeasurement {
                reader_identity: reader.identity.clone(),
                value: bias_values[reader_index],
                signed_share: stable_ratio(bias_values[reader_index], site_values[reader_index]),
                absolute_mass_share: stable_ratio(
                    bias_values[reader_index].abs(),
                    absolute_head_denominators[reader_index],
                ),
            })
            .collect(),
        heads,
        projection_reconstruction,
    })
}

/// A half-open token span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

impl TokenSpan {
    /// Construct a non-empty half-open span.
    pub fn new(start: usize, end: usize) -> Result<Self, AttributionIdentityError> {
        if start >= end {
            return Err(AttributionIdentityError::InvalidSpan { start, end });
        }
        Ok(Self { start, end })
    }

    /// Whether this span contains one token position.
    pub fn contains(self, position: usize) -> bool {
        self.start <= position && position < self.end
    }

    pub(super) fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

/// Construction-time prompt structure used to assign source roles.
///
/// The manifest can be serialized and hashed before executing the model.  Its
/// role assignment does not inspect token text, attention, contribution, or
/// outcomes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptRoleMap {
    pub(super) schema: String,
    pub prompt_token_count: usize,
    pub bos_spans: Vec<TokenSpan>,
    pub relation_spans: Vec<TokenSpan>,
    pub entity_spans: Vec<TokenSpan>,
    pub final_prompt_position: usize,
}
