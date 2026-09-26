use super::*;

#[derive(Clone)]
struct SyntheticProjection {
    bias: Option<Vec<f32>>,
    post: PostAttentionTransform,
}

impl PreparedAttentionProjection for SyntheticProjection {
    fn hidden_size(&self) -> usize {
        2
    }

    fn project_head(&self, query_head: usize, values: &[f32]) -> Result<Vec<f32>, String> {
        if query_head != 0 {
            return Err(format!("unexpected head {query_head}"));
        }
        if values.len() != 2 {
            return Err(format!("unexpected width {}", values.len()));
        }
        // Identity is sufficient to test the decomposition law. The
        // production adapter is responsible for using effective W_O.
        Ok(values.to_vec())
    }

    fn output_bias(&self) -> Result<Option<Vec<f32>>, String> {
        Ok(self.bias.clone())
    }

    fn post_attention_transform(&self) -> Result<PostAttentionTransform, String> {
        Ok(self.post.clone())
    }
}

fn span(start: usize, end: usize) -> TokenSpan {
    TokenSpan::new(start, end).unwrap()
}

fn context() -> SupportCoordinateContext {
    SupportCoordinateContext::new(
        "model-system",
        "logical-plan",
        "text-component",
        "prompt-layout",
        12,
        24,
    )
    .unwrap()
}

fn role_map() -> PromptRoleMap {
    PromptRoleMap::new(8, vec![span(0, 1)], vec![span(2, 4)], vec![span(6, 8)], 7).unwrap()
}

fn small_role_map() -> PromptRoleMap {
    PromptRoleMap::new(2, vec![span(0, 1)], vec![], vec![span(1, 2)], 1).unwrap()
}

fn readers() -> Vec<AttributionReader> {
    vec![
        AttributionReader::new("reader-x", vec![1.0, 0.0]).unwrap(),
        AttributionReader::new("reader-y", vec![0.0, 1.0]).unwrap(),
    ]
}

fn one_head_evidence(recorded_delta: Vec<f32>) -> AttentionSiteEvidence {
    AttentionSiteEvidence {
        expected_query_heads: 1,
        visible_source_range: span(0, 2),
        recorded_delta,
        heads: vec![AttentionHeadEvidence {
            query_head: 0,
            kv_head: 0,
            sink: 0.0,
            // .25*[2,4] + .75*[4,0]
            mixed_values: vec![3.5, 1.0],
            gate: Some(vec![2.0, 0.5]),
            sources: vec![
                AttentionSourceEvidence {
                    position: 0,
                    attention_probability: 0.25,
                    values: vec![2.0, 4.0],
                },
                AttentionSourceEvidence {
                    position: 1,
                    attention_probability: 0.75,
                    values: vec![4.0, 0.0],
                },
            ],
        }],
    }
}

#[test]
fn coordinate_serialization_is_flat_and_excludes_execution_identity() {
    let map = role_map();
    let coordinate = SupportCoordinate::new(
        context(),
        SupportAddress::Source {
            query_head: 3,
            kv_head: 1,
            source_start: 6,
            source_end: 7,
            source_role: SourceRole::Entity,
            role_map_identity: map.identity().unwrap(),
        },
    )
    .unwrap();
    let value = serde_json::to_value(&coordinate).unwrap();
    let object = value.as_object().unwrap();

    assert_eq!(object["coordinate_kind"], "source");
    assert_eq!(object["operator"], "attention");
    assert_eq!(object["source_role"], "ENTITY");
    assert!(!object.contains_key("address"));
    for forbidden in [
        "execution_identity",
        "provenance_fingerprint",
        "prepared_image_fingerprint",
        "observation_receipt_digest",
        "contribution",
        "transition_identity",
        "causal_label",
    ] {
        assert!(!object.contains_key(forbidden), "found {forbidden}");
    }
}

#[test]
fn the_same_coordinate_has_distinct_backend_observations() {
    let coordinate = SupportCoordinate::new(
        context(),
        SupportAddress::Head {
            query_head: 3,
            kv_head: 1,
        },
    )
    .unwrap();
    let reference = SupportObservationIdentity::new(
        &coordinate,
        "reference-execution",
        "reference-provenance",
        "reference-image",
        "reference-receipt",
        42,
    )
    .unwrap();
    let production = SupportObservationIdentity::new(
        &coordinate,
        "production-execution",
        "production-provenance",
        "production-image",
        "production-receipt",
        42,
    )
    .unwrap();

    assert_eq!(
        reference.support_coordinate_id,
        production.support_coordinate_id
    );
    assert_ne!(
        reference.identity().unwrap(),
        production.identity().unwrap()
    );
}

#[test]
fn replay_reproduces_canonical_bytes_and_both_identities() {
    let coordinate = SupportCoordinate::new(
        context(),
        SupportAddress::Head {
            query_head: 3,
            kv_head: 1,
        },
    )
    .unwrap();
    let observation = SupportObservationIdentity::new(
        &coordinate,
        "execution",
        "provenance",
        "prepared-image",
        "receipt",
        19,
    )
    .unwrap();

    let coordinate_replay: SupportCoordinate =
        serde_json::from_slice(&coordinate.canonical_bytes().unwrap()).unwrap();
    let observation_replay: SupportObservationIdentity =
        serde_json::from_slice(&observation.canonical_bytes().unwrap()).unwrap();

    assert_eq!(
        coordinate.identity().unwrap(),
        coordinate_replay.identity().unwrap()
    );
    assert_eq!(
        coordinate.canonical_bytes().unwrap(),
        coordinate_replay.canonical_bytes().unwrap()
    );
    assert_eq!(
        observation.identity().unwrap(),
        observation_replay.identity().unwrap()
    );
    assert_eq!(
        observation.canonical_bytes().unwrap(),
        observation_replay.canonical_bytes().unwrap()
    );
}

#[test]
fn canonical_json_sorts_keys_instead_of_using_struct_order() {
    let coordinate = SupportCoordinate::new(context(), SupportAddress::Site).unwrap();
    let text = String::from_utf8(coordinate.canonical_bytes().unwrap()).unwrap();
    assert!(text.starts_with("{\"component_identity\":"), "{text}");
    assert!(text.find("\"layer\":").unwrap() < text.find("\"operator\":").unwrap());
    assert!(text.find("\"schema\":").unwrap() < text.find("\"site\":").unwrap());
}

#[test]
fn source_coordinates_are_exactly_one_token() {
    let error = SupportCoordinate::new(
        context(),
        SupportAddress::Source {
            query_head: 0,
            kv_head: 0,
            source_start: 3,
            source_end: 5,
            source_role: SourceRole::Other,
            role_map_identity: "role-map".to_string(),
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        AttributionIdentityError::SourceCoordinateNotOneToken { start: 3, end: 5 }
    );
}

#[test]
fn role_map_refuses_any_label_overlap() {
    let error =
        PromptRoleMap::new(8, vec![span(0, 1)], vec![span(2, 5)], vec![span(4, 7)], 7).unwrap_err();
    assert!(matches!(
        error,
        AttributionIdentityError::OverlappingRoleSpans {
            left_role: SourceRole::Relation,
            right_role: SourceRole::Entity,
            ..
        }
    ));
}

#[test]
fn declared_roles_precede_last_and_unlabelled_positions_are_other() {
    let map = role_map();
    let visible = span(0, 10);
    assert_eq!(map.role_at(0, visible).unwrap(), SourceRole::Bos);
    assert_eq!(map.role_at(2, visible).unwrap(), SourceRole::Relation);
    assert_eq!(map.role_at(7, visible).unwrap(), SourceRole::Entity);
    assert_eq!(map.role_at(5, visible).unwrap(), SourceRole::Other);
    assert_eq!(map.role_at(9, visible).unwrap(), SourceRole::Other);

    let last_map =
        PromptRoleMap::new(8, vec![span(0, 1)], vec![span(2, 4)], vec![span(5, 7)], 7).unwrap();
    assert_eq!(last_map.role_at(7, visible).unwrap(), SourceRole::Last);
}

#[test]
fn source_positions_outside_the_visible_range_refuse() {
    let error = role_map().role_at(1, span(2, 8)).unwrap_err();
    assert_eq!(
        error,
        AttributionIdentityError::SourcePositionOutsideVisibleRange {
            position: 1,
            start: 2,
            end: 8,
        }
    );
}

#[test]
fn deserialized_wrong_schemas_cannot_be_hashed() {
    let coordinate = SupportCoordinate::new(context(), SupportAddress::Site).unwrap();
    let mut value = serde_json::to_value(coordinate).unwrap();
    value["schema"] = serde_json::Value::String("not-the-schema".to_string());
    let malformed: SupportCoordinate = serde_json::from_value(value).unwrap();
    assert!(matches!(
        malformed.identity(),
        Err(AttributionIdentityError::WrongSchema { .. })
    ));
}

#[test]
fn descriptive_transform_reconstructs_gated_sources_bias_and_shares() {
    let prepared = SyntheticProjection {
        bias: Some(vec![0.1, -0.2]),
        post: PostAttentionTransform::Identity {
            residual_scale: 0.5,
        },
    };
    // Gated head [7,.5] and bias [.1,-.2], all scaled by .5.
    let evidence = one_head_evidence(vec![3.55, 0.15]);
    let result =
        describe_attention_support(&evidence, &prepared, &readers(), &small_role_map()).unwrap();

    assert!(result.passes_numerical_gates(), "{result:#?}");
    assert!(result.head_reconstruction_relative_l2 < 1e-7);
    assert_eq!(result.heads[0].contribution, vec![3.5, 0.25]);
    assert_eq!(result.bias_contribution, vec![0.05, -0.1]);
    assert_eq!(result.heads[0].sources[0].contribution, vec![0.5, 0.25]);
    assert_eq!(result.heads[0].sources[1].contribution, vec![3.0, 0.0]);
    assert_eq!(result.heads[0].sources[0].source_role, SourceRole::Bos);
    assert_eq!(result.heads[0].sources[1].source_role, SourceRole::Entity);
    assert_eq!(result.heads[0].attention_probability_error, 0.0);
    assert!(result.heads[0].source_reconstruction_relative_l2 < 1e-12);

    // Attention mass is descriptive input, not the contribution: the
    // .25 source carries .5 on reader-x, while the .75 source carries 3.
    assert_eq!(
        result.heads[0].sources[0].selected_token_projection[0].value,
        0.5
    );
    assert_eq!(
        result.heads[0].sources[1].selected_token_projection[0].value,
        3.0
    );
    let signed_share = result.heads[0].selected_token_projection[0]
        .signed_share
        .unwrap();
    assert!((signed_share - 3.5 / 3.55).abs() < 1e-7);
}

#[test]
fn rms_transform_uses_the_once_only_raw_output_scalar() {
    let epsilon = 1e-5;
    let raw_output = [7.1f64, 0.3];
    let alpha =
        1.0 / ((raw_output.iter().map(|value| value * value).sum::<f64>() / 2.0) + epsilon).sqrt();
    let expected = vec![
        (7.1 * alpha * 1.0 * 0.5) as f32,
        (0.3 * alpha * 2.0 * 0.5) as f32,
    ];
    let prepared = SyntheticProjection {
        bias: Some(vec![0.1, -0.2]),
        post: PostAttentionTransform::RmsNorm {
            epsilon,
            weight_offset: 1.0,
            weights: vec![0.0, 1.0],
            residual_scale: 0.5,
        },
    };
    let result = describe_attention_support(
        &one_head_evidence(expected),
        &prepared,
        &readers(),
        &small_role_map(),
    )
    .unwrap();

    assert!(result.passes_numerical_gates(), "{result:#?}");
    assert!(result.head_reconstruction_relative_l2 < 1e-6);
    assert!(result.heads[0].source_reconstruction_relative_l2 < 1e-6);
}

#[test]
fn incomplete_source_surface_refuses_before_projection() {
    let prepared = SyntheticProjection {
        bias: None,
        post: PostAttentionTransform::Identity {
            residual_scale: 1.0,
        },
    };
    let mut evidence = one_head_evidence(vec![7.0, 0.5]);
    evidence.heads[0].sources.pop();
    let error = describe_attention_support(&evidence, &prepared, &readers(), &small_role_map())
        .unwrap_err();
    assert!(matches!(
        error,
        DescriptiveTransformError::SourceCoverage { head: 0, .. }
    ));
}

#[test]
fn source_sidecar_binds_receipt_prepared_image_and_event_sequence() {
    let sidecar = AttentionSourceEvidenceSidecar::new(
        SupportCoordinateContext::new(
            "model-system",
            "logical-plan",
            "text-component",
            "prompt-layout",
            1,
            24,
        )
        .unwrap(),
        "execution",
        "provenance",
        "prepared-image",
        "receipt",
        17,
        one_head_evidence(vec![7.0, 0.5]),
    )
    .unwrap();
    let replay: AttentionSourceEvidenceSidecar =
        serde_json::from_slice(&sidecar.canonical_bytes().unwrap()).unwrap();
    assert_eq!(sidecar.identity().unwrap(), replay.identity().unwrap());
    assert_eq!(
        sidecar.support_coordinate_id,
        sidecar.support_observation.support_coordinate_id
    );

    let mut tampered = replay;
    tampered.support_observation.event_sequence = 18;
    assert!(matches!(
        tampered.validate(),
        Err(AttributionIdentityError::IdentityBindingMismatch {
            field: "support_observation_id"
        })
    ));
}
