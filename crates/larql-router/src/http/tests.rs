use super::*;

#[test]
fn is_binary_content_type_recognises_marker_prefix() {
    assert!(is_binary_content_type(BINARY_CT));
    assert!(is_binary_content_type(
        "application/x-larql-ffn; charset=utf-8"
    ));
    assert!(!is_binary_content_type("application/json"));
    assert!(!is_binary_content_type(""));
}

#[test]
fn extract_layers_from_binary_body() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&42u32.to_le_bytes());
    let (layers, model) = extract_layers_and_model_id(&buf, true).unwrap();
    assert_eq!(layers, vec![42]);
    assert!(model.is_none());
}

#[test]
fn extract_layers_binary_truncated_returns_err() {
    let err = extract_layers_and_model_id(&[], true).unwrap_err();
    assert!(err.contains("truncated"));
}

#[test]
fn extract_layers_from_json_array() {
    let body = br#"{"layers":[0,1,2],"model_id":"gemma"}"#;
    let (layers, model) = extract_layers_and_model_id(body, false).unwrap();
    assert_eq!(layers, vec![0, 1, 2]);
    assert_eq!(model.as_deref(), Some("gemma"));
}

#[test]
fn extract_layers_from_json_scalar() {
    let body = br#"{"layer":7}"#;
    let (layers, model) = extract_layers_and_model_id(body, false).unwrap();
    assert_eq!(layers, vec![7]);
    assert!(model.is_none());
}

#[test]
fn extract_layers_json_missing_fields_errors() {
    let body = br#"{"foo":"bar"}"#;
    let err = extract_layers_and_model_id(body, false).unwrap_err();
    assert!(err.contains("must provide"));
}

#[test]
fn extract_layers_invalid_json_errors() {
    let err = extract_layers_and_model_id(b"not json", false).unwrap_err();
    assert!(err.contains("invalid JSON"));
}

#[test]
fn extract_layers_json_filters_non_numeric_entries() {
    let body = br#"{"layers":[0,"oops",2]}"#;
    let (layers, _) = extract_layers_and_model_id(body, false).unwrap();
    assert_eq!(layers, vec![0, 2]);
}

// ── ADR-0018: extract_request_spec_and_model_id ─────────────────────────

#[test]
fn extract_spec_dense_single_layer() {
    let body = br#"{"layer":3}"#;
    let (spec, model) = extract_request_spec_and_model_id(body, false).unwrap();
    assert_eq!(spec, RequestSpec::Dense(vec![3]));
    assert!(model.is_none());
}

#[test]
fn extract_spec_dense_multi_layer() {
    let body = br#"{"layers":[0,1,2],"model_id":"m"}"#;
    let (spec, model) = extract_request_spec_and_model_id(body, false).unwrap();
    assert_eq!(spec, RequestSpec::Dense(vec![0, 1, 2]));
    assert_eq!(model.as_deref(), Some("m"));
}

#[test]
fn extract_spec_moe_single_layer() {
    let body = br#"{"layer":5,"experts":[0,3,7]}"#;
    let (spec, model) = extract_request_spec_and_model_id(body, false).unwrap();
    assert_eq!(spec, RequestSpec::Moe(vec![(5, vec![0, 3, 7])]));
    assert!(model.is_none());
}

#[test]
fn extract_spec_moe_multi_layer() {
    let body = br#"{"layer_experts":[{"layer":5,"experts":[0,3]},{"layer":6,"experts":[1,5]}]}"#;
    let (spec, _) = extract_request_spec_and_model_id(body, false).unwrap();
    assert_eq!(
        spec,
        RequestSpec::Moe(vec![(5, vec![0, 3]), (6, vec![1, 5])])
    );
}

#[test]
fn extract_spec_moe_layer_experts_takes_priority_over_single_form() {
    // If both shapes are present, the multi-layer form wins.
    let body = br#"{"layer":99,"experts":[0],"layer_experts":[{"layer":5,"experts":[0,3]}]}"#;
    let (spec, _) = extract_request_spec_and_model_id(body, false).unwrap();
    assert_eq!(spec, RequestSpec::Moe(vec![(5, vec![0, 3])]));
}

#[test]
fn extract_spec_moe_experts_without_layer_errors() {
    let body = br#"{"experts":[0,3]}"#;
    let err = extract_request_spec_and_model_id(body, false).unwrap_err();
    assert!(err.contains("requires a 'layer'"));
}

#[test]
fn extract_spec_moe_empty_experts_errors() {
    let body = br#"{"layer":5,"experts":[]}"#;
    let err = extract_request_spec_and_model_id(body, false).unwrap_err();
    assert!(err.contains("empty"));
}

#[test]
fn extract_spec_moe_layer_experts_missing_field_errors() {
    let body = br#"{"layer_experts":[{"layer":5}]}"#;
    let err = extract_request_spec_and_model_id(body, false).unwrap_err();
    assert!(err.contains("'experts' array"));
}

#[test]
fn extract_spec_binary_is_always_dense() {
    // Binary bodies bypass JSON parsing entirely.
    // Encode "single layer 9": just 4 LE bytes of u32 = 9.
    let body = (9u32).to_le_bytes();
    let (spec, _) = extract_request_spec_and_model_id(&body, true).unwrap();
    assert_eq!(spec, RequestSpec::Dense(vec![9]));
}

#[test]
fn request_spec_is_empty_branches() {
    assert!(RequestSpec::Dense(vec![]).is_empty());
    assert!(!RequestSpec::Dense(vec![0]).is_empty());
    assert!(RequestSpec::Moe(vec![]).is_empty());
    // All-empty experts → still treated as empty.
    assert!(RequestSpec::Moe(vec![(5, vec![])]).is_empty());
    assert!(!RequestSpec::Moe(vec![(5, vec![1])]).is_empty());
}

#[test]
fn extract_layers_legacy_helper_rejects_moe_bodies() {
    // The dense-only wrapper around the new parser surfaces a clean
    // error for MoE bodies rather than silently accepting them.
    let body = br#"{"layer":5,"experts":[0]}"#;
    let err = extract_layers_and_model_id(body, false).unwrap_err();
    assert!(err.contains("MoE request"));
}

#[test]
fn metrics_response_renders_text_or_a_named_encoder_failure() {
    use axum::http::{header, StatusCode};

    let ok = metrics_response(Ok("larql_router_build_info 1\n".into()));
    assert_eq!(ok.status(), StatusCode::OK);
    assert_eq!(
        ok.headers()[header::CONTENT_TYPE],
        "text/plain; version=0.0.4; charset=utf-8"
    );

    let failed = metrics_response(Err(prometheus::Error::Msg("bad label".into())));
    assert_eq!(failed.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        failed.headers()[header::CONTENT_TYPE],
        "text/plain; charset=utf-8"
    );
}
