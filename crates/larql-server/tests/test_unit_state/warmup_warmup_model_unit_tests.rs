//! WARMUP — warmup_model unit tests
//! PROBE LABELS (load_probe_labels)
//! RELATIONS CONTENT-TOKEN FILTER
//! SERVER ERROR → HTTP RESPONSE

use super::*;

#[test]
fn warmup_model_skip_weights_sets_loaded_false() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: None,
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert!(!resp.weights_loaded);
    assert_eq!(resp.weights_load_ms, 0);
}

#[test]
fn warmup_model_with_explicit_layers_prefetches_matching() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![0]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.layers_prefetched, 1);
}

#[test]
fn warmup_model_out_of_range_layer_is_skipped() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![999]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.layers_prefetched, 0);
}

#[test]
fn warmup_model_empty_layers_list_prefetches_zero() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.layers_prefetched, 0);
}

#[test]
fn warmup_model_reports_correct_model_name() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![]),
        skip_weights: true,
        warmup_hnsw: false,
    };
    let resp = warmup_model(&model, &req);
    assert_eq!(resp.model, "test/warmup-model");
}

#[test]
fn warmup_model_weight_load_fails_gracefully() {
    use larql_server::routes::warmup::{warmup_model, WarmupRequest};
    let model = make_loaded_model_for_warmup();
    let req = WarmupRequest {
        layers: Some(vec![]),
        skip_weights: false,
        warmup_hnsw: false,
    };
    // Path is /nonexistent so get_or_load_weights fails — should warn but not panic.
    let resp = warmup_model(&model, &req);
    assert!(!resp.weights_loaded);
}

#[test]
fn test_load_probe_labels_from_json_file() {
    let dir = std::env::temp_dir().join("larql_test_labels_01");
    std::fs::create_dir_all(&dir).unwrap();
    let json = r#"{"L0_F0": "capital", "L1_F2": "language", "L5_F10": "continent"}"#;
    std::fs::write(dir.join("feature_labels.json"), json).unwrap();

    let labels = load_probe_labels(&dir);
    assert_eq!(labels.get(&(0, 0)), Some(&"capital".to_string()));
    assert_eq!(labels.get(&(1, 2)), Some(&"language".to_string()));
    assert_eq!(labels.get(&(5, 10)), Some(&"continent".to_string()));
    assert_eq!(labels.len(), 3);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_probe_labels_missing_file_returns_empty() {
    let dir = std::path::Path::new("/nonexistent/path/to/vindex");
    let labels = load_probe_labels(dir);
    assert!(labels.is_empty());
}

#[test]
fn test_load_probe_labels_malformed_json_returns_empty() {
    let dir = std::env::temp_dir().join("larql_test_labels_02");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("feature_labels.json"), b"not valid json").unwrap();

    let labels = load_probe_labels(&dir);
    assert!(labels.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_probe_labels_non_object_json_returns_empty() {
    let dir = std::env::temp_dir().join("larql_test_labels_03");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("feature_labels.json"),
        b"[\"not\",\"an\",\"object\"]",
    )
    .unwrap();

    let labels = load_probe_labels(&dir);
    assert!(labels.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_probe_labels_skips_malformed_keys() {
    let dir = std::env::temp_dir().join("larql_test_labels_04");
    std::fs::create_dir_all(&dir).unwrap();
    // Mix of valid and invalid keys
    let json = r#"{"L0_F0": "capital", "INVALID": "skip", "L_BAD_F": "skip2", "L3_F7": "valid"}"#;
    std::fs::write(dir.join("feature_labels.json"), json).unwrap();

    let labels = load_probe_labels(&dir);
    // Only L0_F0 and L3_F7 should parse.
    assert_eq!(labels.get(&(0, 0)), Some(&"capital".to_string()));
    assert_eq!(labels.get(&(3, 7)), Some(&"valid".to_string()));
    assert_eq!(labels.len(), 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_content_token_valid_words() {
    assert!(is_content_token_test("capital"));
    assert!(is_content_token_test("Paris"));
    assert!(is_content_token_test("language"));
    assert!(is_content_token_test("France"));
    assert!(is_content_token_test("Europe"));
}

#[test]
fn test_content_token_stopwords_rejected() {
    assert!(!is_content_token_test("the"));
    assert!(!is_content_token_test("and"));
    assert!(!is_content_token_test("for"));
    assert!(!is_content_token_test("with"));
    assert!(!is_content_token_test("about"));
    assert!(!is_content_token_test("should"));
}

#[test]
fn test_content_token_too_short_rejected() {
    assert!(!is_content_token_test("ab")); // < 3 chars
    assert!(!is_content_token_test("a"));
    assert!(!is_content_token_test(""));
}

#[test]
fn test_content_token_too_long_rejected() {
    let long = "a".repeat(26);
    assert!(!is_content_token_test(&long));
}

#[test]
fn test_content_token_camelcase_rejected() {
    assert!(!is_content_token_test("camelCase"));
    assert!(!is_content_token_test("camelCaseWord"));
}

#[test]
fn test_content_token_numeric_heavy_rejected() {
    // Less than 2/3 alpha characters
    assert!(!is_content_token_test("a12345"));
}

#[test]
fn test_server_error_not_found_maps_to_404() {
    let resp = ServerError::NotFound("the-thing".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
}

#[test]
fn test_server_error_bad_request_maps_to_400() {
    let resp = ServerError::BadRequest("bad input".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn test_server_error_internal_maps_to_500() {
    let resp = ServerError::Internal("oops".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
}

#[test]
fn test_server_error_conflict_maps_to_409() {
    let resp = ServerError::Conflict("already loading".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::CONFLICT);
}

#[test]
fn test_server_error_unavailable_maps_to_503() {
    #[allow(dead_code)]
    let resp = ServerError::InferenceUnavailable("no weights".into()).into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn test_server_error_display_format() {
    assert!(format!("{}", ServerError::NotFound("x".into())).contains("not found"));
    assert!(format!("{}", ServerError::BadRequest("x".into())).contains("bad request"));
    assert!(format!("{}", ServerError::Internal("x".into())).contains("internal error"));
}
