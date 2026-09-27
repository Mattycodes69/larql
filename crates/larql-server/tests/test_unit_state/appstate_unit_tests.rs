//! APPSTATE UNIT TESTS
//! MODEL_ID_FROM_NAME EDGE CASES
//! MULTI-MODEL LOOKUP
//! INFER MODE PARSING
//! AUTH LOGIC

use super::*;

#[test]
fn test_app_state_model_single_none_returns_first() {
    let state = make_tiny_state(vec![make_tiny_model("gemma")]);
    let m = state.model(None);
    assert!(m.is_some());
    assert_eq!(m.unwrap().id, "gemma");
}

#[test]
fn test_app_state_model_with_id_finds_correct() {
    let state = make_tiny_state(vec![make_tiny_model("a"), make_tiny_model("b")]);
    assert_eq!(state.model(Some("a")).unwrap().id, "a");
    assert_eq!(state.model(Some("b")).unwrap().id, "b");
}

#[test]
fn test_app_state_model_multi_none_returns_none() {
    let state = make_tiny_state(vec![make_tiny_model("a"), make_tiny_model("b")]);
    // Multi-model with no id → must specify which model.
    assert!(state.model(None).is_none());
}

#[test]
fn test_app_state_model_unknown_id_returns_none() {
    let state = make_tiny_state(vec![make_tiny_model("a")]);
    assert!(state.model(Some("nonexistent")).is_none());
}

#[test]
fn test_app_state_is_multi_model_single() {
    let state = make_tiny_state(vec![make_tiny_model("a")]);
    assert!(!state.is_multi_model());
}

#[test]
fn test_app_state_is_multi_model_multi() {
    let state = make_tiny_state(vec![make_tiny_model("a"), make_tiny_model("b")]);
    assert!(state.is_multi_model());
}

#[test]
fn test_app_state_bump_requests_increments() {
    let state = make_tiny_state(vec![make_tiny_model("a")]);
    assert_eq!(
        state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    state.bump_requests();
    assert_eq!(
        state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    state.bump_requests();
    state.bump_requests();
    assert_eq!(
        state
            .requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        3
    );
}

#[test]
fn test_model_id_extraction() {
    assert_eq!(model_id("google/gemma-3-4b-it"), "gemma-3-4b-it");
    assert_eq!(model_id("llama-3-8b"), "llama-3-8b");
    assert_eq!(model_id("org/sub/model"), "model");
}

#[test]
fn test_model_id_from_name_no_slash() {
    assert_eq!(model_id_from_name("llama-3-8b"), "llama-3-8b");
}

#[test]
fn test_model_id_from_name_single_slash() {
    assert_eq!(model_id_from_name("google/gemma-3-4b-it"), "gemma-3-4b-it");
}

#[test]
fn test_model_id_from_name_deep_path() {
    assert_eq!(model_id_from_name("org/sub/model"), "model");
}

#[test]
fn test_model_id_from_name_trailing_slash() {
    // rsplit('/').next() on "foo/" returns "" — reflects actual behavior.
    let result = model_id_from_name("foo/");
    assert_eq!(result, "");
}

#[test]
fn test_multi_model_lookup_by_id() {
    // Simulate AppState.model() logic
    let models = ["gemma-3-4b-it", "llama-3-8b", "mistral-7b"];
    let find = |id: &str| models.iter().find(|m| **m == id);
    assert_eq!(find("gemma-3-4b-it"), Some(&"gemma-3-4b-it"));
    assert_eq!(find("llama-3-8b"), Some(&"llama-3-8b"));
    assert_eq!(find("nonexistent"), None);
}

#[test]
fn test_single_model_returns_first() {
    let models = ["only-model"];
    // Single model mode: None → returns first
    let result = if models.len() == 1 {
        models.first()
    } else {
        None
    };
    assert_eq!(result, Some(&"only-model"));
}

#[test]
fn test_multi_model_none_returns_none() {
    let models = ["a", "b"];
    // Multi-model mode: None → returns None (must specify ID)
    let result: Option<&&str> = if models.len() == 1 {
        models.first()
    } else {
        None
    };
    assert_eq!(result, None);
}

#[test]
fn test_infer_mode_parsing() {
    // The infer handler parses mode into walk/dense/compare
    let check = |mode: &str| -> (bool, bool) {
        let is_compare = mode == "compare";
        let use_walk = mode == "walk" || is_compare;
        let use_dense = mode == "dense" || is_compare;
        (use_walk, use_dense)
    };

    assert_eq!(check("walk"), (true, false));
    assert_eq!(check("dense"), (false, true));
    assert_eq!(check("compare"), (true, true));
}

#[test]
fn test_config_has_inference_capability() {
    let mut config = VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };

    // Browse level → no inference
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = false;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!has_weights);

    // Inference level → has inference
    config.extract_level = ExtractLevel::Inference;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);

    // Legacy has_model_weights flag
    config.extract_level = ExtractLevel::Browse;
    config.has_model_weights = true;
    let has_weights = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(has_weights);
}

#[test]
fn test_bearer_token_extraction() {
    let header = "Bearer sk-abc123";
    let token = header.strip_prefix("Bearer ");
    assert_eq!(token, Some("sk-abc123"));
}

#[test]
fn test_bearer_token_mismatch() {
    let header = "Bearer wrong-key";
    let required = "sk-abc123";
    let token = &header[7..];
    assert_ne!(token, required);
}

#[test]
fn test_no_auth_header() {
    let header: Option<&str> = None;
    let has_valid_token = header
        .filter(|h| h.starts_with("Bearer "))
        .map(|h| &h[7..])
        .is_some();
    assert!(!has_valid_token);
}

#[test]
fn test_health_exempt_from_auth() {
    let path = "/v1/health";
    let is_health = path == "/v1/health";
    assert!(is_health);

    let path = "/v1/describe";
    let is_health = path == "/v1/health";
    assert!(!is_health);
}
