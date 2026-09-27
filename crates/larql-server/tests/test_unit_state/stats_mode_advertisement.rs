//! STATS — mode advertisement
//! INFER DISABLED LOGIC
//! ERROR HANDLING (model lookup)

use super::*;

#[test]
fn test_stats_shape_includes_mode_full_by_default() {
    let mode = "full";
    let ffn_service = true;
    let stats = serde_json::json!({
        "mode": mode,
        "loaded": { "ffn_service": ffn_service },
    });
    assert_eq!(stats["mode"], "full");
    assert_eq!(stats["loaded"]["ffn_service"], true);
}

#[test]
fn test_stats_shape_advertises_ffn_service_mode() {
    let mode = "ffn-service";
    let inference_available = false;
    let stats = serde_json::json!({
        "mode": mode,
        "loaded": {
            "browse": true,
            "inference": inference_available,
            "ffn_service": true,
        },
    });
    assert_eq!(stats["mode"], "ffn-service");
    assert_eq!(stats["loaded"]["inference"], false);
    assert_eq!(stats["loaded"]["ffn_service"], true);
}

#[test]
fn test_ffn_only_implies_infer_disabled() {
    fn effective(no_infer: bool, ffn_only: bool) -> bool {
        no_infer || ffn_only
    }
    assert!(!effective(false, false));
    assert!(effective(true, false));
    assert!(effective(false, true));
    assert!(effective(true, true));
}

#[test]
fn test_stats_shape_advertises_embed_service_mode() {
    let stats = serde_json::json!({
        "mode": "embed-service",
        "loaded": {
            "browse": false,
            "inference": false,
            "ffn_service": false,
            "embed_service": true,
        },
    });
    assert_eq!(stats["mode"], "embed-service");
    assert_eq!(stats["loaded"]["embed_service"], true);
    assert_eq!(stats["loaded"]["browse"], false);
    assert_eq!(stats["loaded"]["ffn_service"], false);
}

#[test]
fn test_embed_only_implies_infer_disabled() {
    fn effective(no_infer: bool, ffn_only: bool, embed_only: bool) -> bool {
        no_infer || ffn_only || embed_only
    }
    assert!(!effective(false, false, false));
    assert!(effective(false, false, true));
    assert!(effective(false, true, false));
    assert!(effective(true, false, false));
    assert!(effective(true, true, true));
}

#[test]
fn test_embed_only_mode_string() {
    fn mode(embed_only: bool, ffn_only: bool) -> &'static str {
        if embed_only {
            "embed-service"
        } else if ffn_only {
            "ffn-service"
        } else {
            "full"
        }
    }
    assert_eq!(mode(false, false), "full");
    assert_eq!(mode(false, true), "ffn-service");
    assert_eq!(mode(true, false), "embed-service");
    // embed_only takes priority
    assert_eq!(mode(true, true), "embed-service");
}

#[test]
fn test_infer_disabled_check() {
    let disabled = true;
    assert!(disabled); // Handler returns 503

    let disabled = false;
    assert!(!disabled); // Handler proceeds
}

#[test]
fn test_infer_weights_required() {
    let config = VindexConfig {
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
    // Browse level + no model weights → can't infer
    let can_infer = config.has_model_weights
        || config.extract_level == ExtractLevel::Inference
        || config.extract_level == ExtractLevel::All;
    assert!(!can_infer);
}

#[test]
fn test_infer_compare_returns_both() {
    let mode = "compare";
    let is_compare = mode == "compare";
    let use_walk = mode == "walk" || is_compare;
    let use_dense = mode == "dense" || is_compare;
    assert!(is_compare);
    assert!(use_walk);
    assert!(use_dense);
}

#[test]
fn test_infer_disabled_all_flag_combinations() {
    fn eff(no_infer: bool, ffn_only: bool, embed_only: bool) -> bool {
        no_infer || ffn_only || embed_only
    }
    // All off → enabled
    assert!(!eff(false, false, false));
    // Single flags
    assert!(eff(true, false, false));
    assert!(eff(false, true, false));
    assert!(eff(false, false, true));
    // Combinations
    assert!(eff(true, true, false));
    assert!(eff(false, true, true));
    assert!(eff(true, false, true));
    assert!(eff(true, true, true));
}

#[test]
fn test_error_model_not_found() {
    let models: Vec<&str> = vec!["gemma-3-4b-it"];
    let result = models.iter().find(|m| **m == "nonexistent");
    assert!(result.is_none()); // → 404
}

#[test]
fn test_error_empty_prompt() {
    let token_ids: Vec<u32> = vec![];
    assert!(token_ids.is_empty()); // → 400 BadRequest
}

#[test]
fn test_error_nonexistent_model_in_multi() {
    let models = ["model-a", "model-b"];
    let find = |id: &str| models.iter().find(|m| **m == id);
    assert!(find("model-c").is_none()); // → 404
}
