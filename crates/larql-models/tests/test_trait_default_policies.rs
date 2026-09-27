//! The trait defaults that answer position and residual-topology questions
//! from `config.json` alone, for a family with no override.

use larql_models::config::{PositionPolicy, ResidualTopology};
use larql_models::{detect_from_json, ModelArchitecture};
use serde_json::{json, Value};

/// A generic (no family override) config with `extra` merged in.
fn arch(extra: Value) -> Box<dyn ModelArchitecture> {
    let mut cfg = json!({
        "model_type": "an-unregistered-family",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 32,
        "rope_theta": 10000.0,
    });
    for (k, v) in extra.as_object().unwrap() {
        cfg[k] = v.clone();
    }
    detect_from_json(&cfg)
}

#[test]
fn a_partial_rotary_with_a_full_mrope_declaration_is_multi_axis() {
    let a = arch(json!({
        "partial_rotary_factor": 0.5,
        "rope_parameters": {"mrope_section": [2, 1, 1], "mrope_interleaved": true},
    }));
    match a.position_policy_for_layer(0) {
        PositionPolicy::MRope {
            section,
            interleaved,
            rotary_fraction,
            ..
        } => {
            assert_eq!(section, [2, 1, 1]);
            assert!(interleaved);
            assert_eq!(rotary_fraction, 0.5);
        }
        other => panic!("expected MRope, got {other:?}"),
    }
}

#[test]
fn an_unpaired_mrope_half_stays_a_partial_rotary() {
    let a = arch(json!({
        "partial_rotary_factor": 0.5,
        "rope_parameters": {"mrope_section": [2, 1, 1]},
    }));
    assert!(matches!(
        a.position_policy_for_layer(0),
        PositionPolicy::PartialRope { .. }
    ));
}

#[test]
fn a_llama3_scaling_declaration_resolves_to_llama3() {
    let a = arch(json!({
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0,
                         "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                         "original_max_position_embeddings": 8192},
    }));
    assert!(matches!(
        a.position_policy_for_layer(0),
        PositionPolicy::Llama3 { .. }
    ));
}

#[test]
fn a_linear_divisor_on_a_partial_rotary_keeps_the_partial_shape() {
    let a = arch(json!({
        "partial_rotary_factor": 0.5,
        "rope_scaling": {"type": "linear", "factor": 2.0},
    }));
    assert!(matches!(
        a.position_policy_for_layer(0),
        PositionPolicy::PartialRope { .. }
    ));
}

#[test]
fn a_relative_scheme_decides_the_policy_outright() {
    let a = arch(json!({"d_rel": 8, "rel_extent": 64}));
    assert!(matches!(
        a.position_policy_for_layer(0),
        PositionPolicy::Relative {
            d_rel: 8,
            extent: 64
        }
    ));
}

#[test]
fn mla_use_nope_means_no_rotation() {
    let a = arch(json!({"mla_use_nope": true}));
    assert!(matches!(
        a.position_policy_for_layer(0),
        PositionPolicy::None
    ));
}

#[test]
fn residual_topology_reads_one_programme_or_refuses() {
    assert!(matches!(
        arch(json!({"attn_res_block_size": 4})).residual_topology(),
        Ok(ResidualTopology::AttentionResidual { block_size: 4 })
    ));
    assert!(matches!(
        arch(json!({"hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6})).residual_topology(),
        Ok(ResidualTopology::HyperConnection(_))
    ));
    let both = arch(json!({"attn_res_block_size": 4, "hc_mult": 4}))
        .residual_topology()
        .unwrap_err();
    assert!(both.contains("two residual topologies"), "{both}");
    let partial = arch(json!({"hc_mult": 4})).residual_topology().unwrap_err();
    assert!(partial.contains("unjudged hyper-connection"), "{partial}");
}

#[test]
fn a_family_with_no_shared_branch_gate_names_no_operand() {
    assert_eq!(arch(json!({})).shared_expert_branch_gate_key(0), None);
}

#[test]
fn bitnet_declares_its_sub_norms_per_layer() {
    let a = detect_from_json(&json!({
        "model_type": "bitnet",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "vocab_size": 32,
    }));
    let keys = a.sub_norm_keys(1);
    assert!(!keys.is_empty());
    assert!(keys.iter().all(|k| k.contains(".1.")), "{keys:?}");
}
