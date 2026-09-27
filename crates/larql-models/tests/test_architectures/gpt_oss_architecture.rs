//! GPT-OSS architecture
//! Mixtral — PerExpert format comparison

use super::*;

#[test]
fn gpt_oss_detection() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.family(), "gpt_oss");
    assert_eq!(arch.config().num_layers, 36);
    assert_eq!(arch.config().hidden_size, 2880);
}

#[test]
fn gpt_oss_is_moe() {
    let arch = gpt_oss_arch();
    assert!(arch.is_moe());
    assert_eq!(arch.num_experts(), 128);
    assert_eq!(arch.num_experts_per_token(), 4);
}

#[test]
fn gpt_oss_expert_format() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.expert_format(), ExpertFormat::PackedMxfp4);
}

#[test]
fn gpt_oss_packed_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.packed_gate_up_blocks_key(5).unwrap(),
        "layers.5.mlp.experts.gate_up_proj_blocks"
    );
    assert_eq!(
        arch.packed_gate_up_scales_key(5).unwrap(),
        "layers.5.mlp.experts.gate_up_proj_scales"
    );
    assert_eq!(
        arch.packed_down_blocks_key(5).unwrap(),
        "layers.5.mlp.experts.down_proj_blocks"
    );
    assert_eq!(
        arch.packed_down_scales_key(5).unwrap(),
        "layers.5.mlp.experts.down_proj_scales"
    );
}

#[test]
fn gpt_oss_router_key() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.moe_router_key(0).unwrap(),
        "layers.0.mlp.router.weight"
    );
}

#[test]
fn gpt_oss_attn_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.attn_q_key(3), "layers.3.self_attn.q_proj.weight");
    assert_eq!(arch.attn_k_key(3), "layers.3.self_attn.k_proj.weight");
    assert_eq!(arch.attn_v_key(3), "layers.3.self_attn.v_proj.weight");
    assert_eq!(arch.attn_o_key(3), "layers.3.self_attn.o_proj.weight");
}

/// GPT-OSS's per-expert keys describe **loaded** state, not the checkpoint.
///
/// On disk there are no per-expert tensors — everything is packed and fused,
/// which is why this test previously asserted `is_none()`. But the safetensors
/// loader dequantises and de-interleaves the experts at load time and stores
/// each one separately, so a compute backend reading `ModelWeights` does see
/// per-expert weights. Advertising them is what lets a generic per-expert FFN
/// backend serve this model without knowing anything about MXFP4 — and until
/// 2026-07-30 nothing could, which is why `shannon score` could not score it.
/// Callers that read the *checkpoint* (extraction) still want `packed_*`.
/// See `docs/k3-funnel.md` §4.7.
#[test]
fn gpt_oss_exposes_dequantised_per_expert_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.expert_ffn_gate_key(0, 5).as_deref(),
        Some("layers.0.block_sparse_moe.experts.5.w1.weight")
    );
    assert_eq!(
        arch.expert_ffn_up_key(0, 5).as_deref(),
        Some("layers.0.block_sparse_moe.experts.5.w3.weight")
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 5).as_deref(),
        Some("layers.0.block_sparse_moe.experts.5.w2.weight")
    );
}

/// The packed keys remain the checkpoint's own layout, unchanged by the above.
#[test]
fn gpt_oss_still_reports_packed_checkpoint_keys() {
    let arch = gpt_oss_arch();
    assert_eq!(
        arch.packed_gate_up_blocks_key(0).as_deref(),
        Some("layers.0.mlp.experts.gate_up_proj_blocks")
    );
    assert_eq!(
        arch.expert_format(),
        larql_models::ExpertFormat::PackedMxfp4
    );
}

#[test]
fn gpt_oss_prefix_strip() {
    let arch = gpt_oss_arch();
    assert_eq!(arch.key_prefixes_to_strip(), &["model."]);
}

#[test]
fn mixtral_expert_format() {
    let arch = mixtral_arch();
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
}

#[test]
fn mixtral_per_expert_keys() {
    let arch = mixtral_arch();
    assert_eq!(
        arch.expert_ffn_gate_key(0, 3).unwrap(),
        "layers.0.block_sparse_moe.experts.3.w1.weight"
    );
    assert_eq!(
        arch.expert_ffn_down_key(0, 3).unwrap(),
        "layers.0.block_sparse_moe.experts.3.w2.weight"
    );
}

#[test]
fn mixtral_no_packed_keys() {
    let arch = mixtral_arch();
    assert!(arch.packed_gate_up_blocks_key(0).is_none());
}

#[test]
fn mixtral_config_accessor_returns_loaded_config() {
    let arch = mixtral_arch();
    let config = arch.config();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.intermediate_size, 14336);
}
