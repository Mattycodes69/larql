//! Gemma 4 hybrid-MoE fixture.

use crate::{detect_from_json, ModelWeights, WeightArray};
use std::collections::HashMap;

use super::*;

//
// Minimum Q4_K-aligned hidden / intermediate / expert-intermediate for
// the Gemma 4 hybrid-MoE fixture. Q4_K requires multiples of 256.

/// Hidden dimension for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_HIDDEN: usize = 256;
/// Intermediate dimension for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_INTER: usize = 256;
/// Expert count for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_NUM_EXPERTS: usize = 4;
/// Top-k experts for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_TOP_K: usize = 2;

/// Build a synthetic Gemma 4 hybrid-MoE `ModelWeights`.
///
/// `enable_moe_block=true` plus all the per-layer dense attention + dense
/// FFN tensors a Gemma 4 26B-A4B variant carries, plus the per-layer MoE
/// pieces:
///
/// - Router projection (`vectors[layers.L.router.proj.weight]`).
/// - Packed BF16 expert `gate_up` (`raw_bytes[layers.L.experts.gate_up_proj]`).
/// - Packed BF16 expert `down`    (`raw_bytes[layers.L.experts.down_proj]`).
///
/// All weights are deterministic LCG ramps. Values are math-meaningless;
/// the fixture's job is to satisfy the runtime checks
/// (`arch.is_hybrid_moe()=true`, `weights.get_packed_bytes(...)` non-None,
/// `weights.vectors[router_key]` non-None) so the MoE forward branches
/// in `pipeline_layer::build_moe_weights` and the kquant_forward MoE
/// guards execute end-to-end on the substrate side.
pub fn make_test_gemma4_moe_weights() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = GEMMA4_MOE_HIDDEN / num_q;
    let num_layers = 2usize;

    let arch_json = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": GEMMA4_MOE_HIDDEN,
            "intermediate_size": GEMMA4_MOE_INTER,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_q,
            "num_key_value_heads": num_kv,
            "head_dim": head_dim,
            "vocab_size": GEMMA4_MOE_HIDDEN,
            "enable_moe_block": true,
            "num_experts": GEMMA4_MOE_NUM_EXPERTS,
            "top_k_experts": GEMMA4_MOE_TOP_K,
            "moe_intermediate_size": GEMMA4_MOE_INTER,
            "rope_theta": 10000.0,
        }
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

    let mut seed = 0xb000_1eef_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let hidden = GEMMA4_MOE_HIDDEN;
    let inter = GEMMA4_MOE_INTER;
    let moe_inter = GEMMA4_MOE_INTER;
    let vocab = GEMMA4_MOE_HIDDEN;

    let embed = rand_mat_seeded(vocab, hidden, 0.05, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    for layer in 0..num_layers {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(hidden, q_dim, 0.05, next_seed()),
        );

        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(inter, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(inter, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, inter, 0.05, next_seed()),
        );

        vectors.insert(arch.input_layernorm_key(layer), vec![0.5; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.5; hidden]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.layer_scalar_key(layer) {
            vectors.insert(k, vec![1.0]);
        }

        let router_key = arch
            .moe_router_key(layer)
            .expect("Gemma 4 MoE arch must produce a router key");
        let router_proj: Vec<f32> = (0..GEMMA4_MOE_NUM_EXPERTS * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect();
        vectors.insert(router_key, router_proj);

        // Packed BF16 expert gate_up.
        let gate_up_floats_per_expert = 2 * moe_inter * hidden;
        let total_gate_up_bytes = GEMMA4_MOE_NUM_EXPERTS * gate_up_floats_per_expert * 2;
        let mut gate_up_blob = vec![0u8; total_gate_up_bytes];
        for (i, chunk) in gate_up_blob.chunks_exact_mut(2).enumerate() {
            let v = (((i & 0xff) as f32 * 0.001 - 0.128) * 0.1).to_bits();
            chunk[0] = (v >> 16) as u8;
            chunk[1] = (v >> 24) as u8;
        }
        let gate_up_key = arch
            .packed_experts_gate_up_key(layer)
            .expect("Gemma 4 MoE arch must produce a packed gate_up key");
        raw_bytes.insert(gate_up_key, gate_up_blob);

        let down_floats_per_expert = hidden * moe_inter;
        let total_down_bytes = GEMMA4_MOE_NUM_EXPERTS * down_floats_per_expert * 2;
        let mut down_blob = vec![0u8; total_down_bytes];
        for (i, chunk) in down_blob.chunks_exact_mut(2).enumerate() {
            let v = (((i & 0xff) as f32 * 0.0007 - 0.09) * 0.1).to_bits();
            chunk[0] = (v >> 16) as u8;
            chunk[1] = (v >> 24) as u8;
        }
        let down_key = arch
            .packed_experts_down_key(layer)
            .expect("Gemma 4 MoE arch must produce a packed down key");
        raw_bytes.insert(down_key, down_blob);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes,
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: vocab,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}
