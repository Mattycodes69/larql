//! Tests for [`super`].
//!
//! Split out of `mod.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

//! End-to-end coverage tests for the three small kquant_forward
//! files (`walk_ffn`, `tensors`, `hooks`) driven against the
//! Q4K fixture index. Each test reaches into the file under test
//! through its public entry point; llvm-cov attributes line
//! execution to the file containing the line, not the test.
use super::*;
use crate::test_fixtures::make_q4k_fixture_index;
use larql_models::test_fixtures::{make_test_q4k_weights, make_test_q4k_weights_silu};
use ndarray::Array2;

//
// Pure MoE has no dense FFN slab; requiring `interleaved_kquant`
// rejected those models before the forward reached its MoE branch
// (the drifted twin of the larql-inference copy). These pin both
// sides of the gate.

/// KvIndex double with attention data only — the shape a pure-MoE
/// vindex presents (its `interleaved_kquant.bin` is empty).
struct AttnOnlyIndex {
    attn: Vec<Vec<u8>>,
}
impl crate::KvIndex for AttnOnlyIndex {
    fn num_features(&self, _l: usize) -> usize {
        16
    }
    fn attn_kquant_layer_data(&self, _l: usize) -> Option<[(&[u8], &str); 4]> {
        Some([
            (self.attn[0].as_slice(), "Q4_K"),
            (self.attn[1].as_slice(), "Q4_K"),
            (self.attn[2].as_slice(), "Q4_K"),
            (self.attn[3].as_slice(), "Q4_K"),
        ])
    }
    fn interleaved_kquant_layer_data(
        &self,
        _l: usize,
    ) -> Option<[(&[u8], &str); crate::FFN_COMPONENTS_PER_LAYER]> {
        None
    }
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        None
    }
}

fn attn_only_fixture() -> (larql_models::ModelWeights, AttnOnlyIndex) {
    // hidden 16, 4 heads × head_dim 4 → every attn matrix is 16×16
    // = 256 elements, one Q4_K super-block.
    let mut weights = larql_models::test_fixtures::make_test_weights();
    weights.arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
    }));
    weights.hidden_size = 16;
    weights.num_layers = 1;
    let ones = vec![1.0f32; 16 * 16];
    let q = crate::cpu::ops::q4_common::quantize_q4_k(&ones);
    let index = AttnOnlyIndex {
        attn: vec![q.clone(), q.clone(), q.clone(), q],
    };
    (weights, index)
}

mod cached_rs_cpu_forward_paths;
mod tensors_rs_the_pure_moe_arm;
