use super::*;
use crate::test_utils::{
    make_test_q4k_vindex, make_test_q4k_weights, make_test_tokenizer, Q4K_TEST_HIDDEN,
    Q4K_TEST_INTER, Q4K_TEST_NUM_LAYERS, Q4K_TEST_VOCAB,
};
use larql_compute::CpuBackend;
use larql_models::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

fn rand_mat(rows: usize, cols: usize, seed: u64) -> WeightArray {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_shared()
}

/// Llama-style fixture: same dimensions as the Gemma 3 fixture but
/// `model_type=llama` so `arch.activation()` returns SiLU instead
/// of GeluTanh. Exercises the SiLU branch in
/// `run_ffn_decode_step_q4k_direct`.
fn make_llama_q4k_weights() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;
    let arch_json = serde_json::json!({
        "model_type": "llama",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": Q4K_TEST_NUM_LAYERS,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "silu",
        "rope_theta": 10000.0,
    });
    let arch = detect_from_json(&arch_json);
    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut seed = 0xc0ffee_u64.wrapping_mul(31);
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let embed = rand_mat(Q4K_TEST_VOCAB, Q4K_TEST_HIDDEN, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());
    vectors.insert(
        arch.final_norm_key().to_string(),
        vec![1.0; Q4K_TEST_HIDDEN],
    );
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    for layer in 0..Q4K_TEST_NUM_LAYERS {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat(q_dim, Q4K_TEST_HIDDEN, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat(kv_dim, Q4K_TEST_HIDDEN, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat(kv_dim, Q4K_TEST_HIDDEN, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat(Q4K_TEST_HIDDEN, q_dim, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat(Q4K_TEST_HIDDEN, Q4K_TEST_INTER, next_seed()),
        );
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; Q4K_TEST_HIDDEN]);
        vectors.insert(
            arch.post_attention_layernorm_key(layer),
            vec![1.0; Q4K_TEST_HIDDEN],
        );
    }
    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: Q4K_TEST_NUM_LAYERS,
        hidden_size: Q4K_TEST_HIDDEN,
        intermediate_size: Q4K_TEST_INTER,
        vocab_size: Q4K_TEST_VOCAB,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

/// Direct decode step on a SiLU-activation arch — exercises the
/// non-GeluTanh branch in `run_ffn_decode_step_q4k_direct`.
#[test]
fn predict_kquant_decode_step_direct_silu_activation_path() {
    let mut weights = make_llama_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let _tok = make_test_tokenizer(weights.vocab_size);
    let token_ids = vec![1u32, 2];
    let (_, mut cache, _) = predict_kquant_prefill(&mut weights, &token_ids, &index);
    let backend = CpuBackend;
    let h_new = predict_kquant_decode_step_direct(
        &mut weights,
        3,
        &index,
        &backend,
        &mut cache,
        token_ids.len(),
    )
    .expect("SiLU direct decode step must succeed");
    assert!(h_new.iter().all(|v| v.is_finite()));
}

/// `predict_kquant_decode_step` (dequant path) on the same SiLU
/// fixture — exercises `run_ffn`'s SiLU branch.
#[test]
fn predict_kquant_decode_step_silu_activation_path() {
    let mut weights = make_llama_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let token_ids = vec![1u32, 2];
    let (_, mut cache, _) = predict_kquant_prefill(&mut weights, &token_ids, &index);
    let (h_new, _) = predict_kquant_decode_step(&weights, 3, &index, &mut cache, token_ids.len())
        .expect("SiLU dequant decode step must succeed");
    assert!(h_new.iter().all(|v| v.is_finite()));
}

/// `CachedTimings::merge` is private; verify the public `add`
/// wrapper covers it (both should sum into `dequant_ms`).
#[test]
fn cached_timings_default_starts_at_zero() {
    let t = CachedTimings::default();
    assert_eq!(t.dequant_ms, 0.0);
}

/// Padded-down handling: when the stored down rows are wider than the
/// layer's `intermediate` (256-padded — the 26B-A4B hybrid-MoE dense
/// slab stores intermediate 2112 as 2304-col rows), the direct FFN
/// step derives the stored width from the byte length, zero-pads the
/// activation, and produces the same output as the unpadded layout:
/// the real 256-element quant blocks are bit-identical and the pad
/// blocks multiply zero activations.
#[test]
fn ffn_decode_step_native_padded_down_matches_unpadded() {
    use crate::test_utils::arc_mmap_from_bytes;
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);
    let hidden = weights.hidden_size;
    let h_post_attn = ndarray::Array2::from_shape_vec(
        (1, hidden),
        (0..hidden)
            .map(|i| ((i as f32) * 0.013).sin() * 0.05)
            .collect(),
    )
    .unwrap();
    let backend = CpuBackend;
    let baseline = ffn_decode_step_native(&weights, &index, &backend, &h_post_attn, 0)
        .expect("unpadded direct FFN step");

    // Rebuild the interleaved storage with every down matrix stored
    // 256-padded: [hidden, inter] → [hidden, inter + 256], zero cols.
    let arch = &*weights.arch;
    let mut payload: Vec<u8> = Vec::new();
    let mut manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..weights.num_layers {
        for (key, pad) in [
            (arch.ffn_gate_key(layer), false),
            (arch.ffn_up_key(layer), false),
            (arch.ffn_down_key(layer), true),
        ] {
            let tensor = weights
                .tensors
                .get(&key)
                .unwrap_or_else(|| panic!("missing tensor {key}"));
            let bytes = if pad {
                let rows = tensor.shape()[0];
                let cols = tensor.shape()[1];
                let padded_cols = cols + 256;
                let mut padded = vec![0.0f32; rows * padded_cols];
                for r in 0..rows {
                    let src = tensor.row(r).to_vec();
                    padded[r * padded_cols..r * padded_cols + cols].copy_from_slice(&src);
                }
                quantize_q4_k(&padded)
            } else {
                quantize_q4_k(tensor.as_slice().expect("contiguous row-major"))
            };
            let offset = payload.len();
            manifest.push((offset, bytes.len(), "Q4_K".to_string()));
            payload.extend_from_slice(&bytes);
        }
    }
    let mut index_padded = make_test_q4k_vindex(&weights);
    {
        let storage = std::sync::Arc::make_mut(&mut index_padded.storage);
        storage.set_interleaved_kquant(arc_mmap_from_bytes(&payload), Some(manifest));
    }
    assert_eq!(
        index_padded.num_features(0),
        index.num_features(0),
        "down padding must not change the derived intermediate width \
         (num_features comes from the gate manifest)"
    );

    let padded_out = ffn_decode_step_native(&weights, &index_padded, &backend, &h_post_attn, 0)
        .expect("padded direct FFN step");

    let max_abs = baseline
        .iter()
        .zip(padded_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 1e-5,
        "padded-down output must match unpadded layout (max_abs={max_abs})"
    );
}

/// The padded-down derivation must reject byte lengths that aren't a
/// whole number of super-blocks per row (corrupt / mismatched store)
/// rather than computing with a truncated width.
#[test]
fn ffn_decode_step_native_rejects_ragged_down_bytes() {
    use crate::test_utils::arc_mmap_from_bytes;
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let weights = make_test_q4k_weights();
    let arch = &*weights.arch;
    let mut payload: Vec<u8> = Vec::new();
    let mut manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..weights.num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let tensor = weights.tensors.get(&key).expect("fixture tensor");
            let mut bytes = quantize_q4_k(tensor.as_slice().expect("contiguous"));
            if key == arch.ffn_down_key(layer) {
                bytes.truncate(bytes.len() - 7); // ragged: not a whole super-block
            }
            let offset = payload.len();
            manifest.push((offset, bytes.len(), "Q4_K".to_string()));
            payload.extend_from_slice(&bytes);
        }
    }
    let mut index = make_test_q4k_vindex(&weights);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_interleaved_kquant(arc_mmap_from_bytes(&payload), Some(manifest));
    }
    let h = ndarray::Array2::<f32>::from_elem((1, weights.hidden_size), 0.01);
    assert!(
        ffn_decode_step_native(&weights, &index, &CpuBackend, &h, 0).is_none(),
        "ragged down byte length must fall back (return None), not mis-stride"
    );
}
