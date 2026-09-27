use super::*;

#[test]
fn make_test_weights_basic_shape() {
    let w = make_test_weights();
    assert_eq!(w.hidden_size, 16);
    assert_eq!(w.intermediate_size, 32);
    assert_eq!(w.vocab_size, 32);
    assert_eq!(w.num_layers, 2);
    assert_eq!(w.num_q_heads, 2);
    assert_eq!(w.num_kv_heads, 1);
    assert_eq!(w.head_dim, 8);
}

#[test]
fn make_test_weights_embed_matrix_shape() {
    let w = make_test_weights();
    assert_eq!(w.embed.shape(), &[32, 16]);
    assert_eq!(w.lm_head.shape(), &[32, 16]);
}

#[test]
fn make_test_weights_per_layer_tensors_present() {
    let w = make_test_weights();
    for layer in 0..w.num_layers {
        assert!(
            w.tensors.contains_key(&w.arch.attn_q_key(layer)),
            "missing q-proj for layer {layer}"
        );
        assert!(
            w.tensors.contains_key(&w.arch.attn_k_key(layer)),
            "missing k-proj for layer {layer}"
        );
        assert!(
            w.tensors.contains_key(&w.arch.attn_v_key(layer)),
            "missing v-proj for layer {layer}"
        );
        assert!(
            w.tensors.contains_key(&w.arch.attn_o_key(layer)),
            "missing o-proj for layer {layer}"
        );
        assert!(
            w.tensors.contains_key(&w.arch.ffn_gate_key(layer)),
            "missing ffn-gate for layer {layer}"
        );
        assert!(
            w.tensors.contains_key(&w.arch.ffn_up_key(layer)),
            "missing ffn-up for layer {layer}"
        );
        assert!(
            w.tensors.contains_key(&w.arch.ffn_down_key(layer)),
            "missing ffn-down for layer {layer}"
        );
        assert!(
            w.vectors.contains_key(&w.arch.input_layernorm_key(layer)),
            "missing input-norm for layer {layer}"
        );
        assert!(
            w.vectors
                .contains_key(&w.arch.post_attention_layernorm_key(layer)),
            "missing post-attn-norm for layer {layer}"
        );
    }
}

#[test]
fn make_test_weights_final_norm_is_ones() {
    let w = make_test_weights();
    let final_norm = w
        .vectors
        .get(w.arch.final_norm_key())
        .expect("final norm missing");
    assert_eq!(final_norm.len(), w.hidden_size);
    assert!(final_norm.iter().all(|v| (*v - 1.0).abs() < 1e-9));
}

#[test]
fn make_test_weights_deterministic_across_calls() {
    // The LCG seed is fixed (0xdeadbeef), so two independent calls
    // must produce identical weight tensors. Pin this so future
    // refactors don't accidentally introduce per-call randomness.
    let a = make_test_weights();
    let b = make_test_weights();
    for layer in 0..a.num_layers {
        let key = a.arch.attn_q_key(layer);
        let ta = a.tensors.get(&key).unwrap();
        let tb = b.tensors.get(&key).unwrap();
        assert_eq!(ta.shape(), tb.shape());
        for (x, y) in ta.iter().zip(tb.iter()) {
            assert!(
                (x - y).abs() < f32::EPSILON,
                "non-deterministic tensor at layer {layer}"
            );
        }
    }
}

#[test]
fn make_test_weights_values_in_expected_range() {
    // All weights are LCG-sampled in [-0.1, 0.1]. Pin the magnitude
    // so future scale tweaks are caught.
    let w = make_test_weights();
    for v in w.embed.iter() {
        assert!(
            v.abs() <= 0.1 + 1e-6,
            "embed value outside [-0.1, 0.1]: {v}"
        );
    }
}
