//! MLA absorption

use super::*;

#[test]
fn mla_projections_are_absorbed_into_standard_qkvo() {
    // DeepSeek-V2-style geometry, tiny: 2 Q heads over 1 KV head.
    const MLA_HIDDEN: usize = 8;
    const QK_NOPE: usize = 2;
    const QK_ROPE: usize = 2;
    const V_HEAD: usize = 2;
    const KV_LORA: usize = 4;
    const Q_LORA: usize = 4;
    const QK_HD: usize = QK_NOPE + QK_ROPE;

    let tmp = tempfile::tempdir().unwrap();
    let mut weights = empty_model_weights(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": MLA_HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": MLA_HIDDEN,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": QK_HD,
        "kv_lora_rank": KV_LORA,
        "q_lora_rank": Q_LORA,
        "qk_nope_head_dim": QK_NOPE,
        "qk_rope_head_dim": QK_ROPE,
        "v_head_dim": V_HEAD,
        "vocab_size": VOCAB,
    }));
    assert!(weights.arch.uses_mla(), "fixture must exercise MLA");
    let inserts = [
        (
            weights.arch.mla_kv_a_key(0).unwrap(),
            fill(KV_LORA + QK_ROPE, MLA_HIDDEN, 1.0),
        ),
        (
            weights.arch.mla_kv_b_key(0).unwrap(),
            fill(NUM_KV_HEADS * (QK_NOPE + V_HEAD), KV_LORA, 2.0),
        ),
        (
            weights.arch.mla_q_a_key(0).unwrap(),
            fill(Q_LORA, MLA_HIDDEN, 3.0),
        ),
        (
            weights.arch.mla_q_b_key(0).unwrap(),
            fill(NUM_Q_HEADS * QK_HD, Q_LORA, 4.0),
        ),
        (
            weights.arch.attn_o_key(0),
            fill(MLA_HIDDEN, NUM_Q_HEADS * V_HEAD, 5.0),
        ),
    ];
    for (key, t) in inserts {
        weights.tensors.insert(key, t.into_shared());
    }

    // Attention-only tier: no FFN / lm_head needed, no index.json update
    // hazards beyond the standard one.
    let scaffold = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &scaffold);
    let opts = WriteWeightsOptions {
        level: crate::ExtractLevel::Attention,
        ..Default::default()
    };
    write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts).unwrap();

    let entries = manifest_entries(tmp.path());
    let shape_of = |key: &str| -> Vec<usize> {
        entries
            .iter()
            .find(|e| e.key == key)
            .unwrap_or_else(|| panic!("missing {key}"))
            .shape
            .clone()
    };
    // Absorbed tensors land under the standard names with dense shapes.
    assert_eq!(
        shape_of(&weights.arch.attn_q_key(0)),
        vec![NUM_Q_HEADS * QK_HD, MLA_HIDDEN]
    );
    assert_eq!(
        shape_of(&weights.arch.attn_k_key(0)),
        vec![NUM_KV_HEADS * QK_HD, MLA_HIDDEN]
    );
    assert_eq!(
        shape_of(&weights.arch.attn_v_key(0)),
        vec![NUM_KV_HEADS * V_HEAD, MLA_HIDDEN]
    );
    // O is passed through untouched — verify bytes.
    let o_key = weights.arch.attn_o_key(0);
    let o_entry = entries.iter().find(|e| e.key == o_key).unwrap();
    let attn_bytes = std::fs::read(tmp.path().join(ATTN_WEIGHTS_BIN)).unwrap();
    let raw = &attn_bytes[o_entry.offset as usize..(o_entry.offset + o_entry.length) as usize];
    let decoded = crate::config::dtype::decode_floats(raw, StorageDtype::F32);
    assert_eq!(
        decoded,
        weights.tensors.get(&o_key).unwrap().as_slice().unwrap()
    );
}
