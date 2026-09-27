//! A Gemma 4 GGUF re-emits the HF keys its heterogeneous attention needs:
//! per-layer KV heads, sliding vs global head width, dual RoPE bases, the
//! sliding window, K-as-V layers and the logit softcap.

use larql_models::loading::gguf::{GgufFile, GgufTensor, GgufValue, GgufWriter};
use serde_json::Value;

const GGML_TYPE_F32: u32 = 0;
const BLOCKS: u32 = 3;

fn tensor(name: &str) -> GgufTensor {
    GgufTensor {
        name: name.into(),
        dims: vec![1],
        ggml_type: GGML_TYPE_F32,
        data: 0f32.to_le_bytes().to_vec(),
    }
}

/// A Gemma 4 GGUF with `extra` metadata and `attn_v` on `v_layers` blocks.
fn gemma4_config(extra: Vec<(&str, GgufValue)>, v_layers: u32) -> Value {
    let mut w = GgufWriter::new();
    w.meta("general.architecture", GgufValue::String("gemma4".into()))
        .meta("gemma4.block_count", GgufValue::U32(BLOCKS))
        .meta("gemma4.embedding_length", GgufValue::U32(8))
        .meta("gemma4.attention.head_count", GgufValue::U32(2));
    for (k, v) in extra {
        w.meta(k, v);
    }
    for layer in 0..v_layers {
        w.tensor(tensor(&format!("blk.{layer}.attn_v.weight")));
    }
    w.tensor(tensor("token_embd.weight"));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gemma4.gguf");
    w.write_to_file(&path).unwrap();
    GgufFile::open(&path).unwrap().to_config_json()
}

#[test]
fn heterogeneous_attention_is_re_emitted_as_hf_keys() {
    let cfg = gemma4_config(
        vec![
            (
                "gemma4.attention.head_count_kv",
                GgufValue::Array(vec![
                    GgufValue::U32(8),
                    GgufValue::U32(8),
                    GgufValue::U32(1),
                ]),
            ),
            ("gemma4.attention.key_length", GgufValue::U32(512)),
            ("gemma4.attention.key_length_swa", GgufValue::U32(256)),
            ("gemma4.rope.freq_base_swa", GgufValue::F32(10_000.0)),
            ("gemma4.attention.sliding_window", GgufValue::U32(1024)),
            ("gemma4.final_logit_softcapping", GgufValue::F32(30.0)),
        ],
        BLOCKS - 1,
    );
    assert_eq!(cfg["num_key_value_heads"], 8);
    assert_eq!(cfg["num_global_key_value_heads"], 1);
    assert_eq!(cfg["head_dim"], 256);
    assert_eq!(cfg["global_head_dim"], 512);
    assert_eq!(cfg["partial_rotary_factor"], 0.25);
    assert_eq!(cfg["rope_local_base_freq"], 10_000.0);
    assert_eq!(cfg["sliding_window"], 1024);
    assert_eq!(cfg["attention_k_eq_v"], true);
    assert_eq!(cfg["final_logit_softcapping"], 30.0);
}

#[test]
fn an_undeclared_sliding_width_takes_the_known_export_width() {
    let cfg = gemma4_config(vec![], BLOCKS);
    assert_eq!(cfg["head_dim"], 256);
    assert!(cfg.get("global_head_dim").is_none());
    // attn_v on every layer: no layer reuses K as V.
    assert!(cfg.get("attention_k_eq_v").is_none());
}

#[test]
fn uniform_kv_heads_declare_no_global_count() {
    let cfg = gemma4_config(
        vec![(
            "gemma4.attention.head_count_kv",
            GgufValue::Array(vec![GgufValue::U32(4); BLOCKS as usize]),
        )],
        BLOCKS,
    );
    assert_eq!(cfg["num_key_value_heads"], 4);
    assert!(cfg.get("num_global_key_value_heads").is_none());
}
