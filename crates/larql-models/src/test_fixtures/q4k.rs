//! Q4_K-quantised fixtures.

use crate::{detect_from_json, ModelWeights, WeightArray};
use std::collections::HashMap;

use super::*;

//
// `make_test_weights` uses hidden=16, below Q4_K's 256-element
// super-block minimum. The cached / direct-matvec decode paths in
// `vindex/kquant_forward/cached.rs` require a vindex with real
// `attn_kquant_layer_data` + `interleaved_kquant_layer_data` manifests,
// so unit tests for those paths can't fit the tiny fixture. The
// helpers below build a hidden=256, intermediate=256 Gemma 3-style
// fixture with synthetic Q4_K bytes that round-trip through
// `larql_compute::cpu::ops::q4_common::quantize_q4_k`.

/// Hidden dimension for the Q4_K test fixture — minimum Q4_K-safe
/// multiple of 256.
pub const Q4K_TEST_HIDDEN: usize = 256;
/// Intermediate dimension for the Q4_K test fixture.
pub const Q4K_TEST_INTER: usize = 256;
/// Vocabulary size for the Q4_K test fixture.
pub const Q4K_TEST_VOCAB: usize = 256;
/// Layer count for the Q4_K test fixture.
pub const Q4K_TEST_NUM_LAYERS: usize = 2;
/// Query-head count for the Q4_K test fixtures.
pub const Q4K_TEST_NUM_Q: usize = 4;
/// K/V-head count for the Q4_K test fixtures (GQA reps = 2).
pub const Q4K_TEST_NUM_KV: usize = 2;
/// Wide FFN width for the Q4_K fixture: threshold tests that route the
/// walk's parallel Q4K-down branch need `hits ≥ 512` while staying
/// below the full-K gemv rewrite at 80% density — so
/// `intermediate > 512 / 0.8 = 640`; 768 is the next 256-multiple
/// (Q4_K super-block constraint).
pub const Q4K_TEST_INTER_WIDE: usize = 768;

/// Build a synthetic `ModelWeights` sized to satisfy Q4_K's 256-element
/// super-block constraint. Uses Gemma 3 architecture so the
/// `has_post_norms` + `GeluTanh` branches in the cached decode path
/// are exercised. `Q4K_TEST_NUM_LAYERS` layers.
pub fn make_test_q4k_weights() -> ModelWeights {
    make_test_q4k_weights_layers(Q4K_TEST_NUM_LAYERS)
}

/// Layer-parametrised sibling of [`make_test_q4k_weights`]. Identical
/// arch / dims, but with `num_layers` decoder layers — for tests that
/// need a depth-fraction layer index to land inside the model (e.g. the
/// LQL FR3 relation resolver, whose probe layer clamps to ≥3, so it needs
/// a model deeper than the default 2-layer fixture).
pub fn make_test_q4k_weights_layers(num_layers: usize) -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    q4k_test_weights_from_json(
        arch_json,
        num_layers,
        Q4K_TEST_INTER,
        Q4K_TEST_HIDDEN,
        Q4K_TEST_NUM_Q,
        Q4K_TEST_NUM_KV,
    )
}

/// Wide-FFN sibling of [`make_test_q4k_weights`]: same Gemma 3 arch and
/// hidden size, but `intermediate_size = Q4K_TEST_INTER_WIDE` (768) and a
/// single layer (the wide FFN triples quantisation cost per layer). Built
/// for walk-engine threshold tests that need a route of ≥ 512 features
/// without triggering the 80%-density full-K gemv rewrite.
pub fn make_test_q4k_weights_wide() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;
    let num_layers = 1usize;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": Q4K_TEST_INTER_WIDE,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    q4k_test_weights_from_json(
        arch_json,
        num_layers,
        Q4K_TEST_INTER_WIDE,
        Q4K_TEST_HIDDEN,
        Q4K_TEST_NUM_Q,
        Q4K_TEST_NUM_KV,
    )
}

/// Rope-scaled sibling of [`make_test_q4k_weights`]: Gemma-3 arch at the
/// same Q4_K-compatible dims, with `sliding_window` + flat linear
/// `rope_scaling` (factor 8) and **6 layers**, so the 5:1 local:global
/// pattern puts a full-attention layer (position divisor 8) inside the
/// model. This is the fixture for direct-vs-staged decode parity on the
/// scaled-RoPE path — [`make_gemma3_rope_scaled_test_weights`] covers the
/// same arch shape but at hidden=16, below Q4_K's 256-element super-block
/// floor, so it cannot drive the direct-matvec kernels.
pub fn make_test_q4k_weights_rope_scaled() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;
    let num_layers = 6usize;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
        "sliding_window": 512,
        "rope_scaling": {"rope_type": "linear", "factor": 8.0},
    });
    q4k_test_weights_from_json(
        arch_json,
        num_layers,
        Q4K_TEST_INTER,
        Q4K_TEST_HIDDEN,
        Q4K_TEST_NUM_Q,
        Q4K_TEST_NUM_KV,
    )
}

pub(super) fn q4k_test_weights_from_json(
    arch_json: serde_json::Value,
    num_layers: usize,
    intermediate: usize,
    hidden: usize,
    num_q: usize,
    num_kv: usize,
) -> ModelWeights {
    let head_dim = hidden / num_q;
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let mut seed = 0xc0ffee_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let embed = rand_mat_seeded(Q4K_TEST_VOCAB, hidden, 0.05, next_seed());
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
            rand_mat_seeded(intermediate, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(intermediate, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, intermediate, 0.05, next_seed()),
        );

        vectors.insert(arch.input_layernorm_key(layer), vec![0.5; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.5; hidden]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        // QK-norm, on the architectures that declare it (Gemma 3 / 4).
        // These were declared-but-absent until 2026-08, which made the
        // fixture claim an architecture it did not actually carry: every
        // consumer that resolves the weight got `None` and silently
        // skipped the stage, so a backend disagreeing about what to do
        // with a missing QK-norm weight had nothing pinning it. Sized per
        // head_dim, not hidden — QK-norm normalises within a head.
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
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
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size: Q4K_TEST_VOCAB,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

/// SiLU sibling of [`make_test_q4k_weights`].
///
/// Uses the TinyModel architecture so the FFN activation is `Silu` and
/// the FFN type is `Gated`. Dimensions match the Q4_K constraints
/// (`Q4K_TEST_HIDDEN` is a multiple of 256) so the same `make_test_q4k_vindex`
/// can wrap the result. Needed by tests that exercise the SiLU branch in
/// quantised forward paths (e.g. `walk_ffn_kquant_dequant`'s `silu_gate_up`
/// arm) without depending on a Gemma3 fixture.
pub fn make_test_q4k_weights_silu() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": Q4K_TEST_NUM_LAYERS,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let mut seed = 0xdeadc0de_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let embed = rand_mat_seeded(Q4K_TEST_VOCAB, Q4K_TEST_HIDDEN, 0.05, next_seed());
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
            rand_mat_seeded(q_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(Q4K_TEST_HIDDEN, q_dim, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(Q4K_TEST_HIDDEN, Q4K_TEST_INTER, 0.05, next_seed()),
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

/// Wrap a byte payload in an anonymous read-only mmap. Used to build
/// in-memory test vindexes without touching the filesystem.
///
/// Public so inference-side fixtures (`make_test_q4k_vindex` etc.)
/// that stay in `larql-inference/src/test_utils.rs` can reuse it.
pub fn arc_mmap_from_bytes(payload: &[u8]) -> std::sync::Arc<memmap2::Mmap> {
    let mut anon = memmap2::MmapMut::map_anon(payload.len().max(1)).expect("anon mmap");
    if !payload.is_empty() {
        anon.copy_from_slice(payload);
    }
    let mmap = anon.make_read_only().expect("freeze");
    std::sync::Arc::new(mmap)
}

/// Gemma-3 Q4_K fixture at **caller-chosen attention dimensions**.
///
/// Every other Q4_K fixture is pinned to `hidden = 256, num_q = 4`, i.e.
/// `head_dim = 64`. That made shape sensitivity untestable: a kernel with
/// a `head_dim` assumption would be wrong on every fixture and right on
/// every real model, which is exactly the signature of the Metal
/// batched-prefill divergence (see `larql-kv`'s
/// `gpu_engine_parity::gemma3_prefill_gap`). This builder is the knob for
/// bisecting that axis, and for any future "does this kernel assume a
/// shape?" question.
///
/// `hidden` must be a multiple of 256 (Q4_K super-block) and divisible by
/// `num_q`; `num_q` must be divisible by `num_kv`.
pub fn make_test_q4k_weights_with_dims(
    hidden: usize,
    num_q: usize,
    num_kv: usize,
    num_layers: usize,
) -> ModelWeights {
    assert!(
        hidden.is_multiple_of(256),
        "Q4_K needs a hidden size that is a multiple of its 256-element super-block, got {hidden}"
    );
    assert!(
        num_q != 0 && hidden.is_multiple_of(num_q),
        "hidden {hidden} must divide evenly into {num_q} query heads"
    );
    assert!(
        num_kv != 0 && num_q.is_multiple_of(num_kv),
        "{num_q} query heads must group evenly onto {num_kv} K/V heads"
    );
    let head_dim = hidden / num_q;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": hidden,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    q4k_test_weights_from_json(arch_json, num_layers, hidden, hidden, num_q, num_kv)
}
