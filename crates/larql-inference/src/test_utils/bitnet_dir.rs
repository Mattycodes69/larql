//! Synthetic BitNet b1.58 `--keep-quant` container.
//!
//! [`write_synthetic_bitnet_model_dir`] writes the smallest on-disk
//! vindex that [`crate::ternary::load_bitnet_model`] accepts, built the
//! same way `larql convert gguf-to-vindex --keep-quant --dense-only`
//! builds a real one: a dense-only vindex (embeddings, norms, lm_head,
//! tokenizer, index.json), then the `bitnet/` I2_S artifacts from
//! `bitnet_writer::write_bitnet_artifacts`, then `bitnet_layout`
//! stamped into index.json. Only the source weights are synthetic, so a
//! test on this fixture exercises the real writer → loader seam.
//!
//! Split out of `test_utils.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use larql_models::{ModelWeights, WeightArray};
use larql_vindex::extract::bitnet_writer::{write_bitnet_artifacts, BitnetArchMeta};
use larql_vindex::format::filenames::INDEX_JSON;
use ndarray::Array2;
use std::collections::HashMap;

/// Vocabulary size of the synthetic BitNet fixture.
pub const BITNET_TEST_VOCAB: usize = 32;
/// Hidden width of the synthetic BitNet fixture.
pub const BITNET_TEST_HIDDEN: usize = 16;
/// FFN width of the synthetic BitNet fixture. A multiple of 4, as I2_S
/// packs four trits per byte along the input dimension.
pub const BITNET_TEST_INTER: usize = 32;
/// Decoder layers in the synthetic BitNet fixture.
pub const BITNET_TEST_NUM_LAYERS: usize = 2;
/// Query heads in the synthetic BitNet fixture.
pub const BITNET_TEST_NUM_Q_HEADS: usize = 2;
/// Key/value heads in the synthetic BitNet fixture (GQA, like the real model).
pub const BITNET_TEST_NUM_KV_HEADS: usize = 1;
/// Per-head width in the synthetic BitNet fixture.
pub const BITNET_TEST_HEAD_DIM: usize = 8;
/// RMSNorm epsilon the fixture's config declares.
pub const BITNET_TEST_RMS_EPS: f64 = 1e-5;
/// RoPE base the fixture's config declares.
pub const BITNET_TEST_ROPE_THETA: f64 = 10_000.0;
/// Per-tensor I2_S magnitude (`W = trit * scale`) every BitLinear carries.
pub const BITNET_TEST_I2S_SCALE: f32 = 0.5;
/// Model name stamped into the fixture's index.json.
pub const BITNET_TEST_MODEL_NAME: &str = "synthetic/bitnet";

/// Trits per I2_S byte (2 bits each).
const TRITS_PER_BYTE: usize = 4;
/// Scale of the dense (non-ternary) embedding / lm_head entries.
const DENSE_ENTRY_SCALE: f32 = 0.1;
/// Seed for the dense embedding / lm_head generator.
const DENSE_SEED: u64 = 0xb17_0e7;

/// The `config.json` the fixture's architecture is detected from.
pub fn synthetic_bitnet_arch_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "bitnet",
        "hidden_size": BITNET_TEST_HIDDEN,
        "num_hidden_layers": BITNET_TEST_NUM_LAYERS,
        "intermediate_size": BITNET_TEST_INTER,
        "head_dim": BITNET_TEST_HEAD_DIM,
        "num_attention_heads": BITNET_TEST_NUM_Q_HEADS,
        "num_key_value_heads": BITNET_TEST_NUM_KV_HEADS,
        "vocab_size": BITNET_TEST_VOCAB,
        "rms_norm_eps": BITNET_TEST_RMS_EPS,
        "rope_theta": BITNET_TEST_ROPE_THETA,
    })
}

/// A `rows × cols` matrix of trits cycling `+1, 0, -1` — every code the
/// I2_S packer emits appears, so a round trip checks all three.
fn ternary_matrix(rows: usize, cols: usize, phase: usize) -> WeightArray {
    const TRITS: [f32; 3] = [1.0, 0.0, -1.0];
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| TRITS[(i + phase) % TRITS.len()])
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .expect("rows * cols elements by construction")
        .into_shared()
}

/// Build the in-memory BitNet `ModelWeights` the way the keep-quant GGUF
/// loader hands them to the writer: every BitLinear projection present as
/// a dequantised ternary tensor, with its raw I2_S bytes and trailing
/// per-tensor scale captured in `raw_bytes`; RMSNorm vectors (including
/// the two BitNet sub-norms) in `vectors`.
pub fn make_test_bitnet_weights() -> ModelWeights {
    let arch = larql_models::detect_from_json(&synthetic_bitnet_arch_json());
    let q_dim = BITNET_TEST_NUM_Q_HEADS * BITNET_TEST_HEAD_DIM;
    let kv_dim = BITNET_TEST_NUM_KV_HEADS * BITNET_TEST_HEAD_DIM;
    let (h, inter) = (BITNET_TEST_HIDDEN, BITNET_TEST_INTER);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

    let embed = rand_mat_seeded(BITNET_TEST_VOCAB, h, DENSE_ENTRY_SCALE, DENSE_SEED);
    tensors.insert(arch.embed_key().to_string(), embed.clone());
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; h]);

    for layer in 0..BITNET_TEST_NUM_LAYERS {
        let bitlinears = [
            (arch.attn_q_key(layer), q_dim, h),
            (arch.attn_k_key(layer), kv_dim, h),
            (arch.attn_v_key(layer), kv_dim, h),
            (arch.attn_o_key(layer), h, q_dim),
            (arch.ffn_gate_key(layer), inter, h),
            (arch.ffn_up_key(layer), inter, h),
            (arch.ffn_down_key(layer), h, inter),
        ];
        for (phase, (key, rows, cols)) in bitlinears.into_iter().enumerate() {
            tensors.insert(key.clone(), ternary_matrix(rows, cols, phase + layer));
            raw_bytes.insert(key.clone(), vec![0u8; rows * cols / TRITS_PER_BYTE]);
            raw_bytes.insert(
                format!("{key}{}", larql_models::I2S_SCALE_SUFFIX),
                BITNET_TEST_I2S_SCALE.to_le_bytes().to_vec(),
            );
        }
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; h]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; h]);
        // The arch declares [attn_sub_norm, ffn_sub_norm]: the first
        // normalises the attention output (width hidden), the second the
        // FFN activation (width inter) — the widths `load_bitnet_model`
        // validates.
        let sub_norms = arch.sub_norm_keys(layer);
        assert_eq!(sub_norms.len(), 2, "bitnet declares two sub-norms");
        for (key, width) in sub_norms.into_iter().zip([h, inter]) {
            vectors.insert(key, vec![1.0; width]);
        }
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
        lm_head: embed.clone(),
        embed,
        position_embed: None,
        arch,
        num_layers: BITNET_TEST_NUM_LAYERS,
        hidden_size: h,
        intermediate_size: inter,
        vocab_size: BITNET_TEST_VOCAB,
        head_dim: BITNET_TEST_HEAD_DIM,
        num_q_heads: BITNET_TEST_NUM_Q_HEADS,
        num_kv_heads: BITNET_TEST_NUM_KV_HEADS,
        rope_base: BITNET_TEST_ROPE_THETA,
    }
}

/// Write a synthetic BitNet `--keep-quant --dense-only` container to
/// `dir`, through the same three steps the convert command runs.
///
/// Layout written:
/// ```text
/// dir/
///   index.json              -- VindexConfig with bitnet_layout, no gate layers
///   tokenizer.json          -- WordLevel "[0]".."[VOCAB-1]"
///   embeddings.bin          -- VOCAB × HIDDEN
///   weight_manifest.json    -- embed / norms / sub-norms / lm_head
///   norms.bin, lm_head.bin  -- the dense parts the ternary forward reads
///   bitnet/*.i2s            -- one packed-trit file per BitLinear
///   bitnet/scales.f32       -- concatenated per-row scales
/// ```
pub fn write_synthetic_bitnet_model_dir(dir: &std::path::Path) -> Result<(), String> {
    let weights = make_test_bitnet_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_dense_only(
        &weights,
        &tokenizer,
        BITNET_TEST_MODEL_NAME,
        dir,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .map_err(|e| format!("build_vindex_dense_only: {e}"))?;

    let meta = BitnetArchMeta::from_architecture(&*weights.arch)
        .map_err(|e| format!("BitnetArchMeta: {e}"))?;
    let layout = write_bitnet_artifacts(dir, &weights, meta)
        .map_err(|e| format!("write_bitnet_artifacts: {e}"))?;

    let mut config =
        larql_vindex::load_vindex_config(dir).map_err(|e| format!("load_vindex_config: {e}"))?;
    config.bitnet_layout = Some(layout);
    let json = serde_json::to_string_pretty(&config).map_err(|e| format!("serialise: {e}"))?;
    std::fs::write(dir.join(INDEX_JSON), json).map_err(|e| format!("write {INDEX_JSON}: {e}"))
}
