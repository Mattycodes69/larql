//! The known dense fixture and header-only shard helpers.

use larql_models::config::{LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_LINEAR_ATTENTION};
use larql_models::inventory::{build_inventory, ArchitectureInventory};
use std::io::Write;
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// A fully-known dense model: recognised family, no unconsumed keys beyond
/// metadata, uniform attention.
/// [`known_dense`] with the caller's config, for gates that turn on one
/// declared key rather than on the shape.
/// Write a config plus one or more **header-only** safetensors shards and
/// build the inventory over them.
///
/// The shards carry the 8-byte length prefix and the header JSON, and stop
/// there — no payload byte is written. `scan_tensors` reads exactly that
/// much (`read_shard_header` never seeks past the header and never checks
/// the file against its own offsets), so every tensor fact — name, dtype,
/// shape, and the byte count derived from `data_offsets` — is the real one
/// while the file on disk stays kilobytes.
///
/// That is what lets a fixture carry a REAL checkpoint's estate at its real
/// sizes. [`known_dense_with_config`] and friends write payloads because
/// encode tests compare bytes; a fixture whose subject is the tensor-address
/// plane needs the names and geometry, and materialising 1.56 TB to get them
/// is not an option. Any test that reads a byte of payload must NOT use this.
///
/// `shards` maps a shard filename to its safetensors header object.
pub fn header_only_shards(
    dir: &Path,
    config: &serde_json::Value,
    shards: &serde_json::Map<String, serde_json::Value>,
) -> ArchitectureInventory {
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    for (name, header) in shards {
        let header_bytes = serde_json::to_vec(header).unwrap();
        let mut file = std::fs::File::create(dir.join(name)).unwrap();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header_bytes).unwrap();
    }
    build_inventory(dir).unwrap()
}

pub fn known_dense_with_config(dir: &Path, config: serde_json::Value) -> ArchitectureInventory {
    let header = serde_json::json!({
        "model.embed_tokens.weight":
            {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 16384]}
    });
    inventory_from(dir, &config, &header)
}

/// [`known_dense`]'s config, for gates that add one declared key to a
/// recognised dense family rather than to the Glimmer shape.
pub fn known_dense_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    })
}

pub fn known_dense(dir: &Path) -> ArchitectureInventory {
    known_dense_with_config(dir, known_dense_config())
}

/// Period of the hybrid interleave: one `full_attention` layer in every
/// [`HYBRID_FULL_ATTENTION_INTERVAL`], the rest recurrent. The 3:1 Qwen3.8
/// / Kimi-Linear cadence.
pub const HYBRID_FULL_ATTENTION_INTERVAL: usize = 4;

/// Depthwise causal conv width over the fused q|k|v channels.
pub const HYBRID_CONV_KERNEL: usize = 4;
/// Dk — key-side head width.
pub const HYBRID_KEY_HEAD_DIM: usize = 16;
/// Dv — value-side head width.
pub const HYBRID_VALUE_HEAD_DIM: usize = 16;
/// Hk — key-side head count.
pub const HYBRID_KEY_HEADS: usize = 2;
/// Hv — value-side head count. Larger than [`HYBRID_KEY_HEADS`], as on
/// Qwen3.8, so a fixture cannot pass by folding the two sides together.
pub const HYBRID_VALUE_HEADS: usize = 4;
/// Precision the recurrence keeps its state at.
pub const HYBRID_STATE_DTYPE: &str = "float32";

/// Whether layer `i` is the full-attention layer of the hybrid cadence.
pub(super) fn is_full_attention_layer(i: usize) -> bool {
    i % HYBRID_FULL_ATTENTION_INTERVAL == HYBRID_FULL_ATTENTION_INTERVAL - 1
}

/// Write the 3:1 `linear_attention` / `full_attention` cadence into a
/// config, the way Qwen3.8 and Kimi Linear write it.
///
/// Declares *that* the stack is hybrid and nothing about which recurrence
/// it runs — [`declare_gated_delta_geometry`] is the separate fact that
/// identifies the operator.
pub fn declare_hybrid_cadence(config: &mut serde_json::Value) {
    let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
        .map(|i| {
            if is_full_attention_layer(i) {
                LAYER_TYPE_FULL_ATTENTION
            } else {
                LAYER_TYPE_LINEAR_ATTENTION
            }
        })
        .collect();
    config["text_config"]["layer_types"] = serde_json::json!(layer_types);
    config["text_config"]["full_attention_interval"] =
        serde_json::json!(HYBRID_FULL_ATTENTION_INTERVAL);
}

/// Declare the geometry that *identifies* the recurrence as Gated
/// DeltaNet.
///
/// Kept separate from [`declare_hybrid_cadence`] because the two are
/// independent facts and the discriminator is precisely whether this one
/// is present: a cadence names a recurrence, and only the geometry names
/// *which* recurrence. `LinearAttentionTopology::from_config` refuses a
/// partial declaration, so these are declared together or not at all.
pub fn declare_gated_delta_geometry(config: &mut serde_json::Value) {
    let text = &mut config["text_config"];
    text["linear_conv_kernel_dim"] = serde_json::json!(HYBRID_CONV_KERNEL);
    text["linear_key_head_dim"] = serde_json::json!(HYBRID_KEY_HEAD_DIM);
    text["linear_value_head_dim"] = serde_json::json!(HYBRID_VALUE_HEAD_DIM);
    text["linear_num_key_heads"] = serde_json::json!(HYBRID_KEY_HEADS);
    text["linear_num_value_heads"] = serde_json::json!(HYBRID_VALUE_HEADS);
    text["mamba_ssm_dtype"] = serde_json::json!(HYBRID_STATE_DTYPE);
}
