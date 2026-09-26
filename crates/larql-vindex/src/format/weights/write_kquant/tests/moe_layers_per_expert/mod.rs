//! Colocated tests for the separate-tensor MoE layer writer.
//!
//! The regression these defend against is specific and was live: extraction of
//! a separate-tensor MoE reported success, verified clean, sliced clean, and
//! then panicked on the first decoded token because no expert store had been
//! written. So the assertions are about *files appearing on disk with the
//! right shape*, not about the quantiser — which `write_layers_parts_tests`
//! covers separately.

use std::collections::HashMap;
use std::path::Path;

use super::super::moe_layers_per_expert::write_per_layer_moe_per_expert;
use crate::format::weights::write_f32::WeightSource;
use crate::format::weights::write_layers::parse_layer_weights_header;

const HIDDEN: usize = 256;
const INTER: usize = 256;
const NUM_LAYERS: usize = 2;
const NUM_EXPERTS: usize = 3;

/// A `WeightSource` backed by an explicit tensor map, so a test can express
/// "this expert is missing" precisely.
struct MapSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    tensors: HashMap<String, (Vec<f32>, usize, usize)>,
}

impl MapSource {
    /// An OLMoE-shaped architecture — the real separate-tensor case.
    fn olmoe(num_experts: usize) -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": "olmoe",
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
        }))
    }

    /// Every expert of every layer present and correctly shaped.
    fn complete() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        for layer in 0..NUM_LAYERS {
            for expert in 0..NUM_EXPERTS {
                insert_expert(&mut tensors, &*arch, layer, expert);
            }
        }
        Self { arch, tensors }
    }

    /// Layer 0 complete; layer 1 missing every expert (a hybrid dense layer).
    fn first_layer_only() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        for expert in 0..NUM_EXPERTS {
            insert_expert(&mut tensors, &*arch, 0, expert);
        }
        Self { arch, tensors }
    }

    /// Layer 0 has expert 0 but not expert 1 — a genuinely malformed layer.
    fn partial_layer() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        insert_expert(&mut tensors, &*arch, 0, 0);
        Self { arch, tensors }
    }
}

fn insert_expert(
    tensors: &mut HashMap<String, (Vec<f32>, usize, usize)>,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    expert: usize,
) {
    // Distinct fill per (layer, expert) so a mixed-up write is detectable.
    let fill = (layer * 10 + expert) as f32;
    if let Some(k) = arch.expert_ffn_gate_key(layer, expert) {
        tensors.insert(k, (vec![fill; INTER * HIDDEN], INTER, HIDDEN));
    }
    if let Some(k) = arch.expert_ffn_up_key(layer, expert) {
        tensors.insert(k, (vec![-fill; INTER * HIDDEN], INTER, HIDDEN));
    }
    if let Some(k) = arch.expert_ffn_down_key(layer, expert) {
        tensors.insert(k, (vec![fill; HIDDEN * INTER], HIDDEN, INTER));
    }
}

impl WeightSource for MapSource {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        self.tensors.get(key).cloned()
    }
    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        NUM_LAYERS
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
}

/// A directory of its own for every call, however many callers share a
/// `name` — within this process (a counter) and across processes (the
/// pid). A fixed name per label let two test processes running at once
/// remove each other's files mid-read.
fn temp_dir(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir()
        .join("moe-per-expert-tests")
        .join(format!(
            "{name}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn layer_file(dir: &Path, layer: usize) -> std::path::PathBuf {
    dir.join(crate::format::filenames::layer_weights_filename(layer))
}

/// A `WeightSource` shaped like `StreamingWeights` against a real GPT-OSS
/// checkpoint: the synthesised per-expert keys resolve to nothing (no shard
/// contains them) and only the raw packed `*_blocks`/`*_scales` answer.
struct RawPackedSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    raw: HashMap<String, (Vec<u8>, Vec<usize>)>,
}

impl WeightSource for RawPackedSource {
    fn get_tensor(&self, _key: &str) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        NUM_LAYERS
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
    fn get_raw_u8(&self, key: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        self.raw.get(key).cloned()
    }
}

fn gpt_oss_arch() -> Box<dyn larql_models::ModelArchitecture> {
    larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
    }))
}

/// Packed MXFP4 fixture tensors for one layer: fused gate_up
/// `[E, 2*INTER, G, 16]` and down `[E, HIDDEN, G, 16]`, plus their scales.
/// Nibble bytes vary with position so a permuted or mis-sliced decode
/// cannot reproduce them, and scale bytes vary per group so a scale/block
/// misalignment shows up in the values (a constant scale would forgive it).
fn packed_layer_fixture(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    raw: &mut HashMap<String, (Vec<u8>, Vec<usize>)>,
) {
    let groups = HIDDEN / larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
    let gu_rows = 2 * INTER;
    let fill = |n: usize, seed: usize| -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + seed * 11) % 251) as u8).collect()
    };
    // E8M0 scale bytes around 2^0 (127): keep within ±3 octaves so the
    // dequantised magnitudes stay ordinary f32.
    let scales = |n: usize, seed: usize| -> Vec<u8> {
        (0..n).map(|i| 124 + ((i + seed) % 7) as u8).collect()
    };
    raw.insert(
        arch.packed_gate_up_blocks_key(layer).unwrap(),
        (
            fill(
                NUM_EXPERTS * gu_rows * groups * larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
                layer,
            ),
            vec![
                NUM_EXPERTS,
                gu_rows,
                groups,
                larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
            ],
        ),
    );
    raw.insert(
        arch.packed_gate_up_scales_key(layer).unwrap(),
        (
            scales(NUM_EXPERTS * gu_rows * groups, layer),
            vec![NUM_EXPERTS, gu_rows, groups],
        ),
    );
    let dn_groups = INTER / larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
    raw.insert(
        arch.packed_down_blocks_key(layer).unwrap(),
        (
            fill(
                NUM_EXPERTS * HIDDEN * dn_groups * larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
                layer + 100,
            ),
            vec![
                NUM_EXPERTS,
                HIDDEN,
                dn_groups,
                larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
            ],
        ),
    );
    raw.insert(
        arch.packed_down_scales_key(layer).unwrap(),
        (
            scales(NUM_EXPERTS * HIDDEN * dn_groups, layer + 100),
            vec![NUM_EXPERTS, HIDDEN, dn_groups],
        ),
    );
}

mod moe_layers_per_expert_basics;
