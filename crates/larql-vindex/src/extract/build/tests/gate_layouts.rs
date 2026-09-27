//! Stage 1's two non-default gate layouts: a MoE layer concatenates every
//! expert's gate matrix (plus the shared expert's), and a non-gated FFN
//! reuses its up projection as the feature-input direction.
//!
//! Every tensor name is asked of the architecture, never spelled here, so
//! the fixture follows whatever keys the family declares.

use super::*;
use larql_models::ModelArchitecture;

/// Routed experts per MoE layer in the fixture.
const EXPERTS: usize = 2;
/// Rows of each routed expert's gate (its intermediate width).
const EXPERT_FEATURES: usize = 3;
/// Rows of the shared expert's gate.
const SHARED_FEATURES: usize = 5;
/// The MoE layer that ships no expert tensors at all.
const EMPTY_MOE_LAYER: usize = 1;
/// Fill value for every fixture matrix: the layout, not the numbers, is
/// what these tests pin.
const FILL: f32 = 0.25;

fn matrix(rows: usize, cols: usize) -> ArcArray2<f32> {
    ndarray::Array2::<f32>::from_elem((rows, cols), FILL).into_shared()
}

/// `ModelWeights` over `arch` holding exactly `tensors`.
fn weights_with(
    arch: Box<dyn ModelArchitecture>,
    tensors: HashMap<String, ArcArray2<f32>>,
    intermediate: usize,
) -> larql_models::ModelWeights {
    let embed = matrix(VOCAB, HIDDEN);
    larql_models::ModelWeights {
        tensors,
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        per_layer_ffn_format: Default::default(),
        per_layer_ffn_arrangement: Default::default(),
        lm_head: embed.clone(),
        embed,
        position_embed: None,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: intermediate,
        vocab_size: VOCAB,
        head_dim: HIDDEN,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

fn moe_weights() -> larql_models::ModelWeights {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "qwen2_moe",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": EXPERT_FEATURES,
        "moe_intermediate_size": EXPERT_FEATURES,
        "shared_expert_intermediate_size": SHARED_FEATURES,
        "num_experts": EXPERTS,
        "num_experts_per_tok": 1,
        "head_dim": HIDDEN,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    }));
    assert!(arch.is_moe(), "fixture must produce a MoE architecture");
    assert_eq!(arch.num_experts(), EXPERTS);
    let mut tensors = HashMap::new();
    for layer in (0..NUM_LAYERS).filter(|l| *l != EMPTY_MOE_LAYER) {
        for expert in 0..EXPERTS {
            let gate = arch.expert_ffn_gate_key(layer, expert).unwrap();
            let down = arch.expert_ffn_down_key(layer, expert).unwrap();
            tensors.insert(gate, matrix(EXPERT_FEATURES, HIDDEN));
            tensors.insert(down, matrix(HIDDEN, EXPERT_FEATURES));
        }
        let shared_gate = arch.shared_expert_gate_key(layer).unwrap();
        let shared_down = arch.shared_expert_down_key(layer).unwrap();
        tensors.insert(shared_gate, matrix(SHARED_FEATURES, HIDDEN));
        tensors.insert(shared_down, matrix(HIDDEN, SHARED_FEATURES));
    }
    weights_with(arch, tensors, EXPERT_FEATURES)
}

fn build_browse(weights: &larql_models::ModelWeights, dir: &std::path::Path) {
    let mut cb = SilentBuildCallbacks;
    build_vindex(
        weights,
        &tokenizer(),
        "test/layout",
        dir,
        3,
        ExtractLevel::Browse,
        StorageDtype::F32,
        &mut cb,
    )
    .unwrap();
}

#[test]
fn moe_gate_layer_concatenates_every_expert_and_the_shared_expert() {
    let dir = TempDir::new().unwrap();
    build_browse(&moe_weights(), dir.path());
    let cfg = crate::load_vindex_config(dir.path()).unwrap();
    // The expert-less layer contributes nothing, so exactly one entry.
    assert_eq!(cfg.layers.len(), 1, "{:?}", cfg.layers);
    let info = &cfg.layers[0];
    assert_eq!(info.layer, 0);
    assert_eq!(
        info.num_features,
        EXPERTS * EXPERT_FEATURES + SHARED_FEATURES
    );
    assert_eq!(info.num_experts, Some(EXPERTS));
    assert_eq!(info.num_features_per_expert, Some(EXPERT_FEATURES));
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    assert_eq!(
        info.length,
        (info.num_features * HIDDEN) as u64 * f32_bytes,
        "every expert's and the shared expert's rows are written"
    );
    let gate_len = std::fs::metadata(dir.path().join(crate::format::filenames::GATE_VECTORS_BIN))
        .unwrap()
        .len();
    assert_eq!(gate_len, info.length);
}

#[test]
fn non_gated_ffn_uses_the_up_projection_as_the_gate() {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "head_dim": HIDDEN,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    }));
    assert_eq!(arch.ffn_type(), larql_models::FfnType::Standard);
    let mut tensors = HashMap::new();
    for layer in 0..NUM_LAYERS {
        tensors.insert(arch.ffn_up_key(layer), matrix(INTERMEDIATE, HIDDEN));
        tensors.insert(arch.ffn_down_key(layer), matrix(HIDDEN, INTERMEDIATE));
    }
    let weights = weights_with(arch, tensors, INTERMEDIATE);
    let dir = TempDir::new().unwrap();
    build_browse(&weights, dir.path());
    let cfg = crate::load_vindex_config(dir.path()).unwrap();
    assert_eq!(cfg.layers.len(), NUM_LAYERS);
    for (layer, info) in cfg.layers.iter().enumerate() {
        assert_eq!(info.layer, layer);
        assert_eq!(info.num_features, INTERMEDIATE);
        assert_eq!(info.num_experts, None);
    }
}

#[test]
fn a_dense_layer_without_its_gate_tensor_is_skipped() {
    let mut weights = make_weights();
    let missing = weights.arch.ffn_gate_key(0);
    assert!(weights.tensors.remove(&missing).is_some());
    let dir = TempDir::new().unwrap();
    build_browse(&weights, dir.path());
    let cfg = crate::load_vindex_config(dir.path()).unwrap();
    let layers: Vec<usize> = cfg.layers.iter().map(|l| l.layer).collect();
    assert_eq!(layers, vec![1]);
}
