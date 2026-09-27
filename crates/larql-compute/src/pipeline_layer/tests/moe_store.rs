//! The per-layer expert store: which bytes each expert resolves to, and
//! **which format authority decides how to read them**.
//!
//! These paths were dark until K3 R1/P5 served GPT-OSS, whose experts are
//! MXFP4 transcoded to Q6_K. A Q6_K store read as Q4_K does not fail — it
//! decodes to plausible-looking garbage and the model generates fluent
//! nonsense. So the rule under test here is that the *store header's own
//! tag* decides, an absent tag means the legacy Q4_K-only era, and an
//! unrecognised tag is a defect that stops the process rather than a
//! fallback. See `docs/k3-funnel.md` §4.11.
//!
//! The legacy BF16-monolith arm is covered by the Gemma 4 fixture tests in
//! the parent module; everything here is the `layers/` arm.

use super::super::moe_build::build_moe_weights;
use super::super::*;
use larql_models::test_fixtures::{make_test_gemma4_moe_weights, make_test_weights};
use larql_models::weights::{per_layer_ffn_key, PER_LAYER_FFN_DOWN, PER_LAYER_FFN_GATE_UP};

/// Byte written into every expert's `gate_up` blob, offset by expert index,
/// so a test can prove the table entry points at *that* expert's bytes and
/// not merely at something of the right length.
const EXPERT_TAG_BASE: u8 = 0x40;
const EXPERT_BLOB_LEN: usize = 8;

/// Rewrite a Gemma 4 MoE fixture into the per-layer (`layers/{l}/{e}/…`)
/// store layout.
///
/// `has_per_layer_ffn()` reads `packed_byte_ranges` while `get_packed_bytes`
/// falls back to `raw_bytes` when the named mmap is absent — the same
/// arrangement `per_layer_ffn_bytes_detects_and_loads_entries` uses in
/// larql-models. That lets a test carry real bytes without a temp file.
fn into_per_layer_store(weights: &mut larql_models::ModelWeights) {
    let num_experts = weights.arch.num_experts();
    for layer in 0..weights.num_layers {
        for e in 0..num_experts {
            let tag = EXPERT_TAG_BASE + e as u8;
            weights.raw_bytes.insert(
                per_layer_ffn_key(layer, e, PER_LAYER_FFN_GATE_UP),
                vec![tag; EXPERT_BLOB_LEN],
            );
            weights.raw_bytes.insert(
                per_layer_ffn_key(layer, e, PER_LAYER_FFN_DOWN),
                vec![tag ^ 0xff; EXPERT_BLOB_LEN],
            );
        }
    }
    // Presence marker only — the file is deliberately absent so resolution
    // falls through to `raw_bytes`.
    weights.packed_byte_ranges.insert(
        per_layer_ffn_key(0, 0, PER_LAYER_FFN_GATE_UP),
        ("absent.bin".into(), 0, EXPERT_BLOB_LEN),
    );
    assert!(
        weights.has_per_layer_ffn(),
        "fixture rewrite must flip the layout probe"
    );
}

fn per_layer_fixture() -> larql_models::ModelWeights {
    let mut weights = make_test_gemma4_moe_weights();
    into_per_layer_store(&mut weights);
    weights
}

/// Every expert resolves to its *own* entry, and the table is as long as the
/// expert count — the per-layer arm's whole job.
#[test]
fn per_layer_store_resolves_one_entry_per_expert() {
    let weights = per_layer_fixture();
    let moe = build_moe_weights(&weights, &*weights.arch, 0)
        .expect("fixture declares an executable router")
        .expect("layer 0 is MoE");

    let num_experts = weights.arch.num_experts();
    assert_eq!(moe.experts_gate_up.len(), num_experts);
    assert_eq!(moe.experts_down.len(), num_experts);
    for e in 0..num_experts {
        let tag = EXPERT_TAG_BASE + e as u8;
        assert_eq!(
            moe.experts_gate_up[e],
            vec![tag; EXPERT_BLOB_LEN],
            "expert {e} gate_up must be expert {e}'s bytes"
        );
        assert_eq!(
            moe.experts_down[e],
            vec![tag ^ 0xff; EXPERT_BLOB_LEN],
            "expert {e} down must be expert {e}'s bytes"
        );
    }
}

/// The store header is the format authority. A Q6_K store must resolve as
/// Q6_K — this is the GPT-OSS case, and reading it as Q4_K is the silent
/// corruption the tag exists to prevent.
#[test]
fn store_header_tag_decides_the_expert_format() {
    let mut weights = per_layer_fixture();
    weights.per_layer_ffn_format.insert(0, "Q6_K".to_string());
    let moe = build_moe_weights(&weights, &*weights.arch, 0)
        .expect("fixture declares an executable router")
        .expect("layer 0 is MoE");
    assert_eq!(moe.expert_data_format, QuantFormat::Q6_K);
}

/// Per-layer, not per-model: layer 1 carrying no tag still falls back
/// independently of layer 0's Q6_K declaration.
#[test]
fn format_authority_is_per_layer_not_per_model() {
    let mut weights = per_layer_fixture();
    weights.per_layer_ffn_format.insert(0, "Q6_K".to_string());
    let l0 = build_moe_weights(&weights, &*weights.arch, 0)
        .expect("fixture declares an executable router")
        .expect("layer 0 is MoE");
    let l1 = build_moe_weights(&weights, &*weights.arch, 1)
        .expect("fixture declares an executable router")
        .expect("layer 1 is MoE");
    assert_eq!(l0.expert_data_format, QuantFormat::Q6_K);
    assert_eq!(l1.expert_data_format, QuantFormat::Q4_K);
}

/// An absent tag means a vindex written before the format was threaded
/// through — those only ever carried Q4_K, so that is the sound default.
#[test]
fn absent_format_tag_falls_back_to_q4k() {
    let weights = per_layer_fixture();
    assert!(weights.per_layer_ffn_format_tag(0).is_none());
    let moe = build_moe_weights(&weights, &*weights.arch, 0)
        .expect("fixture declares an executable router")
        .expect("layer 0 is MoE");
    assert_eq!(moe.expert_data_format, QuantFormat::Q4_K);
}

/// An unrecognised tag is a defect, not a fallback: compute has no decoder,
/// so continuing would produce a confidently wrong forward.
#[test]
#[should_panic(expected = "which compute has no decoder for")]
fn unknown_format_tag_refuses_rather_than_guessing() {
    let mut weights = per_layer_fixture();
    weights
        .per_layer_ffn_format
        .insert(0, "Q3_K_XL_IMAGINARY".to_string());
    let _ = build_moe_weights(&weights, &*weights.arch, 0)
        .expect("fixture declares an executable router");
}

// ── remote-MoE stub patching ──

/// A dense model has nothing to patch — the guard returns before touching
/// any layer, so a dense pipeline cannot acquire phantom MoE weights.
#[test]
fn remote_patch_is_a_no_op_on_a_dense_arch() {
    let weights = make_test_weights();
    assert!(!weights.arch.is_hybrid_moe());
    let mut layers = vec![crate::FullPipelineLayer::default()];
    patch_pipeline_layers_for_remote_moe(&mut layers, &weights)
        .expect("fixture declares an executable router");
    assert!(
        layers[0].moe.is_none(),
        "a dense arch must not be given MoE weights"
    );
}

/// Layers already carrying locally-resolved experts are left alone — the
/// patch fills gaps, it does not overwrite a served layer with a stub.
#[test]
fn remote_patch_preserves_locally_resolved_layers() {
    let weights = make_test_gemma4_moe_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..weights.num_layers)
        .map(|l| {
            build_arch_params(&weights, l, dummy, dummy, dummy, dummy, dummy, dummy, dummy)
                .expect("fixture declares an executable router")
        })
        .collect();
    // Layer 0 keeps its real experts; layer 1 is emptied to look unserved.
    let local_experts = layers[0]
        .moe
        .as_ref()
        .expect("fixture layer 0 resolves experts locally")
        .experts_gate_up
        .len();
    assert!(local_experts > 0, "fixture must resolve real expert bytes");
    layers[1].moe = None;

    patch_pipeline_layers_for_remote_moe(&mut layers, &weights)
        .expect("fixture declares an executable router");

    assert_eq!(
        layers[0]
            .moe
            .as_ref()
            .expect("layer 0 still served")
            .experts_gate_up
            .len(),
        local_experts,
        "a served layer must not be replaced by an empty stub"
    );
    assert!(
        layers[1]
            .moe
            .as_ref()
            .expect("layer 1 patched")
            .experts_gate_up
            .is_empty(),
        "the patched layer is a stub — its experts live remotely"
    );
}

/// The stub's `expert_data_format` follows the same layout question the
/// real builder asks, so a fallback `cpu_moe_forward` (if one ever runs on
/// a stub) decodes with the right reader rather than the BF16 default.
#[test]
fn remote_stub_format_follows_the_store_layout() {
    let weights = make_test_gemma4_moe_weights();
    let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..weights.num_layers)
        .map(|_| crate::FullPipelineLayer::default())
        .collect();
    patch_pipeline_layers_for_remote_moe(&mut layers, &weights)
        .expect("fixture declares an executable router");
    assert_eq!(
        layers[0].moe.as_ref().expect("patched").expert_data_format,
        QuantFormat::BF16,
        "the packed-monolith fixture is BF16"
    );

    let per_layer = per_layer_fixture();
    let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..per_layer.num_layers)
        .map(|_| crate::FullPipelineLayer::default())
        .collect();
    patch_pipeline_layers_for_remote_moe(&mut layers, &per_layer)
        .expect("fixture declares an executable router");
    assert_eq!(
        layers[0].moe.as_ref().expect("patched").expert_data_format,
        QuantFormat::Q4_K,
        "a per-layer store is Q4_K-shaped"
    );
}

// ── Routing rules no policy computes ──────────────────────────────────

/// An OLMoE architecture over the per-layer store, declaring `scoring`
/// as its router activation, with a router projection under its own key
/// when `with_router`. OLMoE takes the router rule from `config.json`, so
/// the declaration is the only thing that differs between the arms.
fn olmoe_over_store(
    scoring: &str,
    with_router: bool,
) -> (
    larql_models::ModelWeights,
    Box<dyn larql_models::ModelArchitecture>,
) {
    let mut weights = per_layer_fixture();
    let gemma = &*weights.arch;
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "olmoe",
        "hidden_size": weights.hidden_size,
        "intermediate_size": gemma.moe_intermediate_size(),
        "num_hidden_layers": weights.num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": gemma.num_experts(),
        "num_experts_per_tok": gemma.num_experts_per_token(),
        "vocab_size": weights.hidden_size,
        "scoring_func": scoring,
    }));
    if with_router {
        let key = arch.moe_router_key(0).expect("OLMoE names a router");
        let rows = arch.num_experts() * weights.hidden_size;
        weights.vectors.insert(key, vec![0.0; rows]);
    }
    (weights, arch)
}

#[test]
fn a_softmax_router_builds_the_block() {
    let (weights, arch) = olmoe_over_store("softmax", true);
    let moe = build_moe_weights(&weights, &*arch, 0)
        .expect("softmax is executable")
        .expect("layer 0 has a router and experts");
    assert_eq!(moe.experts_gate_up.len(), arch.num_experts());
}

/// A declared sigmoid router is refused with its kind, not a panic and not
/// a substitute policy.
#[test]
fn a_sigmoid_router_is_refused_with_its_kind() {
    let (weights, arch) = olmoe_over_store("sigmoid", true);
    let Err(err) = build_moe_weights(&weights, &*arch, 0) else {
        panic!("sigmoid has no policy, so the block must be refused");
    };
    assert_eq!(err.kind, larql_models::MoeRouterKind::Sigmoid);
    assert!(err.to_string().contains("sigmoid"), "{err}");
}

/// The rule is resolved only for a layer that builds a block: a layer with
/// no router is dense, and asks nothing of the routing rule.
#[test]
fn a_layer_without_a_router_never_asks_for_the_rule() {
    let (weights, arch) = olmoe_over_store("sigmoid", false);
    assert!(build_moe_weights(&weights, &*arch, 0)
        .expect("a dense layer is not a refusal")
        .is_none());
}

#[test]
fn every_executable_kind_resolves_and_sigmoid_does_not() {
    use larql_models::MoeRouterKind::*;
    for kind in [TopKSoftmax, TopKThenSoftmax, TopKRenormScaled] {
        assert!(
            crate::MoeRoutingPolicy::for_router_kind(kind).is_ok(),
            "{kind:?}"
        );
    }
    assert_eq!(
        crate::MoeRoutingPolicy::for_router_kind(Sigmoid),
        Err(crate::UnsupportedRouting { kind: Sigmoid })
    );
}
