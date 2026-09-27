//! Attention tables, bindings and expert-bank carving.

use super::super::component::{Component, ComponentRole};
use super::super::edge::HiddenStateEdge;
use super::super::object::{Fidelity, LogicalObject, ObjectKind, Representation, SourceBinding};
use super::super::policy::{
    resolve_layer_kind, AttentionLayerPolicy, AttentionSpan, HeadGeometry, LayerOperator,
};
use larql_models::config::{PositionPolicy, LAYER_TYPE_LINEAR_ATTENTION};
use larql_models::inventory::{ArchitectureInventory, TensorGroup};
use std::collections::BTreeMap;

#[allow(unused_imports)]
use super::*;

pub(super) fn attention_table(inventory: &ArchitectureInventory) -> Vec<AttentionLayerPolicy> {
    let mla = uses_mla(inventory);
    let conv_qkv = inventory.resolved.conv_qkv_attn.is_some();
    inventory
        .resolved
        .layers
        .iter()
        .map(|layer| {
            // Operator and span decided together, in the one place that
            // rule lives — a recurrence gets no span rather than a
            // defaulted `Full`.
            // The checkpoint's own canonical declaration is authoritative
            // when it made one. The resolved boolean is a *derivation* —
            // it answers sliding-or-full from whichever key the parser
            // happened to read — and on a family whose interleave it
            // cannot read it answers "full" for every layer. That is how
            // Inkling-Small's 35 sliding layers (window 512, against a
            // 1,048,576-token context) were reported as retaining an
            // unbounded prefix.
            //
            // `plan::compare` keeps grading the declared array against the
            // boolean, so the comparison it makes stays a real one: the
            // authority moves here, not there.
            let (operator, span) = match layer.declared_kind.as_ref() {
                // A spelling with no kind keeps the layer-blind path: the
                // graph records a softmax layer whose declaration it
                // cannot round-trip to, which is exactly `unexpressed`.
                Some(larql_models::config::LayerKind::Unexpressed { .. }) | None => {
                    resolve_layer_kind(
                        layer.declared_span.as_deref(),
                        layer.attention == RESOLVED_ATTENTION_SLIDING,
                        recurrence_kind(inventory),
                        mla,
                    )
                }
                Some(kind) => operator_and_span(kind, recurrence_kind(inventory), mla, conv_qkv),
            };
            // The architecture's resolved window stays authoritative — it
            // can apply per-family rules the raw config cannot state. The
            // declared window is the FALLBACK, for a family whose window
            // spelling the parser does not know (Inkling-Small writes
            // `sliding_window_size`, not `sliding_window`), where the
            // architecture yields nothing and a sliding layer would
            // otherwise carry no size at all.
            let declared_window = match layer.declared_kind.as_ref() {
                Some(larql_models::config::LayerKind::Sliding { window }) => *window,
                _ => None,
            };
            AttentionLayerPolicy {
                operator,
                span,
                window: layer.window.or(declared_window),
                position: layer.position,
                geometry: Some(HeadGeometry {
                    head_dim: layer.head_dim,
                    num_kv_heads: layer.num_kv_heads,
                }),
                v_from_k: layer.v_from_k,
                // Carried alongside the boolean-derived `span` verbatim, so a
                // declared spelling the vocabulary cannot express (a hybrid
                // linear-attention interleave) is recorded rather than
                // silently lost behind whatever `span` defaulted to. See
                // `AttentionLayerPolicy::declared_span` and
                // `plan::carriage::probe_layer_types`.
                declared_span: layer.declared_span.clone(),
            }
        })
        .collect()
}

/// The per-layer policy of a **nested** component, from that component's
/// own declared topology.
///
/// The same shape as [`attention_table`] one level down: a component's
/// per-layer policy comes from that component's facts. Nested components
/// previously carried `attention: None` — honest at the time, but it
/// meant a perception tower's declared interleave and rope base were
/// parsed into [`ComponentTopology`] and then dropped before the graph,
/// with nothing reporting the loss.
///
/// A component that declares a layer count but no `layer_types` attends
/// fully on every layer — that is what the absence of an interleave means
/// in the HF configs that omit it (Gemma 4's vision tower: 27 layers, one
/// rope base, no `layer_types`), and recording it lets the tower's rope
/// facts be judged against a table instead of vanishing. `None` when the
/// component declares neither a count nor an interleave, or when it names
/// a span the vocabulary does not contain — refusing rather than
/// resolving an unknown spelling to a default, which is the failure this
/// vocabulary exists to prevent.
pub(super) fn nested_attention_table(
    topology: &larql_models::inventory::components::ComponentTopology,
) -> Option<Vec<AttentionLayerPolicy>> {
    let uniform_full;
    let layer_types: &Vec<String> = match (&topology.layer_types, topology.num_layers) {
        (Some(declared), _) => declared,
        (None, Some(n)) => {
            uniform_full = vec![AttentionSpan::Full.declared_name().to_string(); n];
            &uniform_full
        }
        (None, None) => return None,
    };
    // A rope base the component declares for itself; absent means the
    // component states no position policy, which is a fact, not a zero.
    let position = match topology.rope_theta {
        Some(theta) => PositionPolicy::from_declared_theta(theta),
        None => PositionPolicy::None,
    };
    layer_types
        .iter()
        .map(|entry| {
            // A nested component's contract is stricter than the text
            // stack's: it has no resolved boolean to fall back on, so an
            // unrecognised spelling refuses the whole table rather than
            // deferring to a comparison downstream. `linear_attention` is
            // named here so that a perception tower which declares a
            // recurrence records one — no judged tower does today, and
            // the alternative is for it to refuse a spelling the schema
            // now has a home for.
            let (operator, span) = if entry.eq_ignore_ascii_case(LAYER_TYPE_LINEAR_ATTENTION) {
                // A nested component has no linear-attention geometry to
                // resolve — the recurrence keys are text-stack keys — so
                // a tower declaring one is recorded as an unidentified
                // recurrence, never as Gated DeltaNet.
                (LayerOperator::Recurrent, None)
            } else {
                (
                    LayerOperator::Softmax,
                    Some(AttentionSpan::from_declared(entry)?),
                )
            };
            Some(AttentionLayerPolicy {
                operator,
                span,
                // No nested component declares a sequence window today;
                // a spatial window's extent is not a position count.
                window: None,
                position,
                // A nested topology states one head geometry for the
                // whole tower; the surface carries it.
                geometry: None,
                v_from_k: false,
                declared_span: Some(entry.clone()),
            })
        })
        .collect()
}

/// Merge one tensor group into the `(component, kind)` object, creating it
/// on first sight.
pub(super) fn merge_binding(
    objects: &mut BTreeMap<String, LogicalObject>,
    component: &str,
    kind: ObjectKind,
    artifact: &str,
    group: &TensorGroup,
    inventory: &ArchitectureInventory,
) {
    let id = format!("{component}.{}", kind.name());
    let object = objects.entry(id.clone()).or_insert_with(|| LogicalObject {
        id,
        component: component.to_string(),
        kind,
        source_bindings: Vec::new(),
        representations: canonical_representation(inventory, |name| {
            name.starts_with(&group.prefix)
        })
        .into_iter()
        .collect(),
    });
    object.source_bindings.push(SourceBinding {
        artifact: artifact.to_string(),
        tensor_prefix: group.prefix.clone(),
        tensors: group.tensors,
        bytes: group.bytes,
    });
}

/// Carve every routed layer's expert bank out of a just-placed decoder
/// stack into the component's [`ObjectKind::ExpertBank`] object, one
/// binding per layer at the prefix the architecture named
/// (`resolved.layers[L].expert_bank`). Ownership is settled by binding
/// specificity ([`super::super::object::most_specific_owner`]), so the stack keeps
/// its whole-stack binding and simply stops owning what the bank binds;
/// its recorded counts and representation are re-derived over what it
/// still owns, so neither object describes bytes it does not hold.
pub(super) fn carve_expert_banks(
    objects: &mut BTreeMap<String, LogicalObject>,
    component: &str,
    artifact: &str,
    group: &TensorGroup,
    inventory: &ArchitectureInventory,
) {
    let bank_prefixes: Vec<&str> = inventory
        .resolved
        .layers
        .iter()
        .filter_map(|l| l.expert_bank.as_deref())
        .filter(|p| {
            p.strip_prefix(&group.prefix)
                .is_some_and(|r| r.starts_with('.'))
        })
        .collect();
    if bank_prefixes.is_empty() {
        return;
    }
    let in_bank = |name: &str| {
        bank_prefixes
            .iter()
            .any(|p| name == *p || name.strip_prefix(p).is_some_and(|r| r.starts_with('.')))
    };
    let bank_id = format!("{component}.{}", ObjectKind::ExpertBank.name());
    let bank = objects
        .entry(bank_id.clone())
        .or_insert_with(|| LogicalObject {
            id: bank_id,
            component: component.to_string(),
            kind: ObjectKind::ExpertBank,
            source_bindings: Vec::new(),
            representations: canonical_representation(inventory, in_bank)
                .into_iter()
                .collect(),
        });
    let mut carved_tensors = 0usize;
    let mut carved_bytes = 0u64;
    for prefix in &bank_prefixes {
        let (tensors, bytes) = inventory
            .tensors
            .tensors
            .iter()
            .filter(|t| {
                t.name == *prefix
                    || t.name
                        .strip_prefix(prefix)
                        .is_some_and(|r| r.starts_with('.'))
            })
            .fold((0usize, 0u64), |(n, b), t| (n + 1, b + t.bytes));
        carved_tensors += tensors;
        carved_bytes += bytes;
        bank.source_bindings.push(SourceBinding {
            artifact: artifact.to_string(),
            tensor_prefix: (*prefix).to_string(),
            tensors,
            bytes,
        });
    }
    // The stack no longer owns those tensors: correct its counts and its
    // representation to what it still holds.
    let stack_id = format!("{component}.{}", ObjectKind::DecoderStack.name());
    if let Some(stack) = objects.get_mut(&stack_id) {
        if let Some(binding) = stack
            .source_bindings
            .iter_mut()
            .find(|b| b.artifact == artifact && b.tensor_prefix == group.prefix)
        {
            binding.tensors = binding.tensors.saturating_sub(carved_tensors);
            binding.bytes = binding.bytes.saturating_sub(carved_bytes);
        }
        stack.representations = canonical_representation(inventory, |name| {
            name.starts_with(&group.prefix) && !in_bank(name)
        })
        .into_iter()
        .collect();
    }
}

/// The MXFP4 pair suffixes HF writes for a block-quantised tensor: the
/// packed e2m1 nibbles and the e8m0 scales, both stored as `U8`.
pub(super) const MXFP4_BLOCKS_SUFFIX: &str = "_blocks";
pub(super) const MXFP4_SCALES_SUFFIX: &str = "_scales";
/// The encoding name a declared MXFP4 tensor is placed under — the
/// codec's own label, so the graph and the registry cannot spell it apart.
pub(super) const MXFP4_ENCODING: &str =
    crate::format::vindex3::represent::codec::codecs::mxfp4::DTYPE_MXFP4;

/// The encoding one tensor is placed under: its shard dtype, unless the
/// checkpoint's declared stored representation says those bytes are
/// something else. A `U8` `*_blocks` / `*_scales` tensor under an `mxfp4`
/// declaration, outside `modules_to_not_convert`, is MXFP4 — placing it as
/// raw bytes would drop the one fact that gives the bytes meaning.
pub(super) fn tensor_encoding<'a>(
    inventory: &'a ArchitectureInventory,
    name: &str,
    dtype: &'a str,
) -> &'a str {
    let Some(rep) = inventory.stored_representation.as_ref() else {
        return dtype;
    };
    let declared_mxfp4 = rep
        .method
        .eq_ignore_ascii_case(larql_models::inventory::representation::QUANT_METHOD_MXFP4);
    let mxfp4_pair = name.ends_with(MXFP4_BLOCKS_SUFFIX) || name.ends_with(MXFP4_SCALES_SUFFIX);
    if declared_mxfp4 && mxfp4_pair && dtype == "U8" && !rep.excludes(name) {
        MXFP4_ENCODING
    } else {
        dtype
    }
}

/// The canonical representation for an object: encodings actually observed
/// in the shard headers under this prefix — read through the checkpoint's
/// declared stored representation, see [`tensor_encoding`] — falling back
/// to the checkpoint's declared dtype when the per-tensor list was
/// stripped. Never invented.
pub(super) fn canonical_representation(
    inventory: &ArchitectureInventory,
    is_member: impl Fn(&str) -> bool,
) -> Option<Representation> {
    let mut encodings: Vec<&str> = inventory
        .tensors
        .tensors
        .iter()
        .filter(|t| is_member(&t.name))
        .map(|t| tensor_encoding(inventory, &t.name, t.dtype.as_str()))
        .collect();
    encodings.sort_unstable();
    encodings.dedup();
    let encoding = if encodings.is_empty() {
        inventory.identity.dtype.clone()?
    } else {
        encodings.join("+")
    };
    Some(Representation {
        encoding,
        fidelity: Fidelity::Canonical,
    })
}

/// Identify the tap-fusion projector by shape evidence: a 2-D tensor of
/// shape `len(taps)·hidden × hidden` (either orientation). Returns the
/// tensor's first path segment — the adjacency key that claims its sibling
/// groups — or records why identification failed.
pub(super) fn find_projector_segment(
    artifact: &str,
    inventory: &ArchitectureInventory,
    taps: &[usize],
    unresolved: &mut Vec<UnresolvedInterface>,
) -> Option<String> {
    let hidden = inventory.resolved.hidden_size;
    let expected = taps.len() * hidden;
    if inventory.tensors.tensors.is_empty() {
        unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: "per-tensor list absent — projector cannot be identified by shape".to_string(),
        });
        return None;
    }
    let projector_tensor = inventory.tensors.tensors.iter().find(|t| {
        t.shape.len() == 2
            && ((t.shape[0] == expected && t.shape[1] == hidden)
                || (t.shape[0] == hidden && t.shape[1] == expected))
    });
    match projector_tensor {
        Some(tensor) => Some(
            tensor
                .name
                .split('.')
                .next()
                .unwrap_or(&tensor.name)
                .to_string(),
        ),
        None => {
            unresolved.push(UnresolvedInterface {
                artifact: artifact.to_string(),
                reason: format!(
                    "no 2-D tensor of shape {expected}×{hidden} implements the declared \
                     {}-tap interface",
                    taps.len()
                ),
            });
            None
        }
    }
}

/// Wire the interface edge: the producer must be exactly one other
/// component deep enough to own every declared tap.
pub(super) fn wire_edge(
    artifact: &str,
    inventory: &ArchitectureInventory,
    taps: &[usize],
    components: &[Component],
    edges: &mut Vec<HiddenStateEdge>,
    unresolved: &mut Vec<UnresolvedInterface>,
) {
    let Some(consumer) = component_for_artifact(components, artifact) else {
        unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: "declaring artifact has no component".to_string(),
        });
        return;
    };
    let max_tap = taps.iter().copied().max().unwrap_or(0);
    let mut producers = components.iter().filter(|c| {
        c.source_artifact != *artifact
            && c.role == ComponentRole::PrimaryText
            && c.num_layers > max_tap
    });
    match (producers.next(), producers.next()) {
        (Some(producer), None) => edges.push(HiddenStateEdge {
            producer_component: producer.id.clone(),
            producer_layers: taps.to_vec(),
            consumer_component: consumer.clone(),
            consumer_object: format!("{consumer}.{}", ObjectKind::FeatureProjector.name()),
            block_size: declared_block_size(inventory),
        }),
        (None, _) => unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: format!("no component in the set owns layer {max_tap} — producer unresolvable"),
        }),
        (Some(_), Some(_)) => unresolved.push(UnresolvedInterface {
            artifact: artifact.to_string(),
            reason: "multiple candidate producers — refusing to guess".to_string(),
        }),
    }
}
