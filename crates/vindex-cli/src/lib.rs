//! vindex — the format-native facts, as data.
//!
//! Every function here reads only what the container declares —
//! `index.json`, the system graph, the segment headers — and returns
//! one `serde_json::Value`: the same object the binary prints with
//! `--json`, renders as text without it, and the web Explorer renders
//! as a designed panel. One result, three projections; the litmus
//! test for every fact is that an independent VINDEX3 implementation
//! could derive it from the artifact alone.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub mod mixer;

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::artifact;
use larql_vindex::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::plan::capability::Capability;
use larql_vindex::format::vindex3::plan::plan_resolved;
use larql_vindex::format::vindex3::represent::{compile_representation, RepresentSpec};

mod describe;
mod diff;
mod precision;
pub use describe::*;
pub use diff::*;
pub use precision::*;

pub type Facts = Result<Value, String>;

fn read_index(root: &Path) -> Result<Vindex3Index, String> {
    let text = std::fs::read_to_string(root.join(INDEX_JSON))
        .map_err(|e| format!("read {INDEX_JSON}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {INDEX_JSON}: {e}"))
}

fn authority_str(a: &ContainerAuthority) -> &'static str {
    match a {
        ContainerAuthority::Canonical => "canonical",
        ContainerAuthority::Derived => "derived",
    }
}

/// `vindex inspect` — the container, reconstructed from itself.
pub fn inspect_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let index = &inspection.index;
    Ok(json!({
        "container": root.display().to_string(),
        "generation": 3,
        "model": index.model,
        "family": index.family,
        "hidden_size": index.hidden_size,
        "num_layers": index.num_layers,
        "authority": authority_str(&index.authority),
        "derived_from_model": index.derived_from_model,
        "components": inspection.components,
        "objects": inspection.graph.objects.len(),
        "edges": inspection.graph.edges.len(),
        "coherent": inspection.is_coherent(),
        "defects": inspection.defects.len(),
    }))
}

/// `vindex representations` — the physical directory, with the graph's
/// fidelity beside each entry.
pub fn representations_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let fidelity_of = |object: &str, encoding: &str| -> Option<String> {
        inspection
            .graph
            .objects
            .iter()
            .find(|o| o.id == object)
            .and_then(|o| o.representations.iter().find(|r| r.encoding == encoding))
            .map(|r| format!("{:?}", r.fidelity).to_lowercase())
    };
    let entries: Vec<Value> = inspection
        .index
        .representations
        .iter()
        .map(|(id, e)| {
            json!({
                "id": id,
                "object": e.object,
                "encoding": e.encoding,
                "fidelity": fidelity_of(&e.object, &e.encoding),
                "tensor_count": e.tensor_count,
                "payload_bytes": e.payload_bytes,
                "compiled_from": e.compiled_from,
            })
        })
        .collect();
    Ok(json!({ "container": root.display().to_string(), "entries": entries }))
}

/// `vindex verify` — the container against its own recorded hashes:
/// every segment file re-hashed whole, every payload region re-hashed,
/// both compared with what the directory recorded at encode time. This
/// is self-verification — corruption detection from the artifact
/// alone. Proving faithfulness to the *source* additionally needs the
/// source, and that lives with the reference implementation's G4 gate.
pub fn verify_facts(root: &Path) -> Facts {
    let index = read_index(root)?;
    let mut entries: Vec<Value> = Vec::new();
    let mut failures = 0usize;
    for (id, e) in &index.representations {
        let path = root.join(&e.segment);
        let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", e.segment))?;
        let segment_hash = format!("{:x}", Sha256::digest(&bytes));
        let payload_start = 8 + u64::from_le_bytes(
            bytes
                .get(0..8)
                .ok_or_else(|| format!("{}: shorter than its own framing", e.segment))?
                .try_into()
                .map_err(|_| "framing".to_string())?,
        ) as usize;
        let payload = bytes
            .get(payload_start..)
            .ok_or_else(|| format!("{}: payload offset beyond file", e.segment))?;
        let payload_hash = format!("{:x}", Sha256::digest(payload));
        let segment_ok = segment_hash == e.segment_sha256;
        let payload_ok = payload_hash == e.payload_sha256;
        if !segment_ok || !payload_ok {
            failures += 1;
        }
        entries.push(json!({
            "id": id,
            "segment_ok": segment_ok,
            "payload_ok": payload_ok,
        }));
    }
    Ok(json!({
        "container": root.display().to_string(),
        "entries": entries,
        "failures": failures,
        "verified": failures == 0,
        "scope": "self — recorded hashes re-derived from the artifact alone; source faithfulness is the reference implementation's G4",
    }))
}

/// `vindex represent <src> <out>` — compile a representation the spec
/// defines, through the reference compiler. The output container
/// carries every original segment plus the compiled packs; nothing is
/// destroyed, and `vindex verify` holds on the result.
/// `vindex export` — the container's selected representation, compiled
/// to a qwen35 GGUF and verified through the independent reader before
/// the function returns. Every count in the result was observed from
/// the finished file, none predicted.
pub fn export_facts(root: &Path, out: &Path) -> Facts {
    let report = larql_vindex::format::vindex3::gguf::export::export_qwen35(root, out)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "container": root.display().to_string(),
        "out": report.out.display().to_string(),
        "bytes": report.bytes,
        "selected": report.selected_encoding,
        "walk": {
            "source_tensors": report.ledger.source_total,
            "accounted": report.ledger.accounted,
            "geometry_reconciled": report.ledger.geometry_reconciled,
            "scale_siblings": report.ledger.generated_scale_tensors,
        },
        "emitted": {
            "tensors": report.emit.tensors,
            "scale_siblings": report.emit.scale_siblings,
            "metadata_keys": report.emit.metadata_keys,
        },
        "vocab": {
            "tokens": report.vocab_tokens,
            "padded": report.vocab_padded,
            "merges": report.vocab_merges,
        },
        "verified": {
            "tensors": report.verify.tensors,
            "nvfp4_tensors": report.verify.nvfp4_tensors,
            "scale_siblings": report.verify.scale_siblings,
            "metadata_keys": report.verify.metadata_keys,
        },
    }))
}

pub fn represent_facts(src: &Path, out: &Path, encoding: &str) -> Facts {
    let mut spec = RepresentSpec::nvfp4();
    spec.encoding = encoding.to_uppercase();
    let report = compile_representation(src, out, &spec).map_err(|e| format!("represent: {e}"))?;
    let compiled: Vec<Value> = report
        .compiled_objects
        .iter()
        .map(|c| {
            json!({
                "object": c.object,
                "representation": c.representation_id,
                "compiled_tensors": c.compiled_tensors,
                "carried_tensors": c.carried_tensors,
                "source_bytes": c.source_bytes,
                "compiled_bytes": c.compiled_bytes,
                "compression": c.compression(),
                "preserved_roles": c.preserved.iter()
                    .map(|(role, n)| json!({ "role": format!("{role:?}"), "tensors": n }))
                    .collect::<Vec<Value>>(),
            })
        })
        .collect();
    let preserved: Vec<Value> = report
        .preserved_objects
        .iter()
        .map(|p| {
            json!({
                "object": p.object,
                "encoding": p.encoding,
                "bytes": p.bytes,
            })
        })
        .collect();
    Ok(json!({
        "source": src.display().to_string(),
        "out": out.display().to_string(),
        "encoding": spec.encoding,
        "map": spec.map_name(),
        "compiled": compiled,
        "preserved": preserved,
        "linked_segments": report.linked_segments,
    }))
}

// ── Ingest: bringing a model in ──────────────────────────────────────────
//
// Every other verb reads a container that already exists. These two make
// one, and they are the reason `vindex` is a tool you can start with
// rather than a tool you reach for afterwards.
//
// Both resolve their arguments through
// `larql_vindex::format::vindex3::artifact`, which is also what
// `larql vindex3` uses — one authority on what an artifact argument means,
// so the two binaries cannot disagree about a model's identity or produce
// different containers from the same input.

/// The admission verdict for an artifact, without moving its weights.
///
/// A bring-up instrument, not the ordinary path: it answers "what does
/// VINDEX still need to understand about this model?" from configuration
/// and safetensors headers alone. On GLM-5.3-Flash that is ~39 MB of
/// staging against a 328 GB checkpoint.
pub fn plan_facts(artifacts: &[PathBuf]) -> Facts {
    let resolved = artifact::resolve_all(artifacts).map_err(|e| e.to_string())?;
    let staging: Vec<Value> = resolved
        .iter()
        .filter_map(artifact::ResolvedArtifact::staging_json)
        .collect();
    let plan = plan_resolved(artifacts, resolved).map_err(|e| e.to_string())?;
    let mut value = serde_json::to_value(&plan).map_err(|e| e.to_string())?;
    if !staging.is_empty() {
        value["staging"] = Value::Array(staging);
    }
    Ok(value)
}

/// Encode artifacts into a container.
///
/// An `hf://` argument is read over byte ranges: the canonical checkpoint
/// never needs to exist as a complete local file.
pub fn encode_facts(artifacts: &[PathBuf], output: &Path, text_only: bool) -> Facts {
    let resolved = artifact::resolve_all(artifacts).map_err(|e| e.to_string())?;
    let staging: Vec<Value> = resolved
        .iter()
        .filter_map(artifact::ResolvedArtifact::staging_json)
        .collect();
    let pinned: Vec<Value> = resolved
        .iter()
        .filter_map(|a| {
            Some(json!({
                "artifact": a.name,
                "commit": a.commit()?,
            }))
        })
        .collect();
    let unpinned: Vec<Value> = resolved
        .iter()
        .filter_map(|a| {
            Some(json!({
                "artifact": a.name,
                "revision": a.unpinned_revision()?,
            }))
        })
        .collect();

    let capability = text_only.then_some(Capability::TextGeneration);
    let outcome =
        artifact::encode_from_specs(resolved, output, capability).map_err(|e| e.to_string())?;

    Ok(json!({
        "container": outcome.container.display().to_string(),
        "representations": outcome.representations,
        "payload_bytes": outcome.total_payload_bytes,
        "payload": artifact::size(outcome.total_payload_bytes),
        "capabilities": outcome.capabilities,
        "staging": staging,
        "pinned": pinned,
        "unpinned": unpinned,
        "transfers": outcome.transfers.iter().map(|t| json!({
            "artifact": t.name,
            "tensors": t.tensors,
            "fetched_bytes": t.fetched,
            "fetched": artifact::size(t.fetched),
            "declared": artifact::size(t.declared),
            // The ratio IS the claim. Near 1.0 means the plan bound every
            // tensor; well under means the container carries less than the
            // checkpoint holds.
            "fraction": if t.declared == 0 { 0.0 } else { t.fetched as f64 / t.declared as f64 },
        })).collect::<Vec<_>>(),
    }))
}
