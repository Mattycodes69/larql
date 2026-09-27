//! `vindex describe`: one address resolved to its declaration and values.

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::OperandRef;
use mixer::{layer_range, MixerOperands};
use serde_json::{json, Value};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// `vindex describe <address>` — one logical object, in full: identity,
/// bindings, representations, and the head of its tensor table. With
/// `peek`, the first `values` decoded weights of one named tensor —
/// the numbers themselves, read from the canonical bytes.
pub fn describe_facts(root: &Path, address: &str, values: usize, peek: Option<&str>) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    // `layer.N.mixer` — the token mixer as a first-class semantic
    // component: its programme, and every operand with the role the
    // plan assigns it. Architecture-neutral by construction.
    if let Some(n) = address
        .strip_prefix("layer.")
        .and_then(|r| r.strip_suffix(".mixer"))
        .and_then(|n| n.parse::<usize>().ok())
    {
        let (component, plan) = primary_plan(root, &inspection)?;
        let operator = mixer::declared_operator(component, n)?;
        let layer = plan
            .layers
            .get(n)
            .ok_or_else(|| layer_range(&format!("layer {n}"), plan.layers.len()))?;
        let (operands, undescribed) = match mixer::operands(operator, layer) {
            MixerOperands::Named(ops) => (
                ops.into_iter()
                    .map(|(role, op)| {
                        json!({ "role": role, "tensor": op.tensor, "shape": op.shape, "dtype": op.dtype })
                    })
                    .collect::<Vec<Value>>(),
                None,
            ),
            MixerOperands::Undescribed(why) => (Vec::new(), Some(why)),
        };
        return Ok(json!({
            "container": root.display().to_string(),
            "semantic": address,
            "mixer": mixer::label(operator, mixer::has_output_gate(layer)),
            "layer": n,
            "operands": operands,
            "undescribed": undescribed,
        }));
    }
    if let Some(resolved) = resolve_semantic(root, &inspection, address) {
        let (role, op) = resolved?;
        let mut representations: Vec<Value> = Vec::new();
        for entry in inspection
            .index
            .representations
            .values()
            .filter(|e| e.object == op.object)
        {
            let (header, _) = read_segment_header(&root.join(&entry.segment))
                .map_err(|e| format!("segment {}: {e}", entry.segment))?;
            if let Some(t) = header.tensors.iter().find(|t| t.name == op.tensor) {
                let weights: u128 = t.shape.iter().product::<usize>() as u128;
                representations.push(json!({
                    "encoding": entry.encoding,
                    "dtype": t.dtype,
                    "bits_per_weight": if weights > 0 { (t.len as u128 * 8) as f64 / weights as f64 } else { 0.0 },
                    "bytes": t.len,
                }));
            }
        }
        let store = OperandStore::open(root, &inspection).map_err(|e| format!("open: {e}"))?;
        let decoded = load_values(&store, &op)?;
        return Ok(json!({
            "container": root.display().to_string(),
            "semantic": address,
            "role": role,
            "object": op.object,
            "tensor": op.tensor,
            "shape": op.shape,
            "representations": representations,
            "values": decoded.iter().take(values).collect::<Vec<_>>(),
        }));
    }
    let object = find_object(&inspection, address)?;
    let directory: Vec<Value> = inspection
        .index
        .representations
        .iter()
        .filter(|(_, e)| e.object == object.id)
        .map(|(id, e)| {
            let tensors = read_segment_header(&root.join(&e.segment))
                .ok()
                .map(|(header, _)| {
                    header
                        .tensors
                        .iter()
                        .take(values)
                        .map(|t| json!({ "name": t.name, "dtype": t.dtype, "shape": t.shape, "len": t.len }))
                        .collect::<Vec<Value>>()
                });
            json!({
                "id": id,
                "encoding": e.encoding,
                "segment": e.segment,
                "tensor_count": e.tensor_count,
                "payload_bytes": e.payload_bytes,
                "tensor_table_head": tensors,
            })
        })
        .collect();
    let peeked = match peek {
        None => Value::Null,
        Some(tensor) => {
            let store = OperandStore::open(root, &inspection).map_err(|e| format!("open: {e}"))?;
            let entry = inspection
                .index
                .representations
                .values()
                .find(|e| e.object == object.id)
                .ok_or_else(|| format!("`{}` has no directory entry", object.id))?;
            let (header, _) = read_segment_header(&root.join(&entry.segment))
                .map_err(|e| format!("segment {}: {e}", entry.segment))?;
            let t = header
                .tensors
                .iter()
                .find(|t| t.name == tensor || t.name.ends_with(&format!(".{tensor}")))
                .ok_or_else(|| {
                    format!(
                        "no tensor of `{}` matches `{tensor}` — tensors: {}",
                        object.id,
                        header
                            .tensors
                            .iter()
                            .map(|t| t.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            let decoded = load_values(
                &store,
                &OperandRef {
                    object: object.id.clone(),
                    tensor: t.name.clone(),
                    dtype: t.dtype.clone(),
                    shape: t.shape.clone(),
                },
            )?;
            json!({
                "tensor": t.name,
                "dtype": t.dtype,
                "shape": t.shape,
                "values": decoded.iter().take(values).collect::<Vec<_>>(),
            })
        }
    };
    Ok(json!({
        "container": root.display().to_string(),
        "object": serde_json::to_value(object).map_err(|e| e.to_string())?,
        "directory": directory,
        "peek": peeked,
    }))
}
