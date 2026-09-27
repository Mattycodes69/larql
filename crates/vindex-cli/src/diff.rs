//! `vindex diff`: two containers compared by semantic address.

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::graph::Component;
use larql_vindex::format::vindex3::inspect::{inspect_container, SystemInspection};
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, LayerFfn, OperandRef,
};
use larql_vindex::format::vindex3::represent::nvfp4_pack::{split, PackLayout, DTYPE_NVFP4};
use mixer::layer_range;
use serde_json::{json, Value};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

pub(super) fn find_object<'a>(
    inspection: &'a SystemInspection,
    address: &str,
) -> Result<&'a larql_vindex::format::vindex3::graph::object::LogicalObject, String> {
    inspection
        .graph
        .objects
        .iter()
        .find(|o| o.id == address || o.id.ends_with(address))
        .ok_or_else(|| {
            let known: Vec<&str> = inspection
                .graph
                .objects
                .iter()
                .map(|o| o.id.as_str())
                .collect();
            format!(
                "no logical object matches `{address}` — the graph holds: {}",
                known.join(", ")
            )
        })
}

/// A tensor as one side's header declares it: name, dtype, shape.
pub(super) type TensorEntry = (String, String, Vec<usize>);

/// One side of a diff: the store bound to `encoding`, plus that
/// encoding's tensor table for the object. Refuses an encoding the
/// container does not hold, naming what it does.
pub(super) fn open_side(
    root: &Path,
    inspection: &SystemInspection,
    object: &str,
    encoding: &str,
) -> Result<(OperandStore, Vec<TensorEntry>), String> {
    let canonical = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.id == object)
        .and_then(|o| o.representations.first())
        .map(|r| r.encoding.clone())
        .ok_or_else(|| format!("object `{object}` declares no representations"))?;
    let store = if encoding.eq_ignore_ascii_case(&canonical) {
        OperandStore::open(root, inspection).map_err(|e| format!("open {encoding}: {e}"))?
    } else {
        OperandStore::open_for(
            root,
            inspection,
            Some(encoding),
            RepresentationSource::Stored,
        )
        .map_err(|e| format!("open {encoding}: {e}"))?
    };
    let bound = store
        .selection()
        .get(object)
        .map(|s| s.encoding.clone())
        .unwrap_or_default();
    if !bound.eq_ignore_ascii_case(encoding) {
        let held: Vec<String> = inspection
            .index
            .representations
            .values()
            .filter(|e| e.object == object)
            .map(|e| e.encoding.clone())
            .collect();
        return Err(format!(
            "`{object}` has no {encoding} representation — the container holds: {}",
            held.join(", ")
        ));
    }
    let entry = inspection
        .index
        .representations
        .values()
        .find(|e| e.object == object && e.encoding.eq_ignore_ascii_case(encoding))
        .ok_or_else(|| format!("no directory entry for {object}@{encoding}"))?;
    let (header, _) = read_segment_header(&root.join(&entry.segment))
        .map_err(|e| format!("segment {}: {e}", entry.segment))?;
    let tensors = header
        .tensors
        .into_iter()
        .map(|t| (t.name, t.dtype, t.shape))
        .collect();
    Ok((store, tensors))
}

/// The primary component: the graph node that declares it, and the
/// operation plan built over it.
///
/// Both, because the two answer different questions and neither
/// substitutes for the other — the node declares what each layer's
/// token mixer IS, the plan binds the operands that mixer computes
/// with. See [`mixer`].
pub(super) fn primary_plan<'a>(
    root: &Path,
    inspection: &'a SystemInspection,
) -> Result<(&'a Component, ComponentOpPlan), String> {
    let component = inspection
        .graph
        .components
        .first()
        .ok_or("the graph holds no components")?;
    let outcome =
        plan_component_ops(inspection, root, &component.id).map_err(|e| format!("plan: {e}"))?;
    let plan = outcome
        .plan
        .ok_or_else(|| format!("component `{}` has no plan", component.id))?;
    Ok((component, plan))
}

/// A semantic address, resolved through the container's own operation
/// plan — the graph's judgement of what each tensor IS, never a
/// filename convention. `layer.N.ffn.{gate|up|down}` (mlp accepted)
/// and `layer.N.attention.{q|k|v|o}` (attn accepted). Returns the
/// role label with the operand.
pub(super) fn resolve_semantic(
    root: &Path,
    inspection: &SystemInspection,
    address: &str,
) -> Option<Result<(String, OperandRef), String>> {
    let parts: Vec<&str> = address.split('.').collect();
    let [lit, layer, family, role] = parts.as_slice() else {
        return None;
    };
    if *lit != "layer" {
        return None;
    }
    let n: usize = layer.parse().ok()?;
    let family = match *family {
        "ffn" | "mlp" => "ffn",
        "attention" | "attn" => "attention",
        _ => return None,
    };
    let component = inspection.graph.components.first()?.id.clone();
    let go = || -> Result<(String, OperandRef), String> {
        let outcome =
            plan_component_ops(inspection, root, &component).map_err(|e| format!("plan: {e}"))?;
        let plan = outcome
            .plan
            .ok_or_else(|| format!("component `{component}` has no plan"))?;
        let layer_plan = plan
            .layers
            .get(n)
            .ok_or_else(|| layer_range(&format!("layer {n}"), plan.layers.len()))?;
        match family {
            "ffn" => {
                let ffn = layer_plan
                    .ffn
                    .as_ref()
                    .and_then(|f| f.dense())
                    .ok_or_else(|| {
                        let kind = match &layer_plan.ffn {
                            Some(LayerFfn::Routed(_)) => "a routed (mixture-of-experts) FFN",
                            Some(LayerFfn::Hybrid(_)) => "a hybrid FFN",
                            None => "no FFN at all (a mixer-only layer)",
                            Some(LayerFfn::Dense(_)) => unreachable!(),
                        };
                        format!(
                        "layer {n} carries {kind} — per-expert addressing is not yet a CLI surface"
                    )
                    })?;
                let (label, op) = match *role {
                    "gate" => (
                        "FFN GATE PROJECTION",
                        ffn.gate.as_ref().ok_or_else(|| {
                            format!("layer {n}'s FFN is ungated — no gate operand")
                        })?,
                    ),
                    "up" => ("FFN UP PROJECTION", &ffn.up),
                    "down" => ("FFN DOWN PROJECTION", &ffn.down),
                    other => return Err(format!("unknown ffn role `{other}` — gate, up, down")),
                };
                Ok((label.to_string(), op.clone()))
            }
            "attention" => {
                let attn = layer_plan.attention.softmax().ok_or_else(|| {
                    let mixer = inspection
                        .graph
                        .components
                        .first()
                        .and_then(|c| mixer::declared_operator(c, n).ok())
                        .map(|op| mixer::label(op, false))
                        .unwrap_or("a mixer this container does not name");
                    format!(
                        "layer {n} does not attend by softmax — its token mixer is {mixer}, \
                         so its projections are that operator's, not q/k/v/o; \
                         try `describe layer.{n}.mixer`"
                    )
                })?;
                let (label, op) = match *role {
                    "q" => ("ATTENTION QUERY PROJECTION", &attn.q),
                    "k" => ("ATTENTION KEY PROJECTION", &attn.k),
                    "v" => ("ATTENTION VALUE PROJECTION", &attn.v),
                    "o" => ("ATTENTION OUTPUT PROJECTION", &attn.o),
                    other => return Err(format!("unknown attention role `{other}` — q, k, v, o")),
                };
                Ok((label.to_string(), op.clone()))
            }
            _ => unreachable!(),
        }
    };
    Some(go())
}

/// Decode one tensor to f32, whatever the container stored it as. The
/// arithmetic for a packed encoding is the spec's — `tensor_scale ·
/// e4m3(group scale) · e2m1(code)` — so what the diff compares is what
/// a matmul against those bytes would effectively use.
pub(super) fn load_values(store: &OperandStore, operand: &OperandRef) -> Result<Vec<f32>, String> {
    if operand.dtype != DTYPE_NVFP4 {
        return store
            .load(operand)
            .map_err(|e| format!("load {}: {e}", operand.tensor));
    }
    let raw = store
        .load_raw(operand)
        .map_err(|e| format!("read {}: {e}", operand.tensor))?;
    let layout = PackLayout::derive(&operand.shape, &operand.tensor)
        .map_err(|e| format!("{}: {e}", operand.tensor))?;
    let (packed, scales, tensor_scale) = split(&raw.bytes, &layout, &operand.tensor)
        .map_err(|e| format!("{}: {e}", operand.tensor))?;
    let matrix = larql_models::quant::nvfp4::Nvfp4Matrix {
        packed: packed.to_vec(),
        scales: scales.to_vec(),
        tensor_scale,
    };
    let (rows, k) = (operand.shape[0], operand.shape[1]);
    let mut out = vec![0.0f32; rows * k];
    larql_models::quant::nvfp4::dequantize_into(&matrix, rows, k, &mut out)
        .map_err(|e| format!("decode {}: {e:?}", operand.tensor))?;
    Ok(out)
}

/// `vindex diff <a> <b> <address>` — one object decoded under two of
/// the container's own representations, compared value by value. The
/// error is derived, never asserted: both sides decode through the
/// same load path execution uses, and the numbers are whatever the
/// bytes disagree by.
pub fn diff_facts(
    root: &Path,
    encoding_a: &str,
    encoding_b: &str,
    address: &str,
    values: usize,
    tensor_filter: Option<&str>,
) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let mut semantic_tensor: Option<String> = None;
    let object = if let Some(resolved) = resolve_semantic(root, &inspection, address) {
        let (_, op) = resolved?;
        semantic_tensor = Some(op.tensor);
        op.object
    } else {
        find_object(&inspection, address)?.id.clone()
    };
    let tensor_filter = semantic_tensor.as_deref().or(tensor_filter);
    let (encoding_a, encoding_b) = (encoding_a.to_uppercase(), encoding_b.to_uppercase());
    let (store_a, tensors_a) = open_side(root, &inspection, &object, &encoding_a)?;
    let (store_b, tensors_b) = open_side(root, &inspection, &object, &encoding_b)?;
    let dtypes_b: std::collections::BTreeMap<&str, (&str, &Vec<usize>)> = tensors_b
        .iter()
        .map(|(n, d, s)| (n.as_str(), (d.as_str(), s)))
        .collect();

    // A semantic address names ONE tensor: the diff scopes to it, so
    // the result is that tensor's answer rather than the object's.
    let tensors_a: Vec<TensorEntry> = match &semantic_tensor {
        Some(t) => tensors_a.into_iter().filter(|(n, _, _)| n == t).collect(),
        None => tensors_a,
    };
    let mut rows: Vec<Value> = Vec::new();
    let mut head: Option<Value> = None;
    let mut worst: f64 = -1.0;
    let mut sum_sq = 0.0f64;
    let mut max_error = 0.0f64;
    let mut total_weights = 0u64;
    let mut changed_values = 0u64;
    for (name, dtype_a, shape) in &tensors_a {
        let Some((dtype_b, shape_b)) = dtypes_b.get(name.as_str()) else {
            rows.push(json!({ "tensor": name, "note": format!("only in {encoding_a}") }));
            continue;
        };
        if shape != *shape_b {
            rows.push(json!({
                "tensor": name,
                "note": format!("shape differs: {shape:?} vs {shape_b:?}"),
            }));
            continue;
        }
        let make_ref = |dtype: &str| OperandRef {
            object: object.clone(),
            tensor: name.clone(),
            dtype: dtype.to_string(),
            shape: shape.clone(),
        };
        let va = load_values(&store_a, &make_ref(dtype_a))
            .map_err(|e| format!("as {encoding_a}: {e}"))?;
        let vb = load_values(&store_b, &make_ref(dtype_b))
            .map_err(|e| format!("as {encoding_b}: {e}"))?;
        let n = va.len().min(vb.len());
        let mut t_sum_sq = 0.0f64;
        let mut t_max = 0.0f64;
        let mut t_changed = 0u64;
        for i in 0..n {
            let d = (va[i] as f64) - (vb[i] as f64);
            t_sum_sq += d * d;
            if d.abs() > t_max {
                t_max = d.abs();
            }
            if d != 0.0 {
                t_changed += 1;
            }
        }
        let rms = if n > 0 {
            (t_sum_sq / n as f64).sqrt()
        } else {
            0.0
        };
        sum_sq += t_sum_sq;
        total_weights += n as u64;
        changed_values += t_changed;
        if t_max > max_error {
            max_error = t_max;
        }
        rows.push(json!({
            "tensor": name,
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "weights": n as u64,
            "changed": t_changed,
            "rms_error": rms,
            "max_error": t_max,
        }));
        // A suffix only matches at a path boundary: `0.mlp.down` must not
        // match `30.mlp.down`. The first matching tensor wins.
        let wanted = match tensor_filter {
            Some(f) => head.is_none() && (name == f || name.ends_with(&format!(".{f}"))),
            None => rms > worst,
        };
        if wanted {
            worst = rms;
            head = Some(json!({
                "tensor": name,
                "rows": (0..n.min(values)).map(|i| json!({
                    "a": va[i],
                    "b": vb[i],
                    "error": vb[i] - va[i],
                })).collect::<Vec<Value>>(),
            }));
        }
    }
    if let Some(f) = tensor_filter {
        if head.is_none() {
            return Err(format!(
                "no tensor of `{object}` matches `{f}` — tensors: {}",
                tensors_a
                    .iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    Ok(json!({
        "container": root.display().to_string(),
        "object": object,
        "a": encoding_a,
        "b": encoding_b,
        "tensors": rows,
        "values": head,
        "total_weights": total_weights,
        "changed_values": changed_values,
        "rms_error": if total_weights > 0 { (sum_sq / total_weights as f64).sqrt() } else { 0.0 },
        "max_error": max_error,
        "identical": changed_values == 0,
    }))
}
