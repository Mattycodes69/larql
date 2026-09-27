//! Precision facts: per-operand formats and the layer × class precision matrix.

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::{LayerFfn, OperandRef};
use mixer::MixerOperands;
use serde_json::{json, Value};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// `vindex precision` — bits per weight, derived, never asserted: the
/// stored payload bytes over the tensor-table element counts, per
/// representation and effective across the container. The compiled
/// precision map is passed through verbatim when the index carries one.
pub fn precision_facts(root: &Path) -> Facts {
    let index = read_index(root)?;
    let mut total_bits = 0u128;
    let mut total_weights = 0u128;
    let mut per_entry: Vec<Value> = Vec::new();
    for (id, e) in &index.representations {
        let (header, _) = read_segment_header(&root.join(&e.segment))
            .map_err(|err| format!("segment {}: {err}", e.segment))?;
        let weights: u128 = header
            .tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>() as u128)
            .sum();
        let bits = e.payload_bytes as u128 * 8;
        if weights > 0 {
            total_bits += bits;
            total_weights += weights;
        }
        per_entry.push(json!({
            "id": id,
            "encoding": e.encoding,
            "weights": weights as u64,
            "payload_bytes": e.payload_bytes,
            "bits_per_weight": if weights > 0 { (bits as f64) / (weights as f64) } else { 0.0 },
        }));
    }
    Ok(json!({
        "container": root.display().to_string(),
        "entries": per_entry,
        "total_weight_slots": total_weights as u64,
        "stored_bits_per_weight_slot": if total_weights > 0 { (total_bits as f64) / (total_weights as f64) } else { 0.0 },
        "precision_map": index.precision_map,
    }))
}

/// `vindex layers` — every layer's token-mixer programme, from the
/// plan. Three seconds of terminal that says why a hybrid model is an
/// interesting quantization subject.
pub fn layers_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let (component, plan) = primary_plan(root, &inspection)?;
    let mut rows: Vec<Value> = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let ffn = match &l.ffn {
            Some(LayerFfn::Dense(_)) => "dense",
            Some(LayerFfn::Routed(_)) => "routed",
            Some(LayerFfn::Hybrid(_)) => "hybrid",
            // A mixer-only (Mamba2) layer: the mixer is the whole
            // block and no FFN exists to report.
            None => "absent",
        };
        let operator = mixer::declared_operator(component, l.layer)?;
        rows.push(json!({
            "layer": l.layer,
            "mixer": mixer::label(operator, mixer::has_output_gate(l)),
            "ffn": ffn,
        }));
    }
    Ok(json!({
        "container": root.display().to_string(),
        "component": component.id,
        "layers": rows,
    }))
}

/// `vindex precision --matrix` — bits per weight, per layer and per
/// semantic role, read from the representation each object would
/// execute (the compiled pack when one exists, the canonical bytes
/// otherwise). Programme-aware: layers are grouped by their token
/// mixer, each group carrying the columns its programme actually has,
/// and the model's other surfaces (embedding, head, towers) are
/// listed with their own derived bits. No architecture is forced into
/// another's schema.
pub fn precision_matrix_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let (component, plan) = primary_plan(root, &inspection)?;

    // Per object: the representation execution would bind — a compiled
    // pack over the canonical bytes — and its tensor table of
    // (byte length, weight count) per tensor name.
    type TensorSizes = std::collections::BTreeMap<String, (u64, u128)>;
    let mut tables: std::collections::BTreeMap<String, (String, TensorSizes)> =
        std::collections::BTreeMap::new();
    for object in &inspection.graph.objects {
        let canonical = object.representations.first().map(|r| r.encoding.clone());
        let mut chosen: Option<(
            &String,
            &larql_vindex::format::vindex3::index::RepresentationEntry,
        )> = None;
        for (id, e) in &inspection.index.representations {
            if e.object != object.id {
                continue;
            }
            let is_pack = Some(&e.encoding) != canonical.as_ref();
            match &chosen {
                None => chosen = Some((id, e)),
                Some((_, cur)) => {
                    let cur_is_pack = Some(&cur.encoding) != canonical.as_ref();
                    if is_pack && !cur_is_pack {
                        chosen = Some((id, e));
                    }
                }
            }
        }
        if let Some((_, entry)) = chosen {
            let (header, _) = read_segment_header(&root.join(&entry.segment))
                .map_err(|e| format!("segment {}: {e}", entry.segment))?;
            let table = header
                .tensors
                .into_iter()
                .map(|t| {
                    let weights: u128 = t.shape.iter().product::<usize>() as u128;
                    (t.name, (t.len, weights))
                })
                .collect();
            tables.insert(object.id.clone(), (entry.encoding.clone(), table));
        }
    }
    let bits_of = |op: &OperandRef| -> Option<f64> {
        let (_, table) = tables.get(&op.object)?;
        let (len, weights) = table.get(&op.tensor)?;
        if *weights == 0 {
            return None;
        }
        Some((*len as u128 * 8) as f64 / *weights as f64)
    };

    // Group layers by programme; each group's columns are what its
    // programme actually computes with.
    //
    // The empty answer means "no column", and it is deliberate for
    // every per-head or per-channel vector — log decay, timestep
    // bias, norm weights, the Mamba2 skip. A bits-per-weight matrix
    // over a `[Hv]` operand says nothing about a representation and
    // would push a programme's row past readable width; the tensors
    // are still reachable through `describe layer.N.mixer`.
    let short = |role: &str| -> &'static str {
        match role {
            // softmax
            "query" => "q",
            "key" => "k",
            "value" => "v",
            "output" => "o",
            "output gate" => "zgate",
            // gated deltanet
            "fused recurrent q|k|v" => "qkv",
            "decay projection" => "decay",
            "write-strength projection" => "write",
            "output-gate projection" => "zgate",
            "causal conv over q|k|v" => "conv",
            // kda — split where gated deltanet fuses
            "query projection" => "q",
            "key projection" => "k",
            "value projection" => "v",
            "causal conv over q" => "qconv",
            "causal conv over k" => "kconv",
            "causal conv over v" => "vconv",
            "decay gate down" => "fa",
            "decay gate up" => "fb",
            "output gate down" => "ga",
            "output gate up" => "gb",
            // mla — the compressed-kv set
            "compressed kv projection" => "kv_a",
            "kv decompression" => "kv_b",
            // mamba2
            "fused in-projection z|x|B|C|dt" => "in",
            "causal conv over x|B|C" => "conv",
            // shared
            "output projection" => "out",
            _ => "",
        }
    };
    let mut programmes: Vec<(String, Vec<String>, Vec<Value>)> = Vec::new();
    for layer in &plan.layers {
        let operator = mixer::declared_operator(component, layer.layer)?;
        let label = mixer::label(operator, mixer::has_output_gate(layer)).to_string();
        let mut cells = serde_json::Map::new();
        let mut roles: Vec<String> = vec!["gate".into(), "up".into(), "down".into()];
        if let Some(ffn) = layer.ffn.as_ref().and_then(|f| f.dense()) {
            if let Some(g) = &ffn.gate {
                if let Some(b) = bits_of(g) {
                    cells.insert("gate".into(), json!(b));
                }
            }
            if let Some(b) = bits_of(&ffn.up) {
                cells.insert("up".into(), json!(b));
            }
            if let Some(b) = bits_of(&ffn.down) {
                cells.insert("down".into(), json!(b));
            }
        }
        if let MixerOperands::Named(ops) = mixer::operands(operator, layer) {
            for (role, op) in ops {
                let col = short(role);
                if col.is_empty() {
                    continue;
                }
                roles.push(col.to_string());
                if let Some(b) = bits_of(&op) {
                    cells.insert(col.to_string(), json!(b));
                }
            }
        }
        let row = json!({ "layer": layer.layer, "bits": Value::Object(cells) });
        match programmes.iter_mut().find(|(l, _, _)| *l == label) {
            Some((_, _, rows)) => rows.push(row),
            None => programmes.push((label, roles, vec![row])),
        }
    }
    // The model's other surfaces: every object outside the layer plan,
    // at the bits its bound representation derives to.
    let planned: std::collections::BTreeSet<String> = plan
        .layers
        .iter()
        .flat_map(|l| {
            let mut ops: Vec<String> = match mixer::declared_operator(component, l.layer)
                .map(|op| mixer::operands(op, l))
            {
                Ok(MixerOperands::Named(named)) => {
                    named.into_iter().map(|(_, o)| o.object).collect()
                }
                _ => Vec::new(),
            };
            if let Some(ffn) = l.ffn.as_ref().and_then(|f| f.dense()) {
                if let Some(g) = &ffn.gate {
                    ops.push(g.object.clone());
                }
                ops.push(ffn.up.object.clone());
                ops.push(ffn.down.object.clone());
            }
            ops
        })
        .collect();
    let surfaces: Vec<Value> = tables
        .iter()
        .filter(|(object, _)| !planned.contains(*object))
        .map(|(object, (encoding, table))| {
            let (bits, weights) = table.values().fold((0u128, 0u128), |(b, w), (len, n)| {
                (b + *len as u128 * 8, w + n)
            });
            json!({
                "object": object,
                "representation": encoding,
                "bits_per_weight": if weights > 0 { bits as f64 / weights as f64 } else { 0.0 },
            })
        })
        .collect();
    Ok(json!({
        "container": root.display().to_string(),
        "component": component.id,
        "programmes": programmes.into_iter().map(|(label, roles, rows)| json!({
            "label": label,
            "layers": rows.len(),
            "roles": roles,
            "rows": rows,
        })).collect::<Vec<Value>>(),
        "surfaces": surfaces,
    }))
}
