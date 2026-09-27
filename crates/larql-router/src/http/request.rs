//! Request parsing: binary-vs-JSON detection and the ADR-0018 dispatch shape.

use crate::shards::peek_binary;
use serde_json::Value;

#[allow(unused_imports)]
use super::*;

/// Returns `true` when `ct` is the FFN binary protocol marker. Pure;
/// extracted so the binary-vs-JSON branch can be unit-tested without
/// building a full HTTP request.
pub fn is_binary_content_type(ct: &str) -> bool {
    ct.starts_with(BINARY_CT)
}

/// ADR-0018 — request shape after parsing. JSON bodies can be **dense**
/// (just `layer` / `layers`) or **MoE** (`experts` / `layer_experts`).
/// Binary bodies are always dense.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestSpec {
    /// Plain layer list — every owning shard runs the full layer's FFN.
    Dense(Vec<usize>),
    /// Per-layer expert list — the gate scorer upstream emits sparse
    /// `(layer, expert_ids)` pairs and the router dispatches to each
    /// expert-shard.
    Moe(Vec<(usize, Vec<u32>)>),
}

impl RequestSpec {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Dense(layers) => layers.is_empty(),
            Self::Moe(pairs) => pairs.is_empty() || pairs.iter().all(|(_, exs)| exs.is_empty()),
        }
    }
}

/// Pull layer IDs and optional `model_id` out of a request body. For
/// binary bodies the header is peeked; for JSON bodies we look for a
/// `layers` array or a `layer` scalar plus an optional `model_id` field.
///
/// `Err(msg)` is returned to the caller as a 400 reply — the message is
/// already user-facing.
pub fn extract_layers_and_model_id(
    body: &[u8],
    is_binary: bool,
) -> Result<(Vec<usize>, Option<String>), String> {
    match extract_request_spec_and_model_id(body, is_binary)? {
        (RequestSpec::Dense(layers), model_id) => Ok((layers, model_id)),
        (RequestSpec::Moe(_), _) => Err(
            "MoE request (`experts`/`layer_experts`) is not accepted on the dense \
             code path; use `handle_walk_ffn`"
                .to_string(),
        ),
    }
}

/// ADR-0018 — dispatch-shape parser. Handles both dense and MoE JSON
/// shapes; binary bodies are always dense.
///
/// JSON MoE shapes (in priority order — first match wins):
///   - `{"layer_experts": [{"layer": L, "experts": [...]}, ...]}`
///   - `{"layer": L, "experts": [...]}`
///
/// JSON dense shapes (fallback):
///   - `{"layers": [...]}`
///   - `{"layer": L}`
///
/// `model_id` is optional in every shape.
pub fn extract_request_spec_and_model_id(
    body: &[u8],
    is_binary: bool,
) -> Result<(RequestSpec, Option<String>), String> {
    if is_binary {
        // ADR-0018: binary protocol stays dense-only. A future v2 wire
        // format with expert IDs is tracked under ADR-0009.
        let layers =
            peek_binary(body).ok_or_else(|| "binary: truncated or malformed header".to_string())?;
        return Ok((RequestSpec::Dense(layers), None));
    }

    let peek: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
    let model_id = peek
        .get("model_id")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // MoE — multi-layer form takes priority over the single-layer form
    // because `layer_experts` is unambiguous.
    if let Some(arr) = peek.get("layer_experts").and_then(|v| v.as_array()) {
        let mut pairs: Vec<(usize, Vec<u32>)> = Vec::with_capacity(arr.len());
        for item in arr {
            let layer = item
                .get("layer")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "layer_experts: each entry needs a 'layer' field".to_string())?
                as usize;
            let experts_arr = item
                .get("experts")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "layer_experts: each entry needs an 'experts' array".to_string())?;
            let experts: Vec<u32> = experts_arr
                .iter()
                .filter_map(|e| e.as_u64().map(|n| n as u32))
                .collect();
            if experts.is_empty() {
                return Err(format!(
                    "layer_experts: empty 'experts' array for layer {layer}"
                ));
            }
            pairs.push((layer, experts));
        }
        return Ok((RequestSpec::Moe(pairs), model_id));
    }

    // MoE — single-layer form.
    if let Some(arr) = peek.get("experts").and_then(|v| v.as_array()) {
        let layer = peek
            .get("layer")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "moe: 'experts' requires a 'layer' scalar".to_string())?
            as usize;
        let experts: Vec<u32> = arr
            .iter()
            .filter_map(|e| e.as_u64().map(|n| n as u32))
            .collect();
        if experts.is_empty() {
            return Err("moe: 'experts' array is empty".into());
        }
        return Ok((RequestSpec::Moe(vec![(layer, experts)]), model_id));
    }

    // Dense fallback.
    let layers: Vec<usize> = if let Some(arr) = peek.get("layers").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    } else if let Some(n) = peek.get("layer").and_then(|v| v.as_u64()) {
        vec![n as usize]
    } else {
        return Err("must provide 'layer' or 'layers'".into());
    };
    Ok((RequestSpec::Dense(layers), model_id))
}
