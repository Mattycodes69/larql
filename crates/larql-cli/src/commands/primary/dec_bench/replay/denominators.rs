//! Movement ratio and the weight-byte denominators behind it.

use super::super::capture_format::CapturePool;

#[allow(unused_imports)]
use super::*;

// ── Movement ratio ────────────────────────────────────────────────────────────

/// `dec/movement_ratio` = wire bytes crossing the attention↔weights boundary
/// per token ÷ weight bytes the measured endpoint touches per token
/// (docs/dec-funnel.md §1). Offload ≈ 1.0 by construction; this architecture
/// targets 1e-3–1e-4.
pub fn movement_ratio(payload_bytes_tok: f64, weight_bytes_tok: f64) -> Option<f64> {
    if weight_bytes_tok > 0.0 && payload_bytes_tok >= 0.0 {
        Some(payload_bytes_tok / weight_bytes_tok)
    } else {
        None
    }
}

/// Sum the dense FFN weight bytes per token over `layers` from a `/v1/stats`
/// response (`ffn_weights.per_layer_dense_bytes`). This is the exact byte
/// count `/v1/walk-ffn` touches per token on those layers — the movement-
/// ratio denominator for the measured endpoint. Layers with `null` entries
/// (no interleaved k-quant data) contribute nothing and are reported back so
/// the caller can flag partial coverage.
///
/// Returns `(dense_bytes, missing_layer_count)`, or `None` when the stats
/// response has no `ffn_weights` block at all.
pub fn weight_bytes_per_token(stats: &serde_json::Value, layers: &[usize]) -> Option<(f64, usize)> {
    let per_layer = stats
        .get("ffn_weights")?
        .get("per_layer_dense_bytes")?
        .as_array()?;
    let mut total = 0.0f64;
    let mut missing = 0usize;
    for &l in layers {
        match per_layer.get(l).and_then(|v| v.as_u64()) {
            Some(b) => total += b as f64,
            None => missing += 1,
        }
    }
    Some((total, missing))
}

// ── Routed denominators ──────────────────────────────────────────────────────

/// Routed-endpoint movement-ratio denominators for one sweep point, from
/// the captured routing across its (steps × batch rows).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedDenominators {
    /// Σ over (step, layer, row) |E_row| × per_expert_bytes ÷ (steps × batch).
    pub weight_bytes_tok_naive: f64,
    /// Σ over (step, layer) |⋃_rows E| × per_expert_bytes ÷ (steps × batch).
    pub weight_bytes_tok_union: f64,
}

/// Compute both routed denominators. `expert_sets[cell][row]` is one
/// (step, layer) cell's per-row expert-id lists (sentinel-stripped —
/// records may be shorter than top_k). NAIVE is the primary movement-ratio
/// denominator: the server streams each row's experts independently (audit
/// §2 verified no cross-row weight sharing); UNION is reported alongside as
/// the DEC-3 metrology bound.
pub fn routed_weight_bytes_per_token(
    expert_sets: &[Vec<Vec<u32>>],
    per_expert_bytes: f64,
    steps: usize,
    batch: usize,
) -> RoutedDenominators {
    let tokens = (steps * batch) as f64;
    if tokens == 0.0 {
        return RoutedDenominators {
            weight_bytes_tok_naive: 0.0,
            weight_bytes_tok_union: 0.0,
        };
    }
    let mut naive_experts = 0usize;
    let mut union_experts = 0usize;
    for cell in expert_sets {
        let mut union: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
        for row in cell {
            naive_experts += row.len();
            union.extend(row.iter().copied());
        }
        union_experts += union.len();
    }
    RoutedDenominators {
        weight_bytes_tok_naive: naive_experts as f64 * per_expert_bytes / tokens,
        weight_bytes_tok_union: union_experts as f64 * per_expert_bytes / tokens,
    }
}

/// Gather one sweep point's captured expert sets from the pool and compute
/// its [`RoutedDenominators`]. `layers` must already be the MoE subset
/// (non-MoE cells contribute empty rows, which would silently deflate the
/// per-token average — [`routed_layer_subset`] excludes them up front).
pub fn routed_denominators_for_point(
    pool: &CapturePool,
    layers: &[usize],
    steps: usize,
    batch: usize,
    per_expert_bytes: f64,
) -> Result<RoutedDenominators, String> {
    let mut cells = Vec::with_capacity(steps * layers.len());
    for step in 0..steps {
        for &layer in layers {
            let mut rows = Vec::with_capacity(batch);
            for prompt in 0..batch {
                rows.push(
                    pool.routing(prompt, step, layer)?
                        .map(|pairs| pairs.iter().map(|&(id, _)| id).collect())
                        .unwrap_or_default(),
                );
            }
            cells.push(rows);
        }
    }
    Ok(routed_weight_bytes_per_token(
        &cells,
        per_expert_bytes,
        steps,
        batch,
    ))
}

/// Filter `layers` down to those that carried MoE routing in the pool (any
/// (prompt, step) cell non-empty). Non-MoE layers are excluded from the
/// routed sweep's layer set and denominator. Errs on walk-ffn-only pools.
pub fn routed_layer_subset(pool: &CapturePool, layers: &[usize]) -> Result<Vec<usize>, String> {
    let m = &pool.manifest;
    let mut out = Vec::with_capacity(layers.len());
    'layers: for &layer in layers {
        for prompt in 0..m.prompts.len() {
            for step in 0..m.steps {
                if pool.routing(prompt, step, layer)?.is_some() {
                    out.push(layer);
                    continue 'layers;
                }
            }
        }
    }
    Ok(out)
}

/// Routed-denominator facts from `/v1/stats` (`ffn_weights.moe`). A null
/// `per_expert_bytes` is a loud error — without it the routed movement
/// ratio has no denominator (audit §1c: pre-fix servers reported null on
/// sharded topologies).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeWeightStats {
    pub per_expert_bytes: f64,
    pub top_k: Option<u64>,
    pub num_experts: Option<u64>,
}

pub fn moe_weight_stats(stats: &serde_json::Value) -> Result<MoeWeightStats, String> {
    let moe = stats
        .get("ffn_weights")
        .and_then(|f| f.get("moe"))
        .filter(|m| !m.is_null())
        .ok_or(
            "server /v1/stats has no ffn_weights.moe block — routed endpoints need MoE \
             weight accounting (is the server serving a dense model?)",
        )?;
    let per_expert_bytes = moe.get("per_expert_bytes").and_then(|v| v.as_u64()).ok_or(
        "server /v1/stats ffn_weights.moe.per_expert_bytes is null — the routed \
             movement-ratio denominator is unavailable (weights not loaded, or a pre-fix \
             server probing expert entry 0 on a shard that doesn't own it)",
    )? as f64;
    Ok(MoeWeightStats {
        per_expert_bytes,
        top_k: moe.get("top_k").and_then(|v| v.as_u64()),
        num_experts: moe.get("num_experts").and_then(|v| v.as_u64()),
    })
}

/// Parse an inclusive `"A-B"` layer range into a layer list, defaulting to
/// `0..num_layers` when absent.
pub fn parse_layer_range(spec: Option<&str>, num_layers: usize) -> Result<Vec<usize>, String> {
    match spec {
        None => Ok((0..num_layers).collect()),
        Some(s) => {
            let (a, b) = s
                .split_once('-')
                .ok_or_else(|| format!("invalid layer range {s:?} (expected A-B)"))?;
            let a: usize = a
                .trim()
                .parse()
                .map_err(|_| format!("invalid layer range start {a:?}"))?;
            let b: usize = b
                .trim()
                .parse()
                .map_err(|_| format!("invalid layer range end {b:?}"))?;
            if a > b || b >= num_layers {
                return Err(format!(
                    "layer range {a}-{b} out of bounds (model has {num_layers} layers)"
                ));
            }
            Ok((a..=b).collect())
        }
    }
}
