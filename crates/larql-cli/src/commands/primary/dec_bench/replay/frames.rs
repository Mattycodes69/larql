//! Sweep expansion and wire-frame construction through the production codecs.

use super::super::capture_format::CapturePool;
use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use larql_inference::ffn::moe_remote::{
    decode_multi_layer_response, encode_multi_layer_request, encode_multi_layer_request_q8k,
    MultiLayerTask, MultiLayerTaskQ8K, MULTI_LAYER_BATCH_CONTENT_TYPE,
    MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};
use larql_inference::ffn::remote::WireFormat;
use larql_inference::{encode_binary_request_as, encode_q8k_batch_request};

#[allow(unused_imports)]
use super::*;

/// One point of the sweep: (batch, wire, dispatch) on a resolved endpoint.
#[derive(Debug, Clone, Copy)]
pub struct SweepPoint {
    pub batch: usize,
    pub wire: WireSpec,
    pub dispatch: DispatchMode,
    pub endpoint: Endpoint,
}

pub fn parse_batch_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|b| {
            b.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid batch size {b:?}"))
                .and_then(|n| {
                    if n == 0 {
                        Err("batch size 0 is invalid".into())
                    } else {
                        Ok(n)
                    }
                })
        })
        .collect()
}

/// Full cross product, ordered batch-major so all wire/dispatch arms of one
/// batch size run adjacently (comparable thermal window). Errs on wire arms
/// the endpoint family cannot serve (f16/i8 and in/return pairs on
/// `experts` — arg-validation time, before any request is sent).
pub fn expand_sweep(
    batches: &[usize],
    wires: &[WireSpec],
    dispatches: &[DispatchMode],
    kind: EndpointKind,
) -> Result<Vec<SweepPoint>, String> {
    let endpoints: Vec<Endpoint> = wires
        .iter()
        .map(|&w| Endpoint::resolve(kind, w))
        .collect::<Result<_, _>>()?;
    let mut out = Vec::with_capacity(batches.len() * wires.len() * dispatches.len());
    for &batch in batches {
        for (&wire, &endpoint) in wires.iter().zip(endpoints.iter()) {
            for &dispatch in dispatches {
                out.push(SweepPoint {
                    batch,
                    wire,
                    dispatch,
                    endpoint,
                });
            }
        }
    }
    Ok(out)
}

// ── Frame construction ────────────────────────────────────────────────────────

/// Build a B-row `/v1/walk-ffn` request frame for one layer, with the
/// residual payload in `format` (the arm's inbound direction).
///
/// `rows` is `batch × hidden` contiguous f32s. `top_k` is hard-wired to 0:
/// the server's L2 FFN cache engages when `seq_len == 1 && top_k > 0`, which
/// would falsify repeated B=1 measurements with cache hits.
pub fn build_walk_ffn_frame_as(
    format: WireFormat,
    layer: usize,
    rows: &[f32],
    batch: usize,
) -> Vec<u8> {
    encode_binary_request_as(format, Some(layer), None, rows, batch, true, 0)
}

/// f32 twin of [`build_walk_ffn_frame_as`] — the historical (plain-arm)
/// frame shape, kept as the byte-pin reference for the tests below
/// (production goes through [`build_frame`] → `build_walk_ffn_frame_as`).
#[cfg(test)]
pub fn build_walk_ffn_frame(layer: usize, rows: &[f32], batch: usize) -> Vec<u8> {
    build_walk_ffn_frame_as(WireFormat::F32, layer, rows, batch)
}

/// Build a B-row `/v1/walk-ffn-q8k` request frame for one layer: B entries
/// sharing `layer`, each one row quantised through the production
/// `quantize_x_to_q8k` path.
pub fn build_q8k_frame(layer: usize, rows: &[f32], batch: usize, hidden: usize) -> Vec<u8> {
    let q8ks: Vec<_> = (0..batch)
        .map(|i| quantize_x_to_q8k(&rows[i * hidden..(i + 1) * hidden]))
        .collect();
    let entries: Vec<(usize, &_)> = q8ks.iter().map(|q| (layer, q)).collect();
    encode_q8k_batch_request(&entries)
}

/// Build a B-task `/v1/experts/multi-layer-batch` frame for one (step,
/// layer): task `r` is raw post-attention row `r` (the server applies
/// pre_experts_norm itself) plus that row's captured `(expert_id, weight)`
/// routing. Encoded with the production codec (parity discipline).
pub fn build_experts_ml_frame(
    layer: usize,
    rows: &[f32],
    routing: &[Vec<(u32, f32)>],
    batch: usize,
    hidden: usize,
) -> Vec<u8> {
    let tasks: Vec<MultiLayerTask> = (0..batch)
        .map(|r| MultiLayerTask {
            layer,
            residual: rows[r * hidden..(r + 1) * hidden].to_vec(),
            expert_ids: routing[r].iter().map(|&(id, _)| id).collect(),
            weights: routing[r].iter().map(|&(_, w)| w).collect(),
        })
        .collect();
    encode_multi_layer_request(&tasks)
}

/// Build a B-task `/v1/experts/multi-layer-batch-q8k` frame for one (step,
/// layer): task `r` is pre-experts-normed row `r` quantised through the
/// production `quantize_x_to_q8k` path, plus that row's captured routing.
pub fn build_experts_ml_q8k_frame(
    layer: usize,
    normed_rows: &[f32],
    routing: &[Vec<(u32, f32)>],
    batch: usize,
    hidden: usize,
) -> Vec<u8> {
    let tasks: Vec<MultiLayerTaskQ8K> = (0..batch)
        .map(|r| {
            let q8k = quantize_x_to_q8k(&normed_rows[r * hidden..(r + 1) * hidden]);
            MultiLayerTaskQ8K {
                layer,
                hidden,
                qs: q8k.qs,
                d: q8k.d,
                sums: q8k.sums,
                expert_ids: routing[r].iter().map(|&(id, _)| id).collect(),
                weights: routing[r].iter().map(|&(_, w)| w).collect(),
            }
        })
        .collect();
    encode_multi_layer_request_q8k(&tasks)
}

/// Decode + validate a multi-layer-batch response through the production
/// decoder. Returns whether any `h2` value is non-zero — the post-warmup
/// corruption guard (a routed replay whose responses are all-zero means
/// wrong expert ids / shard ownership, the audit §1a failure class).
pub fn check_experts_response(
    body: &[u8],
    layer: usize,
    batch: usize,
    hidden: usize,
) -> Result<bool, String> {
    let results = decode_multi_layer_response(body)
        .ok_or_else(|| format!("experts replay layer {layer}: response failed to decode"))?;
    if results.len() != batch
        || results
            .iter()
            .any(|r| r.layer != layer || r.h2.len() != hidden)
    {
        return Err(format!(
            "experts replay layer {layer}: expected {batch} results × hidden {hidden} for \
             layer {layer}, got {} results",
            results.len()
        ));
    }
    Ok(results.iter().any(|r| r.h2.iter().any(|&v| v != 0.0)))
}

/// Pool-sourced inputs for one (step, layer) request frame. `rows` comes
/// from the endpoint's source plane; `routing` is per-row captured pairs
/// (experts endpoints only).
pub struct FrameInputs {
    pub layer: usize,
    pub rows: Vec<f32>,
    pub routing: Option<Vec<Vec<(u32, f32)>>>,
}

/// Gather one (step, layer) frame's inputs from the pool:
///   * walk-ffn endpoints — dense-prenormed `residuals.bin` rows;
///   * experts f32 — raw post-attention rows (`raw.bin`) + routing;
///   * experts q8k — pre-experts-normed rows (`normed.bin`) + routing.
///
/// A row whose routing record is all-sentinel on a replayed layer becomes a
/// zero-expert task (valid wire, costs nothing server-side — the layer set
/// is pre-filtered to MoE layers via [`routed_layer_subset`]).
pub fn gather_frame_inputs(
    pool: &CapturePool,
    endpoint: Endpoint,
    layer: usize,
    step: usize,
    batch: usize,
) -> Result<FrameInputs, String> {
    let (rows, routing) = match endpoint {
        Endpoint::WalkFfn | Endpoint::WalkFfnQ8k => (pool.rows(batch, step, layer)?, None),
        Endpoint::ExpertsMultiLayer | Endpoint::ExpertsMultiLayerQ8k => {
            let rows = if endpoint == Endpoint::ExpertsMultiLayer {
                pool.raw_rows(batch, step, layer)?
            } else {
                pool.normed_rows(batch, step, layer)?
            };
            let mut routing = Vec::with_capacity(batch);
            for prompt in 0..batch {
                routing.push(
                    pool.routing(prompt, step, layer)?
                        .map(|pairs| pairs.to_vec())
                        .unwrap_or_default(),
                );
            }
            (rows, Some(routing))
        }
    };
    Ok(FrameInputs {
        layer,
        rows,
        routing,
    })
}

/// Build the endpoint's wire frame from gathered inputs, through the same
/// codec functions the production client uses. `wire` selects the inbound
/// residual encoding on the dense walk-ffn endpoint (plain arms stay f32;
/// pairs encode their `input` format); the other endpoints ignore it.
pub fn build_frame(
    endpoint: Endpoint,
    wire: WireSpec,
    inputs: &FrameInputs,
    batch: usize,
    hidden: usize,
) -> Result<Vec<u8>, String> {
    let routing = || {
        inputs
            .routing
            .as_deref()
            .ok_or_else(|| "experts frame build: inputs carry no routing".to_string())
    };
    Ok(match endpoint {
        Endpoint::WalkFfn => {
            build_walk_ffn_frame_as(wire.request_format(), inputs.layer, &inputs.rows, batch)
        }
        Endpoint::WalkFfnQ8k => build_q8k_frame(inputs.layer, &inputs.rows, batch, hidden),
        Endpoint::ExpertsMultiLayer => {
            build_experts_ml_frame(inputs.layer, &inputs.rows, routing()?, batch, hidden)
        }
        Endpoint::ExpertsMultiLayerQ8k => {
            build_experts_ml_q8k_frame(inputs.layer, &inputs.rows, routing()?, batch, hidden)
        }
    })
}

/// Short wire label for a response content-type (unknown CTs pass through
/// verbatim so the run record never hides what the server sent).
pub fn wire_label_for_content_type(ct: &str) -> String {
    match ct {
        larql_inference::BINARY_CT => "f32".into(),
        larql_inference::F16_CT => "f16".into(),
        larql_inference::I8_CT => "i8".into(),
        larql_inference::Q8K_BATCH_CT => "q8k".into(),
        MULTI_LAYER_BATCH_CONTENT_TYPE => "experts-ml".into(),
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE => "experts-ml-q8k".into(),
        other => other.to_string(),
    }
}
