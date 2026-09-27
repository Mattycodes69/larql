//! Per-point request statistics and their summary.

use super::super::super::bench::row::compute_percentiles;

#[allow(unused_imports)]
use super::*;

// ── Per-point statistics ──────────────────────────────────────────────────────

/// One request's measurements (one layer of one step of one repeat).
#[derive(Debug, Clone)]
pub struct RequestSample {
    pub layer: usize,
    /// Wall time of the send→receive window ONLY: frame encode happens
    /// before this timer starts and response decode after it stops, so
    /// `client_ms = transmit + serve` exactly (see [`derive_transmit_us`]).
    pub client_ms: f64,
    /// Legacy field: server compute ms from the f32 walk-ffn embedded
    /// response header; `None` on the other endpoints. Kept so the
    /// pre-two-scoreboard `server_ms_p50/p99` summary keys are unchanged;
    /// `serve_us` is the unified successor covering all four endpoints.
    pub server_ms: Option<f64>,
    /// Server-side serve time in µs, from the endpoint's
    /// [`ServeLatencySource`]: embedded header (walk-ffn) or the opt-in
    /// timing trailer (the other three). `None` when the server predates
    /// the timing extension.
    pub serve_us: Option<f64>,
    /// Client frame-build time in µs (request encode, outside `client_ms`).
    pub encode_us: f64,
    /// Client response-decode time in µs (outside `client_ms`).
    pub client_decode_us: f64,
    /// `client_ms×1000 − serve_us` when serve is known (see
    /// [`derive_transmit_us`]); clamped at 0, never negative.
    pub transmit_us: Option<f64>,
    /// True when the transmit derivation went negative and was clamped —
    /// summaries count these so a clock anomaly is visible, not silent.
    pub transmit_clamped: bool,
    pub bytes_sent: u64,
    pub bytes_recv: u64,
    /// Inbound wire format actually sent (echo of the request
    /// Content-Type label put on the wire — the client is authoritative
    /// for this direction).
    pub served_wire_in: String,
    /// Return wire format the server actually served (from the response
    /// content-type). Accept negotiation may fall back — e.g. an i8 arm
    /// served as f32 — and the run record must say so.
    pub served_wire_out: String,
    /// Whether any decoded response value was non-zero. Consumed by the
    /// post-warmup corruption guard on routed points (audit §1a class);
    /// computed at decode time, outside the timed window.
    pub any_nonzero: bool,
}

/// Derive one request's transmit time (µs) from its send→receive window
/// and the server-reported serve time.
///
/// `client_ms` covers send→receive only (encode runs before the timer,
/// decode after), so the two-scoreboard identity
/// `total = encode + transmit + serve + client_decode` collapses to
/// `transmit_us = client_ms×1000 − serve_us` — `encode_us` and
/// `client_decode_us` are measured outside the window and must NOT be
/// subtracted again. Unknown serve → `(None, false)`. A negative result
/// (clock granularity, or a serve clock that starts before the last
/// request byte lands) clamps to 0 with the flag set — never negative.
pub fn derive_transmit_us(client_ms: f64, serve_us: Option<f64>) -> (Option<f64>, bool) {
    match serve_us {
        None => (None, false),
        Some(s) => {
            let t = client_ms * 1000.0 - s;
            if t < 0.0 {
                (Some(0.0), true)
            } else {
                (Some(t), false)
            }
        }
    }
}

/// Accumulated raw measurements for one sweep point.
#[derive(Debug, Default)]
pub struct SweepPointStats {
    /// One entry per (repeat × step): full-pass step time in ms.
    pub step_ms: Vec<f64>,
    pub samples: Vec<RequestSample>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerSummary {
    pub layer: usize,
    pub client_ms_p50: f64,
    pub client_ms_p99: f64,
}

/// Summarised sweep point, ready for the run record and pulse file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DecPointSummary {
    pub batch: usize,
    /// Endpoint the point measured (`walk-ffn`, `walk-ffn-q8k`,
    /// `experts-ml`, `experts-ml-q8k`).
    pub endpoint: String,
    /// Numeric twin of `endpoint` (0/1/2/3) for numeric-only ingesters.
    pub endpoint_code: u32,
    /// Combined wire-arm label (`dec/wire_format`): plain arms keep their
    /// historical label; asymmetric pairs are `"in/out"` (e.g. `"f16/i8"`).
    pub wire_format: String,
    pub wire_format_code: u32,
    /// Requested INBOUND dtype (request Content-Type direction).
    pub wire_in: String,
    /// Requested RETURN dtype (Accept direction).
    pub wire_out: String,
    /// Inbound wire format(s) actually sent this point (sorted, deduped —
    /// echo of the request Content-Type labels).
    pub served_wire_in: Vec<String>,
    /// Return wire format(s) the server actually served this point
    /// (sorted, deduped). Differs from `wire_out` when Accept negotiation
    /// fell back — the arm's bandwidth number then belongs to the served
    /// format.
    pub served_wire_out: Vec<String>,
    pub dispatch_mode: String,
    pub dispatch_mode_code: u32,
    pub steps: usize,
    pub step_ms_mean: f64,
    pub step_ms_p50: f64,
    pub step_ms_p99: f64,
    /// Rows served per second: `batch × 1000 / step_ms_mean`.
    pub tok_s: f64,
    /// (bytes sent + received) ÷ (steps × batch) — wire cost per token row.
    pub payload_bytes_tok: f64,
    /// bytes sent ÷ (steps × batch) — the inbound (request) direction's
    /// share of `payload_bytes_tok`.
    pub payload_bytes_tok_in: f64,
    /// bytes received ÷ (steps × batch) — the return (response)
    /// direction's share of `payload_bytes_tok`.
    pub payload_bytes_tok_out: f64,
    /// Dense-endpoint denominator: FFN weight bytes the endpoint touches
    /// per token over the replayed layers. Constant across a run's points
    /// (kept per-point so every summary is self-contained); `None` on
    /// routed endpoints and when `/v1/stats` has no `ffn_weights`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bytes_tok: Option<f64>,
    /// Routed-endpoint denominator, per-row naive: Σ over (step, layer,
    /// row) |E_row| × per_expert_bytes ÷ (steps × batch). PRIMARY for the
    /// movement ratio — the server streams each row's experts independently
    /// (audit §2: no cross-row weight sharing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bytes_tok_naive: Option<f64>,
    /// Routed-endpoint batch-union bound: Σ over (step, layer) |⋃_rows E|
    /// × per_expert_bytes ÷ (steps × batch) — the DEC-3 metrology bound.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bytes_tok_union: Option<f64>,
    /// payload_bytes_tok ÷ denominator (dense bytes, or NAIVE for routed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub movement_ratio: Option<f64>,
    /// union ÷ naive — 1.0 = fully disjoint expert sets across rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experts_union_frac: Option<f64>,
    pub server_ms_p50: Option<f64>,
    pub server_ms_p99: Option<f64>,
    // ── Two-scoreboard timing decomposition (dec-funnel §3 DEC-1A) ───────
    /// Driver-side queueing delay before send. Recorded as a schema field
    /// now; always 0.0 at the replay driver's concurrency of 1 (each
    /// worker has exactly one request in flight, so nothing ever waits in
    /// a client-side queue). Becomes meaningful on DEC-2's client-count
    /// axis — the field exists so the schema is stable before that axis.
    pub queue_ms: f64,
    /// p50 of per-request frame-build time (µs, outside `client_ms`).
    pub encode_us_p50: f64,
    /// p50 of per-request response-decode time (µs, outside `client_ms`).
    pub client_decode_us_p50: f64,
    /// p50/p99 of server serve time (µs) from the endpoint's
    /// [`ServeLatencySource`]; absent when the server never reported one
    /// (pre-extension server on a trailer endpoint).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serve_us_p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serve_us_p99: Option<f64>,
    /// p50/p99 of derived transmit time (µs) — `client_ms×1000 − serve_us`
    /// per request; absent whenever serve is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transmit_us_p50: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transmit_us_p99: Option<f64>,
    /// Number of samples whose transmit derivation clamped at 0 (serve ≥
    /// measured window — clock anomaly, counted rather than hidden).
    pub transmit_us_clamped: usize,
    pub per_layer: Vec<LayerSummary>,
}

/// Summarise one sweep point. `dense_weight_bytes_tok` feeds dense
/// endpoints' denominator; `routed` feeds routed endpoints' — the caller
/// passes whichever matches `point.endpoint.denominator()` (the other is
/// ignored).
pub fn summarize(
    point: &SweepPoint,
    stats: &SweepPointStats,
    dense_weight_bytes_tok: Option<f64>,
    routed: Option<&RoutedDenominators>,
) -> DecPointSummary {
    let (mean, p50, p99) = compute_percentiles(&stats.step_ms);
    let tok_s = if mean > 0.0 {
        point.batch as f64 * 1000.0 / mean
    } else {
        0.0
    };
    let sent_bytes: u64 = stats.samples.iter().map(|s| s.bytes_sent).sum();
    let recv_bytes: u64 = stats.samples.iter().map(|s| s.bytes_recv).sum();
    let token_rows = (stats.step_ms.len() * point.batch) as f64;
    let per_tok = |bytes: u64| {
        if token_rows > 0.0 {
            bytes as f64 / token_rows
        } else {
            0.0
        }
    };
    let payload_bytes_tok_in = per_tok(sent_bytes);
    let payload_bytes_tok_out = per_tok(recv_bytes);
    let payload_bytes_tok = per_tok(sent_bytes + recv_bytes);

    let server_samples: Vec<f64> = stats.samples.iter().filter_map(|s| s.server_ms).collect();
    let (server_ms_p50, server_ms_p99) = if server_samples.is_empty() {
        (None, None)
    } else {
        let (_, p50, p99) = compute_percentiles(&server_samples);
        (Some(p50), Some(p99))
    };

    // Two-scoreboard decomposition: percentile pair for an optional per-
    // sample series (present only when at least one sample carried it).
    let optional_percentiles = |vals: Vec<f64>| -> (Option<f64>, Option<f64>) {
        if vals.is_empty() {
            (None, None)
        } else {
            let (_, p50, p99) = compute_percentiles(&vals);
            (Some(p50), Some(p99))
        }
    };
    let (serve_us_p50, serve_us_p99) =
        optional_percentiles(stats.samples.iter().filter_map(|s| s.serve_us).collect());
    let (transmit_us_p50, transmit_us_p99) =
        optional_percentiles(stats.samples.iter().filter_map(|s| s.transmit_us).collect());
    let transmit_us_clamped = stats.samples.iter().filter(|s| s.transmit_clamped).count();
    let encode_us_p50 = {
        let vals: Vec<f64> = stats.samples.iter().map(|s| s.encode_us).collect();
        let (_, p50, _) = compute_percentiles(&vals);
        p50
    };
    let client_decode_us_p50 = {
        let vals: Vec<f64> = stats.samples.iter().map(|s| s.client_decode_us).collect();
        let (_, p50, _) = compute_percentiles(&vals);
        p50
    };

    let sorted_dedup = |mut v: Vec<String>| {
        v.sort();
        v.dedup();
        v
    };
    let served_wire_in = sorted_dedup(
        stats
            .samples
            .iter()
            .map(|s| s.served_wire_in.clone())
            .collect(),
    );
    let served_wire_out = sorted_dedup(
        stats
            .samples
            .iter()
            .map(|s| s.served_wire_out.clone())
            .collect(),
    );

    let mut layers: Vec<usize> = stats.samples.iter().map(|s| s.layer).collect();
    layers.sort_unstable();
    layers.dedup();
    let per_layer = layers
        .into_iter()
        .map(|layer| {
            let vals: Vec<f64> = stats
                .samples
                .iter()
                .filter(|s| s.layer == layer)
                .map(|s| s.client_ms)
                .collect();
            let (_, p50, p99) = compute_percentiles(&vals);
            LayerSummary {
                layer,
                client_ms_p50: p50,
                client_ms_p99: p99,
            }
        })
        .collect();

    let (weight_bytes_tok, weight_bytes_tok_naive, weight_bytes_tok_union, experts_union_frac) =
        match point.endpoint.denominator() {
            DenominatorSource::Dense => (dense_weight_bytes_tok, None, None, None),
            DenominatorSource::RoutedExperts => {
                let naive = routed.map(|d| d.weight_bytes_tok_naive);
                let union = routed.map(|d| d.weight_bytes_tok_union);
                let frac = routed.and_then(|d| {
                    (d.weight_bytes_tok_naive > 0.0)
                        .then(|| d.weight_bytes_tok_union / d.weight_bytes_tok_naive)
                });
                (None, naive, union, frac)
            }
        };
    // Primary denominator: dense bytes for dense endpoints, NAIVE per-row
    // bytes for routed (the server streams per-row — audit §2).
    let ratio = weight_bytes_tok
        .or(weight_bytes_tok_naive)
        .and_then(|w| movement_ratio(payload_bytes_tok, w));

    DecPointSummary {
        batch: point.batch,
        endpoint: point.endpoint.label().into(),
        endpoint_code: point.endpoint.code(),
        wire_format: point.wire.label(),
        wire_format_code: point.wire.code(),
        wire_in: point.wire.in_label().into(),
        wire_out: point.wire.out_label().into(),
        served_wire_in,
        served_wire_out,
        dispatch_mode: point.dispatch.label().into(),
        dispatch_mode_code: point.dispatch.code(),
        steps: stats.step_ms.len(),
        step_ms_mean: mean,
        step_ms_p50: p50,
        step_ms_p99: p99,
        tok_s,
        payload_bytes_tok,
        payload_bytes_tok_in,
        payload_bytes_tok_out,
        weight_bytes_tok,
        weight_bytes_tok_naive,
        weight_bytes_tok_union,
        movement_ratio: ratio,
        experts_union_frac,
        server_ms_p50,
        server_ms_p99,
        // Two-scoreboard fields. queue_ms is structurally 0 at the replay
        // driver's concurrency of 1 (see the field doc) — DEC-2's axis.
        queue_ms: 0.0,
        encode_us_p50,
        client_decode_us_p50,
        serve_us_p50,
        serve_us_p99,
        transmit_us_p50,
        transmit_us_p99,
        transmit_us_clamped,
        per_layer,
    }
}
