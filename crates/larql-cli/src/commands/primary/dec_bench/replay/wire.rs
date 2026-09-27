//! Sweep plan: wire arms, endpoints and dispatch modes.

use larql_inference::ffn::moe_remote::multi_layer_wire::{
    MULTI_LAYER_BATCH_PATH, MULTI_LAYER_BATCH_Q8K_PATH,
};
use larql_inference::ffn::moe_remote::{
    MULTI_LAYER_BATCH_CONTENT_TYPE, MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};
use larql_inference::ffn::remote::{WireFormat, WALK_FFN_PATH, WALK_FFN_Q8K_PATH};

#[allow(unused_imports)]
use super::*;

// ── Sweep plan ────────────────────────────────────────────────────────────────

/// One wire-format arm of the sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireArm {
    F32,
    F16,
    I8,
    Q8k,
}

impl WireArm {
    pub fn parse(s: &str) -> Result<WireArm, String> {
        match s.trim() {
            "f32" => Ok(WireArm::F32),
            "f16" => Ok(WireArm::F16),
            "i8" => Ok(WireArm::I8),
            "q8k" => Ok(WireArm::Q8k),
            other => Err(format!(
                "unknown wire format {other:?} (expected f32, f16, i8, q8k, or an \
                 in/return pair like f16/i8)"
            )),
        }
    }

    /// Arm-only list parsing — superseded by [`WireSpec::parse_list`] (which
    /// adds pair tokens) as the `--wire` entry point; kept for its tests.
    #[cfg(test)]
    pub fn parse_list(s: &str) -> Result<Vec<WireArm>, String> {
        s.split(',').map(WireArm::parse).collect()
    }

    pub fn label(self) -> &'static str {
        match self {
            WireArm::F32 => "f32",
            WireArm::F16 => "f16",
            WireArm::I8 => "i8",
            WireArm::Q8k => "q8k",
        }
    }

    /// Numeric twin of [`Self::label`] for numeric-only metric ingesters.
    pub fn code(self) -> u32 {
        match self {
            WireArm::F32 => 0,
            WireArm::F16 => 1,
            WireArm::I8 => 2,
            WireArm::Q8k => 3,
        }
    }

    /// `Accept` header for the strict arm (no fallback formats offered).
    /// `None` for Q8K, which is its own endpoint rather than a negotiated CT.
    pub fn accept(self) -> Option<&'static str> {
        match self {
            WireArm::F32 => Some(larql_inference::BINARY_CT),
            WireArm::F16 => Some(larql_inference::F16_CT),
            WireArm::I8 => Some(larql_inference::I8_CT),
            WireArm::Q8k => None,
        }
    }
}

/// One `--wire` token: either a plain arm (unchanged behaviour — f32
/// request frames with Accept per arm, or the q8k endpoint) or an explicit
/// asymmetric `in/return` pair on the dense walk-ffn wire (DEC funnel v0.5
/// §3 DEC-1A: request Content-Type and Accept negotiated independently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireSpec {
    /// `"f32"`, `"f16"`, `"i8"`, `"q8k"` — the pre-pair wire axis,
    /// byte-identical behaviour (requests stay f32; Accept per arm).
    Plain(WireArm),
    /// `"in/out"` such as `"f16/i8"`: the request residual is encoded in
    /// `input` (request Content-Type) and the response requested in
    /// `output` (strict Accept). Walk-ffn endpoint only; q8k cannot pair
    /// (own endpoint, not a negotiated CT).
    Pair {
        input: WireFormat,
        output: WireFormat,
    },
}

/// Numeric twin of a [`WireFormat`] for pair codes (f32=0, f16=1, i8=2).
pub(super) fn wire_format_code(f: WireFormat) -> u32 {
    match f {
        WireFormat::F32 => 0,
        WireFormat::F16 => 1,
        WireFormat::I8 => 2,
    }
}

impl WireSpec {
    pub fn parse(tok: &str) -> Result<WireSpec, String> {
        let tok = tok.trim();
        if let Some((i, o)) = tok.split_once('/') {
            let side = |s: &str, which: &str| {
                WireFormat::parse(s).ok_or_else(|| {
                    format!(
                        "invalid wire pair {tok:?}: {which} arm {s:?} must be one of \
                         f32, f16, i8 (q8k is its own endpoint and cannot pair)"
                    )
                })
            };
            Ok(WireSpec::Pair {
                input: side(i, "inbound")?,
                output: side(o, "return")?,
            })
        } else {
            WireArm::parse(tok).map(WireSpec::Plain)
        }
    }

    pub fn parse_list(s: &str) -> Result<Vec<WireSpec>, String> {
        s.split(',').map(WireSpec::parse).collect()
    }

    /// Combined arm label: plain arms keep their historical label; pairs
    /// are `"in/out"` (e.g. `"f16/i8"`). This is `dec/wire_format` — kept
    /// as the one-string axis for run-record continuity.
    pub fn label(self) -> String {
        match self {
            Self::Plain(a) => a.label().into(),
            Self::Pair { input, output } => format!("{}/{}", input.label(), output.label()),
        }
    }

    /// Numeric twin of [`Self::label`]. Plain arms keep their historical
    /// codes (0–3); pairs are `100 + 10×in + out` with f32=0, f16=1, i8=2
    /// (e.g. `f16/i8` → 112, `i8/f16` → 121) — disjoint from the plain
    /// range so numeric-only ingesters can split the axes.
    pub fn code(self) -> u32 {
        match self {
            Self::Plain(a) => a.code(),
            Self::Pair { input, output } => {
                100 + 10 * wire_format_code(input) + wire_format_code(output)
            }
        }
    }

    /// Request-direction dtype label this arm actually puts on the wire:
    /// plain walk-ffn arms send f32 request frames (historical wire), the
    /// q8k arm its own frame format; pairs send their `input` format.
    pub fn in_label(self) -> &'static str {
        match self {
            Self::Plain(WireArm::Q8k) => "q8k",
            Self::Plain(_) => "f32",
            Self::Pair { input, .. } => input.label(),
        }
    }

    /// Return-direction dtype label requested by this arm.
    pub fn out_label(self) -> &'static str {
        match self {
            Self::Plain(a) => a.label(),
            Self::Pair { output, .. } => output.label(),
        }
    }

    /// `Accept` header (strict, no fallback formats offered). Pairs ask
    /// for exactly the return arm.
    pub fn accept(self) -> Option<&'static str> {
        match self {
            Self::Plain(a) => a.accept(),
            Self::Pair { output, .. } => Some(output.content_type()),
        }
    }

    /// Request-frame dtype on the dense walk-ffn endpoint. Plain arms keep
    /// the historical f32 request wire; pairs encode in their `input` arm.
    pub fn request_format(self) -> WireFormat {
        match self {
            Self::Pair { input, .. } => input,
            Self::Plain(_) => WireFormat::F32,
        }
    }

    /// Whether this arm rides the q8k endpoints (own wire, no negotiation).
    pub fn is_q8k(self) -> bool {
        matches!(self, Self::Plain(WireArm::Q8k))
    }
}

// ── Endpoint seam ─────────────────────────────────────────────────────────────

/// CLI-level endpoint family (`--endpoint`). Combined with the wire axis it
/// resolves to a concrete [`Endpoint`] via [`Endpoint::resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointKind {
    /// Dense/shared-expert FFN path (`/v1/walk-ffn[-q8k]`).
    WalkFfn,
    /// Routed-experts multi-layer path (`/v1/experts/multi-layer-batch[-q8k]`).
    Experts,
}

impl EndpointKind {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "walk-ffn" => Ok(Self::WalkFfn),
            "experts" => Ok(Self::Experts),
            other => Err(format!(
                "unknown endpoint {other:?} (expected walk-ffn, experts)"
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::WalkFfn => "walk-ffn",
            Self::Experts => "experts",
        }
    }
}

/// Where an endpoint's server-side serve latency comes from (two-scoreboard
/// schema, dec-funnel §3 DEC-1A). Successor of the boolean `has_server_ms`:
/// every endpoint now has a serve-latency source — the f32 walk-ffn response
/// embeds `latency_ms` in its fixed header; the other three carry the opt-in
/// timing trailer (`x-larql-timing: 1` request header → 8-byte magic +
/// serve_us f32 trailer, absent on pre-extension servers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeLatencySource {
    /// `latency_ms` f32 embedded at bytes 8–11 of the response header
    /// (always present — no opt-in needed).
    EmbeddedMs,
    /// Header-gated timing trailer appended after the payload; `None` at
    /// decode time when the server predates the extension.
    TimingTrailer,
}

/// Which movement-ratio denominator an endpoint's byte accounting uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenominatorSource {
    /// `ffn_weights.per_layer_dense_bytes` summed over replayed layers.
    Dense,
    /// `ffn_weights.moe.per_expert_bytes` × captured routing (naive/union).
    RoutedExperts,
}

/// One concrete server endpoint of the sweep. Owns the request path,
/// content-type/Accept behaviour, frame build + response decode (via the
/// production codecs — parity discipline), `server_ms` availability, and the
/// movement-ratio denominator source. The former two-way `send_one` branch
/// made explicit (audit §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    WalkFfn,
    WalkFfnQ8k,
    ExpertsMultiLayer,
    ExpertsMultiLayerQ8k,
}

impl Endpoint {
    /// Map (endpoint family, wire arm) to a concrete endpoint. The
    /// multi-layer expert wire has no f16/i8 variant and no asymmetric
    /// pairs — those combinations are a loud arg-validation error, not a
    /// silent fallback.
    pub fn resolve(kind: EndpointKind, wire: WireSpec) -> Result<Self, String> {
        match (kind, wire) {
            (EndpointKind::WalkFfn, WireSpec::Plain(WireArm::Q8k)) => Ok(Self::WalkFfnQ8k),
            (EndpointKind::WalkFfn, _) => Ok(Self::WalkFfn),
            (EndpointKind::Experts, WireSpec::Pair { .. }) => Err(format!(
                "--endpoint experts has no {} arm — asymmetric in/return pairs exist only \
                 on the dense walk-ffn wire (the multi-layer expert wire carries f32 or \
                 q8k frames only)",
                wire.label()
            )),
            (EndpointKind::Experts, WireSpec::Plain(WireArm::F32)) => Ok(Self::ExpertsMultiLayer),
            (EndpointKind::Experts, WireSpec::Plain(WireArm::Q8k)) => {
                Ok(Self::ExpertsMultiLayerQ8k)
            }
            (EndpointKind::Experts, WireSpec::Plain(w)) => Err(format!(
                "--endpoint experts has no {} arm — the multi-layer expert wire carries \
                 f32 or q8k frames only (use --wire f32,q8k)",
                w.label()
            )),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::WalkFfn => "walk-ffn",
            Self::WalkFfnQ8k => "walk-ffn-q8k",
            Self::ExpertsMultiLayer => "experts-ml",
            Self::ExpertsMultiLayerQ8k => "experts-ml-q8k",
        }
    }

    /// Numeric twin of [`Self::label`] for numeric-only metric ingesters.
    pub fn code(self) -> u32 {
        match self {
            Self::WalkFfn => 0,
            Self::WalkFfnQ8k => 1,
            Self::ExpertsMultiLayer => 2,
            Self::ExpertsMultiLayerQ8k => 3,
        }
    }

    /// Request path — the production constants, not re-declared strings.
    pub fn path(self) -> &'static str {
        match self {
            Self::WalkFfn => WALK_FFN_PATH,
            Self::WalkFfnQ8k => WALK_FFN_Q8K_PATH,
            Self::ExpertsMultiLayer => MULTI_LAYER_BATCH_PATH,
            Self::ExpertsMultiLayerQ8k => MULTI_LAYER_BATCH_Q8K_PATH,
        }
    }

    /// Request `Content-Type`. On the dense walk-ffn endpoint this is the
    /// arm's INBOUND direction (f32 for plain arms — the historical wire;
    /// the `input` format for pairs); the other endpoints have fixed CTs.
    pub fn request_content_type(self, wire: WireSpec) -> &'static str {
        match self {
            Self::WalkFfn => wire.request_format().content_type(),
            Self::WalkFfnQ8k => larql_inference::Q8K_BATCH_CT,
            Self::ExpertsMultiLayer => MULTI_LAYER_BATCH_CONTENT_TYPE,
            Self::ExpertsMultiLayerQ8k => MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
        }
    }

    /// `Accept` header. Only the walk-ffn endpoint negotiates response
    /// formats (per wire arm / pair return arm); q8k walk-ffn is its own
    /// endpoint with a fixed response, and both multi-layer endpoints
    /// always answer with the f32 multi-layer CT (mirrors the production
    /// shard client's Accept).
    pub fn accept(self, wire: WireSpec) -> Option<&'static str> {
        match self {
            Self::WalkFfn => wire.accept(),
            Self::WalkFfnQ8k => None,
            Self::ExpertsMultiLayer | Self::ExpertsMultiLayerQ8k => {
                Some(MULTI_LAYER_BATCH_CONTENT_TYPE)
            }
        }
    }

    /// How this endpoint's response reports server compute latency. The f32
    /// walk-ffn response embeds `latency_ms` in its header; the q8k
    /// walk-ffn and both multi-layer responses carry the opt-in timing
    /// trailer instead (audit §2 closed by the DEC-1A timing extension).
    pub fn serve_latency_source(self) -> ServeLatencySource {
        match self {
            Self::WalkFfn => ServeLatencySource::EmbeddedMs,
            Self::WalkFfnQ8k | Self::ExpertsMultiLayer | Self::ExpertsMultiLayerQ8k => {
                ServeLatencySource::TimingTrailer
            }
        }
    }

    /// Movement-ratio denominator family.
    pub fn denominator(self) -> DenominatorSource {
        match self {
            Self::WalkFfn | Self::WalkFfnQ8k => DenominatorSource::Dense,
            Self::ExpertsMultiLayer | Self::ExpertsMultiLayerQ8k => {
                DenominatorSource::RoutedExperts
            }
        }
    }

    /// Routed endpoints replay captured routing — the pool must carry the
    /// `--routing` sidecars.
    pub fn requires_routing(self) -> bool {
        self.denominator() == DenominatorSource::RoutedExperts
    }
}

/// How the per-layer requests of one step are dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// Sequential per-layer round trips; step time = Σ layer RTTs.
    Streaming,
    /// All layers fired in parallel (mirrors `forward_predispatch_all`);
    /// step time = fan-out wall time.
    Batch,
}

impl DispatchMode {
    pub fn parse_list(s: &str) -> Result<Vec<DispatchMode>, String> {
        s.split(',')
            .map(|d| match d.trim() {
                "streaming" => Ok(DispatchMode::Streaming),
                "batch" => Ok(DispatchMode::Batch),
                other => Err(format!(
                    "unknown dispatch mode {other:?} (expected streaming, batch)"
                )),
            })
            .collect()
    }

    pub fn label(self) -> &'static str {
        match self {
            DispatchMode::Streaming => "streaming",
            DispatchMode::Batch => "batch",
        }
    }

    pub fn code(self) -> u32 {
        match self {
            DispatchMode::Streaming => 0,
            DispatchMode::Batch => 1,
        }
    }
}
