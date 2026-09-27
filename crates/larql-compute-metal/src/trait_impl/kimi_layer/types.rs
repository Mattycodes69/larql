//! Weights, addressing and scratch types for one Kimi decoder layer.

use super::super::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use super::super::mla::{MlaDeviceState, MlaDeviceWeights, MlaShape};
use crate::shaders::kimi_layer as layer_shader;
use metal::Buffer;

#[allow(unused_imports)]
use super::*;

/// `rms_norm` must be dispatched as ONE threadgroup, this wide.
pub(super) const NORM_THREADS_PER_TG: u64 = 256;
/// Plain `KimiRMSNorm` has no weight offset.
pub(super) const NORM_WEIGHT_OFFSET: f32 = 0.0;
/// `residual_add`'s scale — the residual is unit here.
pub(super) const RESIDUAL_UNIT_SCALE: f32 = 1.0;

/// The MoE half of a layer: the router, and the resident expert bank.
#[derive(Clone, Copy)]
pub struct KimiMoeWeights<'a> {
    /// `[experts, hidden]` f32.
    pub router_weight: &'a [f32],
    /// `[experts]` f32 — the correction bias. It SELECTS and never
    /// weighs; the weights come from the unbiased sigmoid scores.
    pub router_bias: &'a [f32],
    /// The three projection banks. Each carries its OWN addressing and
    /// its own shared-branch region, because a logical expert may sit
    /// at a different physical slot in each of them — and the shared
    /// branch may live in a different allocation entirely.
    pub gate: ProjectionBank<'a>,
    pub up: ProjectionBank<'a>,
    pub down: ProjectionBank<'a>,
    pub inter: usize,
    pub top_k: usize,
    pub renormalize: bool,
    /// `routed_scaling_factor`, folded into the routed weights.
    pub branch_scale: f32,
}

/// One physical region a grouped kernel can read: bytes, and what they
/// ARE.
///
/// Per projection, so `gate/up` at Q6_K with `down` at BF16 is
/// expressible — the precision map can already say it, so the physical
/// vocabulary must be able to carry it.
#[derive(Clone, Copy)]
pub struct EncodedRegion<'a> {
    pub bytes: &'a [u8],
    pub encoding: ExpertEncoding,
}

/// One projection's routed bank, how a logical expert is located in it,
/// and — separately — the shared branch's own region.
///
/// Bank and addressing travel together because they are one physical
/// fact. Splitting them let a caller pair a bank with another
/// projection's coordinates, which is the shared-coordinate bug this
/// whole change exists to make unrepresentable.
///
/// The shared branch is its OWN region, not an offset into the routed
/// bank: `Shared` vs `Routed` is semantic identity and must not imply
/// physical co-location. A source container keeps the shared expert in
/// the decoder stack and the routed experts in an expert bank; a
/// candidate overlay may compile the routed experts to Q6_K while the
/// shared branch stays source BF16. Co-locating the two is a layout an
/// artifact MAY choose (the region can be a subrange of the same
/// allocation), never an invariant execution relies on.
#[derive(Clone, Copy)]
pub struct ProjectionBank<'a> {
    pub routed: EncodedRegion<'a>,
    pub addressing: ExpertAddressing<'a>,
    /// The shared branch's `[n, k]` matrix for this projection, when
    /// the architecture has one. All three projections must agree on
    /// whether it exists — that is one semantic fact — and each binds
    /// its own bytes under its own encoding.
    pub shared: Option<EncodedRegion<'a>>,
}

/// A representation a grouped kernel can execute.
///
/// The backend answers whether it CAN run one; choosing it belongs to a
/// precision map. All three share one binding ABI — weights, byte
/// offsets, X, output, N/K, stride — so this selects a kernel and
/// nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertEncoding {
    Bf16,
    /// Canonical ggml Q8_0: 34-byte blocks of 32 (f16 scale + 32 int8),
    /// 8.5 bpw — the precision ladder's rung between BF16 and Q6_K.
    Q80,
    Q6K,
    Q4K,
}

impl ExpertEncoding {
    pub fn name(self) -> &'static str {
        match self {
            ExpertEncoding::Bf16 => "BF16",
            ExpertEncoding::Q80 => "Q8_0",
            ExpertEncoding::Q6K => "Q6_K",
            ExpertEncoding::Q4K => "Q4_K",
        }
    }

    /// Bytes one `[n, k]` matrix occupies. `None` when the shape cannot
    /// be encoded this way at all.
    pub fn matrix_bytes(self, n: usize, k: usize) -> Option<usize> {
        match self {
            ExpertEncoding::Bf16 => Some(n * k * 2),
            ExpertEncoding::Q80 => k.is_multiple_of(32).then(|| n * k / 32 * 34),
            ExpertEncoding::Q6K | ExpertEncoding::Q4K => k.is_multiple_of(256).then(|| {
                let per = if self == ExpertEncoding::Q6K {
                    210
                } else {
                    144
                };
                n * k / 256 * per
            }),
        }
    }
}

/// How the kernel turns a selected expert id into a bank byte offset.
///
/// **Addressability, not residency.** A full execution-shaped bank
/// addresses by identity and can never fail to answer; a packed subset
/// needs a table and genuinely can. Whether an addressable expert's
/// pages happen to be resident is a paging concern this type does not
/// express — manufacturing a 256-entry table for a full bank would state
/// a residency claim nobody made.
#[derive(Clone, Copy)]
pub enum ExpertAddressing<'a> {
    /// `offset = expert_id * stride`. Nothing is tabulated.
    ///
    /// One stride for all three projections, which holds because gate/up
    /// and down are transposes of one another — the same equality the
    /// shared offset table already relies on.
    Identity { experts: usize, stride: u32 },
    /// `[experts]` byte offsets, or [`layer_shader::NOT_RESIDENT`].
    Table(&'a [u32]),
}

impl ExpertAddressing<'_> {
    /// How many of the checkpoint's experts the router scores.
    pub fn experts(&self) -> usize {
        match self {
            ExpertAddressing::Identity { experts, .. } => *experts,
            ExpertAddressing::Table(t) => t.len(),
        }
    }

    /// The stride the kernel multiplies by, or 0 when it must consult
    /// the table instead.
    pub(super) fn identity_stride(&self) -> u32 {
        match self {
            ExpertAddressing::Identity { stride, .. } => *stride,
            ExpertAddressing::Table(_) => 0,
        }
    }

    /// The offset an expert resolves to, host-side, in 64 bits.
    ///
    /// Wide on purpose: `expert * stride` in `u32` wraps (release) or
    /// panics (debug) once a bank passes 4 GiB, and a wrapped offset is
    /// another expert's weights. The device table is still `u32`, so a
    /// caller that binds this must refuse what does not fit —
    /// [`validate_layer`] does.
    pub fn offset_of(&self, expert: usize) -> Option<u64> {
        match self {
            ExpertAddressing::Identity { experts, stride } => (expert < *experts)
                .then(|| (expert as u64).checked_mul(u64::from(*stride)))
                .flatten(),
            ExpertAddressing::Table(t) => t
                .get(expert)
                .copied()
                .filter(|o| *o != layer_shader::NOT_RESIDENT)
                .map(u64::from),
        }
    }
}

/// Which attention operator a layer runs, with its weights, geometry and
/// resident state.
///
/// Kimi alternates KDA and full-attention layers, and R6c's whole point
/// is that a `KDA -> MLA -> KDA` run can share one command buffer — so
/// the attention is a parameter of the layer rather than a second layer
/// type. Everything after it (residual, norm, router, MoE, residual) is
/// identical either way and is written once.
#[derive(Clone, Copy)]
pub enum AttentionSpec<'a> {
    Kda {
        weights: KdaDeviceWeights<'a>,
        shape: KdaShape,
        state: &'a KdaDeviceState,
    },
    Mla {
        weights: MlaDeviceWeights<'a>,
        shape: MlaShape,
        state: &'a MlaDeviceState,
    },
}

impl AttentionSpec<'_> {
    pub fn hidden(&self) -> usize {
        match self {
            Self::Kda { shape, .. } => shape.hidden,
            Self::Mla { shape, .. } => shape.hidden,
        }
    }
}

/// One layer's weights, attention and MoE together.
#[derive(Clone, Copy)]
pub struct KimiLayerWeights<'a> {
    pub input_norm: &'a [f32],
    pub post_attention_norm: &'a [f32],
    pub attention: AttentionSpec<'a>,
    pub ffn: ffn::FfnSpec<'a>,
    pub norm_eps: f32,
}

thread_local! {
    pub(super) static ENCODE_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    pub(super) static WAIT_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// `(encode, wait)` milliseconds accumulated on this thread since the
/// last call, and reset.
///
/// A diagnostic split, not a production reading: the two have different
/// fixes, and a layer chain's host cost is not obviously one or the
/// other until it is measured.
pub fn take_chain_timing_ms() -> (f64, f64) {
    let e = ENCODE_MS.with(|c| c.replace(0.0));
    let w = WAIT_MS.with(|c| c.replace(0.0));
    (e, w)
}

/// Optional instrumentation for a chain, `None` in serving.
///
/// The router's decisions already exist on device in each layer's
/// `chosen` buffer — 8 `u32` a layer, ~832 bytes for a whole Kimi token.
/// Collecting them costs no extra command buffer, no extra dispatch and
/// **no per-layer synchronisation**: they are read after the chain's
/// single `commit`+`wait`, before the scratch is recycled.
///
/// That is deliberately not the traced path, which reads twelve
/// full-width planes a layer and cost 64 ms a token against 20 ms of
/// GPU work. Reading a handful of 32-byte buffers that are already
/// mapped is free by comparison — the expensive thing was always the
/// width of the planes, never the act of reading after the wait.
///
/// Ids come back in ROUTER ORDER: the raw execution fact. Deciding that
/// ordering is irrelevant belongs to the quality metric, not the engine.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ExecutionTrace {
    /// One entry per layer in the chain, each `top_k` expert ids. Empty
    /// for a dense layer, which routes nothing.
    pub routes: Vec<Vec<u32>>,
    /// Ask for the router's own SELECTION SCORES as well — set before
    /// the call, read after it.
    ///
    /// Off by default because serving must not pay for it, but the cost
    /// is small and worth naming: `experts` floats a layer, ~27 KB for
    /// a whole Kimi token, read after the chain's single wait like the
    /// routes are. That is nothing beside the twelve FULL-WIDTH planes
    /// a layer the traced path reads, which cost 64 ms a token — the
    /// expensive thing was always the width, never the reading.
    pub want_selection_scores: bool,
    /// Per layer, the biased score EVERY expert was ranked by —
    /// `sigmoid(logit) + correction_bias`, which is the quantity the
    /// selection actually used, not a raw pre-policy logit.
    ///
    /// With it a consumer can ask how CLOSE a routing decision was: the
    /// gap between the last selected expert's score and the best
    /// unselected one is the margin the perturbation had to cross.
    /// Empty unless [`Self::want_selection_scores`] was set.
    pub selection_scores: Vec<Vec<f32>>,
    /// Per layer, the combine weights the MoE multiplied by: `top_k`
    /// routed, then the shared branch's unscaled 1.0. Lets a consumer
    /// weigh a routing change by how much MIXTURE MASS moved rather
    /// than counting the change.
    pub combine_weights: Vec<Vec<f32>>,
}

/// One layer in a chain. The state travels inside
/// [`KimiLayerWeights::attention`], because which state a layer carries
/// is decided by which attention it runs.
#[derive(Clone, Copy)]
pub struct KimiLayerCall<'a> {
    pub weights: KimiLayerWeights<'a>,
}

/// One layer's attention scratch, whichever operator it runs.
pub(super) enum AttentionScratch {
    Kda(crate::trait_impl::kda::Scratch),
    Mla(crate::trait_impl::mla::MlaScratch),
}

impl AttentionScratch {
    /// The attention output — the residual's second operand, and the
    /// only plane the layer path itself reads.
    pub(super) fn out(&self) -> &Buffer {
        match self {
            Self::Kda(s) => &s.out,
            Self::Mla(s) => &s.out,
        }
    }
}

/// Scratch for one layer, one pop from the pool each so none alias.
pub(crate) struct LayerScratch {
    pub(super) input_normed: Buffer,
    pub(super) after_attention: Buffer,
    pub(super) post_normed: Buffer,
    pub(super) logits: Buffer,
    pub(super) scores: Buffer,
    pub(super) sel_scores: Buffer,
    pub(super) chosen: Buffer,
    pub(super) gate_offsets: Buffer,
    pub(super) up_offsets: Buffer,
    pub(super) down_offsets: Buffer,
    pub(super) weights: Buffer,
    pub(super) refusals: Buffer,
    pub(super) gate_out: Buffer,
    pub(super) up_out: Buffer,
    pub(super) h: Buffer,
    pub(super) expert_out: Buffer,
    pub(super) out: Buffer,
}
