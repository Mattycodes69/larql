//! Shapes, device weights, state and scratch for one KDA layer.

use super::super::grouped_experts::{ExpertOffset, GroupedError};
use super::super::kimi_layer::ExpertEncoding;
use crate::MetalBackend;
use larql_models::config::KdaGateForm;
use metal::Buffer;

#[allow(unused_imports)]
use super::*;

/// `o_proj` is one slot at offset zero. A `static` so its address is
/// stable and the device table can be cached rather than rebuilt.
pub(super) static O_PROJ_SINGLE_SLOT: [ExpertOffset; 1] = [ExpertOffset(0)];
/// The three convolved streams: q, k, v.
pub(super) const CONV_STREAMS: usize = 3;
/// The narrowest convolution the short-conv kernel defines: it keeps
/// `kernel - 1` inputs of history, and the current input is the last tap.
pub(super) const MIN_CONV_KERNEL: usize = 1;
/// Bytes per bf16 code.
pub(super) const BF16_BYTES: usize = 2;

/// The geometry one KDA layer runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaShape {
    pub hidden: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
}

impl KdaShape {
    /// `num_heads * head_dim` — the width every per-channel stage runs at.
    pub fn width(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Elements of convolution history each stream carries between calls.
    pub(super) fn conv_tail(self) -> usize {
        self.conv_kernel.saturating_sub(1)
    }
}

/// The layer's weights, as the device sees them.
///
/// bf16 for the four wide projections — the checkpoint's own bytes,
/// bound without a widening pass — and f32 for the small gate matrices
/// and per-channel vectors, which is what they are on disk.
#[derive(Clone, Copy)]
pub struct KdaDeviceWeights<'a> {
    /// `q|k|v` concatenated, `[3][width, hidden]`, with each slot's byte
    /// offset. One buffer because the grouped kernel binds one.
    pub qkv_bank: &'a [u8],
    pub qkv_offsets: &'a [ExpertOffset; CONV_STREAMS],
    /// `[hidden, width]`, in the same encoding as `qkv_bank`.
    pub o_proj: &'a [u8],
    /// Physical representation of `qkv_bank` and `o_proj` — the two
    /// wide projections and NOTHING else. The convolutions, the low-rank
    /// decay/output gates, `b_proj`, `A_log`, `dt_bias` and `o_norm`
    /// stay f32 whatever this says: they are a few MB against ~75 MB a
    /// layer, and they feed the numerically delicate recurrence, which
    /// this rung deliberately does not touch. Dispatch selects the
    /// grouped kernel by this value, so bytes can never pair with
    /// another encoding's kernel.
    pub projection_encoding: ExpertEncoding,
    /// `[width, conv_kernel]` each.
    pub q_conv1d: &'a [f32],
    pub k_conv1d: &'a [f32],
    pub v_conv1d: &'a [f32],
    /// `[head_dim, hidden]` then `[width, head_dim]`, each at its own
    /// stored precision — see [`SmallMatrix`].
    pub f_a_proj: SmallMatrix<'a>,
    pub f_b_proj: SmallMatrix<'a>,
    pub g_a_proj: SmallMatrix<'a>,
    pub g_b_proj: SmallMatrix<'a>,
    /// `[num_heads, hidden]`.
    pub b_proj: SmallMatrix<'a>,
    /// `[num_heads]`, `[width]`, `[head_dim]`.
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub o_norm: &'a [f32],
    pub norm_eps: f32,
    /// Which decay gate this checkpoint's FAMILY computes.
    ///
    /// Required, and deliberately not `Option` with a default: Kimi and
    /// GLM both declare `gate_lower_bound: -5.0` and only GLM applies
    /// it, so a default here would be a silent substitution rather than
    /// a missing value. Callers holding a declaration that may be absent
    /// resolve it through [`declared_gate_form`], which refuses by name.
    pub gate_form: KdaGateForm,
}

/// One of KDA's five small matrices, bound at the precision the
/// CHECKPOINT stores it.
///
/// **Dtype-preserving per tensor, not a blanket conversion.** Kimi ships
/// `A_log` and `dt_bias` as F32 while every matrix in the same KDA block
/// is BF16, so "the small tensors are BF16" is false of the block and
/// only true of particular tensors. Widening BF16 codes to f32 at bind
/// time doubles their traffic for no fidelity: these five are 97 % of
/// KDA's non-projection parameters, 0.239 GB/token of avoidable reads
/// across 34 GLM layers.
#[derive(Clone, Copy)]
pub enum SmallMatrix<'a> {
    /// The checkpoint stores f32; bind it as it is.
    F32(&'a [f32]),
    /// The checkpoint stores bf16; bind its own bytes, unwidened.
    Bf16(&'a [u8]),
}

impl SmallMatrix<'_> {
    /// Elements, so a caller can check a shape without knowing the dtype.
    pub fn len(&self) -> usize {
        match self {
            SmallMatrix::F32(v) => v.len(),
            SmallMatrix::Bf16(b) => b.len() / BF16_BYTES,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements, or `None` for a bf16 slice that is not a whole number
    /// of codes — which [`len`](Self::len) would silently round down.
    pub(super) fn exact_len(&self) -> Option<usize> {
        match self {
            SmallMatrix::F32(v) => Some(v.len()),
            SmallMatrix::Bf16(b) => b
                .len()
                .is_multiple_of(BF16_BYTES)
                .then_some(b.len() / BF16_BYTES),
        }
    }
}

/// The four buffers the decay gate binds. Grouped so the encoder's
/// signature stays readable now that the FORM travels with them.
pub(super) struct DecayGateBinding<'a> {
    pub(super) f_low: &'a Buffer,
    pub(super) dt_bias: &'a Buffer,
    pub(super) a_log: &'a Buffer,
    pub(super) decay: &'a Buffer,
}

/// Resolve a DECLARED gate form, refusing rather than defaulting.
///
/// `ExecutionSurface.kda_gate_form` is `Option` because no checkpoint
/// states which branch its reference takes — the fact lives with the
/// family. An unjudged family must not run: picking either form
/// produces a plausible, wrong, compounding recurrence.
pub fn declared_gate_form(declared: Option<KdaGateForm>) -> Result<KdaGateForm, GroupedError> {
    declared.ok_or(GroupedError::KdaGateFormUndeclared)
}

/// The recurrent and convolution state a KDA layer carries between
/// calls, resident on device.
///
/// Nothing here is indexed by position: the recurrent part is one
/// `D x D` matrix per head whatever the sequence length, and the
/// convolution part is the last `kernel - 1` inputs of each stream.
pub struct KdaDeviceState {
    pub(super) shape: KdaShape,
    /// `[heads, dim, dim]` f32.
    pub(super) recurrent: Buffer,
    /// Three `[width, kernel-1]` f32 windows, for q, k and v.
    pub(super) conv: [Buffer; CONV_STREAMS],
}

impl KdaDeviceState {
    /// The zero state a sequence starts from, allocated on device.
    pub fn zeros(metal: &MetalBackend, shape: KdaShape) -> Self {
        let width = shape.width();
        let recurrent = metal
            .bufs()
            .zeroed((shape.num_heads * shape.head_dim * shape.head_dim * 4) as u64);
        let window = (width * shape.conv_tail() * 4) as u64;
        Self {
            shape,
            recurrent,
            conv: [
                metal.bufs().zeroed(window),
                metal.bufs().zeroed(window),
                metal.bufs().zeroed(window),
            ],
        }
    }

    /// Zero the state again, in place, without reallocating.
    ///
    /// A sequence boundary — and, for a measurement, the way to hold the
    /// input constant: the recurrent state advances every step, so a
    /// timed loop over one token would otherwise be scoring a different
    /// hidden state each iteration, and in a MoE layer a different
    /// route.
    ///
    /// Host-side, so it is only legal between steps, after the wait.
    pub fn reset(&self) {
        let g = self.shape;
        let zero = |b: &Buffer| {
            let ptr = b.contents();
            if !ptr.is_null() {
                // SAFETY: shared-storage buffer of exactly this length,
                // and no GPU work is in flight against it — `reset` is
                // only legal between steps.
                unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, b.length() as usize) };
            }
        };
        zero(&self.recurrent);
        for c in &self.conv {
            zero(c);
        }
        let _ = g;
    }

    /// `(recurrent, [q, k, v] windows)` copied to the host.
    ///
    /// For gates only. Production never needs this — the point of the
    /// rung is that the state does not come back — but a state that
    /// silently diverged from the CPU path would show up only many
    /// tokens later, so it has to be checkable.
    pub fn read_back(&self) -> (Vec<f32>, [Vec<f32>; CONV_STREAMS]) {
        let g = self.shape;
        let width = g.width();
        let tail = width * g.conv_tail();
        let read = |b: &Buffer, n: usize| crate::buffers::read_buffer_f32(b, n);
        (
            read(&self.recurrent, g.num_heads * g.head_dim * g.head_dim),
            [
                read(&self.conv[0], tail),
                read(&self.conv[1], tail),
                read(&self.conv[2], tail),
            ],
        )
    }
}

/// Every boundary the device path produces, named exactly as the CPU
/// path's `KdaPlanes` names them, so a disagreement reports the stage it
/// happened in rather than "the layer".
#[derive(Debug, Clone)]
pub struct KdaDevicePlanes {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_conv: Vec<f32>,
    pub k_conv: Vec<f32>,
    pub v_conv: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub f_lowrank: Vec<f32>,
    pub g_decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub recurrent_out: Vec<f32>,
    pub o_gate: Vec<f32>,
    pub o_norm: Vec<f32>,
    pub output: Vec<f32>,
    /// GPU-busy milliseconds for the one command buffer this step used.
    pub gpu_ms: f64,
}

/// One convolved stream's bindings: where its projection sits inside the
/// grouped `q|k|v` output, its depthwise weights, the window it carries
/// between calls, and where the convolved result goes.
///
/// Bundled because the four travel together and a call site that took
/// them positionally is where the q window gets paired with the k
/// weights — a silent wrong answer, since all three streams have the
/// same shape.
pub(super) struct ConvStream<'a> {
    pub(super) src: &'a Buffer,
    pub(super) src_offset: u64,
    pub(super) weight: &'a Buffer,
    pub(super) window: &'a Buffer,
    pub(super) out: &'a Buffer,
}

/// Per-call scratch, one pop from the pool each so no two alias.
pub struct Scratch {
    pub qkv: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub q_norm: Buffer,
    pub k_norm: Buffer,
    pub f_a: Buffer,
    pub f_low: Buffer,
    pub decay: Buffer,
    pub g_a: Buffer,
    pub gate: Buffer,
    pub b_pre: Buffer,
    pub beta: Buffer,
    pub recurrent_out: Buffer,
    pub normed: Buffer,
    pub out: Buffer,
}
