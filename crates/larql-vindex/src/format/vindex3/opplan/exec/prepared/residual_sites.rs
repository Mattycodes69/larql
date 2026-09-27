//! Attention-residual exits and hyper-connection sites.

use super::super::super::{
    AttnResSiteOp, ComponentOpPlan, GatedDeltaOp, HyperConnectionLayerOp, LayerAttention,
    LayerPlan, Mamba2Op, OperandRef,
};
use super::super::accounting::Bound;
use super::super::attention_residual;
use super::super::backend::WeightSlice;
use super::super::experts::FfnOperands;
use super::super::hyper_connection::{HeadWeights, HC_HEAD_SCALE_LEN};
use super::super::operands::OperandSource;
use super::super::weights::{load_weight, LoadedWeight};
use super::super::AttentionOperands;
use crate::error::VindexError;
use larql_models::config::HyperConnection;

#[allow(unused_imports)]
use super::*;

impl PreparedAttnResExit {
    /// Present only on a whole-stack image of an attention-residual
    /// component, and REQUIRED there: see [`ATTN_RES_EXITLESS_WHOLE_STACK`].
    pub(super) fn load(
        plan: &ComponentOpPlan,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Self, VindexError> {
        let Some(op) = &plan.attention_residual_exit else {
            return Err(VindexError::Parse(format!(
                "component `{}` {ATTN_RES_EXITLESS_WHOLE_STACK}",
                plan.component
            )));
        };
        Ok(Self {
            site: PreparedAttnResSite::load(
                &AttnResSiteOp {
                    norm: op.norm.clone(),
                    proj: op.proj.clone(),
                },
                store,
                hidden,
                "the attention-residual exit",
            )?,
            // The exit's RMSNorm is constructed from the component's
            // `rms_norm_eps`, exactly as every site's is, so it is ONE
            // component value — read through the same derivation the
            // hyper-connection head uses, which refuses a stack whose
            // layers disagree rather than picking one of them. Taking
            // the first layer's would have been a silent choice on
            // exactly the plan that needed a loud one.
            norm_eps: component_norm_eps(plan)?,
        })
    }

    pub(in super::super) fn pair(&self) -> attention_residual::SitePair<'_> {
        self.site.pair()
    }

    /// The component's declared norm epsilon, which the exit reduction
    /// scores at.
    pub(in super::super) fn norm_eps(&self) -> f64 {
        self.norm_eps
    }
}

impl PreparedHyperConnection {
    /// The layer's sites under the component's topology — present
    /// exactly when both agree, and a plan where they disagree is one
    /// the builder never produced.
    pub(super) fn for_layer(
        layer: &LayerPlan,
        topology: Option<HyperConnection>,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Option<Self>, VindexError> {
        match (&layer.hyper_connection, topology) {
            (None, None) => Ok(None),
            (Some(sites), Some(hc)) => {
                if layer.layer_scale.is_some() {
                    return Err(VindexError::Parse(format!(
                        "layer {} {HC_WITH_LAYER_SCALE}",
                        layer.layer
                    )));
                }
                Ok(Some(Self::load(sites, hc, hidden, layer.layer, store)?))
            }
            (Some(_), None) => Err(VindexError::Parse(format!(
                "layer {} carries hyper-connection sites but the component declares a single \
                 residual stream; the op plan never produces this",
                layer.layer
            ))),
            (None, Some(_)) => Err(VindexError::Parse(format!(
                "the component declares the hyper-connection topology but layer {} carries no \
                 sites; closure requires them on every layer",
                layer.layer
            ))),
        }
    }

    pub(super) fn load(
        sites: &HyperConnectionLayerOp,
        hc: HyperConnection,
        hidden: usize,
        layer: usize,
        store: OperandSource<'_>,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            attention: PreparedHcSite::load(
                &sites.attention,
                store,
                hc,
                hidden,
                &format!("layer {layer} attention site"),
            )?,
            ffn: PreparedHcSite::load(
                &sites.ffn,
                store,
                hc,
                hidden,
                &format!("layer {layer} ffn site"),
            )?,
        })
    }

    pub(super) fn glue_bytes(&self) -> usize {
        self.attention.glue_bytes() + self.ffn.glue_bytes()
    }
}

/// The head's own reduction operands (a different operation from a
/// site's: one row per stream, one scalar, no Sinkhorn), plus the norm
/// epsilon its mix projection runs at.
pub(in super::super) struct PreparedHcHead {
    pub(super) reduce_fn: Vec<f32>,
    pub(super) base: Vec<f32>,
    pub(super) scale: f32,
    pub(super) norm_eps: f64,
}

impl PreparedHcHead {
    /// Present only on a whole-stack image of a hyper-connected
    /// component, and REQUIRED there: see [`HC_HEADLESS_WHOLE_STACK`].
    pub(super) fn load(
        plan: &ComponentOpPlan,
        hc: HyperConnection,
        hidden: usize,
        store: OperandSource<'_>,
    ) -> Result<Self, VindexError> {
        let Some(op) = &plan.hyper_connection_head else {
            return Err(VindexError::Parse(format!(
                "component `{}` {HC_HEADLESS_WHOLE_STACK}",
                plan.component
            )));
        };
        let reduce_fn = store.load(&op.reduce_fn)?;
        let base = store.load(&op.base)?;
        let scale = store.load(&op.scale)?;
        if reduce_fn.len() != hc.streams * hc.streams * hidden || base.len() != hc.streams {
            return Err(VindexError::Parse(format!(
                "component `{}`: the hyper-connection head's operands do not hold the head's \
                 geometry ([{}, {}] and [{}])",
                plan.component,
                hc.streams,
                hc.streams * hidden,
                hc.streams
            )));
        }
        let scale = match scale[..] {
            [scale] => scale,
            ref other => {
                return Err(VindexError::Parse(format!(
                    "component `{}`: the hyper-connection head's scale holds {} values; the \
                     head reads exactly {HC_HEAD_SCALE_LEN}",
                    plan.component,
                    other.len()
                )))
            }
        };
        Ok(Self {
            reduce_fn,
            base,
            scale,
            norm_eps: component_norm_eps(plan)?,
        })
    }

    pub(in super::super) fn weights(&self) -> HeadWeights<'_> {
        HeadWeights {
            reduce_fn: &self.reduce_fn,
            base: &self.base,
            scale: self.scale,
        }
    }

    /// The component's declared norm epsilon — stage one's `norm_eps`
    /// for the head's mix projection.
    pub(in super::super) fn norm_eps(&self) -> f64 {
        self.norm_eps
    }

    pub(super) fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.reduce_fn[..]) + std::mem::size_of_val(&self.base[..])
    }
}

/// The component's ONE declared norm epsilon, read from the layers that
/// carry it. The head's mix projection runs at the component's
/// `rms_norm_eps`, which the plan carries per layer as a single
/// component fact; layers that disagree are a plan this build has not
/// judged, so the derivation refuses rather than picking one.
pub(super) fn component_norm_eps(plan: &ComponentOpPlan) -> Result<f64, VindexError> {
    let mut layers = plan.layers.iter();
    let Some(first) = layers.next() else {
        return Err(VindexError::Parse(format!(
            "component `{}` has no layers to read a norm epsilon from",
            plan.component
        )));
    };
    let eps = first.declared_norm_eps;
    if let Some(other) = layers.find(|l| l.declared_norm_eps != eps) {
        return Err(VindexError::Parse(format!(
            "component `{}`: layer {} declares norm eps {} where layer {} declares {}; the \
             hyper-connection head needs one component value",
            plan.component, other.layer, other.declared_norm_eps, first.layer, eps
        )));
    }
    Ok(eps)
}

/// One layer's operands, lowered into the backend's execution form.
pub(in super::super) struct PreparedLayer {
    /// `None` under post-norm placement — the sublayer reads the raw
    /// residual and the wrap norm applies to its output instead.
    pub(in super::super) pre_attention: Option<PreparedNorm>,
    pub(in super::super) attention: PreparedAttention,
    pub(in super::super) post_attention: Option<PreparedNorm>,
    /// Absent on a mixer-only (Mamba2) layer — the plan carries no FFN
    /// program there, so there is nothing to prepare and nothing to run.
    pub(in super::super) pre_ffn: Option<PreparedNorm>,
    pub(in super::super) ffn: Option<FfnOperands>,
    pub(in super::super) post_ffn: Option<PreparedNorm>,
    /// The layer's output scalar, when the plan carries one.
    pub(in super::super) layer_scale: Option<f32>,
    /// The two Sinkhorn sites, present exactly when the component
    /// declares the topology (wave 19a).
    pub(in super::super) hyper_connection: Option<PreparedHyperConnection>,
    /// The two attention-residual sites, present exactly when the
    /// component declares the topology.
    pub(in super::super) attention_residual: Option<PreparedAttentionResidual>,
}

impl PreparedLayer {
    /// This layer's norm weights — f32 glue, counted so the census adds
    /// up to the whole image rather than to the parts that were easy.
    /// The hyper-connection sites count here too, by the decision that
    /// made them f32 glue.
    pub(super) fn glue_bytes(&self) -> usize {
        let norm = |n: &PreparedNorm| std::mem::size_of_val(&n.weight[..]);
        self.pre_attention.as_ref().map_or(0, norm)
            + self.pre_ffn.as_ref().map_or(0, norm)
            + self.post_attention.as_ref().map_or(0, norm)
            + self.post_ffn.as_ref().map_or(0, norm)
            + self
                .attention_residual
                .as_ref()
                .map_or(0, PreparedAttentionResidual::glue_bytes)
            + self
                .hyper_connection
                .as_ref()
                .map_or(0, PreparedHyperConnection::glue_bytes)
    }
}

/// Which attention-class operator a prepared layer holds operands for.
///
/// An enum, not `Option<AttentionOperands>` and not "softmax unless
/// proven otherwise": a layer runs exactly one operator, and the
/// alternative spellings both make "I could not tell" indistinguishable
/// from "it is softmax". Qwen3.8 is 48 layers where that difference is
/// the whole model.
///
/// Chosen from the op plan's `LayerAttention`, which the op builder
/// derived from operand EVIDENCE — so the operands loaded here and the
/// operator dispatched later cannot disagree.
pub(in super::super) enum PreparedAttention {
    Softmax(Box<AttentionOperands>),
    GatedDelta(Box<GatedDeltaOperands>),
    Mamba2(Box<Mamba2Operands>),
    ConvQkv(Box<ConvQkvOperands>),
    Kda(Box<KdaOperands>),
    Mla(Box<MlaOperands>),
}

impl PreparedAttention {
    /// Every bound operand, paired by the loader that bound it — refusing
    /// a plan whose attention is a different program from the prepared one.
    pub(in super::super) fn bound<'a>(
        &'a self,
        attention: &'a LayerAttention,
    ) -> Result<Vec<Bound<'a>>, VindexError> {
        Ok(match (self, attention) {
            (Self::Softmax(ops), LayerAttention::Softmax(op)) => ops.bound(op),
            (Self::GatedDelta(ops), LayerAttention::GatedDelta(op)) => ops.bound(op),
            (Self::Mamba2(ops), LayerAttention::Mamba2(op)) => ops.bound(op),
            (Self::ConvQkv(ops), LayerAttention::ConvQkv(op)) => ops.bound(op),
            (Self::Kda(ops), LayerAttention::Kda(op)) => ops.bound(op),
            (Self::Mla(ops), LayerAttention::Mla(op)) => ops.bound(op),
            _ => {
                return Err(VindexError::Parse(
                    "the prepared attention and the plan's attention are different programs"
                        .to_string(),
                ))
            }
        })
    }

    /// Every matrix this attention holds resident — what a pinned
    /// projection realization is checked against.
    pub(in super::super) fn matrices(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::Softmax(ops) => ops.loaded_matrices(),
            Self::GatedDelta(ops) => ops.loaded_matrices().to_vec(),
            Self::Mamba2(ops) => ops.loaded_matrices().to_vec(),
            Self::ConvQkv(ops) => ops.loaded_matrices().to_vec(),
            Self::Kda(ops) => ops.loaded_matrices().to_vec(),
            Self::Mla(ops) => ops.loaded_matrices().to_vec(),
        }
    }

    /// Matrix operands for device placement.
    ///
    /// A recurrence contributes none: its nine operands are elementwise
    /// glue and a depthwise convolution, not the matrix traffic a device
    /// backend holds resident — and no device backend runs this operator
    /// yet, so placing them would reserve memory nothing reads.
    pub(super) fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::Softmax(ops) => ops.weight_slices(),
            Self::GatedDelta(_)
            | Self::Mamba2(_)
            | Self::ConvQkv(_)
            | Self::Kda(_)
            | Self::Mla(_) => Vec::new(),
        }
    }
}

/// The nine operands a Gated DeltaNet layer reads, loaded once.
///
/// The five projections carry a `LoadedWeight` and the four glue
/// operands a `Vec<f32>`, which is the split the measurements draw: 11.1
/// GB of matrix against 6 MB of convolution kernel, gate bias and norm.
pub(in super::super) struct GatedDeltaOperands {
    pub(in super::super) op: GatedDeltaOp,
    pub(super) in_proj_qkv: LoadedWeight,
    pub(super) in_proj_a: LoadedWeight,
    pub(super) in_proj_b: LoadedWeight,
    pub(super) in_proj_z: LoadedWeight,
    pub(super) out_proj: LoadedWeight,
    pub(super) conv1d: Vec<f32>,
    pub(super) a_log: Vec<f32>,
    pub(super) dt_bias: Vec<f32>,
    pub(super) norm: Vec<f32>,
    pub(super) norm_eps: f32,
}

impl GatedDeltaOperands {
    pub(in super::super) fn bound<'a>(&'a self, op: &'a GatedDeltaOp) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.in_proj_qkv, &self.in_proj_qkv),
            Bound::one(&op.in_proj_a, &self.in_proj_a),
            Bound::one(&op.in_proj_b, &self.in_proj_b),
            Bound::one(&op.in_proj_z, &self.in_proj_z),
            Bound::one(&op.out_proj, &self.out_proj),
        ]
    }

    pub(super) fn load(
        op: &GatedDeltaOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
        norm_eps: f32,
    ) -> Result<Self, VindexError> {
        // Per operand, and the answers differ WITHIN this layer: at
        // Qwen3.8's shapes `in_proj_qkv` is 105 MB and stays compact
        // while `in_proj_a` is 0.5 MB and does not. A single format for
        // the layer could not express that, and the version of this that
        // loaded everything f32 is what left 48 of 64 layers widened.
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj_qkv: matrix(&op.in_proj_qkv)?,
            in_proj_a: matrix(&op.in_proj_a)?,
            in_proj_b: matrix(&op.in_proj_b)?,
            in_proj_z: matrix(&op.in_proj_z)?,
            out_proj: matrix(&op.out_proj)?,
            conv1d: glue(&op.conv1d)?,
            a_log: glue(&op.a_log)?,
            dt_bias: glue(&op.dt_bias)?,
            norm: glue(&op.norm)?,
            norm_eps,
        })
    }

    /// The five matrices, for residency ACCOUNTING — not for device
    /// placement, which [`PreparedAttention::weight_slices`] still
    /// declines to offer for a recurrence no device kernel runs.
    pub(in super::super) fn loaded_matrices(&self) -> [&LoadedWeight; 5] {
        [
            &self.in_proj_qkv,
            &self.in_proj_a,
            &self.in_proj_b,
            &self.in_proj_z,
            &self.out_proj,
        ]
    }

    /// The four f32 operands that are not matrix traffic.
    pub(in super::super) fn glue_bytes(&self) -> usize {
        [&self.conv1d, &self.a_log, &self.dt_bias, &self.norm]
            .iter()
            .map(|v| std::mem::size_of_val(&v[..]))
            .sum()
    }

    pub(in super::super) fn weights(
        &self,
    ) -> Result<super::super::gated_delta::GatedDeltaWeights<'_>, VindexError> {
        // Geometry from the op, never from the slice length: a resident
        // slab is page-padded and can be longer than the matrix.
        Ok(super::super::gated_delta::GatedDeltaWeights {
            in_proj_qkv: matrix_rows(&self.in_proj_qkv, &self.op.in_proj_qkv)?,
            in_proj_a: matrix_rows(&self.in_proj_a, &self.op.in_proj_a)?,
            in_proj_b: matrix_rows(&self.in_proj_b, &self.op.in_proj_b)?,
            in_proj_z: matrix_rows(&self.in_proj_z, &self.op.in_proj_z)?,
            out_proj: matrix_rows(&self.out_proj, &self.op.out_proj)?,
            conv1d: &self.conv1d,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
            norm_eps: self.norm_eps,
        })
    }
}

/// The nine operands a Mamba2 layer reads, loaded once.
///
/// The two projections carry a `LoadedWeight`; the seven glue operands
/// are f32 — the same matrix/glue split the delta operands draw, at this
/// family's shapes (a 6448×1536 fused projection against kilobytes of
/// conv taps and per-head scalars).
pub(in super::super) struct Mamba2Operands {
    pub(in super::super) op: Mamba2Op,
    pub(super) in_proj: LoadedWeight,
    pub(super) out_proj: LoadedWeight,
    pub(super) conv1d: Vec<f32>,
    pub(super) conv1d_bias: Option<Vec<f32>>,
    pub(super) a_log: Vec<f32>,
    pub(super) d: Vec<f32>,
    pub(super) dt_bias: Vec<f32>,
    pub(super) norm: Option<Vec<f32>>,
    pub(super) norm_eps: f32,
}
