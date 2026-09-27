//! Mamba2, conv-QKV and KDA operands.

use super::super::super::conv_qkv::ConvQkvOp;
use super::super::super::KdaOutputGate;
use super::super::super::{KdaOp, Mamba2Op, OperandRef};
use super::super::accounting::Bound;
use super::super::kda::KdaOutputGateWeights;
use super::super::operands::OperandSource;
use super::super::weights::{load_weight, LoadedWeight};
use crate::error::VindexError;

#[allow(unused_imports)]
use super::*;

impl Mamba2Operands {
    pub(in super::super) fn bound<'a>(&'a self, op: &'a Mamba2Op) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.in_proj, &self.in_proj),
            Bound::one(&op.out_proj, &self.out_proj),
        ]
    }

    pub(super) fn load(
        op: &Mamba2Op,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj: matrix(&op.in_proj)?,
            out_proj: matrix(&op.out_proj)?,
            conv1d: glue(&op.conv1d)?,
            conv1d_bias: op.conv1d_bias.as_ref().map(glue).transpose()?,
            a_log: glue(&op.a_log)?,
            d: glue(&op.d)?,
            dt_bias: glue(&op.dt_bias)?,
            norm: op
                .gated_norm
                .as_ref()
                .map(|n| glue(&n.weight))
                .transpose()?,
            // The epsilon travels with the gated norm's own NormOp; a
            // mixer with `rms_norm: false` has no norm and the value is
            // never read.
            norm_eps: op.gated_norm.as_ref().map_or(0.0, |n| n.eps as f32),
        })
    }

    /// The two matrices, for residency accounting.
    pub(in super::super) fn loaded_matrices(&self) -> [&LoadedWeight; 2] {
        [&self.in_proj, &self.out_proj]
    }

    /// The f32 operands that are not matrix traffic.
    pub(in super::super) fn glue_bytes(&self) -> usize {
        let opt = |v: &Option<Vec<f32>>| v.as_ref().map_or(0, |v| std::mem::size_of_val(&v[..]));
        [&self.conv1d, &self.a_log, &self.d, &self.dt_bias]
            .iter()
            .map(|v| std::mem::size_of_val(&v[..]))
            .sum::<usize>()
            + opt(&self.conv1d_bias)
            + opt(&self.norm)
    }

    pub(in super::super) fn weights(
        &self,
    ) -> Result<super::super::mamba2::Mamba2Weights<'_>, VindexError> {
        Ok(super::super::mamba2::Mamba2Weights {
            in_proj: matrix_rows(&self.in_proj, &self.op.in_proj)?,
            out_proj: matrix_rows(&self.out_proj, &self.op.out_proj)?,
            conv1d: &self.conv1d,
            conv1d_bias: self.conv1d_bias.as_deref(),
            a_log: &self.a_log,
            d: &self.d,
            dt_bias: &self.dt_bias,
            norm: self.norm.as_deref(),
            norm_eps: self.norm_eps,
        })
    }
}

/// The four operands a conv-QKV attention layer reads, loaded once —
/// the same matrix/glue split as the mixer's: two dense projections
/// against kilobytes of conv taps.
pub(in super::super) struct ConvQkvOperands {
    pub(in super::super) op: super::super::super::conv_qkv::ConvQkvOp,
    pub(super) in_proj: LoadedWeight,
    pub(super) out_proj: LoadedWeight,
    pub(super) conv1d: Vec<f32>,
    pub(super) conv1d_bias: Option<Vec<f32>>,
}

impl ConvQkvOperands {
    pub(in super::super) fn bound<'a>(&'a self, op: &'a ConvQkvOp) -> Vec<Bound<'a>> {
        vec![
            Bound::one(&op.in_proj, &self.in_proj),
            Bound::one(&op.out_proj, &self.out_proj),
        ]
    }

    pub(super) fn load(
        op: &super::super::super::conv_qkv::ConvQkvOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj: matrix(&op.in_proj)?,
            out_proj: matrix(&op.out_proj)?,
            conv1d: glue(&op.conv1d)?,
            conv1d_bias: op.conv1d_bias.as_ref().map(glue).transpose()?,
        })
    }

    /// The two matrices, for residency accounting.
    pub(in super::super) fn loaded_matrices(&self) -> [&LoadedWeight; 2] {
        [&self.in_proj, &self.out_proj]
    }

    /// The f32 operands that are not matrix traffic.
    pub(in super::super) fn glue_bytes(&self) -> usize {
        std::mem::size_of_val(&self.conv1d[..])
            + self
                .conv1d_bias
                .as_ref()
                .map_or(0, |v| std::mem::size_of_val(&v[..]))
    }

    pub(in super::super) fn weights(
        &self,
    ) -> Result<super::super::conv_qkv::ConvQkvWeights<'_>, VindexError> {
        Ok(super::super::conv_qkv::ConvQkvWeights {
            in_proj: matrix_rows(&self.in_proj, &self.op.in_proj)?,
            out_proj: matrix_rows(&self.out_proj, &self.op.out_proj)?,
            conv1d: &self.conv1d,
            conv1d_bias: self.conv1d_bias.as_deref(),
        })
    }
}

/// The fifteen operands a KDA layer reads, loaded once.
///
/// The same matrix/glue split every recurrence here draws, at KDA's own
/// proportions: four wide projections (q, k, v and the output, the whole
/// of this layer's matrix traffic) against fifteen kilobytes of
/// convolution taps, low-rank gate factors, decay parameters and the
/// gated norm's weight. The gate factorisations are matrices too, but at
/// `[rank, hidden]` and `[width, rank]` they are three orders of
/// magnitude smaller than the four, and the executor consumes them f32 —
/// so they load as glue, which is what they cost.
pub(in super::super) struct KdaOperands {
    pub(in super::super) op: super::super::super::KdaOp,
    pub(super) q_proj: LoadedWeight,
    pub(super) k_proj: LoadedWeight,
    pub(super) v_proj: LoadedWeight,
    pub(super) o_proj: LoadedWeight,
    pub(super) q_conv1d: Vec<f32>,
    pub(super) k_conv1d: Vec<f32>,
    pub(super) v_conv1d: Vec<f32>,
    pub(super) f_a_proj: Vec<f32>,
    pub(super) f_b_proj: Vec<f32>,
    pub(super) output_gate: KdaGateOperands,
    pub(super) b_proj: Vec<f32>,
    pub(super) a_log: Vec<f32>,
    pub(super) dt_bias: Vec<f32>,
    pub(super) o_norm: Vec<f32>,
    pub(super) norm_eps: f32,
}

/// The output gate's loaded operands, one variant per declared form: the
/// low-rank pair is glue, the full-rank projection is a matrix like the
/// four wide ones (on Kimi-K3 it is their size).
pub(super) enum KdaGateOperands {
    LowRank {
        g_a_proj: Vec<f32>,
        g_b_proj: Vec<f32>,
    },
    FullRank {
        g_proj: LoadedWeight,
    },
}

impl KdaOperands {
    pub(in super::super) fn bound<'a>(&'a self, op: &'a KdaOp) -> Vec<Bound<'a>> {
        let mut bound = vec![
            Bound::one(&op.q_proj, &self.q_proj),
            Bound::one(&op.k_proj, &self.k_proj),
            Bound::one(&op.v_proj, &self.v_proj),
            Bound::one(&op.out_proj, &self.o_proj),
        ];
        if let (KdaOutputGate::FullRank { g_proj: r }, KdaGateOperands::FullRank { g_proj: w }) =
            (&op.output_gate, &self.output_gate)
        {
            bound.push(Bound::one(r, w));
        }
        bound
    }

    pub(super) fn load(
        op: &super::super::super::KdaOp,
        store: OperandSource<'_>,
        format: FormatFor<'_>,
        norm_eps: f32,
    ) -> Result<Self, VindexError> {
        let matrix = |r: &OperandRef| load_weight(store, r, format(r)?);
        let glue = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            q_proj: matrix(&op.q_proj)?,
            k_proj: matrix(&op.k_proj)?,
            v_proj: matrix(&op.v_proj)?,
            o_proj: matrix(&op.out_proj)?,
            q_conv1d: glue(&op.q_conv1d)?,
            k_conv1d: glue(&op.k_conv1d)?,
            v_conv1d: glue(&op.v_conv1d)?,
            f_a_proj: glue(&op.f_a_proj)?,
            f_b_proj: glue(&op.f_b_proj)?,
            output_gate: match &op.output_gate {
                KdaOutputGate::LowRank { g_a_proj, g_b_proj } => KdaGateOperands::LowRank {
                    g_a_proj: glue(g_a_proj)?,
                    g_b_proj: glue(g_b_proj)?,
                },
                KdaOutputGate::FullRank { g_proj } => KdaGateOperands::FullRank {
                    g_proj: matrix(g_proj)?,
                },
            },
            b_proj: glue(&op.b_proj)?,
            a_log: glue(&op.a_log)?,
            dt_bias: glue(&op.dt_bias)?,
            o_norm: glue(&op.o_norm)?,
            norm_eps,
        })
    }

    /// The four matrices, for residency accounting.
    pub(in super::super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut matrices = vec![&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj];
        if let KdaGateOperands::FullRank { g_proj } = &self.output_gate {
            matrices.push(g_proj);
        }
        matrices
    }

    /// The f32 operands that are not matrix traffic.
    pub(in super::super) fn glue_bytes(&self) -> usize {
        [
            &self.q_conv1d,
            &self.k_conv1d,
            &self.v_conv1d,
            &self.f_a_proj,
            &self.f_b_proj,
            &self.b_proj,
            &self.a_log,
            &self.dt_bias,
            &self.o_norm,
        ]
        .into_iter()
        .chain(match &self.output_gate {
            KdaGateOperands::LowRank { g_a_proj, g_b_proj } => vec![g_a_proj, g_b_proj],
            KdaGateOperands::FullRank { .. } => vec![],
        })
        .map(|v| std::mem::size_of_val(&v[..]))
        .sum()
    }

    pub(in super::super) fn weights(
        &self,
    ) -> Result<super::super::kda::KdaWeights<'_>, VindexError> {
        Ok(super::super::kda::KdaWeights {
            q_proj: matrix_rows(&self.q_proj, &self.op.q_proj)?,
            k_proj: matrix_rows(&self.k_proj, &self.op.k_proj)?,
            v_proj: matrix_rows(&self.v_proj, &self.op.v_proj)?,
            o_proj: matrix_rows(&self.o_proj, &self.op.out_proj)?,
            q_conv1d: &self.q_conv1d,
            k_conv1d: &self.k_conv1d,
            v_conv1d: &self.v_conv1d,
            f_a_proj: &self.f_a_proj,
            f_b_proj: &self.f_b_proj,
            output_gate: match (&self.output_gate, &self.op.output_gate) {
                (
                    KdaGateOperands::LowRank { g_a_proj, g_b_proj },
                    KdaOutputGate::LowRank { .. },
                ) => KdaOutputGateWeights::LowRank { g_a_proj, g_b_proj },
                (KdaGateOperands::FullRank { g_proj }, KdaOutputGate::FullRank { g_proj: r }) => {
                    KdaOutputGateWeights::FullRank {
                        g_proj: matrix_rows(g_proj, r)?,
                    }
                }
                // `load` builds the operands FROM the op's form, so the two
                // cannot disagree; this is a construction error, never a
                // runtime condition.
                _ => unreachable!("KDA gate operands loaded for a form the op does not declare"),
            },
            // Refused, not defaulted: an unjudged family reaches this
            // arm and must not be served either form. The two observed
            // checkpoints declare the same bound and disagree on what it
            // means, so "pick the common one" is exactly the silent
            // mis-service this refusal exists to prevent.
            gate_form: self
                .op
                .gate_form
                .ok_or(VindexError::UnsupportedArchitecture {
                    family: "unjudged".to_string(),
                    feature: "KDA decay-gate form (whether the reference applies the \
                              declared `gate_lower_bound`; Kimi Linear and GLM-5.3-Flash \
                              both declare -5.0 and compute different gates, so the value \
                              does not settle it)"
                        .to_string(),
                    surface: "KDA executor".to_string(),
                })?,
            b_proj: &self.b_proj,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            o_norm: &self.o_norm,
            norm_eps: self.norm_eps,
            gate_rank: self.op.gate_rank,
        })
    }
}

/// The five operands an MLA layer reads, loaded once — plus the one
/// epsilon its own latent norm runs at.
///
/// That epsilon is why this struct exists in this form. It is not the
/// layer's `rms_norm_eps`: `kv_a_layernorm` takes its class default
/// (`1e-6` against the layer's `1e-5`), and until lift 2 the container
/// could not carry that at all — it lived as a constant inside a
/// family-shaped loader, where deleting the checkpoint could not restore
/// it. Loading REFUSES a container that carries no judged value rather
/// than borrowing the layer's: a norm at the wrong epsilon computes a
/// different function with every shape still closing.
pub(in super::super) struct MlaOperands {
    pub(in super::super) op: super::super::super::MlaOp,
    pub(super) kv_a_proj: LoadedWeight,
    pub(super) kv_b_proj: LoadedWeight,
    pub(super) o_proj: LoadedWeight,
    pub(super) kv_a_norm: Vec<f32>,
    pub(super) kv_a_norm_eps: f64,
    /// The query form's own operands (K3-MLA-Q-LORA-1).
    pub(super) query: MlaQueryOperands,
    /// The declared output gate's projection (K3-REP-GATE-1), a matrix
    /// the size of `o_proj`; `None` on an ungated layer.
    pub(super) output_gate: Option<LoadedWeight>,
}

/// The loaded operands of whichever query form the layer declared.
///
/// Mirrors [`MlaQueryProjection`] one-for-one so that "both" and
/// "neither" stay unrepresentable on this side of the load too.
pub(super) enum MlaQueryOperands {
    Direct {
        q_proj: LoadedWeight,
    },
    LowRank {
        q_a_proj: LoadedWeight,
        q_a_norm: Vec<f32>,
        q_b_proj: LoadedWeight,
        q_a_norm_eps: f64,
    },
}
