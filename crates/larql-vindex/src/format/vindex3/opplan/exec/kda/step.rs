//! The KDA recurrence step.

use super::super::continuation::RecurrentState;
use super::super::timing::{timed, OpClass};
use larql_models::config::{KdaGateForm, KdaGeometry};

#[allow(unused_imports)]
use super::*;

/// L2-normalise each head's slice in place.
///
/// Applied to q and k inside the reference's kernel
/// (`use_qk_l2norm_in_kernel=True`), which is why it appears nowhere in
/// the checkpoint's modeling file — and why it is the easiest operation in
/// the whole block to omit by accident.
pub(super) fn l2_normalise_heads(v: &mut [f32], heads: usize, dim: usize) {
    for h in 0..heads {
        let head = &mut v[h * dim..(h + 1) * dim];
        let norm = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Matches `F.normalize`'s clamp: a zero head stays zero rather
        // than becoming NaN.
        let inv = 1.0 / norm.max(1e-12);
        for x in head.iter_mut() {
            *x *= inv;
        }
    }
}

pub(super) fn bf16_round(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xFFFF_0000)
}

/// One position through the block, advancing `state`.
///
/// Returns the layer output for this position and appends every boundary
/// to `planes`.
#[allow(clippy::too_many_arguments)]
pub fn step(
    x: &[f32],
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    planes: &mut KdaPlanes,
    mutation: Mutation,
) -> Vec<f32> {
    step_with(&CpuKdaProjections, x, w, g, state, planes, mutation)
}

/// [`step`] with the projections executed somewhere the caller chooses.
#[allow(clippy::too_many_arguments)]
pub fn step_with(
    projections: &dyn KdaProjections,
    x: &[f32],
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    planes: &mut KdaPlanes,
    mutation: Mutation,
) -> Vec<f32> {
    let (heads, dim) = (g.num_heads, g.head_dim);
    let width = g.value_width();

    // q/k/v/o_proj are BF16, executed ONE AT A TIME through the executor's
    // row-parallel path (P4c-4 — see `matvec_bf16`'s own doc comment for
    // why sequential, not concurrent). The gate arithmetic stays plain f32
    // via the single-call `matvec`, unchanged from before P4c-2a — this
    // rung deliberately narrows to the four wide projections only.
    // All three at once. They are mutually independent — each feeds its
    // own convolution window and its own norm — so hoisting them above
    // the convolutions reorders nothing observable, and it is what lets a
    // batching backend see them together.
    let [q_p, k_p, v_p] = projections.qkv(w, x, width);
    let mut q = {
        let _t = timed(OpClass::KdaConv);
        short_conv(
            &q_p,
            w.q_conv1d,
            state.buffer_mut(CONV_Q).cells_mut(),
            width,
            g.conv_kernel,
        )
    };
    planes.q_conv.extend_from_slice(&q);
    {
        let _t = timed(OpClass::KdaQkNorm);
        if mutation != Mutation::NoQNorm {
            l2_normalise_heads(&mut q, heads, dim);
        }
    }
    planes.q_norm.extend_from_slice(&q);

    let mut k = {
        let _t = timed(OpClass::KdaConv);
        short_conv(
            &k_p,
            w.k_conv1d,
            state.buffer_mut(CONV_K).cells_mut(),
            width,
            g.conv_kernel,
        )
    };
    planes.k_conv.extend_from_slice(&k);
    {
        let _t = timed(OpClass::KdaQkNorm);
        if mutation != Mutation::NoKNorm {
            l2_normalise_heads(&mut k, heads, dim);
        }
    }
    planes.k_norm.extend_from_slice(&k);

    let mut v = {
        let _t = timed(OpClass::KdaConv);
        short_conv(
            &v_p,
            w.v_conv1d,
            state.buffer_mut(CONV_V).cells_mut(),
            width,
            g.conv_kernel,
        )
    };
    planes.v_conv.extend_from_slice(&v);

    // The decay gate. Everything from here is f32 by construction: the
    // gate is an exponential of a softplus, and the recurrence multiplies
    // by it every step, so a narrower accumulator compounds.
    let decay = {
        let _t = timed(OpClass::KdaDecayGate);
        let f_low = matvec(w.f_b_proj, &matvec(w.f_a_proj, x, w.gate_rank), width);
        planes.f_lowrank.extend_from_slice(&f_low);
        let mut decay = vec![0.0f32; width];
        for h in 0..heads {
            let a = w.a_log[h].exp();
            for d in 0..dim {
                let i = h * dim + d;
                let pre = f_low[i] + w.dt_bias[i];
                // The DECLARED form, unless a control overrides it. Both
                // overrides exist so the choice can be falsified from
                // either side.
                let form = match mutation {
                    Mutation::ApplyGateLowerBound(bound) => {
                        KdaGateForm::ClampedSigmoid { lower_bound: bound }
                    }
                    Mutation::ForceSoftplusGate => KdaGateForm::Softplus,
                    _ => w.gate_form,
                };
                decay[i] = match form {
                    KdaGateForm::Softplus => -a * softplus(pre),
                    // `lower_bound * sigmoid(exp(A_log) * pre)`, bounding
                    // the decay below at `exp(lower_bound)`.
                    KdaGateForm::ClampedSigmoid { lower_bound } => {
                        lower_bound * (1.0 / (1.0 + (-(a * pre)).exp()))
                    }
                };
            }
        }
        planes.g_decay.extend_from_slice(&decay);
        decay
    };

    // The output gate's PROJECTION, in the declared form. `project` times
    // the full-rank matvec itself under the same class, so only the
    // low-rank composition is timed here.
    let mut gate = match w.output_gate {
        KdaOutputGateWeights::LowRank { g_a_proj, g_b_proj } => {
            let _t = timed(OpClass::KdaOutputGate);
            matvec(g_b_proj, &matvec(g_a_proj, x, w.gate_rank), width)
        }
        KdaOutputGateWeights::FullRank { g_proj } => {
            project(OpClass::KdaOutputGate, g_proj, x, width)
        }
    };
    if mutation == Mutation::GateSkipped {
        gate.iter_mut().for_each(|g| *g = 0.0);
    }
    planes.o_gate.extend_from_slice(&gate);
    if mutation == Mutation::GateOnValueBeforeRecurrence {
        for (vi, gi) in v.iter_mut().zip(&gate) {
            *vi /= 1.0 + (-gi).exp();
        }
    }

    let beta: Vec<f32> = {
        let _t = timed(OpClass::KdaBProj);
        let beta: Vec<f32> = matvec(w.b_proj, x, heads)
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        planes.beta.extend_from_slice(&beta);
        beta
    };

    // The delta rule, per head. `q` is scaled by `D^-1/2` at the readout.
    //
    // FUSED, P4c-3: the reference is four full `D×D` state traversals —
    // decay-write, predict-read, update-write, readout-read — but two of
    // those pairs are FUSABLE without changing a single summed value.
    //
    // (1) decay + predict: `pred[vv] = Σ_kk kh[kk] * (decay[kk]*s_old[kk,vv])`.
    // Decaying row `kk` is entirely local to that row, so decaying it and
    // immediately folding its contribution into every `pred[vv]` — before
    // moving to row `kk+1` — reads EXACTLY the values a separate later
    // predict pass would have read, and accumulates them in the SAME
    // kk-ascending order the original `.sum()` did. Not an approximation:
    // the identical sequence of additions in the identical order is
    // bit-identical under IEEE-754.
    //
    // (2) write + readout: `out[vv] = Σ_kk qh[kk]*scale*s_new[kk,vv]` where
    // `s_new[kk,vv]` is written earlier in the SAME kk iteration — same
    // argument, same conclusion.
    //
    // `Mutation::ReadBeforeWrite` is the one case that CANNOT take fusion
    // (2): its entire point is that the readout must see the PRE-write
    // state, so write and readout stay two passes there, exactly as
    // before — falling back to the unfused form for one control is not a
    // performance regression on the decode path, since production
    // decoding never uses this mutation.
    let out = {
        let _t = timed(OpClass::KdaRecurrence);
        let scale = (dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; width];
        for h in 0..heads {
            let s =
                &mut state.buffer_mut(RECURRENT).cells_mut()[h * dim * dim..(h + 1) * dim * dim];
            let (qh, kh, vh) = (&q[h * dim..], &k[h * dim..], &v[h * dim..]);

            let mut pred = vec![0.0f32; dim];
            for kk in 0..dim {
                if mutation != Mutation::NoDecay {
                    let d = decay[h * dim + kk].exp();
                    for vv in 0..dim {
                        s[kk * dim + vv] *= d;
                    }
                }
                let kv = kh[kk];
                for vv in 0..dim {
                    pred[vv] += kv * s[kk * dim + vv];
                }
            }
            // The prediction error `v - kᵀS`, which is what the delta
            // rule writes against. Writing `v` instead agrees at T=1
            // from a zero state — see `Mutation::WriteValueNotError`.
            let mut err = vec![0.0f32; dim];
            for vv in 0..dim {
                err[vv] = match mutation {
                    Mutation::WriteValueNotError => vh[vv],
                    _ => vh[vv] - pred[vv],
                };
            }
            let b = if mutation == Mutation::NoBeta {
                1.0
            } else {
                beta[h]
            };

            if mutation == Mutation::ReadBeforeWrite {
                for vv in 0..dim {
                    out[h * dim + vv] = (0..dim).map(|kk| qh[kk] * scale * s[kk * dim + vv]).sum();
                }
                for kk in 0..dim {
                    let write = b * kh[kk];
                    for vv in 0..dim {
                        let cell = &mut s[kk * dim + vv];
                        *cell += write * err[vv];
                        if mutation == Mutation::Bf16Recurrence {
                            *cell = bf16_round(*cell);
                        }
                    }
                }
            } else {
                for kk in 0..dim {
                    let write = b * kh[kk];
                    let qv = qh[kk] * scale;
                    for vv in 0..dim {
                        let cell = &mut s[kk * dim + vv];
                        *cell += write * err[vv];
                        if mutation == Mutation::Bf16Recurrence {
                            *cell = bf16_round(*cell);
                        }
                        out[h * dim + vv] += qv * *cell;
                    }
                }
            }
        }
        planes.recurrent_out.extend_from_slice(&out);
        out
    };

    // `gate` was already computed above — it depends only on `x`, never
    // on `out`.

    // Gated RMSNorm: normalise over ONE head's width, scale by the
    // weight, then gate by `sigmoid(gate)`.
    let normed = {
        let _t = timed(OpClass::KdaGatedNorm);
        let mut normed = vec![0.0f32; width];
        // `GateBeforeNorm` gates `out` first and norms the gated vector;
        // every other arm norms `out` and applies the gate factor after.
        let pre: Vec<f32> = if mutation == Mutation::GateBeforeNorm {
            out.iter()
                .zip(&gate)
                .map(|(o, g)| o / (1.0 + (-g).exp()))
                .collect()
        } else {
            out.clone()
        };
        for h in 0..heads {
            let slice = &pre[h * dim..(h + 1) * dim];
            let ms = slice.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = (ms + w.norm_eps).sqrt().recip();
            for (d, (sv, nv)) in slice.iter().zip(w.o_norm).enumerate() {
                let i = h * dim + d;
                let factor = match mutation {
                    Mutation::GateBeforeNorm | Mutation::GateOnValueBeforeRecurrence => 1.0,
                    Mutation::SigmoidOmitted => gate[i],
                    _ => 1.0 / (1.0 + (-gate[i]).exp()),
                };
                normed[i] = sv * inv * nv * factor;
            }
        }
        planes.o_norm.extend_from_slice(&normed);
        normed
    };

    planes.q_proj.extend_from_slice(&q_p);
    planes.k_proj.extend_from_slice(&k_p);
    planes.v_proj.extend_from_slice(&v_p);

    // `o_proj` is `[hidden, Hv·Dv]`, so its output width is the hidden
    // width this position arrived at — read from the ACTIVATION, never
    // from the weight's slice length: a resident slab is page-padded and
    // a compact representation is not even f32-shaped, so `len / width`
    // was only ever right for one residency.
    let y = projections.o(w.o_proj, &normed, x.len());
    planes.output.extend_from_slice(&y);
    y
}

/// A whole sequence through the block, from `state`.
pub fn layer_forward(
    x: &[f32],
    hidden: usize,
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    mutation: Mutation,
) -> KdaPlanes {
    layer_forward_with(&CpuKdaProjections, x, hidden, w, g, state, mutation)
}

/// [`layer_forward`] with the projections executed somewhere the caller
/// chooses. Everything but those four matvecs is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_with(
    projections: &dyn KdaProjections,
    x: &[f32],
    hidden: usize,
    w: KdaWeights<'_>,
    g: KdaGeometry,
    state: &mut RecurrentState,
    mutation: Mutation,
) -> KdaPlanes {
    let mut planes = KdaPlanes::default();
    for pos in x.chunks_exact(hidden) {
        step_with(projections, pos, w, g, state, &mut planes, mutation);
    }
    planes
}
