//! Slow, naive reference kernels the backends are compared against.

use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::Activation;

#[allow(unused_imports)]
use super::*;

// ── Reference impls (slow + naive) ────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn reference_one_expert(
    h: &[f32],
    gu_bytes: &[u8],
    dn_bytes: &[u8],
    hidden: usize,
    inter: usize,
    inter_padded: usize,
    pre_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    activation: Activation,
    verbose: bool,
) -> Vec<f32> {
    let h_norm = naive_rms_norm(h, pre_norm, eps, norm_offset);
    if verbose {
        dump3("ref h_norm", &h_norm);
    }
    let gate_up_w = dequantize_q4_k(gu_bytes, 2 * inter * hidden);
    let down_w = dequantize_q4_k(dn_bytes, hidden * inter_padded);

    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

    let gate_out = naive_matvec(&h_norm, gate_w, inter, hidden);
    let up_out = naive_matvec(&h_norm, up_w, inter, hidden);
    if verbose {
        dump3("ref gate_out", &gate_out);
        dump3("ref up_out  ", &up_out);
    }

    let mut hidden_state = vec![0.0f32; inter_padded];
    for j in 0..inter {
        hidden_state[j] = match activation {
            Activation::GeluTanh => naive_gelu_tanh(gate_out[j]) * up_out[j],
            _ => naive_silu(gate_out[j]) * up_out[j],
        };
    }
    naive_matvec(&hidden_state, &down_w, hidden, inter_padded)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reference_moe_block(
    h: &[f32],
    experts_gate_up: &[&[u8]],
    experts_down: &[&[u8]],
    router_proj: &[f32],
    router_per_expert_scale: &[f32],
    router_norm: &[f32],
    router_norm_parameter_free: bool,
    router_input_scalar: f32,
    pre_norm: &[f32],
    post_norm: &[f32],
    hidden: usize,
    inter: usize,
    inter_padded: usize,
    num_experts: usize,
    top_k: usize,
    activation: Activation,
    norm_offset: f32,
    eps: f32,
    verbose: bool,
) -> Vec<f32> {
    // 1. Pre-experts norm — for the expert matmuls.
    let h_norm = naive_rms_norm(h, pre_norm, eps, norm_offset);
    if verbose {
        dump3("ref h_norm        ", &h_norm);
    }

    // 2. Router input norm — applied to h_norm (matching Metal's
    //    `cpu_moe_route(&h_norm, ...)` and the routing-convention fix
    //    in `cpu_moe_forward`). Empirically the trained 26B-A4B weights
    //    expect this even though HF's modeling_gemma4.py uses raw h.
    let router_in_normed = if !router_norm.is_empty() {
        naive_rms_norm(&h_norm, router_norm, eps, norm_offset)
    } else if router_norm_parameter_free {
        naive_rms_norm(&h_norm, &[], eps, 0.0)
    } else {
        h_norm.clone()
    };
    let mut router_in = router_in_normed;
    if router_input_scalar != 1.0 && router_input_scalar != 0.0 {
        for v in router_in.iter_mut() {
            *v *= router_input_scalar;
        }
    }
    if verbose {
        dump3("ref router_in     ", &router_in);
    }

    // 3. Router projection [hidden → num_experts].
    let mut logits = naive_matvec(&router_in, router_proj, num_experts, hidden);
    naive_softmax(&mut logits);

    // 4. Top-K + renormalisation.
    let (indices, mut weights) = naive_top_k(&logits, top_k);
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    if !router_per_expert_scale.is_empty() {
        for (i, &ei) in indices.iter().enumerate() {
            if ei < router_per_expert_scale.len() {
                weights[i] *= router_per_expert_scale[ei];
            }
        }
    }
    if verbose {
        println!(
            "  ref top_k indices: {:?}  weights: {:?}",
            indices,
            weights
                .iter()
                .map(|w| format!("{w:.4}"))
                .collect::<Vec<_>>()
        );
    }

    // 5. Sum K weighted expert outputs.
    let mut moe_out = vec![0.0f32; hidden];
    for (k, &ei) in indices.iter().enumerate() {
        let w = weights[k];
        if w == 0.0 {
            continue;
        }
        let contrib = reference_one_expert(
            h,
            experts_gate_up[ei],
            experts_down[ei],
            hidden,
            inter,
            inter_padded,
            pre_norm,
            norm_offset,
            eps,
            activation,
            false,
        );
        for (acc, &v) in moe_out.iter_mut().zip(contrib.iter()) {
            *acc += w * v;
        }
    }
    if verbose {
        dump3("ref pre-post-norm ", &moe_out);
    }

    // 6. Post-experts norm.
    if !post_norm.is_empty() {
        moe_out = naive_rms_norm(&moe_out, post_norm, eps, norm_offset);
    }
    moe_out
}

/// Run only the routing portion of the MoE block — return top-K indices +
/// renormalised weights. Used by the routing-convention diff to expose
/// whether two router-input variants pick different experts.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_top_k(
    router_in_pre: &[f32],
    router_proj: &[f32],
    router_per_expert_scale: &[f32],
    router_norm: &[f32],
    router_norm_parameter_free: bool,
    router_input_scalar: f32,
    num_experts: usize,
    top_k: usize,
    hidden: usize,
    eps: f32,
    norm_offset: f32,
) -> (Vec<usize>, Vec<f32>) {
    let router_in_normed = if !router_norm.is_empty() {
        naive_rms_norm(router_in_pre, router_norm, eps, norm_offset)
    } else if router_norm_parameter_free {
        naive_rms_norm(router_in_pre, &[], eps, 0.0)
    } else {
        router_in_pre.to_vec()
    };
    let mut router_in = router_in_normed;
    if router_input_scalar != 1.0 && router_input_scalar != 0.0 {
        for v in router_in.iter_mut() {
            *v *= router_input_scalar;
        }
    }
    let mut logits = naive_matvec(&router_in, router_proj, num_experts, hidden);
    naive_softmax(&mut logits);
    let (indices, mut weights) = naive_top_k(&logits, top_k);
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    if !router_per_expert_scale.is_empty() {
        for (i, &ei) in indices.iter().enumerate() {
            if ei < router_per_expert_scale.len() {
                weights[i] *= router_per_expert_scale[ei];
            }
        }
    }
    (indices, weights)
}

// ── Naive primitives (f64 accumulators, no BLAS) ──────────────────────────────

pub(super) fn naive_matvec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_rows];
    for r in 0..out_rows {
        let mut s = 0.0f64;
        for c in 0..in_cols {
            s += (w[r * in_cols + c] as f64) * (x[c] as f64);
        }
        out[r] = s as f32;
    }
    out
}

pub(super) fn naive_rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let rms = (x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64 + eps as f64)
        .sqrt() as f32;
    if w.is_empty() {
        return x.iter().map(|v| v / rms).collect();
    }
    x.iter()
        .zip(w.iter())
        .map(|(v, ww)| (v / rms) * (ww + offset))
        .collect()
}

pub(super) fn naive_softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    if sum > 0.0 {
        let inv = (1.0 / sum) as f32;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

pub(super) fn naive_top_k(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(k);
    let weights: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
    (idx, weights)
}

pub(super) fn naive_gelu_tanh(x: f32) -> f32 {
    // sqrt(2 / π); precision capped at f32 range — the tanh approx
    // already saturates well before more digits would matter.
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

pub(super) fn naive_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
