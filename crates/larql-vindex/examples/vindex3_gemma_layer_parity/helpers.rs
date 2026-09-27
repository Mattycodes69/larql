//! Row readers, diff reports and operation binding for the layer parity ladder.

use larql_compute::cpu::ops::moe::{
    quantize_x_to_q8k, run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::MoeLayerWeights;
use larql_vindex::format::capability::binding::{ComponentView, RepresentationIdentity};
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::runtime::consts::COL_DIM;
use larql_vindex::runtime::{
    BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation, BoundReduction,
    BoundRouter, BoundTensor, ExpertKernel, RouterKernel,
};

#[allow(unused_imports)]
use super::*;

/// Read an f32 dump and return its final row — the token being decoded.
pub(super) fn last_row(path: &str, width: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if !values.len().is_multiple_of(width) || values.is_empty() {
        return Err(format!(
            "{path}: {} values is not a whole number of {width}-wide rows",
            values.len()
        ));
    }
    let rows = values.len() / width;
    Ok(values[(rows - 1) * width..].to_vec())
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub(super) fn report(label: &str, a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        println!(
            "  {label:<LADDER_LABEL_WIDTH$} LENGTH {} vs {}",
            a.len(),
            b.len()
        );
        return false;
    }
    let diff = max_abs_diff(a, b);
    let ok = diff <= WEIGHT_TOLERANCE;
    println!(
        "  {label:<LADDER_LABEL_WIDTH$} max|Δ| = {diff:.3e}  {}",
        if ok { "within tolerance" } else { "OUTSIDE" }
    );
    ok
}

/// Largest absolute value in a vector — the scale a difference is judged against.
pub(super) fn magnitude(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

/// Compare two vectors relative to the **reference's** own magnitude.
///
/// An absolute band would have to be re-derived for every layer, because a
/// residual-scale quantity's size is a property of the layer rather than of the
/// arithmetic. The ratio is the same question asked in units that travel.
///
/// # The denominator is fixed, and it is stated
///
/// `max|reference|`, never `max|found|` and never the larger of the two. A
/// denominator that moved with the thing being measured would let a kernel that
/// inflates its output report a *smaller* percentage for a larger error, and
/// would make two layers' figures incomparable. The oracle is the fixed point;
/// the bound path is what is being judged against it.
///
/// # Zero-near-zero is a separate verdict, not a divide
///
/// Below [`ORACLE_SCALE_FLOOR`] the ratio stops meaning anything — a reference
/// of 1e-9 turns any rounding difference into thousands of percent — so this
/// reports the absolute difference against the floor and labels it, rather than
/// dividing and emitting a number that looks like the others but is not
/// comparable to them. A layer whose output is genuinely near zero should read
/// as such rather than as a catastrophic relative error.
pub(super) fn report_relative(label: &str, reference: &[f32], found: &[f32]) -> bool {
    if reference.len() != found.len() {
        println!(
            "  {label:<LADDER_LABEL_WIDTH$} LENGTH {} vs {}",
            reference.len(),
            found.len()
        );
        return false;
    }
    let diff = max_abs_diff(reference, found);
    let scale = magnitude(reference);
    if scale < ORACLE_SCALE_FLOOR {
        let ok = diff < ORACLE_SCALE_FLOOR;
        println!(
            "  {label:<LADDER_LABEL_WIDTH$} max|Δ| = {diff:.3e}, reference magnitude \
             {scale:.3e} below the {ORACLE_SCALE_FLOOR:.0e} floor — absolute, not a ratio  {}",
            if ok { "within floor" } else { "OUTSIDE" }
        );
        return ok;
    }
    let relative = diff / scale;
    let ok = relative <= ORACLE_RELATIVE_BAND;
    println!(
        "  {label:<LADDER_LABEL_WIDTH$} max|Δ| = {diff:.3e} of {scale:.3e} = {:.2}%  {}",
        relative * 100.0,
        if ok { "within band" } else { "OUTSIDE" }
    );
    ok
}

/// Bind one expert's Q4_K bytes as dequantised f32.
///
/// Materialising rather than binding Q4_K directly: the reference decoder
/// implements the directly-readable encodings only, and a quantised region is
/// a missing kernel rather than bad bytes. The incumbent's own fallback path
/// dequantises the same way, which is what makes the two comparable at all.
pub(super) fn dequantised(
    role: RegionRole,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, ComponentContract), String> {
    let values = dequantize_q4_k(bytes, rows * cols);
    if values.len() < rows * cols {
        return Err(format!(
            "{}: dequantised {} of {} expected elements",
            role.name(),
            values.len(),
            rows * cols
        ));
    }
    let raw: Vec<u8> = values[..rows * cols]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    Ok((raw, ComponentContract::matrix(rows as u32, cols as u32)))
}

pub(super) fn tensor<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    contract: ComponentContract,
) -> Result<BoundTensor<'a>, String> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        format,
        contract,
    )
    .map_err(|e| e.to_string())
}

/// Bind a stored matrix whose trailing columns are quantisation padding.
///
/// The role sees `[rows, keep]`; the bytes remain `[rows, stored_cols]`. No
/// repacking, no copy — the view resolves the difference at read time.
///
/// Both kernels read this one binding and take from it what each needs: the
/// reference reads the `keep` live columns and never touches the padding,
/// while the Q4_K kernel — which decodes whole super-blocks and cannot stop
/// inside one — reads all `stored_cols` and lets the zero-padded activation
/// cancel the rest. That is why the view belongs to the operand rather than
/// being something each kernel is told separately.
pub(super) fn sliced_tensor<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    storage: ComponentContract,
    keep: usize,
) -> Result<BoundTensor<'a>, String> {
    BoundTensor::new(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        format,
        storage,
        ComponentView::Slice {
            dim: COL_DIM,
            start: 0,
            len: keep as u32,
        },
    )
    .map_err(|e| e.to_string())
}

/// Owns the dequantised expert buffers so bound tensors can borrow them.
pub(super) struct ExpertBuffers {
    pub(super) expert_id: u32,
    pub(super) gate_up: Vec<u8>,
    pub(super) down: Vec<u8>,
    pub(super) gate_up_contract: ComponentContract,
    pub(super) down_contract: ComponentContract,
}

/// Assemble the operation around one bank of experts.
///
/// Written once and called twice. Two hand-written literals meant to differ in
/// exactly one field is how a parity harness acquires a second difference
/// nobody notices.
pub(super) fn bind_operation<'a>(
    layer: usize,
    hidden: usize,
    moe: &MoeLayerWeights<'_>,
    router_bytes: &'a [u8],
    scale_bytes: &'a [u8],
    experts: Vec<BoundExpert<'a>>,
    kernel: ExpertKernel,
) -> Result<BoundMoeOperation<'a>, String> {
    Ok(BoundMoeOperation {
        router: BoundRouter {
            weight: tensor(
                ROUTER_REGION_SET,
                router_bytes,
                RegionFormat::F32,
                ComponentContract::matrix(moe.num_experts as u32, hidden as u32),
            )?,
            top_k: moe.top_k,
            selected_weight: moe.routing_policy.selected_weight,
            scaling: if scale_bytes.is_empty() {
                BoundExpertScaling::None
            } else {
                BoundExpertScaling::PerExpert {
                    scales: tensor(
                        PER_EXPERT_SCALE_REGION_SET,
                        scale_bytes,
                        RegionFormat::F32,
                        ComponentContract::vector(moe.num_experts as u32),
                    )?,
                }
            },
            // Rung 1: bind the production scoring kernel, not a lookalike.
            kernel: RouterKernel::Incumbent,
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(layer as u32, BANK_ID),
            experts,
            intermediate_dim: moe.intermediate_size,
            hidden_dim: hidden,
            activation: match moe.gate_rule {
                larql_compute::MoeGateRule::Gated(a) => a,
                rule => {
                    return Err(format!(
                        "bound MoE kernels are gated-only, layer declares {rule:?}"
                    ))
                }
            },
            kernel,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: hidden,
    })
}

/// Run the incumbent's own expert kernel over each selected expert, in
/// selection order, sharing one Q8_K activation exactly as production does.
pub(super) fn incumbent_expert_outputs(
    moe: &MoeLayerWeights<'_>,
    expert_input: &[f32],
    selected: &[usize],
) -> Result<Vec<Vec<f32>>, String> {
    let hidden = expert_input.len();
    let inter = moe.intermediate_size;
    let q8k = quantize_x_to_q8k(expert_input);
    let mut scratch = ExpertScratch::new(hidden, inter, moe.inter_padded());
    selected
        .iter()
        .map(|&e| {
            let gate_up = *moe
                .experts_gate_up
                .get(e)
                .ok_or_else(|| format!("expert {e} has no gate_up bytes"))?;
            let down = *moe
                .experts_down
                .get(e)
                .ok_or_else(|| format!("expert {e} has no down bytes"))?;
            Ok(run_single_expert_q4k_q8k_into(
                &mut scratch,
                &q8k,
                gate_up,
                down,
                inter,
                moe.expert_mlp(e),
            )
            .to_vec())
        })
        .collect()
}

/// `sum_i weight_i * output_i`, in selection order.
///
/// The incumbent's parallel paths reach this same sum through a rayon
/// tree-reduce or a spin-pool slot scan, neither of which fixes an order. So
/// the reduction is compared against *this* — the incumbent's own per-expert
/// outputs, combined in the order VINDEX3 combines them — which isolates the
/// combine from the schedule. Blaming a scheduling difference on the reduction
/// is exactly the false diagnosis a ladder exists to prevent.
pub(super) fn weighted_sum(outputs: &[Vec<f32>], weights: &[f32], width: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; width];
    for (out, &w) in outputs.iter().zip(weights) {
        for (slot, &v) in acc.iter_mut().zip(out) {
            *slot += w * v;
        }
    }
    acc
}

/// Print one ladder rung and say whether it was bit-identical.
pub(super) fn rung(step: usize, label: &str, a: &[f32], b: &[f32]) -> bool {
    let identical = a == b;
    let verdict = if identical {
        "BIT-IDENTICAL".to_string()
    } else if a.len() == b.len() {
        format!("max|Δ| = {:.3e}", max_abs_diff(a, b))
    } else {
        format!("LENGTH {} vs {}", a.len(), b.len())
    };
    println!("  {step} {label:<LADDER_LABEL_WIDTH$} {verdict}");
    identical
}
