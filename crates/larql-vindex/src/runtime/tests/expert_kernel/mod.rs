//! Tests for `expert_kernel` — binding the production expert kernel.
//!
//! The headline is [`the_bound_kernel_reproduces_the_incumbent_call_bit_for_bit`].
//! Everything else exists so that a bit-identical result cannot be a
//! bit-identical *mistake*: two calls of one function agree whatever operands
//! they are handed, so the refusals below are what establish that the operands
//! reaching it are the right ones.

use larql_compute::cpu::ops::moe::{
    quantize_x_to_q8k, run_single_expert_q4k_q8k_into, ExpertScratch,
};
use larql_compute::cpu::ops::q4_common::dequantize_q4_k;
use larql_compute::{Activation, MoeTopKWeightPolicy};

use crate::format::capability::binding::{ComponentView, RepresentationIdentity};
use crate::format::capability::component::ComponentContract;
use crate::format::capability::coordinate::BankCoordinate;
use crate::format::lyrw2::region_format::RegionFormat;

use crate::runtime::axis::Axis;
use crate::runtime::bank::{BoundBankOperation, BoundExpert};
use crate::runtime::consts::{COL_DIM, FUSED_PROJECTION_HALVES};
use crate::runtime::error::{ExecutionError, OperandUnsuitability};
use crate::runtime::execute::execute_traced;
use crate::runtime::expert_kernel::ExpertKernel;
use crate::runtime::inputs::MoeInputs;
use crate::runtime::operation::BoundMoeOperation;
use crate::runtime::projection::BoundProjection;
use crate::runtime::reduction::BoundReduction;
use crate::runtime::router::{BoundExpertScaling, BoundRouter, RouterKernel};
use crate::runtime::tensor::BoundTensor;

use super::support::{q4k_bytes, Q4K_BLOCK_ELEMS};

/// One super-block wide, so the Q8_K activation quantiser is satisfied and the
/// fixture stays small enough to reason about.
const HIDDEN: usize = Q4K_BLOCK_ELEMS;
/// Half a super-block, so `down` is stored padded — the shape every real
/// k-quant MoE layer has, and the one a kernel binding gets wrong.
const INTERMEDIATE: usize = Q4K_BLOCK_ELEMS / 2;
const INTERMEDIATE_PADDED: usize = Q4K_BLOCK_ELEMS;
const POPULATION: usize = 2;
const TOP_K: usize = 1;
const LAYER: u32 = 0;
const BANK_ID: u16 = 0;
/// Gemma's activation, and one of the two the incumbent kernel implements.
const ACTIVATION: Activation = Activation::GeluTanh;

const VARIANT: &str = "test";
const GATE_UP_REGION_SET: &str = "gate_up_fused";
const DOWN_REGION_SET: &str = "down";
const ROUTER_REGION_SET: &str = "router";

const GATE_UP_SEED: usize = 1;
const DOWN_SEED: usize = 7;

/// The reference dequantises to f32 while the incumbent keeps an integer dot
/// against a Q8_K activation, so the two agree to quantisation noise, not to
/// the bit. A band, not a target.
const KERNEL_TOLERANCE: f32 = 5e-2;

fn identity(region_set: &str) -> RepresentationIdentity {
    RepresentationIdentity::new(region_set, VARIANT)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn direct<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    rows: u32,
    cols: u32,
) -> BoundTensor<'a> {
    BoundTensor::direct(
        identity(region_set),
        bytes,
        format,
        ComponentContract::matrix(rows, cols),
    )
    .expect("well-formed fixture operand")
}

fn column_prefix<'a>(
    region_set: &str,
    bytes: &'a [u8],
    format: RegionFormat,
    rows: u32,
    stored_cols: u32,
    role_cols: u32,
) -> BoundTensor<'a> {
    BoundTensor::new(
        identity(region_set),
        bytes,
        format,
        ComponentContract::matrix(rows, stored_cols),
        ComponentView::Slice {
            dim: COL_DIM,
            start: 0,
            len: role_cols,
        },
    )
    .expect("well-formed fixture operand")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Q4_K bytes, plus the same values dequantised, so both kernels can be bound
/// over one set of weights.
struct Fixture {
    gate_up: Vec<u8>,
    gate_up_f32: Vec<u8>,
    down: Vec<u8>,
    down_f32: Vec<u8>,
    router: Vec<u8>,
    input: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let gate_up = q4k_bytes(FUSED_PROJECTION_HALVES * INTERMEDIATE, HIDDEN, GATE_UP_SEED);
        let down = q4k_bytes(HIDDEN, INTERMEDIATE_PADDED, DOWN_SEED);
        let gate_up_f32 = f32_bytes(&dequantize_q4_k(
            &gate_up,
            FUSED_PROJECTION_HALVES * INTERMEDIATE * HIDDEN,
        ));
        let down_f32 = f32_bytes(&dequantize_q4_k(&down, HIDDEN * INTERMEDIATE_PADDED));

        // A signed input, so the kernel exercises both halves of the 4-bit
        // range rather than a one-sided ramp.
        let input: Vec<f32> = (0..HIDDEN).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();

        // Expert 0's router row *is* the input, so its logit is ‖x‖² while
        // expert 1's is zero. An earlier fixture filled row 0 with ones, which
        // scored `sum(x)` — and this input sums to approximately zero, so the
        // two logits tied and selection fell to whichever side the float
        // residue landed. A fixture that is accidentally testing a tie is not
        // testing the kernel.
        let mut router = vec![0.0f32; POPULATION * HIDDEN];
        router[..HIDDEN].copy_from_slice(&input);
        Self {
            gate_up,
            gate_up_f32,
            down,
            down_f32,
            router: f32_bytes(&router),
            input,
        }
    }

    /// The store's own Q4_K blocks, `down` bound at its stored padded width
    /// with the role seeing the live columns.
    fn q4k_expert(&self) -> BoundExpert<'_> {
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &self.gate_up,
                    RegionFormat::Q4K,
                    (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                    HIDDEN as u32,
                ),
            },
            down: column_prefix(
                DOWN_REGION_SET,
                &self.down,
                RegionFormat::Q4K,
                HIDDEN as u32,
                INTERMEDIATE_PADDED as u32,
                INTERMEDIATE as u32,
            ),
        }
    }

    /// The same weights dequantised, for the reference kernel.
    fn f32_expert(&self) -> BoundExpert<'_> {
        BoundExpert {
            expert_id: 0,
            projection: BoundProjection::Fused {
                gate_up: direct(
                    GATE_UP_REGION_SET,
                    &self.gate_up_f32,
                    RegionFormat::F32,
                    (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                    HIDDEN as u32,
                ),
            },
            down: column_prefix(
                DOWN_REGION_SET,
                &self.down_f32,
                RegionFormat::F32,
                HIDDEN as u32,
                INTERMEDIATE_PADDED as u32,
                INTERMEDIATE as u32,
            ),
        }
    }

    fn operation<'a>(
        &'a self,
        expert: BoundExpert<'a>,
        kernel: ExpertKernel,
    ) -> BoundMoeOperation<'a> {
        self.operation_with(expert, kernel, ACTIVATION)
    }

    fn operation_with<'a>(
        &'a self,
        expert: BoundExpert<'a>,
        kernel: ExpertKernel,
        activation: Activation,
    ) -> BoundMoeOperation<'a> {
        BoundMoeOperation {
            router: BoundRouter {
                weight: direct(
                    ROUTER_REGION_SET,
                    &self.router,
                    RegionFormat::F32,
                    POPULATION as u32,
                    HIDDEN as u32,
                ),
                top_k: TOP_K,
                selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
                scaling: BoundExpertScaling::None,
                kernel: RouterKernel::default(),
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(LAYER, BANK_ID),
                experts: vec![expert],
                intermediate_dim: INTERMEDIATE,
                hidden_dim: HIDDEN,
                activation,
                kernel,
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: HIDDEN,
        }
    }

    /// The incumbent's own function, called directly on the same bytes.
    fn incumbent_expert_output(&self) -> Vec<f32> {
        let q8k = quantize_x_to_q8k(&self.input);
        let mut scratch = ExpertScratch::new(HIDDEN, INTERMEDIATE, INTERMEDIATE_PADDED);
        run_single_expert_q4k_q8k_into(
            &mut scratch,
            &q8k,
            &self.gate_up,
            &self.down,
            INTERMEDIATE,
            larql_compute::ExpertMlp::gated(ACTIVATION),
        )
        .to_vec()
    }

    fn run(&self, operation: &BoundMoeOperation<'_>) -> Vec<f32> {
        let (_, trace) = execute_traced(operation, MoeInputs::shared(&self.input))
            .expect("the fixture executes");
        trace
            .expert_output(0)
            .expect("expert 0 was selected")
            .to_vec()
    }
}

//
// `execute` does not call `validate` — re-checking a binding per token is the
// resolution creep the bound object exists to prevent. So the kernel's operand
// checks have to hold on their own, and these reach them by executing an
// operation that was never validated. Every one of them would otherwise be a
// stride read at the wrong offset producing well-shaped noise.

/// Run without validating first, and return the refusal.
fn execute_unvalidated(fixture: &Fixture, expert: BoundExpert<'_>) -> ExecutionError {
    let operation = fixture.operation(expert, ExpertKernel::IncumbentQ4kQ8k);
    execute_traced(&operation, MoeInputs::shared(&fixture.input))
        .expect_err("the kernel should refuse this operand")
}

mod refusals_the_operands;
mod the_claim;
mod the_kernel_checks_its_own_operands_even;
