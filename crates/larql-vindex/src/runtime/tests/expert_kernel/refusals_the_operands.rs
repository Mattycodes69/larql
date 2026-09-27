//! Refusals: the operands
//! The session

use super::*;

#[test]
fn a_decomposed_projection_is_refused_by_arrangement() {
    // Remedy: bind the fused variant, or a kernel that takes two regions.
    // Stitching them into one temporary would make the parity result a
    // statement about the stitching.
    let fixture = Fixture::new();
    let half = fixture.gate_up.len() / FUSED_PROJECTION_HALVES;
    let expert = BoundExpert {
        expert_id: 0,
        projection: BoundProjection::Decomposed {
            gate: direct(
                GATE_UP_REGION_SET,
                &fixture.gate_up[..half],
                RegionFormat::Q4K,
                INTERMEDIATE as u32,
                HIDDEN as u32,
            ),
            up: direct(
                GATE_UP_REGION_SET,
                &fixture.gate_up[half..],
                RegionFormat::Q4K,
                INTERMEDIATE as u32,
                HIDDEN as u32,
            ),
        },
        down: fixture.q4k_expert().down,
    };
    let err = fixture
        .operation(expert, ExpertKernel::IncumbentQ4kQ8k)
        .validate()
        .unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::Arrangement { .. },
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn dequantised_operands_are_refused_by_format() {
    // Remedy: bind the Q4_K variant, or the reference kernel. Reading f32 bytes
    // as super-blocks is exactly the substitution this refuses.
    let fixture = Fixture::new();
    let err = fixture
        .operation(fixture.f32_expert(), ExpertKernel::IncumbentQ4kQ8k)
        .validate()
        .unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::ElementFormat { .. },
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn a_down_region_padded_to_the_wrong_width_is_refused_by_the_kernel() {
    // Shape-valid — the role still exposes `INTERMEDIATE` columns — but stored
    // at a width the kernel would not stride by. Only the kernel binding can
    // catch this one, which is why the stored extent travels with the operand.
    let fixture = Fixture::new();
    let wide = INTERMEDIATE_PADDED + Q4K_BLOCK_ELEMS;
    let bytes = q4k_bytes(HIDDEN, wide, DOWN_SEED);
    let expert = BoundExpert {
        expert_id: 0,
        down: column_prefix(
            DOWN_REGION_SET,
            &bytes,
            RegionFormat::Q4K,
            HIDDEN as u32,
            wide as u32,
            INTERMEDIATE as u32,
        ),
        ..fixture.q4k_expert()
    };
    let err = fixture
        .operation(expert, ExpertKernel::IncumbentQ4kQ8k)
        .validate()
        .unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::DimensionMismatch { axis, expected, found, .. }
                if axis == Axis::InputWidth && expected == INTERMEDIATE_PADDED && found == wide
        ),
        "{err}"
    );
}

#[test]
fn a_bank_input_that_is_not_whole_blocks_is_refused_at_execution() {
    // The activation-side counterpart. The incumbent guards this by falling
    // back to its f32 path; VINDEX3 refuses, because a silent change of kernel
    // is a silent change of answer.
    //
    // It fires at execution rather than at bind because the bank input is a
    // property of the token, not of the binding.
    let ragged = HIDDEN - 1;
    let gate_up = q4k_bytes(FUSED_PROJECTION_HALVES * INTERMEDIATE, HIDDEN, GATE_UP_SEED);
    let down = q4k_bytes(HIDDEN, INTERMEDIATE_PADDED, DOWN_SEED);
    let router = f32_bytes(&vec![1.0f32; POPULATION * ragged]);
    let operation = BoundMoeOperation {
        router: BoundRouter {
            weight: direct(
                ROUTER_REGION_SET,
                &router,
                RegionFormat::F32,
                POPULATION as u32,
                ragged as u32,
            ),
            top_k: TOP_K,
            selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
            scaling: BoundExpertScaling::None,
            kernel: RouterKernel::default(),
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(LAYER, BANK_ID),
            experts: vec![BoundExpert {
                expert_id: 0,
                projection: BoundProjection::Fused {
                    gate_up: direct(
                        GATE_UP_REGION_SET,
                        &gate_up,
                        RegionFormat::Q4K,
                        (FUSED_PROJECTION_HALVES * INTERMEDIATE) as u32,
                        ragged as u32,
                    ),
                },
                down: column_prefix(
                    DOWN_REGION_SET,
                    &down,
                    RegionFormat::Q4K,
                    ragged as u32,
                    INTERMEDIATE_PADDED as u32,
                    INTERMEDIATE as u32,
                ),
            }],
            intermediate_dim: INTERMEDIATE,
            hidden_dim: ragged,
            activation: ACTIVATION,
            kernel: ExpertKernel::IncumbentQ4kQ8k,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: ragged,
    };

    let input = vec![0.1f32; ragged];
    let err = execute_traced(&operation, MoeInputs::shared(&input)).unwrap_err();
    assert!(
        matches!(
            err,
            ExecutionError::KernelOperandUnsuitable {
                reason: OperandUnsuitability::BlockAlignment { found, .. },
                ..
            } if found == ragged
        ),
        "{err}"
    );
}

#[test]
fn every_expert_in_a_bank_reads_the_one_quantised_activation() {
    // Two experts over identical weights, both selected. Each must reproduce
    // the incumbent's output for the *bank's* input — which is what the shared
    // Q8_K session buys, and what a per-expert quantisation would only
    // accidentally match.
    let fixture = Fixture::new();
    let mut operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    let second = BoundExpert {
        expert_id: 1,
        ..fixture.q4k_expert()
    };
    operation.banks[0].experts.push(second);
    operation.router.top_k = POPULATION;
    operation.validate().expect("both experts bind");

    let (_, trace) = execute_traced(&operation, MoeInputs::shared(&fixture.input))
        .expect("the fixture executes");
    let incumbent = fixture.incumbent_expert_output();
    assert_eq!(trace.expert_outputs.len(), POPULATION);
    for output in &trace.expert_outputs {
        assert_eq!(
            output.values, incumbent,
            "expert {} read a different activation",
            output.expert_id
        );
    }
}
