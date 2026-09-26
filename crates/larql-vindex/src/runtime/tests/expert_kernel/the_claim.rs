//! The claim
//! Naming
//! Refusals: the activation

use super::*;

#[test]
fn the_bound_kernel_reproduces_the_incumbent_call_bit_for_bit() {
    // The rung's whole point, at unit scale and without a checkpoint: the same
    // Q4_K bytes, the same activation, the same production function — reached
    // through a bound operation rather than by calling it directly.
    let fixture = Fixture::new();
    let operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    operation.validate().expect("the fixture binds");

    let bound = fixture.run(&operation);
    let incumbent = fixture.incumbent_expert_output();

    assert_eq!(bound.len(), HIDDEN);
    assert_eq!(
        bound,
        incumbent,
        "bound kernel diverged from the incumbent call by {}",
        max_abs_diff(&bound, &incumbent)
    );
}

#[test]
fn the_bound_kernel_is_not_producing_zeros() {
    // Guards the guard. The incumbent's short-slab branch zeroes its output and
    // returns successfully, so an all-zero agreement would be two failures
    // agreeing rather than a parity result.
    let fixture = Fixture::new();
    let operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    let bound = fixture.run(&operation);
    assert!(
        bound.iter().any(|v| v.abs() > f32::EPSILON),
        "an all-zero expert output means the kernel took its refusal branch"
    );
    assert!(bound.iter().all(|v| v.is_finite()));
}

#[test]
fn the_reference_kernel_agrees_with_the_bound_one_within_quantisation_noise() {
    // The independent leg. Two calls of one function agree whatever they are
    // handed; this says the operands were the right ones, because a swapped
    // gate/up half or a mis-strided `down` would not land inside a band.
    let fixture = Fixture::new();
    let bound =
        fixture.run(&fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k));
    let reference = fixture.run(&fixture.operation(fixture.f32_expert(), ExpertKernel::Reference));

    let diff = max_abs_diff(&bound, &reference);
    assert!(
        diff <= KERNEL_TOLERANCE,
        "kernels disagree by {diff}, beyond quantisation noise"
    );
}

#[test]
fn the_two_kernels_are_not_trivially_identical() {
    // Guards the tolerance. If the reference somehow ran the same arithmetic,
    // the band above would be vacuous and would keep passing through a real
    // regression.
    let fixture = Fixture::new();
    let bound =
        fixture.run(&fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k));
    let reference = fixture.run(&fixture.operation(fixture.f32_expert(), ExpertKernel::Reference));
    assert_ne!(
        bound, reference,
        "an integer-dot kernel and an f32 reference should not be bit-identical"
    );
}

#[test]
fn each_kernel_has_a_distinct_name_and_the_reference_is_the_default() {
    assert_eq!(ExpertKernel::default(), ExpertKernel::Reference);
    assert_ne!(
        ExpertKernel::Reference.name(),
        ExpertKernel::IncumbentQ4kQ8k.name()
    );
    assert!(ExpertKernel::IncumbentQ4kQ8k.name().contains("q4k"));
}

#[test]
fn the_bank_reports_which_kernel_it_bound() {
    let fixture = Fixture::new();
    let operation = fixture.operation(fixture.q4k_expert(), ExpertKernel::IncumbentQ4kQ8k);
    assert!(operation
        .describe()
        .contains(ExpertKernel::IncumbentQ4kQ8k.name()));
}

#[test]
fn an_activation_the_kernel_does_not_implement_is_refused() {
    // The most dangerous case, because nothing else would signal it: the
    // incumbent's loop branches on GeluTanh and falls through to SiLU, so a
    // ReLU bank would run SiLU and return finite plausible values.
    let fixture = Fixture::new();
    let operation = fixture.operation_with(
        fixture.q4k_expert(),
        ExpertKernel::IncumbentQ4kQ8k,
        Activation::ReLU,
    );
    let err = operation.validate().unwrap_err();
    assert!(
        matches!(err, ExecutionError::KernelActivationUnsupported { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("ReLU"), "{err}");
}

#[test]
fn both_activations_the_kernel_implements_are_accepted() {
    let fixture = Fixture::new();
    for activation in [Activation::Silu, Activation::GeluTanh] {
        fixture
            .operation_with(
                fixture.q4k_expert(),
                ExpertKernel::IncumbentQ4kQ8k,
                activation,
            )
            .validate()
            .unwrap_or_else(|e| panic!("{activation:?} should bind: {e}"));
    }
}

#[test]
fn the_reference_kernel_accepts_every_activation() {
    // The counter-case: the restriction belongs to the incumbent kernel, not to
    // the runtime.
    let fixture = Fixture::new();
    fixture
        .operation_with(
            fixture.f32_expert(),
            ExpertKernel::Reference,
            Activation::ReLU,
        )
        .validate()
        .expect("the reference implements all four");
}
