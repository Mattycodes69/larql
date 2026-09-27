//! The vocabulary itself

use super::*;

#[test]
fn every_form_names_its_backend_its_form_and_its_resident_representation() {
    let expected_format = [
        WeightFormat::Bf16,
        WeightFormat::F32,
        WeightFormat::Q8,
        WeightFormat::F16,
        WeightFormat::F32,
        WeightFormat::Nvfp4,
    ];
    let expected_plan = [
        Some(PhysicalProjectionPlan::FusedBf16),
        Some(PhysicalProjectionPlan::BlasF32),
        Some(PhysicalProjectionPlan::FusedQ8),
        None,
        None,
        None,
    ];
    for ((form, format), plan) in every_form()
        .into_iter()
        .zip(expected_format)
        .zip(expected_plan)
    {
        let cpu = RealizationId::cpu(form);
        assert_eq!(cpu.format(), format, "{form:?}");
        assert_eq!(cpu.cpu_plan(), plan, "{form:?}");
        assert!(cpu.name().starts_with("cpu:"), "{}", cpu.name());
        let device = RealizationId {
            backend: RealizationBackend::Device,
            form,
        };
        assert!(device.name().starts_with("device:"), "{}", device.name());
        assert_ne!(cpu, device);
    }
    let names: std::collections::BTreeSet<String> = every_form()
        .into_iter()
        .map(|f| RealizationId::cpu(f).name())
        .collect();
    assert_eq!(
        names.len(),
        every_form().len(),
        "every form renders distinctly"
    );
}

#[test]
fn resident_profiles_are_priced_from_the_executor_s_own_forms() {
    let f32_width = std::mem::size_of::<f32>() as f64;
    let cases = [
        (
            WeightFormat::F32,
            ResidencyClass::TransientDecoded,
            f32_width,
        ),
        (WeightFormat::Bf16, ResidencyClass::Rebound, 2.0),
        (WeightFormat::F16, ResidencyClass::TransientRequantised, 2.0),
        (
            WeightFormat::Q8,
            ResidencyClass::TransientRequantised,
            1.0 + f32_width / Q8_BLOCK as f64,
        ),
        (
            WeightFormat::Q4,
            ResidencyClass::TransientRequantised,
            0.5 + f32_width / Q4_BLOCK as f64,
        ),
        (WeightFormat::Nvfp4, ResidencyClass::Rebound, 4.5 / 8.0),
        (WeightFormat::Mxfp4, ResidencyClass::Stored, 4.25 / 8.0),
        (WeightFormat::KQuant, ResidencyClass::Stored, 8.5 / 8.0),
    ];
    for (format, class, bytes) in cases {
        let profile = resident_profile(format);
        assert_eq!(profile.class, class, "{format:?}");
        assert!(
            (profile.bytes_per_weight - bytes).abs() < 1e-12,
            "{format:?}: {profile:?}"
        );
    }
}

#[test]
fn reasons_and_refusal_kinds_carry_distinct_names() {
    let reasons = [
        SelectionReason::DirectDeclared,
        SelectionReason::NoDirectRealization,
        SelectionReason::ArmPrefersDecode,
        SelectionReason::SizePolicy,
        SelectionReason::BankSlicedAtLoad,
        SelectionReason::DeviceClassTable,
        SelectionReason::EmbeddingGather,
        SelectionReason::ReferenceOracle,
        SelectionReason::OverlaidEdit,
        SelectionReason::SourcePrecisionHeld,
        SelectionReason::CompiledPrecisionHeld,
    ];
    let names: std::collections::BTreeSet<&str> = reasons.iter().map(|r| r.name()).collect();
    assert_eq!(names.len(), reasons.len());
    let kinds = [
        RefusalKind::UnregisteredRepresentation,
        RefusalKind::AccessRefused,
        RefusalKind::MissingRealization,
    ];
    let names: std::collections::BTreeSet<&str> = kinds.iter().map(|k| k.name()).collect();
    assert_eq!(names.len(), kinds.len());
}

#[test]
fn a_refusal_renders_the_operand_the_kind_and_every_candidate_it_considered() {
    let operand = synthetic(Operation::ExpertBankSlice, SMALL);
    let slice = RealizationId::cpu(RealizationForm::SliceStored {
        convert: WeightFormat::F32,
    });
    let decode = RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32));
    let refusal = SelectionRefusal {
        operand: operand.operand.clone(),
        operation: operand.operation,
        representation: "BF16_ZLIB".into(),
        requested: RequiredAccess::RowRandom,
        kind: RefusalKind::AccessRefused,
        considered: vec![
            (slice, "provides sequential access".into()),
            (decode, "not offered for a bank".into()),
        ],
    };
    let text = refusal.to_string();
    for expected in [
        "`0.w`",
        "expert-bank-slice",
        "row-random",
        "BF16_ZLIB",
        "access refused",
        &slice.name(),
        "provides sequential access",
        &decode.name(),
        "not offered for a bank",
    ] {
        assert!(text.contains(expected), "{expected} missing from: {text}");
    }
    let bare = SelectionRefusal {
        considered: vec![],
        kind: RefusalKind::MissingRealization,
        ..refusal.clone()
    };
    assert!(bare.to_string().contains("no realization to consider"));
    let all = SelectionRefusals(vec![refusal, bare]).to_string();
    assert!(
        all.starts_with("2 planned operand(s) have no admissible realization"),
        "{all}"
    );
    assert_eq!(
        all.matches("\n  ").count(),
        2,
        "one line per refusal: {all}"
    );
}

#[test]
fn facts_resolve_through_a_registry_and_an_overlay_empties_the_direct_candidates() {
    let scratch = CodecRegistry::new().register(Box::new(BF16)).unwrap();
    let bf16 = RepresentationFacts::resolve_in(&scratch, "BF16");
    assert_eq!(bf16.label, "BF16");
    assert!(bf16.registered.is_some());
    assert_eq!(
        bf16.direct_cpu_plans(),
        vec![
            PhysicalProjectionPlan::FusedBf16,
            PhysicalProjectionPlan::Bf16xQ8
        ]
    );
    assert!(bf16.provides(RequiredAccess::ElementRandom));
    assert_eq!(
        bf16.direct_residency(PhysicalProjectionPlan::FusedBf16),
        Some(ResidencyProfile::stored(16.0))
    );
    assert_eq!(
        bf16.direct_residency(PhysicalProjectionPlan::FusedNvfp4),
        None
    );
    bf16.admit_row_slicing().unwrap();
    // An overlay edit: still registered, still row-addressable in storage,
    // but nothing direct can honour an f32-space edit.
    let edited = bf16.clone().overlaid();
    assert!(edited.overlaid);
    assert!(edited.direct_cpu_plans().is_empty());
    assert!(edited.provides(RequiredAccess::RowRandom));
    // A label the scratch registry does not carry: no decode, no
    // capabilities, and a bank dialect the loader judges itself.
    let alien = RepresentationFacts::resolve_in(&scratch, "NVFP4");
    assert!(alien.registered.is_none());
    assert!(!alien.provides(RequiredAccess::Sequential));
    assert!(alien.direct_cpu_plans().is_empty());
    assert_eq!(
        alien.direct_residency(PhysicalProjectionPlan::FusedNvfp4),
        None
    );
    alien.admit_row_slicing().unwrap();
    // Candidates follow the facts: a bf16 source offers its kernels, the
    // decode, and the executor's re-quantised forms; an unregistered
    // label offers nothing.
    let requantise = [PhysicalProjectionPlan::FusedQ8];
    let candidates = cpu_projection_candidates(&bf16, PhysicalProjectionPlan::BlasF32, &requantise);
    assert_eq!(candidates.len(), 2 + 1 + 1);
    assert!(
        cpu_projection_candidates(&alien, PhysicalProjectionPlan::BlasF32, &requantise).is_empty()
    );
    let f16 = RepresentationFacts::resolve("F16");
    assert_eq!(
        cpu_projection_candidates(&f16, PhysicalProjectionPlan::BlasF32, &requantise),
        vec![RealizationId::cpu(RealizationForm::Decode(
            PhysicalProjectionPlan::BlasF32
        ))]
    );
}

#[test]
fn the_common_selections_cover_the_table_the_bank_and_the_shared_expert() {
    let registered = RepresentationFacts::resolve("BF16");
    let unregistered = RepresentationFacts::resolve("U8");
    let embed = synthetic(Operation::Embed, SMALL);
    let gathered = common_selection(&embed, &registered, WeightFormat::F32)
        .unwrap()
        .unwrap();
    assert_eq!(gathered.realization.form, RealizationForm::DecodedGather);
    assert_eq!(gathered.reason, SelectionReason::EmbeddingGather);
    let refused = common_selection(&embed, &unregistered, WeightFormat::F32)
        .unwrap()
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);

    let bank = synthetic(Operation::ExpertBankSlice, SMALL);
    let sliced = common_selection(&bank, &registered, WeightFormat::F16)
        .unwrap()
        .unwrap();
    assert_eq!(
        sliced.realization.form,
        RealizationForm::SliceStored {
            convert: WeightFormat::F16
        }
    );
    assert_eq!(sliced.residency, resident_profile(WeightFormat::F16));
    let sequential = RepresentationFacts::resolve("BF16_ZLIB");
    let refused = common_selection(&bank, &sequential, WeightFormat::F32)
        .unwrap()
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::AccessRefused);
    assert_eq!(refused.considered.len(), 1);

    // A shared expert's projections are whole matrices: the backend
    // chooses, as for any dense projection (V1). The scalar branch gate is
    // one row read whole as f32 and applied by the same literal dot on
    // every backend: one candidate, chosen here, and an unregistered
    // representation still refuses.
    let gate = synthetic(Operation::SharedExpertBranchGate, SMALL);
    let selected = common_selection(&gate, &registered, WeightFormat::F32)
        .unwrap()
        .unwrap();
    assert_eq!(
        selected.realization.form,
        RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32)
    );
    assert_eq!(selected.reason, SelectionReason::ScalarBranchGate);
    assert_eq!(selected.candidates, vec![selected.realization]);
    let refused = common_selection(
        &gate,
        &RepresentationFacts::resolve("U8"),
        WeightFormat::F32,
    )
    .unwrap()
    .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);

    // A per-expert bank binds its stored bytes as a mapping, in the stored
    // form, when the CPU runs that form in place: bf16 and f32 do, an
    // entropy-coded or quantised form does not, and an unregistered
    // label cannot be mapped at all.
    let bank = synthetic(
        Operation::ExpertProject {
            experts: 4,
            top_k: 2,
        },
        SMALL,
    );
    let mapped = common_selection(&bank, &registered, WeightFormat::F32)
        .unwrap()
        .unwrap();
    assert_eq!(
        mapped.realization.form,
        RealizationForm::MappedStored {
            format: WeightFormat::Bf16,
            access: MappedAccess::Demand,
        }
    );
    assert_eq!(mapped.reason, SelectionReason::BankMappedAsStored);
    assert_eq!(mapped.residency, resident_profile(WeightFormat::Bf16));
    assert_eq!(mapped.candidates, vec![mapped.realization]);
    let f32 = common_selection(
        &bank,
        &RepresentationFacts::resolve("F32"),
        WeightFormat::F32,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        f32.realization.form,
        RealizationForm::MappedStored {
            format: WeightFormat::F32,
            access: MappedAccess::Demand,
        }
    );
    for label in ["Q8_0", "BF16_ZLIB"] {
        let refused = common_selection(
            &bank,
            &RepresentationFacts::resolve(label),
            WeightFormat::F32,
        )
        .unwrap()
        .unwrap_err();
        assert_eq!(refused.kind, RefusalKind::MissingRealization, "{label}");
        assert_eq!(refused.considered.len(), 1, "{label}");
        assert!(
            refused.considered[0].1.contains("never decoded or copied"),
            "{label}"
        );
    }
    let refused = common_selection(&bank, &unregistered, WeightFormat::F32)
        .unwrap()
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);

    for projection in [
        Operation::Project(MatrixClass::AttentionProjection),
        Operation::OutputHead,
        Operation::SharedExpertProject,
    ] {
        assert!(common_selection(
            &synthetic(projection, SMALL),
            &registered,
            WeightFormat::F32
        )
        .is_none());
    }
    assert_eq!(
        class_of(Operation::OutputHead),
        Some(MatrixClass::OutputHead)
    );
    assert_eq!(
        class_of(Operation::SharedExpertProject),
        Some(MatrixClass::FfnProjection)
    );
    assert_eq!(class_of(Operation::Embed), None);
    assert_eq!(
        class_of(Operation::ExpertProject {
            experts: 4,
            top_k: 2
        }),
        None
    );
}

#[test]
fn the_cpu_selector_refuses_an_unregistered_projection_and_realizes_a_branch_gate() {
    let unregistered = RepresentationFacts::resolve("U8");
    let refused = select_cpu(
        &synthetic(Operation::Project(MatrixClass::FfnProjection), SMALL),
        &unregistered,
        KQuantExecution::Direct,
    )
    .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);
    // A shared expert's projection is any dense FFN projection to the
    // CPU: chosen by the size policy like the routed layer's neighbours.
    let shared = select_cpu(
        &synthetic(Operation::SharedExpertProject, SMALL),
        &RepresentationFacts::resolve("BF16"),
        KQuantExecution::Direct,
    )
    .unwrap();
    assert_eq!(
        shared.realization.form,
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)
    );
    assert_eq!(shared.reason, SelectionReason::SizePolicy);
    let gate = select_cpu(
        &synthetic(Operation::SharedExpertBranchGate, SMALL),
        &RepresentationFacts::resolve("BF16"),
        KQuantExecution::Direct,
    )
    .unwrap();
    assert_eq!(gate.reason, SelectionReason::ScalarBranchGate);
    // An overlay edit on a bf16 operand decodes, and the reason says so.
    let edited = RepresentationFacts::resolve("BF16").overlaid();
    let selected = select_cpu(
        &synthetic(Operation::Project(MatrixClass::FfnProjection), LARGE),
        &edited,
        KQuantExecution::Direct,
    )
    .unwrap();
    assert_eq!(
        selected.realization.form,
        RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)
    );
    assert_eq!(selected.reason, SelectionReason::OverlaidEdit);
}

#[test]
fn the_reference_backend_refuses_an_unregistered_representation_before_any_byte() {
    let fixture = dense(Some(Transcode::Unregistered));
    let before = fixture.store.load_count();
    let err = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .err()
    .map(|e| e.to_string())
    .expect("the oracle cannot decode what nothing decodes");
    assert!(err.contains("unregistered representation"), "{err}");
    assert_eq!(fixture.store.load_count(), before);
}
