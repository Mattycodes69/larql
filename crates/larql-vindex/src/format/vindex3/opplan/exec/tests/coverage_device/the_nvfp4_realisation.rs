//! The NVFP4 realisation

use super::*;

/// A single NVFP4 projection through the seam equals the round-tripped
/// matrix against the same input: the seam hands the device the codes,
/// the E4M3 scales *and* the tensor scale intact, with the geometry.
#[test]
fn an_nvfp4_projection_matches_the_round_tripped_matrix() {
    let backend =
        DevicePlanBackend::new(Nvfp4LoopDevice, "nvfp4-loop-project", WeightFormat::Nvfp4);
    let values = lcg_values(ROWS * COLS, 21);
    let x = lcg_values(COLS, 22);
    let LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
        ..
    } = quantize_nvfp4(&values, ROWS, COLS, "nv").unwrap()
    else {
        unreachable!()
    };
    let got = project_on(
        &backend,
        WeightSlice::Nvfp4 {
            packed: packed.as_slice(),
            scales: scales.as_slice(),
            tensor_scale,
            activation: Default::default(),
        },
        &x,
    )
    .unwrap();
    let reconstructed = round_trip(&values, ROWS, COLS).unwrap();
    let expected = matvec_rows(&reconstructed, ROWS, COLS, &x);
    assert!(
        max_abs(&got, &expected) < LOOP_NOISE_CEILING,
        "seam-routed NVFP4 gemv diverges from the reference reconstruction"
    );
    // The realisation is lossy but must not be degenerate: the tensor
    // scale was applied (an all-zero output would mean it was dropped).
    assert!(got.iter().any(|v| *v != 0.0));
}

/// The NVFP4 realisation runs the dense plan end to end — attention
/// Q/K/V through the NVFP4 multi path, the FFN pair likewise, single
/// projections through the NVFP4 gemv — stays in the reference's
/// neighbourhood, and its decode session reproduces its own batch
/// traversal bit for bit (the step path with no gate, on the device).
#[test]
fn an_nvfp4_device_backend_executes_and_decodes_the_dense_plan() {
    let (_c, plan, store) = dense_fixture();
    let backend = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-loop-dense",
        WeightFormats::uniform(WeightFormat::Nvfp4),
    );
    // The class table is the device backend's own candidate: an FFN
    // projection of a registered representation lands on NVFP4 whatever
    // the container stores it as.
    let ffn = plan
        .planned_operands()
        .into_iter()
        .find(|p| {
            p.operation
                == crate::format::vindex3::opplan::planned::Operation::Project(
                    MatrixClass::FfnProjection,
                )
        })
        .expect("the dense plan has an FFN projection");
    let facts =
        crate::format::vindex3::opplan::exec::realization::RepresentationFacts::resolve("F32");
    let selection = backend.select(&ffn, &facts).unwrap();
    assert_eq!(selection.realization.format(), WeightFormat::Nvfp4);
    assert_eq!(
        selection.reason,
        crate::format::vindex3::opplan::exec::realization::SelectionReason::DeviceClassTable
    );
    let on_device: ExecutionTrace = execute_plan(&plan, &store, &DENSE_TOKENS, &backend).unwrap();
    let on_reference =
        execute_plan(&plan, &store, &DENSE_TOKENS, &ReferenceBackend::new()).unwrap();
    let logits = on_device.logits.as_ref().unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
    let cos = cosine(logits, on_reference.logits.as_ref().unwrap());
    assert!(
        cos > NVFP4_COSINE_FLOOR,
        "nvfp4 logits decorrelated from reference: cos {cos}"
    );

    let stepped = decode_logits(&plan, &store, &backend);
    assert_eq!(
        logits.as_slice(),
        stepped.as_slice(),
        "nvfp4 decode-session logits differ from the batch traversal"
    );
    // Every matrix went through the device: the batch pass and the
    // decode pass both submitted work.
    assert!(backend.dispatch_stats().unwrap().submissions > 0);
}

/// Rung 3b: the device backend answers the prepared plan's question the
/// way every backend does — its class table is its own single candidate
/// for a projection, the table and the bank take the common selections,
/// and what it cannot bind it refuses by name.
#[test]
fn the_device_backend_selects_by_its_class_table_and_refuses_what_it_cannot_bind() {
    use crate::format::vindex3::opplan::exec::realization::{
        RealizationBackend, RealizationForm, RefusalKind, RepresentationFacts, SelectionReason,
    };
    use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
    use crate::format::vindex3::opplan::OperandRef;
    use crate::format::vindex3::represent::codec::RepresentationExtent;

    let backend = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-loop-select",
        WeightFormats::uniform(WeightFormat::Nvfp4),
    );
    let planned = |operation: Operation| PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.w".into(),
            dtype: String::new(),
            shape: vec![8, 8],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: 64,
    };
    let bf16 = RepresentationFacts::resolve("BF16");
    let head = backend
        .select(&planned(Operation::OutputHead), &bf16)
        .unwrap();
    assert_eq!(head.realization.backend, RealizationBackend::Device);
    assert_eq!(
        head.realization.form,
        RealizationForm::DeviceResident(WeightFormat::Nvfp4)
    );
    assert_eq!(head.reason, SelectionReason::DeviceClassTable);
    assert_eq!(head.candidates, vec![head.realization]);
    let table = backend.select(&planned(Operation::Embed), &bf16).unwrap();
    assert_eq!(table.realization.form, RealizationForm::DecodedGather);
    let bank = backend
        .select(&planned(Operation::ExpertBankSlice), &bf16)
        .unwrap();
    assert_eq!(
        bank.realization.form,
        RealizationForm::SliceStored {
            convert: WeightFormat::Nvfp4
        }
    );
    let refused = backend
        .select(
            &planned(Operation::Project(MatrixClass::FfnProjection)),
            &RepresentationFacts::resolve("U8"),
        )
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::UnregisteredRepresentation);
    let refused = backend
        .select(&planned(Operation::SharedExpertProject), &bf16)
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::MissingRealization);
    // A per-expert bank is a HOST mapping; the device binds packed banks
    // only, and says so before any common arm could admit it.
    let refused = backend
        .select(
            &planned(Operation::ExpertProject {
                experts: 4,
                top_k: 2,
            }),
            &bf16,
        )
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::MissingRealization);
    assert!(refused.considered.is_empty());
}

/// A class table asking NVFP4 over an operand the container holds at
/// source precision pins f16, naming why; the same table over any other
/// operand, and an f16 class over a held one, answer as before.
#[test]
fn an_nvfp4_class_over_a_source_held_operand_pins_f16() {
    use crate::format::vindex3::opplan::exec::realization::{
        RealizationForm, RepresentationFacts, SelectionReason,
    };
    use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
    use crate::format::vindex3::opplan::OperandRef;
    use crate::format::vindex3::represent::codec::RepresentationExtent;

    let planned = |operation: Operation| PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.w".into(),
            dtype: String::new(),
            shape: vec![8, 8],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: 64,
    };
    let held = RepresentationFacts::resolve("BF16").with_nvfp4_at_source(true);
    let free = RepresentationFacts::resolve("BF16");
    let nvfp4 = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-loop-held",
        WeightFormats::uniform(WeightFormat::Nvfp4),
    );
    for operation in [
        Operation::OutputHead,
        Operation::Project(MatrixClass::FfnProjection),
        Operation::Project(MatrixClass::AttentionProjection),
    ] {
        let pinned = nvfp4.select(&planned(operation), &held).unwrap();
        assert_eq!(
            pinned.realization.form,
            RealizationForm::DeviceResident(WeightFormat::F16),
            "{operation:?}"
        );
        assert_eq!(pinned.reason, SelectionReason::SourcePrecisionHeld);
        let asked = nvfp4.select(&planned(operation), &free).unwrap();
        assert_eq!(
            asked.realization.form,
            RealizationForm::DeviceResident(WeightFormat::Nvfp4),
            "{operation:?}"
        );
        assert_eq!(asked.reason, SelectionReason::DeviceClassTable);
    }
    // An f16 class never asked for NVFP4, so the fact changes nothing.
    let f16 = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "f16-loop-held",
        WeightFormats::uniform(WeightFormat::F16),
    );
    let head = f16.select(&planned(Operation::OutputHead), &held).unwrap();
    assert_eq!(head.reason, SelectionReason::DeviceClassTable);
}

/// End to end, the failure this fixes: an NVFP4 pack compiled under the
/// conservative default keeps some operands at source precision, and an
/// all-NVFP4 device arm used to pin NVFP4 on them while the loader bound
/// f16 — `verify_pins` refused ("the loader drifted from the selector").
/// Preparation now succeeds, and every source-held operand is pinned
/// and resident at f16, under `auto` and under `stored` alike.
#[test]
fn an_all_nvfp4_device_prepares_a_conservative_pack_with_held_operands_at_f16() {
    use crate::format::vindex3::opplan::exec::operands::RepresentationSource;
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::realization::SelectionReason;
    use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
    use crate::format::vindex3::represent::{compile_representation, policy, RepresentSpec};

    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    crate::format::vindex3::fixtures::encode_fixture_container(
        dense_f32_model,
        &checkpoint,
        &src,
        "target",
    );
    compile_representation(
        &src,
        &out,
        &RepresentSpec {
            encoding: DTYPE_NVFP4.to_string(),
            objects: Vec::new(),
            roles: policy::RolePolicy::default(),
            deployment: false,
            protect: policy::Protections::default(),
        },
    )
    .expect("the fixture compiles to NVFP4");
    let inspection = inspect_container(&out, false).unwrap();
    let plan = plan_component_ops(&inspection, &out, "target")
        .unwrap()
        .plan
        .expect("a plan");
    let backend = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-loop-conservative",
        WeightFormats::uniform(WeightFormat::Nvfp4),
    );
    for source in [RepresentationSource::Auto, RepresentationSource::Stored] {
        let store = OperandStore::open_for(&out, &inspection, Some(DTYPE_NVFP4), source).unwrap();
        let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)
            .unwrap_or_else(|e| panic!("{source:?}: preparation refused: {e}"));
        ops.verify_pins().unwrap();
        let held: Vec<_> = ops
            .realizations()
            .iter()
            .filter(|r| r.selection.reason == SelectionReason::SourcePrecisionHeld)
            .collect();
        assert!(
            !held.is_empty(),
            "{source:?}: the conservative pack holds nothing at source — the witness is vacuous"
        );
        for r in held {
            assert_eq!(
                r.selection.realization.format(),
                WeightFormat::F16,
                "{source:?}"
            );
        }
    }
}

/// End to end, the mirror of the held-at-source failure: `nvfp4-ffn`
/// asks f16 attention, the NVFP4 pack stores attention compiled and holds
/// no source bytes for it, and the F16 loader had no judged narrowing for
/// an NVFP4 pack ("no judged f16 narrowing for dtype NVFP4"). Preparation
/// now binds the compiled bytes as stored, and the pin says so.
#[test]
fn an_nvfp4_ffn_device_prepares_a_pack_whose_attention_is_compiled() {
    use crate::format::vindex3::opplan::exec::operands::RepresentationSource;
    use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
    use crate::format::vindex3::opplan::exec::realization::SelectionReason;
    use crate::format::vindex3::opplan::exec::weights::load_weight;
    use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
    use crate::format::vindex3::represent::{compile_representation, policy, RepresentSpec};

    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    crate::format::vindex3::fixtures::encode_fixture_container(
        dense_f32_model,
        &checkpoint,
        &src,
        "target",
    );
    compile_representation(
        &src,
        &out,
        &RepresentSpec {
            encoding: DTYPE_NVFP4.to_string(),
            objects: Vec::new(),
            roles: policy::RolePolicy::default(),
            deployment: false,
            protect: policy::Protections::default(),
        },
    )
    .expect("the fixture compiles to NVFP4");
    let inspection = inspect_container(&out, false).unwrap();
    let plan = plan_component_ops(&inspection, &out, "target")
        .unwrap()
        .plan
        .expect("a plan");
    let backend = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-ffn-loop",
        WeightFormats {
            attention: WeightFormat::F16,
            ffn: WeightFormat::Nvfp4,
            head: WeightFormat::F16,
        },
    );
    for source in [RepresentationSource::Auto, RepresentationSource::Stored] {
        let store = OperandStore::open_for(&out, &inspection, Some(DTYPE_NVFP4), source).unwrap();
        let ops = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full)
            .unwrap_or_else(|e| panic!("{source:?}: preparation refused: {e}"));
        ops.verify_pins().unwrap();
        let compiled: Vec<_> = ops
            .realizations()
            .iter()
            .filter(|r| r.selection.reason == SelectionReason::CompiledPrecisionHeld)
            .collect();
        assert!(
            !compiled.is_empty(),
            "{source:?}: the pack compiled no attention operand — the witness is vacuous"
        );
        for r in &compiled {
            assert_eq!(
                r.selection.realization.format(),
                WeightFormat::Nvfp4,
                "{source:?}"
            );
            // The loader, asked f16 directly as a class-table caller asks
            // it, binds the same compiled bytes an NVFP4 request does.
            let operand = &r.planned.operand;
            let asked = load_weight((&store).into(), operand, WeightFormat::F16)
                .unwrap_or_else(|e| panic!("{source:?}: f16 over a compiled operand: {e}"));
            let stored = load_weight((&store).into(), operand, WeightFormat::Nvfp4).unwrap();
            match (&asked, &stored) {
                (
                    LoadedWeight::Nvfp4 {
                        packed: a,
                        scales: sa,
                        tensor_scale: ta,
                        ..
                    },
                    LoadedWeight::Nvfp4 {
                        packed: b,
                        scales: sb,
                        tensor_scale: tb,
                        ..
                    },
                ) => {
                    assert_eq!(a.as_slice(), b.as_slice(), "{source:?}");
                    assert_eq!(sa.as_slice(), sb.as_slice(), "{source:?}");
                    assert_eq!(ta.to_bits(), tb.to_bits(), "{source:?}");
                }
                other => panic!("{source:?}: expected two NVFP4 bindings, got {other:?}"),
            }
        }
        // The source-held head is untouched: an f16 class over bf16 bytes.
        assert!(
            ops.realizations()
                .iter()
                .all(|r| r.selection.reason != SelectionReason::SourcePrecisionHeld),
            "{source:?}: nothing here asked NVFP4 over a held operand"
        );
    }
}

/// An f16 class over an operand stored compiled to NVFP4 pins NVFP4,
/// naming why; an NVFP4 class over it, and an f16 class over anything
/// else, answer as before.
#[test]
fn an_f16_class_over_a_compiled_nvfp4_operand_pins_nvfp4() {
    use crate::format::vindex3::opplan::exec::realization::{
        RealizationForm, RepresentationFacts, SelectionReason,
    };
    use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
    use crate::format::vindex3::opplan::OperandRef;
    use crate::format::vindex3::represent::codec::RepresentationExtent;

    let planned = |operation: Operation| PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.w".into(),
            dtype: String::new(),
            shape: vec![8, 8],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::BASE,
        layer: Some(0),
        declared_representation: None,
        logical_elements: 64,
    };
    let compiled = RepresentationFacts::resolve("BF16").with_nvfp4_compiled(true);
    let free = RepresentationFacts::resolve("BF16");
    let f16 = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "f16-loop-compiled",
        WeightFormats::uniform(WeightFormat::F16),
    );
    for operation in [
        Operation::OutputHead,
        Operation::Project(MatrixClass::FfnProjection),
        Operation::Project(MatrixClass::AttentionProjection),
    ] {
        let pinned = f16.select(&planned(operation), &compiled).unwrap();
        assert_eq!(
            pinned.realization.form,
            RealizationForm::DeviceResident(WeightFormat::Nvfp4),
            "{operation:?}"
        );
        assert_eq!(pinned.reason, SelectionReason::CompiledPrecisionHeld);
        let asked = f16.select(&planned(operation), &free).unwrap();
        assert_eq!(
            asked.realization.form,
            RealizationForm::DeviceResident(WeightFormat::F16),
            "{operation:?}"
        );
        assert_eq!(asked.reason, SelectionReason::DeviceClassTable);
    }
    // An NVFP4 class already asks for what is stored, so the fact changes nothing.
    let nvfp4 = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-loop-compiled",
        WeightFormats::uniform(WeightFormat::Nvfp4),
    );
    let head = nvfp4
        .select(&planned(Operation::OutputHead), &compiled)
        .unwrap();
    assert_eq!(head.reason, SelectionReason::DeviceClassTable);
}
