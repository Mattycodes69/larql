//! Rung 3c: declarations bound AGAINST observations. The expectation side
//! is priced from the pin, the codec's declared residency, the executor's
//! block geometry and the container's recorded lengths; the observation
//! side is read off the objects the loader bound. Neither reads the other.

use super::super::accounting::{
    execution_touch, expectations, ledger_correspondence, reconcile, render_selection_summary,
    resident_profile_with, stored_footprint, BlockGeometry, Bound, Observed,
};
use super::super::backend::{MatrixClass, WeightFormat};
use super::super::cpu::ledger::{ledger, thread_projection_calls};
use super::super::cpu::physical::{KQuantExecution, PhysicalProjectionPlan};
use super::super::lowering::LoweringIdentity;
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::{select_cpu, ProductionBackend};
use super::super::realization::{
    ExtentPin, RealizationForm, RealizationId, RealizationRecord, RepresentationFacts, Selection,
    SelectionReason,
};
use super::super::weights::{load_weight, DEVICE_PAGE_ALIGN};
use super::super::{execute_prepared_streaming, PlaneEvent};
use super::bf16_zlib_execution::{transcode, Transcode};
use super::kquant_projection::{a_stored_matrix, compiled, open as open_pack};
use crate::format::vindex3::fixtures::{
    dense_f32_model, dense_f32_model_with, encode_fixture_container, HeadStorage,
};
use crate::format::vindex3::inspect::{inspect_container, SystemInspection};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::codecs::{float, kquant, mxfp4, nvfp4};
use crate::format::vindex3::represent::codec::{
    CodecRegistry, RepresentationExtent, ResidencyProfile,
};
use crate::format::vindex3::represent::kquant::Q6_K;

const F32_WIDTH: u64 = std::mem::size_of::<f32>() as u64;
const BF16_WIDTH: u64 = std::mem::size_of::<u16>() as u64;
const PROJECTION_SUFFIX: &str = "_proj.weight";
const TOKENS: [u32; 3] = [3, 17, 28];

struct Fixture {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
    inspection: SystemInspection,
    plan: ComponentOpPlan,
    store: OperandStore,
}

impl Fixture {
    /// The same container, decoding through `registry`.
    fn store_through(&self, registry: &'static CodecRegistry) -> OperandStore {
        OperandStore::open(self.container.path(), &self.inspection)
            .unwrap()
            .with_registry(registry)
            .unwrap()
    }
}

fn fixture(write: fn(&std::path::Path), into: Option<Transcode>) -> Fixture {
    let src = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(write, src.path(), container.path(), "dense");
    if let Some(into) = into {
        let done = transcode(
            container.path(),
            |name, shape| shape.len() == 2 && name.ends_with(PROJECTION_SUFFIX),
            into,
        );
        assert!(!done.is_empty());
    }
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _src: src,
        container,
        inspection,
        plan,
        store,
    }
}

fn tied_head() -> Fixture {
    fixture(|d| dense_f32_model_with(d, HeadStorage::Tied), None)
}

fn prepared(f: &Fixture) -> PreparedOperands {
    PreparedOperands::load(
        &f.plan,
        &f.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
}

/// A record for one operand under one realization, as the selector would
/// have pinned it, with the residency a codec would have declared.
fn record(
    operand: &OperandRef,
    realization: RealizationId,
    residency: ResidencyProfile,
) -> RealizationRecord {
    let operation = Operation::Project(MatrixClass::FfnProjection);
    RealizationRecord {
        planned: PlannedOperand {
            operand: operand.clone(),
            operation,
            access: operation.access(),
            extent: RepresentationExtent::BASE,
            layer: Some(0),
            declared_representation: None,
            logical_elements: operand.shape.iter().product(),
        },
        representation: operand.dtype.clone(),
        codec_provider: None,
        lowering_provider: LoweringIdentity::cpu_production(),
        // A terminal representation: one extent, unpriced here because
        // this helper's subject is residency, not what a plane costs.
        extent: ExtentPin::unknown(),
        // Nothing attested, so nothing was verified.
        verified_bytes: 0,
        // And it depends on nothing, like every codec but one.
        dependencies: Vec::new(),
        selection: Selection {
            realization,
            residency,
            reason: SelectionReason::SizePolicy,
            candidates: vec![realization],
        },
    }
}

/// The first FFN projection of the dense plan.
fn an_ffn_projection(plan: &ComponentOpPlan) -> OperandRef {
    plan.planned_operands()
        .into_iter()
        .find(|p| p.operation == Operation::Project(MatrixClass::FfnProjection))
        .map(|p| p.operand)
        .unwrap()
}

fn rung_one_registry(with_f32: bool) -> CodecRegistry {
    let mut r = CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .unwrap();
    if with_f32 {
        r = r.register(Box::new(float::F32)).unwrap();
    }
    r.register(Box::new(kquant::Q4_K))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .unwrap()
}

fn exact_ledger_correspondence() {
    let f = fixture(dense_f32_model, None);
    let ops = prepared(&f);
    let planned = f.plan.planned_operands();
    let pins = ops.realizations();
    let bound = ops.bound(&f.plan).unwrap();
    // One pin per planned operand, one bound object per pin.
    assert_eq!(pins.len(), planned.len());
    assert_eq!(bound.len(), pins.len());
    let projections = pins
        .iter()
        .filter(|r| r.selection.realization.cpu_plan().is_some())
        .count();

    ledger().reset();
    let backend = ProductionBackend::new();
    let mut sink = |_: PlaneEvent| Ok(());
    execute_prepared_streaming(&f.plan, &ops, &TOKENS, &backend, None, &mut sink).unwrap();
    let account = ledger_correspondence(pins, ledger()).unwrap();
    assert_eq!(
        account.len(),
        1,
        "one CPU plan on an f32 fixture: {account:?}"
    );
    let (plan, pinned, tally) = account[0];
    assert_eq!(plan, PhysicalProjectionPlan::BlasF32);
    assert_eq!(pinned, projections);
    eprintln!(
        "BlasF32 over {projections} pinned projections: calls {} slabs {} positions {} (this thread: {})",
        tally.calls,
        tally.slabs,
        tally.positions,
        thread_projection_calls()
    );
    // Positions are the unit that corresponds: every layer projection
    // processed every token, and the head processed the last one. Calls
    // are batched differently per site and threaded, so they are not.
    let in_layers = pins
        .iter()
        .filter(|r| r.planned.layer.is_some() && r.selection.realization.cpu_plan().is_some())
        .count() as u64;
    let heads = projections as u64 - in_layers;
    assert_eq!(
        tally.positions,
        in_layers * TOKENS.len() as u64 + heads,
        "{tally:?}"
    );
    // The same ledger held against no pins: a plan that ran with nothing
    // pinned to it is refused, not ignored.
    let err = ledger_correspondence(&[], ledger())
        .unwrap_err()
        .to_string();
    assert!(err.contains("no operand was pinned to it"), "{err}");
}

use super::gemma4::{closure as gemma4_closure, encoded as gemma4_encoded, miniature_gemma4};
use crate::format::vindex3::represent::codec::{
    AccessGranularity, CodecCapabilities, CodecError, CodecOperands, ExtentCertificate,
    RepresentationCodec, StreamSpec,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// A codec that claims someone else's label under its own identity —
/// what a provider that CHANGED looks like to a prepared image.
pub(super) struct ProviderStub {
    pub label: &'static str,
}

impl RepresentationCodec for ProviderStub {
    fn encoding_label(&self) -> &'static str {
        self.label
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: format!("stub-{}", self.label),
            revision: 7,
            group_elems: 1,
            element: "stub".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "stub".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &[crate::format::vindex3::represent::codec::streams::VALUES]
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(32.0)]
    }
    fn stored_bytes(
        &self,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<u64, CodecError> {
        Ok(0)
    }
    fn validate(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: RepresentationExtent,
        _: &str,
    ) -> Result<(), CodecError> {
        Ok(())
    }
    fn decode_rows(
        &self,
        _: &CodecOperands<'_>,
        _: &[usize],
        _: std::ops::Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        _: &str,
    ) -> Result<(), CodecError> {
        dst.fill(0.0);
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

mod every_resident_form_the_cpu_loader_produ;
mod presentation_over_the_records;
mod reconcile_refusals;
mod the_prepared_plan_s_pairing_and_provider;
