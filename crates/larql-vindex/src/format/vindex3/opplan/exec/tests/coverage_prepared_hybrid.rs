//! A prepared KDA + MLA hybrid accounts for itself and executes, in both
//! of the forms K3 introduced: the full-rank KDA gate beside a gated MLA
//! layer, and the factorised MLA query.
//!
//! Each image is asked every accounting question the serve path asks —
//! what it bound, whether that reconciles with what it pinned, where its
//! bytes live — and then run, so the prepared operands' `weights()` view
//! is the one the traversal actually consumed.

use crate::format::vindex3::fixtures::encode_fixture_container;
use crate::format::vindex3::fixtures_kimi::{
    hybrid_kda_mla_f32_model_with, hybrid_kda_mla_f32_model_with_query, HybridGateForms,
    HybridQueryForms,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::operands::{OperandSource, OperandStore};
use crate::format::vindex3::opplan::exec::prefill_prepared;
use crate::format::vindex3::opplan::exec::prepared::{
    select_realizations, ExecutionSlice, PreparedOperands,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerAttention};

/// A prompt inside the hybrid fixture's 64-token vocabulary.
const TOKENS: [u32; 3] = [3, 17, 40];
/// A head index and width for the per-head projection probe; the probe
/// is refused before either is used on a non-softmax layer.
const PROBE_HEAD: usize = 0;
const PROBE_HEAD_DIM: usize = 8;
const PROBE_HEADS: usize = 4;

struct Hybrid {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

impl Hybrid {
    fn build(write: impl FnOnce(&std::path::Path)) -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(write, src.path(), container.path(), "kkkm");
        let inspection = inspect_container(container.path(), false).unwrap();
        let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
        assert!(outcome.closed(), "defects: {:?}", outcome.defects);
        let plan = outcome.plan.unwrap();
        let store = OperandStore::open(container.path(), &inspection).unwrap();
        Self {
            _src: src,
            _container: container,
            plan,
            store,
        }
    }

    fn gated() -> Self {
        Self::build(|dir| hybrid_kda_mla_f32_model_with(dir, HybridGateForms::KIMI_K3))
    }

    fn factorised_query() -> Self {
        Self::build(|dir| hybrid_kda_mla_f32_model_with_query(dir, HybridQueryForms::KIMI_K3))
    }

    fn source(&self) -> OperandSource<'_> {
        (&self.store).into()
    }

    fn prepare(&self) -> PreparedOperands {
        PreparedOperands::load(
            &self.plan,
            &self.store,
            &ReferenceBackend::new(),
            ExecutionSlice::Full,
        )
        .expect("the hybrid prepares")
    }

    /// Indices of the first KDA and the first MLA layer.
    fn kda_and_mla(&self) -> (usize, usize) {
        let find = |want: fn(&LayerAttention) -> bool| {
            self.plan
                .layers
                .iter()
                .position(|l| want(&l.attention))
                .expect("the hybrid has both operators")
        };
        (
            find(|a| matches!(a, LayerAttention::Kda(_))),
            find(|a| matches!(a, LayerAttention::Mla(_))),
        )
    }
}

/// Every accounting surface answers for both operators, and the
/// observation side reconciles with the pins.
fn accounts_for_itself(hybrid: &Hybrid) {
    let ops = hybrid.prepare();
    let bound = ops
        .bound(&hybrid.plan)
        .expect("the image names what it bound");
    for layer in 0..hybrid.plan.layers.len() {
        assert!(
            bound.iter().any(|o| o.layer == Some(layer)),
            "layer {layer} bound nothing"
        );
    }
    ops.reconcile(&hybrid.plan, hybrid.source())
        .expect("declared and resident agree");
    ops.verify_pins().expect("the loader kept its pins");

    let residency = ops.residency_census();
    assert!(
        residency.delta.total() > 0,
        "the KDA projections are counted"
    );
    assert!(
        residency.attention.total() > 0,
        "the MLA projections are counted"
    );
    assert!(
        residency.glue.widened_f32 > 0,
        "the recurrence glue is counted"
    );
    assert_eq!(
        ops.mapped_residency().mapped_bytes,
        0,
        "an owned image maps nothing"
    );
    assert!(ops.allocation_census().bytes > 0);

    let source = hybrid.source();
    assert_eq!(ops.source_stamp(), source.stamp());
    ops.ensure_current_for(&source)
        .expect("the image describes its own source");

    let selected = select_realizations(
        &hybrid.plan,
        hybrid.source(),
        &ReferenceBackend::new(),
        &ExecutionSlice::Full,
    )
    .expect("selection is the loader's own");
    assert!(!selected.is_empty());
}

/// Neither operator has softmax heads: the per-head surfaces say so
/// rather than projecting a matrix that is not an attention output.
fn per_head_surfaces_refuse_a_non_softmax_layer(hybrid: &Hybrid) {
    let ops = hybrid.prepare();
    let backend = ReferenceBackend::new();
    let (kda, mla) = hybrid.kda_and_mla();
    for layer in [kda, mla] {
        assert!(!ops.attention_has_heads(layer).unwrap());
        assert!(ops.attention_output_bias(layer).unwrap().is_none());
        let err = ops
            .head_projection(
                &backend,
                layer,
                PROBE_HEAD,
                PROBE_HEAD_DIM,
                PROBE_HEADS,
                &[0.0; PROBE_HEAD_DIM],
            )
            .expect_err("a non-softmax layer has no heads to project")
            .to_string();
        assert!(err.contains("no softmax heads"), "{err}");
    }
}

fn executes(hybrid: &Hybrid) -> Vec<f32> {
    let ops = hybrid.prepare();
    let mut kv = RowKvState::default();
    let out = prefill_prepared(
        &hybrid.plan,
        &ops,
        &TOKENS,
        &ReferenceBackend::new(),
        &mut kv,
    )
    .expect("the hybrid executes");
    let logits = out.logits.expect("the fixture carries a head");
    assert!(logits.iter().all(|v| v.is_finite()));
    logits
}

#[test]
fn a_gated_hybrid_accounts_for_its_full_rank_and_output_gates() {
    accounts_for_itself(&Hybrid::gated());
}

#[test]
fn a_factorised_query_hybrid_accounts_for_its_query_triple() {
    accounts_for_itself(&Hybrid::factorised_query());
}

#[test]
fn a_hybrid_refuses_per_head_probes_on_its_recurrent_and_latent_layers() {
    per_head_surfaces_refuse_a_non_softmax_layer(&Hybrid::gated());
}

/// The gated and the factorised hybrids both execute to finite logits,
/// and they differ — the two forms are different programs.
#[test]
fn both_k3_forms_execute_and_compute_different_functions() {
    let gated = executes(&Hybrid::gated());
    let factorised = executes(&Hybrid::factorised_query());
    assert_eq!(gated.len(), factorised.len());
    assert_ne!(gated, factorised);
}
