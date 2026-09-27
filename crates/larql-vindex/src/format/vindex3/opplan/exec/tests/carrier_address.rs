//! RESIDUAL-BUS-2 A1, A2 and forecast F2 (`docs/residual-bus-2.md`).
//!
//! A1: every transition carries one address, `(position, layer, site,
//! form)`, and batch and decode name the same one. A2: batch positions
//! are absolute, so a prefill split into chunks over one provider names
//! every transition, and writes every carrier value, exactly as the
//! unchunked prefill and decode do. F2 is that agreement on every
//! subject; KDA/MLA is the subject that could fail, because its
//! recurrent state crosses each chunk boundary inside the provider.

use std::collections::BTreeMap;

use super::attn_res_2b_batch::TOKENS as ATTN_RES_TOKENS;
use super::attn_res_substrate;
use super::carrier_write::gemma4_fixture;
use super::decode::fixture as golden_fixture;
use super::wave19_hc_decode;
use super::wave19_hc_substrate::{self, Variant};
use crate::format::vindex3::fixtures::G_TOKENS;
use crate::format::vindex3::opplan::exec::address::CarrierAddress;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    CarrierForm, CarrierTransition, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::portability::{carrier_form_of, ensure_portable};
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{prefill_prepared_observed, PlaneEvent};
use crate::format::vindex3::opplan::tests::kda_mla_exec::kimi_fixture;
use crate::format::vindex3::opplan::ComponentOpPlan;

/// Tokens inside every miniature's vocabulary.
const SMALL_TOKENS: [u32; 3] = [3, 1, 5];
/// Long enough that the KDA/MLA prefill is split more than once.
const KIMI_TOKENS: [u32; 6] = [3, 17, 5, 9, 12, 1];

/// Every transition, grouped by absolute position, in emission order.
type Addressed = BTreeMap<usize, Vec<(CarrierAddress, CarrierTransition)>>;
/// One single-stream write's values, as bits, keyed by its address.
type Writes = BTreeMap<(usize, usize, SublayerSite), (Vec<u32>, Vec<u32>)>;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Decode's side: one step per token, every transition and every
/// single-stream write.
#[derive(Default)]
struct DecodeLog {
    transitions: Addressed,
    writes: Writes,
}

impl StepObserver for DecodeLog {
    fn event(&mut self, _event: StepEvent) {}

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        self.writes.insert(
            (record.position, record.layer, record.site),
            (bits(record.delta), bits(record.after)),
        );
    }

    fn transition(&mut self, address: CarrierAddress, transition: CarrierTransition) {
        self.transitions
            .entry(address.position)
            .or_default()
            .push((address, transition));
    }
}

fn decode<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
) -> DecodeLog {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    let mut log = DecodeLog::default();
    for &token in tokens {
        session.step_observed(token, &mut log).unwrap();
    }
    log
}

/// A prefill of `tokens` split at `cuts`, over ONE provider, observed.
/// No cuts is the unchunked prefill.
fn prefill<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    cuts: &[usize],
) -> DecodeLog {
    let mut kv = RowKvState::default();
    let mut log = DecodeLog::default();
    let mut bounds: Vec<usize> = cuts.to_vec();
    bounds.push(tokens.len());
    let mut start = 0;
    for end in bounds {
        let mut sink = |event: PlaneEvent| {
            match event {
                PlaneEvent::Transition {
                    address,
                    transition,
                } => log
                    .transitions
                    .entry(address.position)
                    .or_default()
                    .push((address, transition)),
                PlaneEvent::CarrierWrite(write) => {
                    for (row, (delta, after)) in write.deltas.iter().zip(write.after).enumerate() {
                        log.writes.insert(
                            (write.base + row, write.layer, write.site),
                            (bits(delta), bits(after)),
                        );
                    }
                }
                _ => {}
            }
            Ok(())
        };
        prefill_prepared_observed(plan, ops, &tokens[start..end], backend, &mut kv, &mut sink)
            .unwrap();
        start = end;
    }
    log
}

/// F2 on one subject: decode, the unchunked prefill and every chunking
/// name the same addressed transitions at every absolute position, every
/// address carries `form`, and every single-stream write is bit-identical.
fn assert_f2<B: PlanBackend>(
    name: &str,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    form: CarrierForm,
) {
    let mut chunkings: Vec<Vec<usize>> = vec![Vec::new()];
    for cut in 1..tokens.len() {
        chunkings.push(vec![cut]);
    }
    if tokens.len() > 2 {
        chunkings.push((1..tokens.len()).collect());
    }
    assert_f2_over(name, plan, ops, backend, tokens, form, &chunkings);
}

/// [`assert_f2`] on a real container, with fewer chunkings because each
/// is a full prefill: unchunked, halved, and one token at a time.
pub(super) fn assert_f2_on_real<B: PlanBackend>(
    name: &str,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
) {
    let chunkings = vec![
        Vec::new(),
        vec![tokens.len() / 2],
        (1..tokens.len()).collect(),
    ];
    assert_f2_over(
        name,
        plan,
        ops,
        backend,
        tokens,
        CarrierForm::Single,
        &chunkings,
    );
}

fn assert_f2_over<B: PlanBackend>(
    name: &str,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    form: CarrierForm,
    chunkings: &[Vec<usize>],
) {
    let truth = decode(plan, ops, backend, tokens);
    assert_eq!(
        truth.transitions.keys().copied().collect::<Vec<_>>(),
        (0..tokens.len()).collect::<Vec<_>>(),
        "{name}: decode names every position"
    );
    for (addressed, _) in truth.transitions.values().flatten() {
        assert_eq!(addressed.form, form, "{name}: A1, decode's form");
    }
    for cuts in chunkings {
        let run = prefill(plan, ops, backend, tokens, cuts);
        assert_eq!(
            run.transitions, truth.transitions,
            "{name}: A1/A2, chunks {cuts:?} name different addressed transitions from decode"
        );
        assert_eq!(
            run.writes, truth.writes,
            "{name}: F2, chunks {cuts:?} write different values from decode"
        );
    }
}

fn full_ops<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> PreparedOperands {
    PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap()
}

#[test]
fn f2_the_plain_stack_agrees_at_every_absolute_position() {
    fn check<B: PlanBackend>(name: &str, backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let ops = full_ops(&plan, &store, backend);
        assert_f2(name, &plan, &ops, backend, &G_TOKENS, CarrierForm::Single);
    }
    check("plain, reference", &ReferenceBackend::new());
    check("plain, production", &ProductionBackend::new());
}

#[test]
fn f2_the_layer_scale_agrees_at_every_absolute_position() {
    let (_c, plan, store) = gemma4_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    assert_f2(
        "gemma4",
        &plan,
        &ops,
        &backend,
        &SMALL_TOKENS,
        CarrierForm::Single,
    );
}

#[test]
fn f2_a_bundle_agrees_at_every_absolute_position() {
    let sub = wave19_hc_substrate::build(Variant::HeadBearing);
    let ops = wave19_hc_decode::prepare(&sub, ExecutionSlice::Full).unwrap();
    assert_f2(
        "bundle",
        &sub.plan,
        &ops,
        &ReferenceBackend::new(),
        &SMALL_TOKENS,
        CarrierForm::Bundle,
    );
}

#[test]
fn f2_a_history_agrees_at_every_absolute_position() {
    let sub = attn_res_substrate::substrate();
    let (_store, ops) = attn_res_substrate::prepare(&sub);
    assert_f2(
        "history",
        &sub.plan,
        &ops,
        &ReferenceBackend::new(),
        &ATTN_RES_TOKENS,
        CarrierForm::History,
    );
}

/// The falsifier: recurrent (KDA) and latent (MLA) state cross every chunk
/// boundary inside the provider.
#[test]
fn f2_mixed_kda_mla_agrees_at_every_absolute_position() {
    let (_d, _c, plan, store) = kimi_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    assert_f2(
        "kda/mla",
        &plan,
        &ops,
        &backend,
        &KIMI_TOKENS,
        CarrierForm::Single,
    );
}

/// Control for A2: a batch that forgot its base names a chunk's
/// transitions at positions another chunk already used.
#[test]
fn control_a_batch_without_its_base_collides_with_an_earlier_chunk() {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    let truth = decode(&plan, &ops, &backend, &G_TOKENS);
    let mut chunked = prefill(&plan, &ops, &backend, &G_TOKENS, &[1]);
    // Re-number the second chunk as if its base had been dropped.
    let second: Vec<(CarrierAddress, CarrierTransition)> = chunked
        .transitions
        .split_off(&1)
        .into_values()
        .flatten()
        .map(|(mut address, transition)| {
            address.position -= 1;
            (address, transition)
        })
        .collect();
    for (address, transition) in second {
        chunked
            .transitions
            .entry(address.position)
            .or_default()
            .push((address, transition));
    }
    assert_ne!(chunked.transitions, truth.transitions);
}

/// Control for A1: a transition relabelled with the wrong form is caught.
#[test]
fn control_a_transition_with_the_wrong_form_is_caught() {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    let truth = decode(&plan, &ops, &backend, &G_TOKENS);
    let mut run = prefill(&plan, &ops, &backend, &G_TOKENS, &[]);
    assert_eq!(run.transitions, truth.transitions);
    run.transitions.get_mut(&0).unwrap()[0].0.form = CarrierForm::Bundle;
    assert_ne!(run.transitions, truth.transitions);
}

/// A3: one authority says which carrier forms may cross a process
/// boundary. Rows may; bundles and histories are addressable but not
/// portable, and are refused by name.
#[test]
fn a3_only_a_single_stream_carrier_is_portable() {
    let (_c, plain, _s) = golden_fixture();
    let bundle = wave19_hc_substrate::build(Variant::HeadBearing);
    let history = attn_res_substrate::substrate();
    assert_eq!(
        carrier_form_of(plain.residual_topology),
        CarrierForm::Single
    );
    assert_eq!(
        carrier_form_of(bundle.plan.residual_topology),
        CarrierForm::Bundle
    );
    assert_eq!(
        carrier_form_of(history.plan.residual_topology),
        CarrierForm::History
    );
    ensure_portable(CarrierForm::Single).unwrap();
    for (form, names) in [
        (CarrierForm::Bundle, "bundle"),
        (CarrierForm::History, "history"),
    ] {
        let err = ensure_portable(form).unwrap_err().to_string();
        assert!(err.contains(names), "{err}");
        assert!(err.contains("not portable"), "{err}");
    }
}

/// The four subjects above other than the plain stack, on the Production
/// backend (the plain stack already runs on both).
#[test]
fn f2_every_subject_agrees_on_the_production_backend() {
    let backend = ProductionBackend::new();
    let (_c, plan, store) = gemma4_fixture();
    let ops = full_ops(&plan, &store, &backend);
    assert_f2(
        "gemma4, production",
        &plan,
        &ops,
        &backend,
        &SMALL_TOKENS,
        CarrierForm::Single,
    );

    let sub = wave19_hc_substrate::build(Variant::HeadBearing);
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = full_ops(&sub.plan, &store, &backend);
    assert_f2(
        "bundle, production",
        &sub.plan,
        &ops,
        &backend,
        &SMALL_TOKENS,
        CarrierForm::Bundle,
    );

    let sub = attn_res_substrate::substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = full_ops(&sub.plan, &store, &backend);
    assert_f2(
        "history, production",
        &sub.plan,
        &ops,
        &backend,
        &ATTN_RES_TOKENS,
        CarrierForm::History,
    );

    let (_d, _c, plan, store) = kimi_fixture();
    let ops = full_ops(&plan, &store, &backend);
    assert_f2(
        "kda/mla, production",
        &plan,
        &ops,
        &backend,
        &KIMI_TOKENS,
        CarrierForm::Single,
    );
}
