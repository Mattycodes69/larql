//! RESIDUAL-BUS-1 step 1 witnesses (`docs/residual-bus-1.md`): the batch
//! traversal's single-stream carrier-write record equals decode's.
//!
//! T6: for every (position, layer, site), batch's `CarrierWritePlane` row
//! equals decode's `CarrierWriteRecord` — `delta`, `after` and
//! `layer_scale`, bit for bit — on the plain stack (Reference and
//! Production) and through the Gemma 4 layer scale. F1: the batch emits
//! exactly the writes decode does. T7: subscribing to the record leaves
//! the batch result bit-identical. Negative controls: one ulp in a batch
//! `delta` or `after`, and a swapped position pair, are each caught at
//! exactly the rows they alter.
//!
//! Bundle, history and KDA/MLA subjects are later steps of the same rung.
use std::collections::BTreeMap;

use super::carrier_write::{gemma4_fixture, observed, writes_declared_by};
use super::decode::fixture as golden_fixture;
use crate::format::vindex3::fixtures::G_TOKENS;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{ContinuationProvider, KvState, RowKvState};
use crate::format::vindex3::opplan::exec::observe::{CarrierTransition, HistoryWriteMode};
use crate::format::vindex3::opplan::exec::observe::{
    CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{
    execute_plan_streaming, execute_plan_streaming_in, PlaneEvent,
};
use crate::format::vindex3::opplan::exec::{
    execute_prepared_streaming, execute_prepared_streaming_in,
};
use crate::format::vindex3::opplan::tests::kda_mla_exec::kimi_fixture;

use super::attn_res_2b_batch::TOKENS as ATTN_RES_TOKENS;
use super::attn_res_substrate;
use super::wave19_hc_decode;
use super::wave19_hc_substrate::{self, Variant};
use crate::format::vindex3::opplan::ComponentOpPlan;

/// Tokens inside every miniature's vocabulary (the smallest is 7).
const SMALL_TOKENS: [u32; 3] = [3, 1, 5];

/// The prompt `kda_mla_exec` steps through its miniature Kimi (vocab 23).
const KIMI_TOKENS: [u32; 6] = [3, 17, 5, 9, 12, 1];

/// One write's identity: T6 compares per (position, layer, site).
type Key = (usize, usize, SublayerSite);

/// One write's values, owned for comparison.
#[derive(Clone, Debug)]
struct Row {
    delta: Vec<f32>,
    after: Vec<f32>,
    layer_scale: Option<f32>,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

impl Row {
    fn bit_equal(&self, other: &Row) -> bool {
        bits(&self.delta) == bits(&other.delta)
            && bits(&self.after) == bits(&other.after)
            && self.layer_scale.map(f32::to_bits) == other.layer_scale.map(f32::to_bits)
    }
}

/// Decode's side: every `CarrierWriteRecord`, keyed.
#[derive(Default)]
struct DecodeRecords(BTreeMap<Key, Row>);

impl StepObserver for DecodeRecords {
    fn event(&mut self, _event: StepEvent) {}

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let previous = self.0.insert(
            (record.position, record.layer, record.site),
            Row {
                delta: record.delta.to_vec(),
                after: record.after.to_vec(),
                layer_scale: record.layer_scale,
            },
        );
        assert!(previous.is_none(), "decode wrote one site twice");
    }
}

/// Batch's side: every `CarrierWritePlane` row, keyed, plus the logits of
/// the run that produced them.
fn batch_records<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
) -> (BTreeMap<Key, Row>, Option<Vec<f32>>) {
    batch_records_in(plan, store, backend, tokens, None)
}

/// As [`batch_records`], over caller-prepared continuation state: a plan
/// with recurrent or latent layers (KDA, MLA) cannot execute without it.
fn batch_records_in<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    state: Option<&mut dyn KvState>,
) -> (BTreeMap<Key, Row>, Option<Vec<f32>>) {
    let mut records = BTreeMap::new();
    let mut sink = |event: PlaneEvent| {
        if let PlaneEvent::CarrierWrite(write) = event {
            assert_eq!(write.deltas.len(), write.after.len());
            for (position, (delta, after)) in write.deltas.iter().zip(write.after).enumerate() {
                let previous = records.insert(
                    (position, write.layer, write.site),
                    Row {
                        delta: delta.clone(),
                        after: after.clone(),
                        layer_scale: write.layer_scale,
                    },
                );
                assert!(previous.is_none(), "batch wrote one site twice");
            }
        }
        Ok(())
    };
    let out = match state {
        Some(state) => {
            execute_plan_streaming_in(plan, store, tokens, backend, None, &mut sink, state)
        }
        None => execute_plan_streaming(plan, store, tokens, backend, None, &mut sink),
    }
    .unwrap();
    (records, out.logits)
}

/// The keys whose rows differ between the two sides, and any key present
/// on only one side.
fn mismatches(decode: &BTreeMap<Key, Row>, batch: &BTreeMap<Key, Row>) -> Vec<Key> {
    let mut out: Vec<Key> = decode
        .iter()
        .filter(|(key, row)| batch.get(key).is_none_or(|b| !b.bit_equal(row)))
        .map(|(key, _)| *key)
        .collect();
    out.extend(batch.keys().filter(|k| !decode.contains_key(k)).copied());
    out.sort_unstable();
    out
}

fn both_sides<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
) -> (BTreeMap<Key, Row>, BTreeMap<Key, Row>) {
    let mut decode = DecodeRecords::default();
    observed(plan, store, backend, tokens, &mut decode);
    let (batch, _) = batch_records(plan, store, backend, tokens);
    (decode.0, batch)
}

// ── T6 + F1 ─────────────────────────────────────────────────────────

#[test]
fn t6_batch_and_decode_write_records_are_bit_identical_on_the_plain_stack() {
    fn check<B: PlanBackend>(name: &str, backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let (decode, batch) = both_sides(&plan, &store, backend, &G_TOKENS);
        let declared = writes_declared_by(&plan) * G_TOKENS.len();
        assert_eq!(decode.len(), declared, "{name}: decode's write count");
        assert_eq!(batch.len(), declared, "{name}: F1, batch's write count");
        assert_eq!(mismatches(&decode, &batch), Vec::<Key>::new(), "{name}");
    }
    check("reference", &ReferenceBackend::new());
    check("production", &ProductionBackend::new());
}

/// T4 / C5: on a component with a layer scalar, the batch FFN record
/// carries the scale and a PRE-scale `after`, exactly as decode's does.
#[test]
fn t6_batch_and_decode_write_records_agree_through_the_gemma4_layer_scale() {
    let (_c, plan, store) = gemma4_fixture();
    assert!(plan.layers.iter().all(|l| l.layer_scale.is_some()));
    let (decode, batch) = both_sides(&plan, &store, &ReferenceBackend::new(), &SMALL_TOKENS);
    assert_eq!(batch.len(), writes_declared_by(&plan) * SMALL_TOKENS.len());
    assert_eq!(mismatches(&decode, &batch), Vec::<Key>::new());
    for ((_, _, site), row) in &batch {
        assert_eq!(
            row.layer_scale.is_some(),
            *site == SublayerSite::Ffn,
            "the scale rides on the FFN write only"
        );
    }
}

/// The first per-carrier-state batch/decode witness for mixed KDA/MLA.
/// Until now the two traversals were compared on logits only
/// (`opplan/tests/kda_mla_exec.rs`), where a state region that silently
/// reset would still produce finite logits of the right length.
#[test]
fn t6_batch_and_decode_write_records_are_bit_identical_on_mixed_kda_mla() {
    fn check<B: PlanBackend>(name: &str, backend: &B) {
        let (_d, _c, plan, store) = kimi_fixture();
        let tokens = KIMI_TOKENS;
        let mut decode = DecodeRecords::default();
        observed(&plan, &store, backend, &tokens, &mut decode);
        let mut state = RowKvState::default();
        state
            .prepare_continuation(&plan_continuation_geometry(&plan).unwrap())
            .unwrap();
        let (batch, _) = batch_records_in(&plan, &store, backend, &tokens, Some(&mut state));
        let declared = writes_declared_by(&plan) * tokens.len();
        assert_eq!(decode.0.len(), declared, "{name}: decode's write count");
        assert_eq!(batch.len(), declared, "{name}: F1, batch's write count");
        assert_eq!(mismatches(&decode.0, &batch), Vec::<Key>::new(), "{name}");
    }
    check("reference", &ReferenceBackend::new());
    check("production", &ProductionBackend::new());
}

// ── T7 ──────────────────────────────────────────────────────────────

#[test]
fn t7_subscribing_to_the_batch_record_leaves_the_result_bit_identical() {
    let (_c, plan, store) = golden_fixture();
    let backend = ProductionBackend::new();
    let ignoring =
        execute_plan_streaming(&plan, &store, &G_TOKENS, &backend, None, &mut |_| Ok(())).unwrap();
    let (_, subscribed) = batch_records(&plan, &store, &backend, &G_TOKENS);
    assert_eq!(
        ignoring.logits.as_deref().map(bits),
        subscribed.as_deref().map(bits)
    );
}

// ── Negative controls ───────────────────────────────────────────────

fn one_ulp(v: &mut f32) {
    *v = f32::from_bits(v.to_bits() ^ 1);
}

#[test]
fn control_one_ulp_in_a_batch_delta_or_after_is_caught_at_exactly_that_row() {
    let (_c, plan, store) = golden_fixture();
    let (decode, batch) = both_sides(&plan, &store, &ReferenceBackend::new(), &G_TOKENS);
    let target: Key = (1, 1, SublayerSite::Ffn);
    for perturb_delta in [true, false] {
        let mut altered = batch.clone();
        let row = altered.get_mut(&target).unwrap();
        one_ulp(if perturb_delta {
            &mut row.delta[0]
        } else {
            &mut row.after[0]
        });
        assert_eq!(mismatches(&decode, &altered), vec![target]);
    }
}

#[test]
fn control_a_swapped_position_pair_is_caught_at_exactly_those_rows() {
    let (_c, plan, store) = golden_fixture();
    let (decode, batch) = both_sides(&plan, &store, &ReferenceBackend::new(), &G_TOKENS);
    let (a, b): (Key, Key) = (
        (0, 0, SublayerSite::Attention),
        (1, 0, SublayerSite::Attention),
    );
    let mut altered = batch.clone();
    let (row_a, row_b) = (altered[&a].clone(), altered[&b].clone());
    altered.insert(a, row_b);
    altered.insert(b, row_a);
    assert_eq!(mismatches(&decode, &altered), vec![a, b]);
}

// ── T1: one transition vocabulary, the same sequence on both paths ───

/// Every position's transitions, in the order they fired.
type Sequences = BTreeMap<usize, Vec<(usize, CarrierTransition)>>;

#[derive(Default)]
struct TransitionLog(Sequences);

impl StepObserver for TransitionLog {
    fn event(&mut self, _event: StepEvent) {}

    fn transition(&mut self, position: usize, layer: usize, transition: CarrierTransition) {
        self.0
            .entry(position)
            .or_default()
            .push((layer, transition));
    }
}

/// Both traversals' transition sequences over one prepared image. Decode
/// steps every token through a session; batch runs them in one call,
/// over `state` when the plan keeps recurrent or latent state.
fn transition_sequences<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    state: Option<&mut dyn KvState>,
) -> (Sequences, Sequences) {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    let mut decode = TransitionLog::default();
    for &token in tokens {
        session.step_observed(token, &mut decode).unwrap();
    }
    let mut batch = Sequences::new();
    let mut sink = |event: PlaneEvent| {
        if let PlaneEvent::Transition {
            layer,
            position,
            transition,
        } = event
        {
            batch.entry(position).or_default().push((layer, transition));
        }
        Ok(())
    };
    match state {
        Some(state) => {
            execute_prepared_streaming_in(plan, ops, tokens, backend, None, &mut sink, state)
        }
        None => execute_prepared_streaming(plan, ops, tokens, backend, None, &mut sink),
    }
    .unwrap();
    (decode.0, batch)
}

/// T1's claim for one subject: per position, the batch sequence IS the
/// decode sequence, every position enters first, and the subject shows
/// each transition it exists to exercise. Returns the flattened kinds for
/// the caller's subject-specific checks.
fn assert_same_sequences(
    name: &str,
    decode: &Sequences,
    batch: &Sequences,
    positions: usize,
) -> Vec<CarrierTransition> {
    assert_eq!(decode.len(), positions, "{name}: decode positions");
    assert_eq!(
        batch.keys().collect::<Vec<_>>(),
        decode.keys().collect::<Vec<_>>(),
        "{name}: positions"
    );
    for (position, sequence) in decode {
        assert_eq!(
            sequence.first().map(|(_, t)| *t),
            Some(CarrierTransition::Enter),
            "{name}: position {position} enters first"
        );
        assert_eq!(
            &batch[position], sequence,
            "{name}: position {position}'s transitions differ between batch and decode"
        );
    }
    decode.values().flatten().map(|(_, t)| *t).collect()
}

fn full_ops<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> PreparedOperands {
    PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap()
}

#[test]
fn t1_the_plain_stack_names_the_same_transitions_on_both_paths() {
    fn check<B: PlanBackend>(name: &str, backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let ops = full_ops(&plan, &store, backend);
        let (decode, batch) = transition_sequences(&plan, &ops, backend, &G_TOKENS, None);
        let kinds = assert_same_sequences(name, &decode, &batch, G_TOKENS.len());
        let adds = kinds
            .iter()
            .filter(|t| matches!(t, CarrierTransition::Add { .. }))
            .count();
        assert_eq!(
            adds,
            writes_declared_by(&plan) * G_TOKENS.len(),
            "{name}: F1"
        );
    }
    check("reference", &ReferenceBackend::new());
    check("production", &ProductionBackend::new());
}

#[test]
fn t1_the_layer_scale_is_its_own_transition_on_both_paths() {
    let (_c, plan, store) = gemma4_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    let (decode, batch) = transition_sequences(&plan, &ops, &backend, &SMALL_TOKENS, None);
    let kinds = assert_same_sequences("gemma4", &decode, &batch, SMALL_TOKENS.len());
    let scales = kinds
        .iter()
        .filter(|t| **t == CarrierTransition::Scale)
        .count();
    assert_eq!(
        scales,
        plan.layers.len() * SMALL_TOKENS.len(),
        "one Scale per layer"
    );
}

#[test]
fn t1_a_bundle_updates_and_never_adds_on_both_paths() {
    let sub = wave19_hc_substrate::build(Variant::HeadBearing);
    let ops = wave19_hc_decode::prepare(&sub, ExecutionSlice::Full).unwrap();
    let (decode, batch) = transition_sequences(
        &sub.plan,
        &ops,
        &ReferenceBackend::new(),
        &SMALL_TOKENS,
        None,
    );
    let kinds = assert_same_sequences("bundle", &decode, &batch, SMALL_TOKENS.len());
    assert!(kinds
        .iter()
        .any(|t| matches!(t, CarrierTransition::HcUpdate { .. })));
    assert!(
        !kinds
            .iter()
            .any(|t| matches!(t, CarrierTransition::Add { .. })),
        "T5: a bundle update is never named an add"
    );
}

/// The two transitions no record named before BUS-1 T1 — a boundary's
/// prefix RESET, and a history write that REPLACES — now fire, the same
/// way, on both paths.
#[test]
fn t1_a_history_names_its_resets_and_both_write_modes_on_both_paths() {
    let sub = attn_res_substrate::substrate();
    let (_store, ops) = attn_res_substrate::prepare(&sub);
    let (decode, batch) = transition_sequences(
        &sub.plan,
        &ops,
        &ReferenceBackend::new(),
        &ATTN_RES_TOKENS,
        None,
    );
    let kinds = assert_same_sequences("history", &decode, &batch, ATTN_RES_TOKENS.len());
    for wanted in [
        CarrierTransition::HistorySnapshot,
        CarrierTransition::HistoryReset,
    ] {
        assert!(kinds.contains(&wanted), "{wanted:?} fires");
    }
    for mode in [HistoryWriteMode::Add, HistoryWriteMode::Replace] {
        assert!(
            kinds.iter().any(
                |t| matches!(t, CarrierTransition::HistoryWrite { mode: m, .. } if *m == mode)
            ),
            "a {mode:?} history write fires"
        );
    }
}

#[test]
fn t1_mixed_kda_mla_names_the_same_transitions_on_both_paths() {
    let (_d, _c, plan, store) = kimi_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    let mut state = RowKvState::default();
    state
        .prepare_continuation(&plan_continuation_geometry(&plan).unwrap())
        .unwrap();
    let (decode, batch) =
        transition_sequences(&plan, &ops, &backend, &KIMI_TOKENS, Some(&mut state));
    assert_same_sequences("kda/mla", &decode, &batch, KIMI_TOKENS.len());
}

/// The sequence witness is not vacuous: a batch stream missing one
/// transition at one position disagrees at exactly that position.
#[test]
fn control_a_dropped_transition_is_caught_at_exactly_its_position() {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    let (decode, mut batch) = transition_sequences(&plan, &ops, &backend, &G_TOKENS, None);
    batch.get_mut(&1).unwrap().pop();
    let differing: Vec<usize> = decode
        .keys()
        .filter(|position| batch[position] != decode[position])
        .copied()
        .collect();
    assert_eq!(differing, vec![1]);
}
