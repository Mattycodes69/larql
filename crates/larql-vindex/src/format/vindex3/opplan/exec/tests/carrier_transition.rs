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
use crate::format::vindex3::opplan::exec::observe::{
    CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan_streaming, PlaneEvent};
use crate::format::vindex3::opplan::ComponentOpPlan;

/// Tokens inside every miniature's vocabulary (the smallest is 7).
const SMALL_TOKENS: [u32; 3] = [3, 1, 5];

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
    let mut records = BTreeMap::new();
    let out = execute_plan_streaming(plan, store, tokens, backend, None, &mut |event| {
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
    })
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
