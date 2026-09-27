//! RESIDUAL-BUS-2 S1–S3 (`docs/residual-bus-2.md`): one sequence guard
//! accepts every unaltered batch, decode and chunked-prefill stream on
//! every subject, and refuses each defect at exactly the rule it breaks.

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
    CarrierForm, CarrierTransition, StepEvent, StepObserver,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::sequence::{
    declared_transitions, SequenceGuard, SequenceRefusal, Sequenced, Sequencer, StreamId,
};
use crate::format::vindex3::opplan::exec::{
    execute_prepared_streaming, execute_prepared_streaming_in, prefill_prepared_observed,
    PlaneEvent,
};
use crate::format::vindex3::opplan::tests::kda_mla_exec::kimi_fixture;
use crate::format::vindex3::opplan::ComponentOpPlan;

const SMALL_TOKENS: [u32; 3] = [3, 1, 5];
const KIMI_TOKENS: [u32; 6] = [3, 17, 5, 9, 12, 1];
/// Stand-in identity digests: the guard compares them, it does not derive
/// them (that is `identity::ExecutionIdentity`, tested on its own).
const IDENTITY: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const FOREIGN: &str = "2222222222222222222222222222222222222222222222222222222222222222";

type Stream = Vec<(CarrierAddress, CarrierTransition)>;

fn stream_id(identity: &str) -> StreamId {
    StreamId {
        identity: identity.to_string(),
        run: 7,
    }
}

fn stamp(stream: &Stream, identity: &str) -> Vec<Sequenced> {
    let mut sequencer = Sequencer::new(stream_id(identity));
    stream
        .iter()
        .map(|(address, transition)| sequencer.stamp(*address, *transition))
        .collect()
}

#[derive(Default)]
struct DecodeStream(Stream);

impl StepObserver for DecodeStream {
    fn event(&mut self, _event: StepEvent) {}
    fn transition(&mut self, address: CarrierAddress, transition: CarrierTransition) {
        self.0.push((address, transition));
    }
}

fn decode<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
) -> Stream {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    let mut log = DecodeStream::default();
    for &token in tokens {
        session.step_observed(token, &mut log).unwrap();
    }
    log.0
}

fn transitions_of(
    stream: &mut Stream,
) -> impl FnMut(PlaneEvent) -> Result<(), crate::error::VindexError> + '_ {
    |event| {
        if let PlaneEvent::Transition {
            address,
            transition,
        } = event
        {
            stream.push((address, transition));
        }
        Ok(())
    }
}

/// One batch traversal, with a provider when the plan keeps state.
fn batch<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    stateful: bool,
) -> Stream {
    let mut stream = Stream::new();
    {
        let mut sink = transitions_of(&mut stream);
        if stateful {
            execute_prepared_streaming_in(
                plan,
                ops,
                tokens,
                backend,
                None,
                &mut sink,
                Box::new(RowKvState::default()),
            )
            .unwrap();
        } else {
            execute_prepared_streaming(plan, ops, tokens, backend, None, &mut sink).unwrap();
        }
    }
    stream
}

/// Open a guard over `domain` for `ops`, and run `stream` through it.
fn guard(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    form: CarrierForm,
    domain: std::ops::Range<usize>,
    stream: &[Sequenced],
) -> Result<(), SequenceRefusal> {
    let mut guard = SequenceGuard::open(
        stream_id(IDENTITY),
        0..plan.layers.len(),
        form,
        domain,
        declared_transitions(plan, ops),
    );
    for next in stream {
        guard.accept(next)?;
    }
    guard.close()
}

/// S1/S2 acceptance on one subject: batch, decode and a prefill chunked
/// at every cut all pass one guard per operation.
fn assert_accepted<B: PlanBackend>(
    name: &str,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    form: CarrierForm,
    stateful: bool,
) {
    let all = 0..tokens.len();
    let decoded = stamp(&decode(plan, ops, backend, tokens), IDENTITY);
    guard(plan, ops, form, all.clone(), &decoded)
        .unwrap_or_else(|e| panic!("{name}: decode refused: {e}"));
    let batched = stamp(&batch(plan, ops, backend, tokens, stateful), IDENTITY);
    guard(plan, ops, form, all, &batched).unwrap_or_else(|e| panic!("{name}: batch refused: {e}"));
    // F3: batch (layer-major) and decode (position-major) interleave
    // positions differently, and still stamp every transition with the
    // same ordinal at the same address.
    let by_position = |stream: &[Sequenced]| {
        let mut keyed: Vec<(usize, usize, CarrierAddress, CarrierTransition)> = stream
            .iter()
            .map(|s| (s.address.position, s.ordinal, s.address, s.transition))
            .collect();
        keyed.sort_by_key(|k| (k.0, k.1));
        keyed
    };
    assert_eq!(
        by_position(&batched),
        by_position(&decoded),
        "{name}: F3, batch and decode ordinals differ"
    );
    for cut in 1..tokens.len() {
        let mut kv = RowKvState::default();
        for chunk in [0..cut, cut..tokens.len()] {
            let mut stream = Stream::new();
            prefill_prepared_observed(
                plan,
                ops,
                &tokens[chunk.clone()],
                backend,
                &mut kv,
                &mut transitions_of(&mut stream),
            )
            .unwrap();
            guard(plan, ops, form, chunk.clone(), &stamp(&stream, IDENTITY))
                .unwrap_or_else(|e| panic!("{name}: chunk {chunk:?} of cut {cut} refused: {e}"));
        }
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
fn s2_the_plain_stack_is_accepted_on_every_path() {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    assert_accepted(
        "plain",
        &plan,
        &ops,
        &backend,
        &G_TOKENS,
        CarrierForm::Single,
        false,
    );
}

#[test]
fn s2_the_layer_scale_is_accepted_on_every_path() {
    let (_c, plan, store) = gemma4_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    assert_accepted(
        "gemma4",
        &plan,
        &ops,
        &backend,
        &SMALL_TOKENS,
        CarrierForm::Single,
        false,
    );
}

#[test]
fn s2_a_bundle_is_accepted_on_every_path() {
    let sub = wave19_hc_substrate::build(Variant::HeadBearing);
    let ops = wave19_hc_decode::prepare(&sub, ExecutionSlice::Full).unwrap();
    assert_accepted(
        "bundle",
        &sub.plan,
        &ops,
        &ReferenceBackend::new(),
        &SMALL_TOKENS,
        CarrierForm::Bundle,
        false,
    );
}

#[test]
fn s2_a_history_is_accepted_on_every_path() {
    let sub = attn_res_substrate::substrate();
    let (_store, ops) = attn_res_substrate::prepare(&sub);
    assert_accepted(
        "history",
        &sub.plan,
        &ops,
        &ReferenceBackend::new(),
        &ATTN_RES_TOKENS,
        CarrierForm::History,
        false,
    );
}

#[test]
fn s2_mixed_kda_mla_is_accepted_on_every_path() {
    let (_d, _c, plan, store) = kimi_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    assert_accepted(
        "kda/mla",
        &plan,
        &ops,
        &backend,
        &KIMI_TOKENS,
        CarrierForm::Single,
        true,
    );
}

/// The plain stack's decode stream, stamped, with its declared count.
fn plain() -> (Vec<Sequenced>, ComponentOpPlan, PreparedOperands) {
    let (_c, plan, store) = golden_fixture();
    let backend = ReferenceBackend::new();
    let ops = full_ops(&plan, &store, &backend);
    let stream = stamp(&decode(&plan, &ops, &backend, &G_TOKENS), IDENTITY);
    (stream, plan, ops)
}

fn refusal(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    domain: std::ops::Range<usize>,
    stream: &[Sequenced],
) -> SequenceRefusal {
    guard(plan, ops, CarrierForm::Single, domain, stream).expect_err("the defect must be refused")
}

#[test]
fn control_each_immediate_defect_is_refused_by_its_rule() {
    let (stream, plan, ops) = plain();
    let all = 0..G_TOKENS.len();

    let mut duplicated = stream.clone();
    duplicated.insert(3, duplicated[2].clone());
    assert!(matches!(
        refusal(&plan, &ops, all.clone(), &duplicated),
        SequenceRefusal::Duplicate { .. }
    ));

    let mut dropped = stream.clone();
    dropped.remove(2);
    assert!(matches!(
        refusal(&plan, &ops, all.clone(), &dropped),
        SequenceRefusal::Gap { .. }
    ));

    let mut swapped = stream.clone();
    swapped.swap(2, 3);
    assert!(matches!(
        refusal(&plan, &ops, all.clone(), &swapped),
        SequenceRefusal::Gap { .. }
    ));

    assert!(matches!(
        refusal(&plan, &ops, 0..1, &stream),
        SequenceRefusal::PositionOutsideDomain { .. }
    ));

    let foreign = stamp(
        &stream.iter().map(|s| (s.address, s.transition)).collect(),
        FOREIGN,
    );
    assert!(matches!(
        refusal(&plan, &ops, all.clone(), &foreign),
        SequenceRefusal::StaleIdentity { .. }
    ));

    let mut relabelled = stream.clone();
    relabelled[1].address.form = CarrierForm::Bundle;
    assert!(matches!(
        refusal(&plan, &ops, all.clone(), &relabelled),
        SequenceRefusal::WrongForm { .. }
    ));

    let mut narrow = SequenceGuard::open(
        stream_id(IDENTITY),
        0..1,
        CarrierForm::Single,
        all.clone(),
        declared_transitions(&plan, &ops),
    );
    let beyond = stream
        .iter()
        .find(|s| s.address.layer >= 1)
        .expect("a transition past layer 0");
    for next in stream.iter().take_while(|s| s.address.layer < 1) {
        narrow.accept(next).unwrap();
    }
    assert!(matches!(
        narrow.accept(beyond),
        Err(SequenceRefusal::LayerOutOfRange { .. })
    ));
    assert_eq!(narrow.accept(&stream[0]), Err(SequenceRefusal::StreamEnded));
}

#[test]
fn control_a_withheld_or_truncated_position_is_refused_at_close() {
    let (stream, plan, ops) = plain();
    let all = 0..G_TOKENS.len();

    let withheld: Vec<Sequenced> = stream
        .iter()
        .filter(|s| s.address.position != 1)
        .cloned()
        .collect();
    assert_eq!(
        refusal(&plan, &ops, all.clone(), &withheld),
        SequenceRefusal::MissingPosition { position: 1 }
    );

    let last_of_one = stream
        .iter()
        .rposition(|s| s.address.position == 1)
        .unwrap();
    let mut truncated = stream.clone();
    truncated.remove(last_of_one);
    assert!(matches!(
        refusal(&plan, &ops, all, &truncated),
        SequenceRefusal::Truncated { position: 1, .. }
    ));
}

#[test]
fn control_an_ordinal_past_the_declared_count_is_refused() {
    let (stream, plan, ops) = plain();
    let declared = declared_transitions(&plan, &ops);
    let mut guard = SequenceGuard::open(
        stream_id(IDENTITY),
        0..plan.layers.len(),
        CarrierForm::Single,
        0..G_TOKENS.len(),
        declared,
    );
    let at_zero: Vec<&Sequenced> = stream.iter().filter(|s| s.address.position == 0).collect();
    for next in &at_zero {
        guard.accept(next).unwrap();
    }
    let mut extra = (*at_zero.last().unwrap()).clone();
    extra.ordinal = declared;
    assert!(matches!(
        guard.accept(&extra),
        Err(SequenceRefusal::BeyondDeclared { .. })
    ));
}

/// Every refusal names its cause: a receiver's log must say which rule a
/// stream broke, not merely that it was refused.
#[test]
fn every_refusal_says_which_rule_it_broke() {
    let cases = [
        (
            SequenceRefusal::StaleIdentity {
                bound: IDENTITY.into(),
                got: FOREIGN.into(),
            },
            "stream bound to",
        ),
        (
            SequenceRefusal::LayerOutOfRange {
                layer: 3,
                bound: 0..2,
            },
            "outside the bound layer range",
        ),
        (
            SequenceRefusal::WrongForm {
                bound: CarrierForm::Single,
                got: CarrierForm::History,
            },
            "History carrier",
        ),
        (
            SequenceRefusal::PositionOutsideDomain {
                position: 9,
                domain: 0..3,
            },
            "declared domain",
        ),
        (
            SequenceRefusal::Duplicate {
                position: 1,
                ordinal: 2,
            },
            "arrived twice",
        ),
        (
            SequenceRefusal::Gap {
                position: 1,
                expected: 2,
                got: 4,
            },
            "missing or out of order",
        ),
        (
            SequenceRefusal::BeyondDeclared {
                position: 1,
                ordinal: 9,
                declared: 9,
            },
            "the plan declares",
        ),
        (
            SequenceRefusal::MissingPosition { position: 2 },
            "never arrived",
        ),
        (
            SequenceRefusal::Truncated {
                position: 2,
                received: 3,
                declared: 9,
            },
            "stopped after 3 of its 9",
        ),
        (SequenceRefusal::StreamEnded, "already ended"),
    ];
    for (refusal, names) in cases {
        let text = refusal.to_string();
        assert!(text.contains(names), "{refusal:?} rendered as {text}");
    }
}

/// The four subjects other than the plain stack, on the Production backend.
#[test]
fn s2_every_subject_is_accepted_on_the_production_backend() {
    let backend = ProductionBackend::new();
    let (_c, plan, store) = gemma4_fixture();
    let ops = full_ops(&plan, &store, &backend);
    assert_accepted(
        "gemma4, production",
        &plan,
        &ops,
        &backend,
        &SMALL_TOKENS,
        CarrierForm::Single,
        false,
    );

    let sub = wave19_hc_substrate::build(Variant::HeadBearing);
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = full_ops(&sub.plan, &store, &backend);
    assert_accepted(
        "bundle, production",
        &sub.plan,
        &ops,
        &backend,
        &SMALL_TOKENS,
        CarrierForm::Bundle,
        false,
    );

    let sub = attn_res_substrate::substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = full_ops(&sub.plan, &store, &backend);
    assert_accepted(
        "history, production",
        &sub.plan,
        &ops,
        &backend,
        &ATTN_RES_TOKENS,
        CarrierForm::History,
        false,
    );

    let (_d, _c, plan, store) = kimi_fixture();
    let ops = full_ops(&plan, &store, &backend);
    assert_accepted(
        "kda/mla, production",
        &plan,
        &ops,
        &backend,
        &KIMI_TOKENS,
        CarrierForm::Single,
        true,
    );
}
