//! V3-LENS-1 on the record

use super::*;

#[test]
fn lens_readouts_are_recorded_in_sequence_priced_on_the_receipt_and_replay_equal() {
    use crate::vindex3::{LensLayers, LensSites, LogitLens};
    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let plain = run(&f.plan, &ops, &backend, &mut NoopObserver);
    let sites = LensSites {
        layers: LensLayers::All,
        attention: false,
        ffn: true,
    };
    let lens = LogitLens::new(&ops, &backend, sites, vec![3, 17], 2);
    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_lens(Box::new(lens));
    let recorded = run(&f.plan, &ops, &backend, &mut recorder);
    assert_eq!(plain, recorded, "the lens on the record changed the logits");
    let readouts: Vec<&RecordedEvent> = recorder
        .events()
        .iter()
        .filter(|e| matches!(e.event, EventKind::Readout { .. }))
        .collect();
    assert_eq!(
        readouts.len(),
        G_LAYERS * G_TOKENS.len(),
        "one readout per armed site"
    );
    // A readout follows its write's stats and precedes the structural write.
    let first = recorder
        .events()
        .iter()
        .position(|e| matches!(e.event, EventKind::Readout { .. }))
        .unwrap();
    assert!(matches!(
        recorder.events()[first - 1].event,
        EventKind::CarrierStats { .. }
    ));
    assert!(matches!(
        recorder.events()[first + 1].event,
        EventKind::CarrierWrite { .. }
    ));
    for r in &readouts {
        let EventKind::Readout {
            site,
            method,
            tokens,
            top,
            ..
        } = &r.event
        else {
            unreachable!()
        };
        assert!(matches!(site, crate::vindex3::Site::Ffn));
        assert_eq!(method, crate::vindex3::LENS_METHOD);
        assert_eq!(tokens.iter().map(|t| t.id).collect::<Vec<_>>(), vec![3, 17]);
        assert!(tokens.iter().all(|t| t.logprob <= 0.0 && t.rank >= 1));
        assert_eq!(top.len(), 2);
    }
    // The last layer's readout of the greedy argmax has rank 1: the lens
    // at the exit is the executor's own distribution.
    let last = readouts.last().unwrap();
    let EventKind::Readout { top, .. } = &last.event else {
        unreachable!()
    };
    let argmax = plain
        .last()
        .unwrap()
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0 as u32;
    assert_eq!(top[0].id, argmax);

    recorder.complete();
    let total = recorder.events().len();
    let record = recorder.finish();
    assert_eq!(
        record.receipt.head_passes as usize,
        G_LAYERS * G_TOKENS.len()
    );
    assert_eq!(record.receipt.lens_failure, None);
    assert_eq!(record.events.len(), total);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lens.jsonl");
    record.write_jsonl(&path).unwrap();
    assert_record_eq(&RunRecord::read_jsonl(&path).unwrap(), &record);
}

#[test]
fn a_failing_lens_is_named_on_the_receipt_and_the_record_is_still_complete() {
    use crate::vindex3::{LensSites, LogitLens};
    let f = fixture();
    let backend = ReferenceBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let lens = LogitLens::new(&ops, &backend, LensSites::every_ffn(), vec![u32::MAX], 0);
    let mut recorder = RunRecorder::for_image(identity(), &ops, None).with_lens(Box::new(lens));
    run(&f.plan, &ops, &backend, &mut recorder);
    recorder.complete();
    let record = recorder.finish();
    assert_eq!(
        record.receipt.head_passes, 1,
        "it stopped after the first refusal"
    );
    assert!(record
        .receipt
        .lens_failure
        .as_deref()
        .unwrap()
        .contains("outside the head's vocabulary"));
    assert!(!record
        .events
        .iter()
        .any(|e| matches!(e.event, EventKind::Readout { .. })));
    assert!(record.receipt.complete);
}

/// V3-HEAD-OBS-1, HP6: with a head reader armed, the record carries one
/// `HeadSum` and one `HeadWrite` per head beneath every attention write,
/// the structural `HeadsObserved` event, counts on the receipt, an
/// uncovered layer named once, and the whole record reads back equal.
#[test]
fn hp6_an_observed_run_records_its_head_rows_and_counts_them_on_the_receipt() {
    use larql_vindex::format::vindex3::fixtures::{G_LAYERS, G_Q_HEADS};
    use larql_vindex::format::vindex3::opplan::exec::observe::StepEvent;
    use larql_vindex::format::vindex3::opplan::exec::observe_heads::HeadStats;

    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let heads = HeadStats::new(
        &ops,
        &f.plan,
        &backend,
        Some(FixedBasis::seeded(G_HIDDEN, DIMS, SEED).unwrap()),
        2,
    );
    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_heads(Box::new(heads));
    run(&f.plan, &ops, &backend, &mut recorder);
    recorder.complete();
    let written = recorder.finish();

    let writes = G_TOKENS.len() * G_LAYERS;
    let sums = written
        .events
        .iter()
        .filter(|e| matches!(e.event, EventKind::HeadSum { .. }))
        .count();
    let rows = written
        .events
        .iter()
        .filter(|e| matches!(e.event, EventKind::HeadWrite { .. }))
        .count();
    let observed = written
        .events
        .iter()
        .filter(|e| matches!(e.event, EventKind::HeadsObserved { .. }))
        .count();
    assert_eq!(sums, writes, "one head-sum per attention write");
    assert_eq!(
        rows,
        writes * G_Q_HEADS,
        "one row per head per attention write"
    );
    assert_eq!(observed, writes);
    for e in &written.events {
        match &e.event {
            EventKind::HeadSum {
                method,
                residual,
                heads,
                ..
            } => {
                assert_eq!(method, "head-sum-through-post-norm/v1");
                assert!(*residual <= 1e-5, "residual {residual}");
                assert_eq!(*heads, G_Q_HEADS);
            }
            EventKind::HeadWrite {
                projection,
                sources,
                sink,
                norm,
                ..
            } => {
                assert_eq!(projection.len(), DIMS);
                assert!(!sources.is_empty() && sources.len() <= 2);
                assert_eq!(*sink, 0.0);
                assert!(norm.is_finite());
            }
            _ => {}
        }
    }
    // Order beneath the write: HeadsObserved, then HeadSum and the rows,
    // then the write's stats and its structural event.
    let first_sum = written
        .events
        .iter()
        .position(|e| matches!(e.event, EventKind::HeadSum { .. }))
        .unwrap();
    assert!(matches!(
        written.events[first_sum - 1].event,
        EventKind::HeadsObserved { .. }
    ));
    assert!(
        matches!(
            written.events[first_sum + G_Q_HEADS + 1].event,
            EventKind::CarrierStats { .. }
        ),
        "{:?}",
        written.events[first_sum.saturating_sub(2)..first_sum + G_Q_HEADS + 2]
            .iter()
            .map(|e| format!("{:?}", e.event)
                .chars()
                .take(60)
                .collect::<String>())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        written.receipt.head_records,
        u64::try_from(writes * G_Q_HEADS).unwrap()
    );
    assert!(written.receipt.head_layers_uncovered.is_empty());
    assert_eq!(written.receipt.head_failure, None);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("heads.jsonl");
    written.write_jsonl(&path).unwrap();
    let read = RunRecord::read_jsonl(&path).unwrap();
    assert_record_eq(&read, &written);
    assert_eq!(read.receipt, written.receipt);

    // An unarmed recorder asks for nothing and counts nothing.
    let mut plain = RunRecorder::for_image(identity(), &ops, None);
    assert!(!plain.wants_attention_heads());
    run(&f.plan, &ops, &backend, &mut plain);
    let plain = plain.finish();
    assert_eq!(plain.receipt.head_records, 0);
    assert!(!plain.events.iter().any(|e| matches!(
        e.event,
        EventKind::HeadSum { .. } | EventKind::HeadsObserved { .. }
    )));

    // An uncovered layer, as the executor would name it on a mixed plan,
    // reaches the receipt once however often it fires.
    let mut mixed = RunRecorder::for_image(identity(), &ops, None);
    mixed.event(StepEvent::HeadsUncovered { layer: 1 });
    mixed.event(StepEvent::HeadsUncovered { layer: 1 });
    mixed.event(StepEvent::HeadsUncovered { layer: 0 });
    let mixed = mixed.finish();
    assert_eq!(mixed.receipt.head_layers_uncovered, vec![0, 1]);
    assert_eq!(
        mixed
            .events
            .iter()
            .filter(|e| matches!(e.event, EventKind::HeadsUncovered { .. }))
            .count(),
        3
    );
}

/// V3-INTERVENE-1, IP6: an intervened run's record carries the
/// declaration in its identity and receipt, the firing as an event at its
/// position ahead of the write it changed, counts firings against
/// declarations, reads back equal, and names a declared address the run
/// never reached — only once the run is complete.
#[test]
fn ip6_an_intervened_record_carries_its_declaration_firings_and_refusal() {
    use crate::vindex3::record::{Kind, Site};
    use larql_vindex::format::vindex3::opplan::exec::intervene::{
        Address, Intervention, InterventionPlan,
    };
    use larql_vindex::format::vindex3::opplan::exec::observe::SublayerSite;

    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let layer = G_LAYERS - 1;
    let fired_at = 2usize;
    let never = 40usize;
    let declared = InterventionPlan::none()
        .with(Intervention::zero(
            Address::new(layer, SublayerSite::Ffn, [fired_at, never]).unwrap(),
        ))
        .unwrap();

    let step_all = |plan: &InterventionPlan, recorder: &mut RunRecorder| -> usize {
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&f.plan, &ops, &backend, &mut kv).unwrap();
        let heads = larql_vindex::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan::none();
        G_TOKENS
            .iter()
            .map(|&t| {
                session
                    .step_intervened(t, recorder, plan, &heads)
                    .unwrap()
                    .firings
                    .len()
            })
            .sum()
    };

    let mut recorder =
        RunRecorder::for_image(identity(), &ops, Some(stats())).with_interventions(&declared);
    assert_eq!(
        step_all(&declared, &mut recorder),
        1,
        "one position reached, one firing"
    );
    recorder.complete();
    let written = recorder.finish();

    // Identity: the declaration's hash joins it, and a baseline differs.
    assert_eq!(
        written.identity.intervention_sha256,
        declared.declaration_sha256()
    );
    assert!(written.identity.intervention_sha256.is_some());
    let baseline = RunRecorder::for_image(identity(), &ops, None).finish();
    assert_eq!(baseline.identity.intervention_sha256, None);
    assert_ne!(
        baseline.identity.intervention_sha256, written.identity.intervention_sha256,
        "an intervened run is never read as a baseline"
    );

    // The event: at its position, before the write's stats and its
    // structural event, exactly once.
    let idx = written
        .events
        .iter()
        .position(|e| matches!(e.event, EventKind::Intervened { .. }))
        .expect("the firing is on the record");
    assert_eq!(written.events[idx].position, fired_at);
    assert_eq!(
        written.events[idx].event,
        EventKind::Intervened {
            layer,
            site: Site::Ffn,
            intervention: Kind::Zero,
        }
    );
    assert!(matches!(
        &written.events[idx + 1].event,
        EventKind::CarrierStats { layer: l, site: Site::Ffn, .. } if *l == layer
    ));
    assert!(matches!(
        &written.events[idx + 2].event,
        EventKind::CarrierWrite { layer: l, site: Site::Ffn, .. } if *l == layer
    ));
    assert_eq!(
        written
            .events
            .iter()
            .filter(|e| matches!(e.event, EventKind::Intervened { .. }))
            .count(),
        1
    );

    // The receipt: counted, and the unreached position named.
    assert_eq!(written.receipt.interventions_declared, 1);
    assert_eq!(written.receipt.interventions_applied, 1);
    let refusal = written
        .receipt
        .intervention_refusal
        .as_deref()
        .expect("position 40 was never reached");
    assert!(refusal.contains("position 40"), "{refusal}");
    assert!(refusal.contains("executed 5 position(s)"), "{refusal}");

    // Round trip: the record reads back equal, identity and receipt included.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("intervened.jsonl");
    written.write_jsonl(&path).unwrap();
    let read = RunRecord::read_jsonl(&path).unwrap();
    assert_record_eq(&read, &written);
    assert_eq!(read.identity, written.identity);
    assert_eq!(read.receipt, written.receipt);

    // An incomplete record is a prefix: it has not failed to reach anything.
    let mut prefix = RunRecorder::for_image(identity(), &ops, None).with_interventions(&declared);
    step_all(&declared, &mut prefix);
    let prefix = prefix.finish();
    assert!(!prefix.receipt.complete);
    assert_eq!(prefix.receipt.intervention_refusal, None);
    assert_eq!(prefix.receipt.interventions_applied, 1);

    // A declaration the run fully reaches carries no refusal.
    let reached = InterventionPlan::none()
        .with(Intervention::zero(
            Address::new(layer, SublayerSite::Ffn, [fired_at]).unwrap(),
        ))
        .unwrap();
    let mut recorder = RunRecorder::for_image(identity(), &ops, None).with_interventions(&reached);
    assert_eq!(step_all(&reached, &mut recorder), 1);
    recorder.complete();
    let complete = recorder.finish();
    assert_eq!(complete.receipt.intervention_refusal, None);
    assert_eq!(complete.receipt.interventions_declared, 1);
    assert_eq!(complete.receipt.interventions_applied, 1);

    // An unintervened record still says so on its receipt.
    assert_eq!(baseline.receipt.interventions_declared, 0);
    assert_eq!(baseline.receipt.interventions_applied, 0);
    assert_eq!(baseline.receipt.intervention_refusal, None);
}

/// V3-INTERVENE-2, J5/JP3: zeroing one head's `ctx_h` on a plan WITH a
/// post-attention norm and a gate produces a measurable shortcut gap —
/// the subtractive shortcut (`delta_base − c′_h`) is not the model's own
/// counterfactual, computed from this SAME run's pre-intervention head
/// records, never a second run.
#[test]
fn jp3_zeroing_a_head_on_a_gated_post_norm_plan_produces_a_measurable_shortcut_gap() {
    use larql_vindex::format::vindex3::opplan::exec::intervene_heads::{
        HeadAddress, HeadIntervention, HeadInterventionPlan,
    };
    use larql_vindex::format::vindex3::opplan::exec::observe_heads::HeadStats;

    let f = fixture();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let layer = 0;
    let head = 0;
    let position = 3;
    let declared = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(
            HeadAddress::new(layer, head, [position]).unwrap(),
        ))
        .unwrap();

    let heads = HeadStats::new(&ops, &f.plan, &backend, None, 3).retaining_children();
    let mut recorder = RunRecorder::for_image(identity(), &ops, None)
        .with_heads(Box::new(heads))
        .with_head_interventions(&declared);
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&f.plan, &ops, &backend, &mut kv).unwrap();
    for &t in G_TOKENS.iter() {
        session
            .step_intervened(
                t,
                &mut recorder,
                &larql_vindex::format::vindex3::opplan::exec::intervene::InterventionPlan::none(),
                &declared,
            )
            .unwrap();
    }
    recorder.complete();
    let record = recorder.finish();

    assert!(record.identity.head_intervention_sha256.is_some());
    assert_eq!(record.receipt.head_interventions_declared, 1);
    assert_eq!(record.receipt.head_interventions_applied, 1);

    let gap = record
        .events
        .iter()
        .find_map(|e| match &e.event {
            EventKind::HeadInterventionGap {
                layer: l,
                head: h,
                gap,
            } if *l == layer && *h == head => Some(*gap),
            _ => None,
        })
        .expect("the zero firing's gap is on the record");
    assert!(
        gap.is_finite() && gap >= 0.0,
        "gap {gap} is not a valid relative norm"
    );
    assert!(
        gap > 1e-6,
        "gap {gap} is suspiciously small for a gated post-norm layer; the shortcut and the \
         real counterfactual should visibly differ"
    );

    // The event precedes the write it explains, same as HeadSum/HeadWrite.
    let gap_idx = record
        .events
        .iter()
        .position(|e| matches!(e.event, EventKind::HeadInterventionGap { .. }))
        .unwrap();
    let write_idx = record
        .events
        .iter()
        .position(|e| {
            matches!(
                &e.event,
                EventKind::CarrierWrite {
                    site: Site::Attention,
                    ..
                }
            ) && e.position == position
        })
        .unwrap();
    assert!(
        gap_idx < write_idx,
        "the gap precedes the write it explains"
    );
}
