//! Events, coverage, refusal

use super::*;

#[test]
fn heads_observed_precedes_the_attention_write_once_per_layer_and_only_when_asked() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let (_, plain) = run(&plan, &store, &backend, false);
    assert!(!plain.events.iter().any(|(_, e)| matches!(
        e,
        StepEvent::HeadsObserved { .. } | StepEvent::HeadsUncovered { .. }
    )));
    let (_, tapped) = run(&plan, &store, &backend, true);
    let observed: Vec<&(usize, StepEvent)> = tapped
        .events
        .iter()
        .filter(|(_, e)| matches!(e, StepEvent::HeadsObserved { .. }))
        .collect();
    assert_eq!(observed.len(), G_TOKENS.len() * G_LAYERS);
    assert!(!tapped
        .events
        .iter()
        .any(|(_, e)| matches!(e, StepEvent::HeadsUncovered { .. })));
    for (i, (position, event)) in tapped.events.iter().enumerate() {
        if let StepEvent::HeadsObserved { layer, heads } = event {
            assert_eq!(*heads, G_Q_HEADS);
            let next_write = tapped.events[i + 1..]
                .iter()
                .find(|(_, e)| matches!(e, StepEvent::CarrierWrite { .. }))
                .unwrap();
            assert_eq!(
                next_write,
                &(
                    *position,
                    StepEvent::CarrierWrite {
                        layer: *layer,
                        site: SublayerSite::Attention,
                        carrier: crate::format::vindex3::opplan::exec::observe::CarrierForm::Single,
                    }
                ),
                "the heads event precedes its own attention write"
            );
        }
    }
    // Every record for a layer precedes that layer's HeadsObserved event
    // is a property of the kernel order; here: records exist for every
    // (layer, position) the event names.
    for (position, event) in &tapped.events {
        if let StepEvent::HeadsObserved { layer, .. } = event {
            assert_eq!(
                tapped
                    .records
                    .iter()
                    .filter(|r| r.layer == *layer && r.position == *position)
                    .count(),
                G_Q_HEADS
            );
        }
    }
}

#[test]
fn a6_a_mixed_family_plan_names_its_uncovered_layers_and_is_not_refused() {
    let (_c, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let softmax: Vec<usize> = plan
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| l.attention.softmax().is_some())
        .map(|(i, _)| i)
        .collect();
    let other: Vec<usize> = (0..plan.layers.len())
        .filter(|i| !softmax.contains(i))
        .collect();
    assert!(
        !softmax.is_empty() && !other.is_empty(),
        "the fixture mixes families"
    );
    let (_, tapped) = run(&plan, &store, &backend, true);
    for &layer in &softmax {
        assert_eq!(
            tapped
                .events
                .iter()
                .filter(|(_, e)| *e
                    == StepEvent::HeadsObserved {
                        layer,
                        heads: plan.layers[layer].attention.softmax().unwrap().num_q_heads
                    })
                .count(),
            G_TOKENS.len()
        );
    }
    for &layer in &other {
        assert_eq!(
            tapped
                .events
                .iter()
                .filter(|(_, e)| *e == StepEvent::HeadsUncovered { layer })
                .count(),
            G_TOKENS.len()
        );
        assert!(!tapped.records.iter().any(|r| r.layer == layer));
    }
    // And the head-sum law still holds on the covered layers.
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 3);
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    for &t in G_TOKENS.iter() {
        session.step_observed(t, &mut stats).unwrap();
    }
    assert!(stats.failure.is_none(), "{:?}", stats.failure);
    assert_eq!(stats.writes.len(), G_TOKENS.len() * softmax.len());
    assert!(
        stats.writes.iter().all(|w| w.residual <= 1e-5),
        "{:?}",
        stats.writes.iter().map(|w| w.residual).collect::<Vec<_>>()
    );
}

#[test]
fn a8_a_backend_that_does_not_serve_heads_refuses_before_the_first_token() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-heads", WeightFormat::F32);
    assert!(!device.serves_attention_heads());
    let mut session = DecodeSession::new(
        &plan,
        &store,
        &device,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    let mut witness = Witness {
        heads: true,
        ..Witness::default()
    };
    let err = match session.step_observed(G_TOKENS[0], &mut witness) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a backend that does not serve heads must refuse"),
    };
    assert!(err.contains("not served by the"), "{err}");
    assert_eq!(session.position(), 0, "refused before any token executed");
    assert!(witness.events.is_empty());
    // Without heads the same backend runs.
    let mut plain = Witness::default();
    session.step_observed(G_TOKENS[0], &mut plain).unwrap();
    assert_eq!(session.position(), 1);
}

/// The reader trait is what a recorder drives; drive it by hand.
#[test]
fn the_reader_decomposes_only_the_write_whose_heads_it_saw() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 1);
    let delta = vec![0.0f32; ops.hidden()];
    let ffn_write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Ffn,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(
        stats.finish_write(&ffn_write).is_none(),
        "an FFN write has no heads"
    );
    let attention_write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(
        stats.finish_write(&attention_write).is_none(),
        "no heads seen yet"
    );
    assert_eq!(HeadReader::records(&stats), 0);
    assert!(HeadReader::failure(&stats).is_none());
}
