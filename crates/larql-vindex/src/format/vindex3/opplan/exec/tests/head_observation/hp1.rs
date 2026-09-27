//! HP1
//! HP2

use super::*;

#[test]
fn hp1_heads_armed_leave_logits_writes_and_events_bit_identical() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let (plain_logits, plain) = run(&plan, &store, backend, false);
        let (head_logits, tapped) = run(&plan, &store, backend, true);
        assert_eq!(plain_logits.len(), head_logits.len());
        for (p, (a, b)) in plain_logits.iter().zip(&head_logits).enumerate() {
            assert!(
                bits_equal(a, b),
                "{}: logits differ at position {p}",
                backend.name()
            );
        }
        assert_eq!(plain.delta.len(), tapped.delta.len());
        for (key, delta) in &plain.delta {
            assert!(
                bits_equal(delta, &tapped.delta[key]),
                "delta moved at {key:?}"
            );
            assert!(
                bits_equal(&plain.after[key], &tapped.after[key]),
                "after moved at {key:?}"
            );
            assert_eq!(plain.layer_scale[key], tapped.layer_scale[key]);
        }
        assert_eq!(structural(&plain.events), structural(&tapped.events));
        assert!(plain.records.is_empty(), "nothing asked, nothing fired");
        assert_eq!(
            tapped.records.len(),
            G_TOKENS.len() * G_LAYERS * G_Q_HEADS,
            "one record per head per softmax layer per position"
        );
    });
}

/// A narrowed capture is a capture-cost filter, never an arithmetic
/// change: it fires only at the selected address, leaves the logits
/// bit-identical, and hands over exactly the records a full capture
/// fires there — including the conditioned query and key rows the GW
/// head replay reconstructs from.
#[test]
fn a_narrowed_capture_fires_only_where_asked_and_changes_nothing() {
    let (_c, plan, store) = fixture();
    let target = (G_LAYERS - 1, G_TOKENS.len() - 1);
    on_both_backends!(|backend| {
        let (plain_logits, _) = run(&plan, &store, backend, false);
        let (_, full) = run(&plan, &store, backend, true);
        let (narrow_logits, narrow) = run_narrowed(&plan, &store, backend, true, Some(target));
        for (p, (a, b)) in plain_logits.iter().zip(&narrow_logits).enumerate() {
            assert!(
                bits_equal(a, b),
                "{}: narrowing moved logits at position {p}",
                backend.name()
            );
        }
        assert_eq!(narrow.records.len(), G_Q_HEADS, "one record per head, once");
        let expected: Vec<&Head> = full
            .records
            .iter()
            .filter(|h| (h.layer, h.position) == target)
            .collect();
        assert_eq!(narrow.records.iter().collect::<Vec<_>>(), expected);
        let observed: Vec<_> = narrow
            .events
            .iter()
            .filter(|(_, e)| matches!(e, StepEvent::HeadsObserved { .. }))
            .collect();
        assert_eq!(
            observed,
            vec![&(
                target.1,
                StepEvent::HeadsObserved {
                    layer: target.0,
                    heads: G_Q_HEADS
                }
            )]
        );
        for head in &full.records {
            assert_eq!(head.query.len(), G_HEAD_DIM);
            assert_eq!(head.source_keys.len(), head.weights.len());
            assert!(head.source_keys.iter().all(|k| k.len() == G_HEAD_DIM));
        }
    });
}

#[test]
fn hp2_every_record_says_what_the_kernel_computed() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let (_, tapped) = run(&plan, &store, backend, true);
        let group = G_Q_HEADS / G_KV_HEADS;
        let mut seen = std::collections::BTreeSet::new();
        let mut truncated = false;
        for r in &tapped.records {
            assert!(
                seen.insert((r.layer, r.position, r.head)),
                "duplicate record"
            );
            assert_eq!(r.kv_head, r.head / group);
            assert_eq!(r.values.len(), G_HEAD_DIM);
            assert!(r.values.iter().all(|v| v.is_finite()));
            // The golden plan declares an output gate: the record carries
            // its activated slice, one value per element of the head.
            let gate = r
                .gate
                .as_ref()
                .expect("the golden plan declares an output gate");
            assert_eq!(gate.len(), G_HEAD_DIM);
            assert!(
                gate.iter().all(|g| *g > 0.0 && *g < 1.0),
                "a sigmoid gate: {gate:?}"
            );
            assert_eq!(r.weights.len(), r.position + 1 - r.source_start);
            assert_eq!(r.source_values.len(), r.weights.len());
            assert!(r.weights.iter().all(|w| (0.0..=1.0).contains(w)));
            let mass: f32 = r.weights.iter().sum::<f32>() + r.sink;
            assert!((mass - 1.0).abs() < 1e-6, "mass {mass} at {r:?}");
            assert_eq!(r.sink, 0.0, "no sinks on the golden plan");
            // The window: layer 0 slides with G_WINDOW, layer 1 is full.
            if r.layer == 0 && r.position + 1 > G_WINDOW {
                assert_eq!(r.source_start, r.position + 1 - G_WINDOW);
                assert_eq!(r.weights.len(), G_WINDOW);
                truncated = true;
            } else {
                assert_eq!(r.source_start, 0);
            }
            // A3: the mixed value is the weighted sum of the source rows.
            let rebuilt: Vec<f32> = (0..G_HEAD_DIM)
                .map(|i| {
                    r.weights
                        .iter()
                        .zip(&r.source_values)
                        .map(|(w, v)| w * v[i])
                        .sum::<f32>()
                })
                .collect();
            for (a, b) in rebuilt.iter().zip(&r.values) {
                assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-6), "A3: {a} vs {b}");
            }
        }
        assert!(truncated, "the fixture crosses the sliding window");
    });
}
