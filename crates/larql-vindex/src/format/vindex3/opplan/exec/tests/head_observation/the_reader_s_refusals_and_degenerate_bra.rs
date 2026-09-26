//! The reader's refusals and degenerate branches, driven directly

use super::*;

#[test]
fn the_reader_refuses_a_write_whose_head_count_is_not_the_layers_and_stays_failed() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 1);
    let weights = [1.0f32];
    let values = [0.5f32; G_HEAD_DIM];
    let row: &[f32] = &values;
    let sources = [row];
    // Two records for a three-head layer.
    for head in 0..2 {
        HeadReader::attention_head(
            &mut stats,
            0,
            synthetic_record(head, 0, &weights, &values, &sources),
        );
    }
    assert_eq!(HeadReader::records(&stats), 2);
    let delta = vec![0.25f32; ops.hidden()];
    let write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(stats.finish_write(&write).is_none());
    let failure = HeadReader::failure(&stats)
        .expect("the count mismatch is a failure")
        .to_string();
    assert!(failure.contains("2 head records for 3 heads"), "{failure}");
    // Failed stays failed: later records and writes are refused, the
    // failure is not overwritten, and the pending heads were cleared.
    for head in 0..G_Q_HEADS {
        HeadReader::attention_head(
            &mut stats,
            1,
            synthetic_record(head, 0, &weights, &values, &sources),
        );
    }
    let write_1 = CarrierWriteRecord {
        layer: 1,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(stats.finish_write(&write_1).is_none());
    assert!(HeadReader::failure(&stats)
        .unwrap()
        .to_string()
        .contains("2 head records for 3 heads"));
    assert_eq!(
        HeadReader::records(&stats),
        2 + G_Q_HEADS,
        "records are still counted"
    );
}

#[test]
fn the_reader_refuses_a_layer_without_softmax_heads() {
    let (_c, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let other = (0..plan.layers.len())
        .find(|&i| plan.layers[i].attention.softmax().is_none())
        .expect("the hybrid fixture has a non-softmax layer");
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 1);
    let weights = [1.0f32];
    let values = [0.5f32; G_HEAD_DIM];
    let row: &[f32] = &values;
    let sources = [row];
    HeadReader::attention_head(
        &mut stats,
        other,
        synthetic_record(0, 0, &weights, &values, &sources),
    );
    let delta = vec![0.25f32; ops.hidden()];
    let write = CarrierWriteRecord {
        layer: other,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(stats.finish_write(&write).is_none());
    let failure = HeadReader::failure(&stats).unwrap().to_string();
    assert!(failure.contains("has no softmax attention"), "{failure}");
}

#[test]
fn a_zero_delta_yields_an_absolute_residual_and_the_projection_is_empty_without_a_basis() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 2).retaining_children();
    let weights = [0.25f32, 0.75];
    let values = [0.5f32; G_HEAD_DIM];
    let row: &[f32] = &values;
    let sources = [row, row];
    for head in 0..G_Q_HEADS {
        HeadReader::attention_head(
            &mut stats,
            0,
            synthetic_record(head, 1, &weights, &values, &sources),
        );
    }
    let zero = vec![0.0f32; ops.hidden()];
    let write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 1,
        delta: &zero,
        after: &zero,
        layer_scale: None,
    };
    let decomposed = stats
        .finish_write(&write)
        .expect("a full set of heads decomposes");
    assert!(HeadReader::failure(&stats).is_none());
    // With a zero delta the residual is the absolute head-sum norm, not a ratio.
    let children = decomposed.children.as_ref().unwrap();
    let hidden = ops.hidden();
    let mut sum = vec![0.0f64; hidden];
    for child in children {
        for (acc, c) in sum.iter_mut().zip(child) {
            *acc += f64::from(*c);
        }
    }
    let expected: f64 = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((decomposed.residual - expected).abs() <= 1e-9 * expected.max(1.0));
    assert!(
        decomposed.residual > 0.0,
        "the heads wrote something the zero delta did not"
    );
    for row in &decomposed.rows {
        assert!(row.projection.is_empty(), "no basis, no projection");
        assert_eq!(row.sources.len(), 2);
        assert_eq!(row.sources[0], (1, 0.75), "sources descend by weight");
        assert_eq!(row.sources[1], (0, 0.25));
    }
    // The pending heads were consumed: the same write again decomposes nothing.
    assert!(stats.finish_write(&write).is_none());
    assert!(stats.basis().is_none());
}
