//! HP3 / HP4

use super::*;

#[test]
fn hp3_the_head_sum_law_holds_through_the_post_norm_on_the_golden_plan() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let ops = prepared(&plan, &store, backend);
        let basis = FixedBasis::seeded(ops.hidden(), 3, 42).unwrap();
        let mut stats = HeadStats::new(&ops, &plan, backend, Some(basis), 2).retaining_children();
        let (_, plain) = run(&plan, &store, backend, false);
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, backend, &mut kv).unwrap();
        for &t in G_TOKENS.iter() {
            session.step_observed(t, &mut stats).unwrap();
        }
        assert!(stats.failure.is_none(), "{:?}", stats.failure);
        assert_eq!(
            stats.writes.len(),
            G_TOKENS.len() * G_LAYERS,
            "one decomposition per attention write"
        );
        assert_eq!(stats.records, G_TOKENS.len() * G_LAYERS * G_Q_HEADS);
        let basis = stats.basis().unwrap().rows().to_vec();
        for write in &stats.writes {
            assert!(
                write.residual <= 1e-5,
                "{}: head-sum residual {} at layer {} position {}",
                backend.name(),
                write.residual,
                write.layer,
                write.position
            );
            assert_eq!(write.rows.len(), G_Q_HEADS);
            let delta = &plain.delta[&(write.layer, SublayerSite::Attention, write.position)];
            // The reader-projected form: Σ_h ⟨r, c′_h⟩ = ⟨r, delta⟩. The bound is
            // relative to the L1 scale of the f32 terms being summed, never to the
            // cancelled sum: each stored ⟨r, c′_h⟩ carries f32 rounding proportional
            // to its own magnitude, so the identity's error scales with Σ_h |⟨r, c′_h⟩|
            // (the largest term alone is only a factor ≤ H tighter, with no analytic
            // reason). At the golden plan's first write the heads cancel 44x
            // (Σ_h |·| = 0.948 against a sum of 0.0217); bounded on |⟨r, delta⟩| this
            // read 1.07e-5 on Windows against 3.3e-6 / 7.2e-6 (production / reference)
            // on macOS — reduction order, not a defect. Against the term scale the
            // worst observed margin is 2.4e-7, and a wrong head moves the sum by ~1e-1.
            for (d, row) in basis.iter().enumerate() {
                let lhs: f64 = write.rows.iter().map(|r| f64::from(r.projection[d])).sum();
                let term_scale: f64 = write
                    .rows
                    .iter()
                    .map(|r| f64::from(r.projection[d]).abs())
                    .sum();
                let rhs: f64 = row
                    .iter()
                    .zip(delta)
                    .map(|(r, x)| f64::from(*r) * f64::from(*x))
                    .sum();
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * term_scale.max(1e-3),
                    "projection {d}: {lhs} vs {rhs} (term scale {term_scale})"
                );
            }
            for row in &write.rows {
                assert_eq!(row.sources.len(), 2.min(write.position + 1));
                assert!(
                    row.sources.windows(2).all(|w| w[0].1 >= w[1].1),
                    "sources descend"
                );
                assert!(row.norm.is_finite());
            }
            let children = write.children.as_ref().unwrap();
            assert_eq!(children.len(), G_Q_HEADS);
        }
    });
}

#[test]
fn hp4_the_source_split_sums_to_the_head_through_the_projection() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let ops = prepared(&plan, &store, backend);
        let (_, tapped) = run(&plan, &store, backend, true);
        for r in &tapped.records {
            let whole = ops
                .head_projection(backend, r.layer, r.head, G_HEAD_DIM, G_Q_HEADS, &r.values)
                .unwrap();
            let mut summed = vec![0.0f64; whole.len()];
            for (w, v) in r.weights.iter().zip(&r.source_values) {
                let scaled: Vec<f32> = v.iter().map(|x| w * x).collect();
                let part = ops
                    .head_projection(backend, r.layer, r.head, G_HEAD_DIM, G_Q_HEADS, &scaled)
                    .unwrap();
                for (acc, p) in summed.iter_mut().zip(&part) {
                    *acc += f64::from(*p);
                }
            }
            let err: f64 = summed
                .iter()
                .zip(&whole)
                .map(|(s, w)| (s - f64::from(*w)).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale: f64 = whole
                .iter()
                .map(|w| f64::from(*w).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                err <= 1e-5 * scale.max(1e-6),
                "HP4 at {:?}: {err} vs {scale}",
                (r.layer, r.position, r.head)
            );
        }
    });
}

#[test]
fn head_projection_refuses_what_it_cannot_represent() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let x = vec![0.5f32; G_HEAD_DIM];
    let err = ops
        .head_projection(&backend, G_LAYERS, 0, G_HEAD_DIM, G_Q_HEADS, &x)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("outside this image's executed layers"),
        "{err}"
    );
    let err = ops
        .head_projection(&backend, 0, G_Q_HEADS, G_HEAD_DIM, G_Q_HEADS, &x)
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside this layer's"), "{err}");
    let err = ops
        .head_projection(&backend, 0, 0, G_HEAD_DIM, G_Q_HEADS, &x[..G_HEAD_DIM - 1])
        .unwrap_err()
        .to_string();
    assert!(err.contains("-wide head input"), "{err}");
    assert!(ops.attention_has_heads(0).unwrap());
    assert!(
        ops.attention_output_bias(0).unwrap().is_none(),
        "the golden plan has no O bias"
    );
}
