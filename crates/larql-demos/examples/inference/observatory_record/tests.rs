use super::*;

#[test]
fn parity_detects_one_bit_nonfinite_shape_and_signed_zero() {
    let a = vec![vec![1.0, 0.0]];
    assert!(bit_parity(&a, &a));
    for b in [
        vec![vec![f32::from_bits(1.0f32.to_bits() + 1), 0.0]],
        vec![vec![1.0, -0.0]],
        vec![vec![1.0]],
        vec![vec![f32::NAN, 0.0]],
    ] {
        assert!(!bit_parity(&a, &b));
    }
}

#[test]
fn vocabulary_uses_full_mass_stable_ties_and_declared_targets() {
    let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let row = vocabulary_row(&[1000., 1000., -1000.], &tokenizer, &[(1, "target".into())]).unwrap();
    assert_eq!(row["top"][0]["token_id"], 0);
    assert_eq!(row["targets"][0]["rank"], 2);
    assert_eq!(row["targets"][0]["probability"], 0.5);
    assert_eq!(row["top"][1], row["targets"][0]);
    assert!((row["entropy"].as_f64().unwrap() - 2.0f64.ln()).abs() < 1e-12);
    assert_eq!(row["top"][2]["probability"], 0.0);
    assert!(vocabulary_row(&[f32::NAN], &tokenizer, &[]).is_err());
    assert!(vocabulary_row(&[1.], &tokenizer, &[(4, "bad".into())]).is_err());
}

#[test]
fn callbacks_pair_once_and_ffn_boundary_does_not_invent_a_write() {
    let mut r = Recorder::new(
        StatsObserver::new(FixedBasis::seeded(3, 3, SEED).unwrap(), None),
        "test".into(),
    );
    r.event(StepEvent::Embedded { position: 0 });
    r.carrier_write(CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 0,
        delta: &[1., 2., 3.],
        after: &[2., 3., 4.],
        layer_scale: None,
    });
    r.event(StepEvent::CarrierWrite {
        layer: 0,
        site: SublayerSite::Attention,
        carrier: CarrierForm::Single,
    });
    r.event(StepEvent::FfnDone { layer: 0 });
    assert!(r.error.is_none());
    assert_eq!(r.writes, 1);
    assert_eq!(
        r.events
            .iter()
            .filter(|e| e["kind"] == "CarrierWrite")
            .count(),
        1
    );
    r.event(StepEvent::CarrierWrite {
        layer: 0,
        site: SublayerSite::Ffn,
        carrier: CarrierForm::Single,
    });
    assert!(r.error.is_some());
}
