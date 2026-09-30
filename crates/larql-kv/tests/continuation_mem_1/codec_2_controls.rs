//! CONTINUATION-CODEC-2 W1: known-answer controls for the frozen
//! statistics, guards and verdict. Hand-built vectors and fixtures only —
//! no model arm runs here.

use std::collections::BTreeMap;

use larql_vindex::format::vindex3::opplan::exec::operands::SelectedRepresentation;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::{
    summarise, PositionMetrics,
};

use super::codec_2::{
    adjudicate, bank, full_precision_guard, guard_a, rules, Outcome, Verdict, BANK_LEN, DECODE,
    N_MIN, RESUME, RUNGS,
};
use super::codec_2_stats::{
    mcnemar, nearest_rank, stratified_bootstrap, upper_bounds, SplitMix64, BLOCK, SEED, Z_95,
};
use super::*;

/// A scored position against R.
fn pos(position: usize, kl: f64, top1_agree: bool, margin: f64) -> PositionMetrics {
    PositionMetrics {
        sample: 0,
        position,
        category: "fixture".into(),
        kl,
        top1_agree,
        top5_overlap: 5,
        delta_nll: None,
        reference_margin: margin,
        reference_entropy: 0.0,
        max_abs_delta: 0.0,
        mean_abs_delta: 0.0,
    }
}

/// `agree` of `n` paired positions: the first `b` only the candidate
/// agrees, the next `c` only the baseline, the rest both.
fn paired(n: usize, b: usize, c: usize) -> (Vec<bool>, Vec<bool>) {
    let cand = (0..n).map(|i| i < b || i >= b + c).collect();
    let base = (0..n).map(|i| i >= b).collect();
    (cand, base)
}

/// The freeze's formula, written out independently of the module.
fn closed_form_lower(n: f64, b: f64, c: f64) -> f64 {
    (b - c) / n - Z_95 * ((b + c) - (b - c) * (b - c) / n).sqrt() / n
}

#[test]
fn the_ladder_fits_the_bank_and_scores_whole_blocks() {
    let _serial = serial();
    let ids = bank().expect("the frozen bank 2");
    assert_eq!(ids.len(), BANK_LEN);
    assert_eq!(ids[0], 2, "position 0 is <bos>");
    for n in RUNGS {
        assert!(n > DECODE + RESUME, "rung {n} holds its prefill");
        assert!(
            n < BANK_LEN,
            "rung {n}'s last scored position has a next id"
        );
    }
    assert_eq!(DECODE % BLOCK, 0, "a rung is whole bootstrap blocks");
}

#[test]
fn splitmix64_is_the_published_generator() {
    let _serial = serial();
    let mut rng = SplitMix64::new(0);
    assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
    assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
}

#[test]
fn nearest_rank_is_the_metrics_definition() {
    let _serial = serial();
    let kls: Vec<f64> = (0..1_537)
        .map(|i| ((i * 7_919) % 1_000) as f64 / 97.0)
        .collect();
    let positions: Vec<_> = kls
        .iter()
        .enumerate()
        .map(|(i, &k)| pos(i, k, true, 0.5))
        .collect();
    let summary = summarise(&positions).unwrap();
    assert_eq!(nearest_rank(&kls, 0.99), summary.all.kl_p99);
    assert_eq!(nearest_rank(&kls, 0.50), summary.all.kl_p50);
}

#[test]
fn b1_matches_its_closed_form_and_splits_at_minus_delta() {
    let _serial = serial();
    for (b, c) in [(40, 50), (40, 54), (40, 55), (0, 0)] {
        let (cand, base) = paired(1_000, b, c);
        let m = mcnemar(&cand, &base);
        assert_eq!((m.b, m.c, m.n), (b, c, 1_000));
        let expect = if b + c == 0 {
            0.0
        } else {
            closed_form_lower(1_000.0, b as f64, c as f64)
        };
        assert!(
            (m.lower - expect).abs() < 1e-12,
            "b {b} c {c}: {} vs {expect}",
            m.lower
        );
    }
    // 54 losses sit just above −δ (−0.02993), 55 just below (−0.03101).
    let lower = |c| mcnemar(&paired(1_000, 40, c).0, &paired(1_000, 40, c).1).lower;
    assert!(lower(54) >= -0.03 && lower(55) < -0.03);
}

/// Three strata of `len` positions: C agrees on 80%, KL 0.2 + a spread;
/// CH = C with `dkl` added and the first `lose` agreements of each stratum
/// lost; margins 0.95 on the first `confident` positions overall.
fn strata(
    len: usize,
    dkl: f64,
    lose: usize,
    confident: usize,
) -> (Vec<Vec<PositionMetrics>>, Vec<Vec<PositionMetrics>>) {
    let mut c = Vec::new();
    let mut ch = Vec::new();
    for s in 0..3 {
        let base = s * len;
        let cs: Vec<_> = (0..len)
            .map(|i| {
                let g = base + i;
                let margin = if g < confident { 0.95 } else { 0.3 };
                pos(g, 0.2 + (i % 10) as f64 / 100.0, i % 5 != 0, margin)
            })
            .collect();
        let hs = cs
            .iter()
            .enumerate()
            .map(|(i, m)| PositionMetrics {
                kl: m.kl + dkl,
                top1_agree: m.top1_agree && !(i < lose * 5 && i % 5 == 1),
                ..m.clone()
            })
            .collect();
        c.push(cs);
        ch.push(hs);
    }
    (c, ch)
}

fn refs(v: &[Vec<PositionMetrics>]) -> Vec<&[PositionMetrics]> {
    v.iter().map(Vec::as_slice).collect()
}

#[test]
fn b4_below_n_min_is_uninformative_and_at_n_min_is_adjudicated() {
    let _serial = serial();
    let (c, ch) = strata(128, 0.0, 0, N_MIN - 1);
    let r = rules(&refs(&c), &refs(&ch));
    assert_eq!(r.b4.n, N_MIN - 1);
    assert_eq!(r.b4_outcome, Outcome::Unsupported);
    assert_eq!(
        [r.b1_outcome, r.b2_outcome, r.b3_outcome],
        [Outcome::Holds; 3]
    );
    assert_eq!(r.verdict(), Verdict::Uninformative, "support, not failure");

    let (c, ch) = strata(128, 0.0, 0, N_MIN);
    let r = rules(&refs(&c), &refs(&ch));
    assert_eq!((r.b4.n, r.b4_outcome), (N_MIN, Outcome::Holds));
    assert_eq!(r.verdict(), Verdict::Acceptable);
}

#[test]
fn every_verdict_path_is_reachable() {
    let _serial = serial();
    let (c, ch) = strata(128, 0.0, 0, 384);
    assert_eq!(
        adjudicate(&refs(&c), &refs(&ch))["verdict"],
        json!(Verdict::Acceptable)
    );

    // KV damage above a quarter of C's mean KL: B2 fails.
    let (c, ch) = strata(128, 1.0, 0, 384);
    let r = rules(&refs(&c), &refs(&ch));
    assert_eq!(
        (r.b2_outcome, r.b3_outcome),
        (Outcome::Fails, Outcome::Fails)
    );
    assert_eq!(r.verdict(), Verdict::NotAcceptable);

    // Top-1 losses well past δ: B1 fails, even with B4 unsupported.
    let (c, ch) = strata(128, 0.0, 20, 10);
    let r = rules(&refs(&c), &refs(&ch));
    assert_eq!(
        (r.b1_outcome, r.b4_outcome),
        (Outcome::Fails, Outcome::Unsupported)
    );
    assert_eq!(
        r.verdict(),
        Verdict::NotAcceptable,
        "a failure outranks missing support"
    );
}

#[test]
fn guard_a_admits_at_its_bounds_and_refuses_past_them() {
    let _serial = serial();
    let at = |agree: usize, kl: f64| -> Vec<PositionMetrics> {
        (0..1_000).map(|i| pos(i, kl, i < agree, 0.5)).collect()
    };
    assert!(guard_a(&at(700, 0.50)).is_ok());
    assert!(guard_a(&at(699, 0.50)).unwrap_err().contains("guard A"));
    assert!(guard_a(&at(700, 0.501)).is_err());

    let broken: Vec<_> = (0..3)
        .map(|s| {
            (0..128)
                .map(|i| pos(s * 128 + i, 0.2, i % 2 == 0, 0.95))
                .collect::<Vec<_>>()
        })
        .collect();
    let out = adjudicate(&refs(&broken), &refs(&broken));
    assert_eq!(out["verdict"], json!(Verdict::Uninformative));
    assert!(out["rules"].is_null(), "a failed guard adjudicates nothing");
}

#[test]
fn the_bootstrap_is_reproducible_and_matches_a_closed_form() {
    let _serial = serial();
    let d: Vec<Vec<f64>> = (0..3)
        .map(|s| {
            (0..DECODE)
                .map(|i| ((i * 31 + s * 17) % 97) as f64 / 50.0 - 0.9)
                .collect()
        })
        .collect();
    let strata: Vec<&[f64]> = d.iter().map(Vec::as_slice).collect();
    let a = stratified_bootstrap(&strata, 500, SEED);
    assert_eq!(
        a,
        stratified_bootstrap(&strata, 500, SEED),
        "one seed, one answer"
    );
    assert_ne!(a, stratified_bootstrap(&strata, 500, SEED + 1));

    let k = [vec![0.5; DECODE], vec![0.5; DECODE], vec![0.5; DECODE]];
    let constant: Vec<&[f64]> = k.iter().map(Vec::as_slice).collect();
    let bounds = upper_bounds(&stratified_bootstrap(&constant, 500, SEED));
    assert_eq!((bounds.mean_upper, bounds.p99_upper), (0.5, 0.5));
}

#[test]
fn the_stratified_bootstrap_never_reweights_a_rung() {
    let _serial = serial();
    let k: Vec<Vec<f64>> = (0..3).map(|s| vec![s as f64; DECODE]).collect();
    let strata: Vec<&[f64]> = k.iter().map(Vec::as_slice).collect();
    let replicates = stratified_bootstrap(&strata, 2_000, SEED);
    assert!(
        replicates.iter().all(|r| r.mean == 1.0),
        "each rung contributes exactly a third of every replicate"
    );

    // NEGATIVE control: the same fixture sampled without strata (all 96
    // blocks drawn freely) reweights the rungs, and the check sees it.
    let pooled: Vec<f64> = k.concat();
    let blocks = pooled.len() / BLOCK;
    let mut rng = SplitMix64::new(SEED);
    let free: Vec<f64> = (0..2_000)
        .map(|_| {
            (0..blocks)
                .map(|_| {
                    let s = rng.below(blocks) * BLOCK;
                    pooled[s..s + BLOCK].iter().sum::<f64>()
                })
                .sum::<f64>()
                / pooled.len() as f64
        })
        .collect();
    assert!(
        free.iter().any(|&m| m != 1.0),
        "an unstratified sampler must fail the check"
    );
}

#[test]
fn the_binding_guards_tell_full_precision_from_a_pack() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "codec2-full-precision");
    assert!(full_precision_guard(subject.store.selection()).is_ok());

    let pack = BTreeMap::from([(
        "target.decoder_stack".to_string(),
        SelectedRepresentation {
            encoding: "Q4_K".into(),
            stored: true,
            codec: None,
        },
    )]);
    assert!(full_precision_guard(&pack)
        .unwrap_err()
        .contains("UNINFORMATIVE"));
    assert!(
        full_precision_guard(&BTreeMap::new()).is_err(),
        "a missing binding is refused"
    );
}
