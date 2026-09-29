//! Concurrency changes wall time only: a batch scores each sample exactly as
//! it would be scored alone, and any batch size measures the same positions.

use super::*;
use crate::format::vindex3::represent::measure::plan::scoring::{measure_samples, null_arm};
use crate::format::vindex3::represent::token_bank::TokenBank;

fn all_samples(f: &Fixture) -> (TokenBank, Vec<Vec<u32>>) {
    let bank = TokenBank::open(&f.bank).unwrap();
    let ids = (0..bank.sample_count())
        .map(|i| bank.read(i).unwrap())
        .collect();
    (bank, ids)
}

fn bits(logits: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<u32>>> {
    logits
        .iter()
        .map(|s| {
            s.iter()
                .map(|p| p.iter().map(|x| x.to_bits()).collect())
                .collect()
        })
        .collect()
}

#[test]
fn a_batch_scores_each_sample_exactly_as_alone() {
    let f = fixture();
    let (_, samples) = all_samples(&f);
    let (_, mut candidate) = reference_and_candidate(&f);
    let alone: Vec<Vec<Vec<f32>>> = samples
        .iter()
        .map(|ids| candidate.score(ids).unwrap())
        .collect();
    let batched = candidate.score_batch(&samples).unwrap();
    assert_eq!(bits(&batched), bits(&alone));
}

#[test]
fn every_batch_size_measures_the_same_positions() {
    let f = fixture();
    let (bank, samples) = all_samples(&f);
    let measure = |batch: usize| {
        let (mut r, mut c) = reference_and_candidate(&f);
        let cache = null_arm(&bank, &mut r, 1, |e| panic!("{e:?}")).unwrap();
        let measured = measure_samples(
            &bank,
            samples.len(),
            batch,
            &mut r,
            &mut c,
            cache,
            None,
            |e| panic!("{e:?}"),
        )
        .unwrap();
        assert_eq!(measured.samples_read, samples.len());
        serde_json::to_string(&measured.positions).unwrap()
    };
    let one_at_a_time = measure(1);
    for batch in [0, 3, samples.len(), 8] {
        assert_eq!(measure(batch), one_at_a_time, "batch {batch}");
    }
}
