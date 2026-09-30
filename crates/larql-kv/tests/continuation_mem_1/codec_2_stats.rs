//! CONTINUATION-CODEC-2: the frozen statistics, pure and model-free.
//!
//! SplitMix64, the nearest-rank percentile, the paired McNemar lower bound
//! (B1, B4) and the stratified block bootstrap B2 and B3 share. Every
//! definition here is stated in docs/represent/forecasts/continuation-codec-2.json
//! (rule) and continuation-codec-2-notes.json (W1 decisions).

/// One-sided 95% normal quantile, as the freeze writes it.
pub(super) const Z_95: f64 = 1.6449;
/// Consecutive scored positions per bootstrap block.
pub(super) const BLOCK: usize = 32;
/// Bootstrap replicates and the frozen seed.
pub(super) const REPLICATES: usize = 10_000;
pub(super) const SEED: u64 = 0xC0DEC2;
/// The bound's quantile over replicates, and the tail the rules read.
pub(super) const UPPER_QUANTILE: f64 = 0.95;
pub(super) const TAIL_QUANTILE: f64 = 0.99;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MUL_1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MUL_2: u64 = 0x94D0_49BB_1331_11EB;

/// SplitMix64: a fixed, documented generator, so every bound is
/// reproducible from the seed alone.
pub(super) struct SplitMix64(u64);

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(SPLITMIX_GAMMA);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MUL_1);
        z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MUL_2);
        z ^ (z >> 31)
    }

    /// An index below `n` (modulo bias ≤ n / 2^64, stated not corrected).
    pub(super) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Nearest-rank percentile: rank = max(1, ceil(p · n)) of the sorted
/// values — represent/bank.rs's definition, which a test target cannot
/// reach (pub(crate)).
pub(super) fn nearest_rank(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p * sorted.len() as f64).ceil() as usize).max(1);
    sorted[rank.min(sorted.len()) - 1]
}

/// The paired McNemar comparison of a candidate's and a baseline's
/// agreement with the reference, position by position.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(super) struct McNemar {
    pub n: usize,
    /// Candidate agrees, baseline does not.
    pub b: usize,
    /// Baseline agrees, candidate does not.
    pub c: usize,
    pub diff: f64,
    pub se: f64,
    pub lower: f64,
}

pub(super) fn mcnemar(candidate: &[bool], baseline: &[bool]) -> McNemar {
    assert_eq!(candidate.len(), baseline.len(), "paired positions");
    let n = candidate.len();
    let b = candidate
        .iter()
        .zip(baseline)
        .filter(|&(&x, &y)| x && !y)
        .count();
    let c = candidate
        .iter()
        .zip(baseline)
        .filter(|&(&x, &y)| !x && y)
        .count();
    if n == 0 {
        return McNemar {
            n,
            b,
            c,
            diff: 0.0,
            se: 0.0,
            lower: 0.0,
        };
    }
    let nf = n as f64;
    let (bf, cf) = (b as f64, c as f64);
    let diff = (bf - cf) / nf;
    let se = ((bf + cf) - (bf - cf).powi(2) / nf).max(0.0).sqrt() / nf;
    McNemar {
        n,
        b,
        c,
        diff,
        se,
        lower: diff - Z_95 * se,
    }
}

/// One bootstrap replicate's statistics of the pooled d.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Replicate {
    pub mean: f64,
    pub p99: f64,
}

/// The stratified block bootstrap: every replicate draws, with
/// replacement, exactly as many blocks from each stratum (rung) as that
/// stratum holds, and pools them — so no replicate reweights a stratum.
/// Strata are drawn in the order given; each must be a whole number of
/// blocks.
pub(super) fn stratified_bootstrap(
    strata: &[&[f64]],
    replicates: usize,
    seed: u64,
) -> Vec<Replicate> {
    for s in strata {
        assert!(
            !s.is_empty() && s.len() % BLOCK == 0,
            "a stratum of {} positions is not whole blocks of {BLOCK}",
            s.len()
        );
    }
    let mut rng = SplitMix64::new(seed);
    let total: usize = strata.iter().map(|s| s.len()).sum();
    let mut pooled = Vec::with_capacity(total);
    (0..replicates)
        .map(|_| {
            pooled.clear();
            for s in strata {
                let blocks = s.len() / BLOCK;
                for _ in 0..blocks {
                    let start = rng.below(blocks) * BLOCK;
                    pooled.extend_from_slice(&s[start..start + BLOCK]);
                }
            }
            Replicate {
                mean: pooled.iter().sum::<f64>() / pooled.len() as f64,
                p99: nearest_rank(&pooled, TAIL_QUANTILE),
            }
        })
        .collect()
}

/// The frozen upper bounds of the mean and the p99 of d over replicates.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub(super) struct Bounds {
    pub mean_upper: f64,
    pub p99_upper: f64,
}

pub(super) fn upper_bounds(replicates: &[Replicate]) -> Bounds {
    let means: Vec<f64> = replicates.iter().map(|r| r.mean).collect();
    let p99s: Vec<f64> = replicates.iter().map(|r| r.p99).collect();
    Bounds {
        mean_upper: nearest_rank(&means, UPPER_QUANTILE),
        p99_upper: nearest_rank(&p99s, UPPER_QUANTILE),
    }
}
