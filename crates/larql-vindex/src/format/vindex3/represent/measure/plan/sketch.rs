//! **The logit sketch**: a fixed random projection of each position's
//! centred logit difference, written beside `positions.jsonl` so a linearity
//! test over many runs needs no full-vocabulary logits
//! (AUTO-REP-LANDSCAPE-1, `docs/auto-rep-landscape-1.md`).
//!
//! For a position with reference logits `r` and candidate logits `c` over a
//! vocabulary of `V`, the sketch is `R · centre(c − r)`, where `centre`
//! subtracts the vector's mean over the vocabulary (softmax is
//! shift-invariant, so this is also the centred log-probability difference)
//! and `R` is `dim × V` with entries `±1/√dim`.
//!
//! `R` is Rademacher, not Gaussian, so that it is generated from integers
//! alone and is bit-identical on every platform: a Gaussian needs `ln` and
//! `cos`, which are not correctly rounded. The projection is linear, so the
//! sketch of a sum is the sum of the sketches, and a nonzero vector has a zero
//! sketch with probability at most `2^-dim` (Littlewood–Offord: each row
//! independently hits zero with probability at most one half).
//!
//! **The generator, stated so an independent implementation can reproduce
//! it.** Column `j` of `R` takes its signs from words `n = j·W + b` for
//! `b in 0..W`, `W = ceil(dim / 64)`, where word `n` is the `n`-th output of
//! SplitMix64 seeded with `seed` (its first output is `n = 0`). Row `k` reads
//! bit `k mod 64` of word `b = k / 64`: set means `+1/√dim`. Sums run in
//! `f64` in vocabulary order and are rounded to `f32` once, at the end.

use serde::{Deserialize, Serialize};

/// The sketch's file, written into the request's output directory:
/// `positions × dim` little-endian `f32`, in `positions.jsonl`'s row order.
pub const SKETCH_FILE: &str = "sketch.f32";

/// The generator's name and version, recorded in the report. A change to
/// anything the module header states is a new version.
pub const SKETCH_GENERATOR: &str = "rademacher-splitmix64/v1";

/// Signs one SplitMix64 word supplies.
const SIGNS_PER_WORD: usize = u64::BITS as usize;

/// SplitMix64's increment (the golden-ratio constant).
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// SplitMix64's two finaliser multipliers.
const SPLITMIX_MIX_1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MIX_2: u64 = 0x94D0_49BB_1331_11EB;

/// Which sketch to write: its dimension and the seed that fixes `R`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SketchSpec {
    pub dim: usize,
    pub seed: u64,
}

impl SketchSpec {
    /// A spec, refused when it would project onto nothing.
    pub fn new(dim: usize, seed: u64) -> Result<Self, String> {
        if dim == 0 {
            return Err("a sketch of dimension 0 records nothing".into());
        }
        Ok(Self { dim, seed })
    }

    /// Words each vocabulary column draws its signs from.
    fn words_per_column(&self) -> usize {
        self.dim.div_ceil(SIGNS_PER_WORD)
    }

    /// Sketch one position. The caller has already checked that both rows
    /// are finite and the same length (`metrics::position_metrics`).
    pub fn project(&self, reference: &[f32], candidate: &[f32]) -> Vec<f32> {
        let vocabulary = reference.len().min(candidate.len());
        if vocabulary == 0 {
            return vec![0.0; self.dim];
        }
        let delta = |j: usize| f64::from(candidate[j]) - f64::from(reference[j]);
        let mean = (0..vocabulary).map(delta).sum::<f64>() / vocabulary as f64;
        let words = self.words_per_column();
        let mut sums = vec![0.0f64; self.dim];
        for j in 0..vocabulary {
            let centred = delta(j) - mean;
            for (b, rows) in sums.chunks_mut(SIGNS_PER_WORD).enumerate() {
                let signs = splitmix64(self.seed, (j * words + b) as u64);
                for (bit, sum) in rows.iter_mut().enumerate() {
                    if (signs >> bit) & 1 == 1 {
                        *sum += centred;
                    } else {
                        *sum -= centred;
                    }
                }
            }
        }
        let scale = (self.dim as f64).sqrt();
        sums.into_iter().map(|s| (s / scale) as f32).collect()
    }

    /// The report's description of this sketch.
    pub fn describe(&self, positions: usize) -> serde_json::Value {
        serde_json::json!({
            "file": SKETCH_FILE,
            "generator": SKETCH_GENERATOR,
            "dim": self.dim,
            "seed": self.seed,
            "positions": positions,
            "layout": "positions x dim, f32 little-endian, in positions.jsonl row order",
            "input": "candidate minus reference logits, mean over the vocabulary subtracted",
            "scale": "1/sqrt(dim)",
        })
    }
}

/// The `n`-th output of SplitMix64 seeded with `seed`, counting from zero.
pub fn splitmix64(seed: u64, n: u64) -> u64 {
    let mut z = seed.wrapping_add(n.wrapping_add(1).wrapping_mul(SPLITMIX_GAMMA));
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MIX_1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MIX_2);
    z ^ (z >> 31)
}

/// Every position's sketch, in order, as little-endian `f32` bytes: the
/// contents of [`SKETCH_FILE`].
pub fn sketch_bytes(sketches: &[Vec<f32>]) -> Vec<u8> {
    sketches
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests;
