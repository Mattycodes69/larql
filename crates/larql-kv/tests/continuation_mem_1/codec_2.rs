//! CONTINUATION-CODEC-2 W1: the held-out bank's authority, the binding
//! guards, guard A and the frozen tri-state verdict of CH against C.
//!
//! Frozen: docs/represent/forecasts/continuation-codec-2.json. CODEC-1's
//! helpers (decode_rows, score, null_guard, yardstick_guard) are used as
//! they are; CODEC-1's modules are not edited.

use std::collections::BTreeMap;

use larql_vindex::format::vindex3::opplan::exec::operands::SelectedRepresentation;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::PositionMetrics;
use sha2::{Digest, Sha256};

use super::codec_2_stats::{
    mcnemar, nearest_rank, stratified_bootstrap, upper_bounds, McNemar, REPLICATES, SEED,
    TAIL_QUANTILE,
};
use super::*;

/// The held-out bank, its length and digest (sha256 over u32 LE ids).
const BANK_PATH: &str = "../../docs/represent/forecasts/continuation-codec-2-token-bank.json";
pub(super) const BANK_LEN: usize = 8193;
pub(super) const BANK_IDS_SHA256: &str =
    "5120445c411424f83e6c0f7c6eb349f3a3d13748746e42b4af5c05c85081119e";

/// The frozen ladder: prefill n − RESUME − DECODE, resume, decode scored.
pub(super) const RUNGS: [usize; 3] = [2_048, 4_096, 8_192];
pub(super) const RESUME: usize = 16;
pub(super) const DECODE: usize = 1_024;

/// The frozen rule.
pub(super) const DELTA: f64 = 0.03;
pub(super) const EPSILON_FRACTION: f64 = 0.25;
pub(super) const CONFIDENT_MARGIN: f64 = 0.9;
pub(super) const N_MIN: usize = 350;
/// Guard A: C is usable as a baseline.
pub(super) const GUARD_A_TOP1_MIN: f64 = 0.70;
pub(super) const GUARD_A_KL_MEAN_MAX: f64 = 0.50;

const DECODER_STACK: &str = "target.decoder_stack";

/// Bank 2, refused unless it is the frozen one.
pub(super) fn bank() -> Result<Vec<u32>, String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(BANK_PATH);
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let ids: Vec<u32> = doc["ids"]
        .as_array()
        .ok_or("the bank has no ids")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("an id is not a u32")
        })
        .collect::<Result<_, _>>()?;
    let mut hash = Sha256::new();
    for id in &ids {
        hash.update(id.to_le_bytes());
    }
    let digest = format!("{:x}", hash.finalize());
    if ids.len() != BANK_LEN || digest != BANK_IDS_SHA256 {
        return Err(format!(
            "token bank is not the frozen one: {} ids, sha256 {digest}",
            ids.len()
        ));
    }
    Ok(ids)
}

/// R, NULL and H must run the full-precision stack: the decoder stack is
/// not a stored pack.
pub(super) fn full_precision_guard(
    selection: &BTreeMap<String, SelectedRepresentation>,
) -> Result<(), String> {
    match selection.get(DECODER_STACK) {
        Some(s) if !s.stored => Ok(()),
        other => Err(format!(
            "UNINFORMATIVE: a full-precision arm bound `{DECODER_STACK}` as {other:?}"
        )),
    }
}

/// A rule's outcome: adjudicated either way, or lacking the support the
/// freeze requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(super) enum Outcome {
    Holds,
    Fails,
    Unsupported,
}

impl Outcome {
    fn from(holds: bool) -> Self {
        if holds {
            Self::Holds
        } else {
            Self::Fails
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(super) enum Verdict {
    Acceptable,
    NotAcceptable,
    Uninformative,
}

/// Guard A over the pooled R↔C positions.
pub(super) fn guard_a(c: &[PositionMetrics]) -> Result<(), String> {
    let (top1, kl) = (top1_rate(c), mean_kl(c));
    if top1 >= GUARD_A_TOP1_MIN && kl <= GUARD_A_KL_MEAN_MAX {
        Ok(())
    } else {
        Err(format!(
            "UNINFORMATIVE: guard A — R↔C top-1 {top1:.4} (min {GUARD_A_TOP1_MIN}), \
             KL mean {kl:.4} (max {GUARD_A_KL_MEAN_MAX})"
        ))
    }
}

fn top1_rate(p: &[PositionMetrics]) -> f64 {
    p.iter().filter(|m| m.top1_agree).count() as f64 / p.len().max(1) as f64
}

fn mean_kl(p: &[PositionMetrics]) -> f64 {
    p.iter().map(|m| m.kl).sum::<f64>() / p.len().max(1) as f64
}

fn agreement(p: &[PositionMetrics]) -> Vec<bool> {
    p.iter().map(|m| m.top1_agree).collect()
}

/// B1–B4 over paired strata (one per rung, in ladder order): `c[i]` and
/// `ch[i]` are the same positions scored against R. Thresholds come from
/// C over the same strata.
#[derive(Debug, Clone, serde::Serialize)]
pub(super) struct Rules {
    pub b1: McNemar,
    pub b1_outcome: Outcome,
    pub b2_mean_d_upper: f64,
    pub b2_epsilon: f64,
    pub b2_outcome: Outcome,
    pub b3_p99_d_upper: f64,
    pub b3_bound: f64,
    pub b3_outcome: Outcome,
    pub b4: McNemar,
    pub b4_outcome: Outcome,
}

pub(super) fn rules(c: &[&[PositionMetrics]], ch: &[&[PositionMetrics]]) -> Rules {
    assert_eq!(c.len(), ch.len(), "one CH stratum per C stratum");
    for (cs, hs) in c.iter().zip(ch) {
        assert_eq!(cs.len(), hs.len(), "paired strata");
        for (x, y) in cs.iter().zip(hs.iter()) {
            assert_eq!(x.position, y.position, "C and CH score the same positions");
            assert_eq!(
                x.reference_margin, y.reference_margin,
                "C and CH are scored against one R"
            );
        }
    }
    let pooled_c: Vec<PositionMetrics> = c.iter().flat_map(|s| s.iter().cloned()).collect();
    let pooled_ch: Vec<PositionMetrics> = ch.iter().flat_map(|s| s.iter().cloned()).collect();

    let b1 = mcnemar(&agreement(&pooled_ch), &agreement(&pooled_c));
    let b1_outcome = Outcome::from(b1.lower >= -DELTA);

    let d: Vec<Vec<f64>> = c
        .iter()
        .zip(ch)
        .map(|(cs, hs)| cs.iter().zip(hs.iter()).map(|(x, y)| y.kl - x.kl).collect())
        .collect();
    let strata: Vec<&[f64]> = d.iter().map(Vec::as_slice).collect();
    let bounds = upper_bounds(&stratified_bootstrap(&strata, REPLICATES, SEED));
    let c_kls: Vec<f64> = pooled_c.iter().map(|m| m.kl).collect();
    let b2_epsilon = EPSILON_FRACTION * mean_kl(&pooled_c);
    let b3_bound = EPSILON_FRACTION * nearest_rank(&c_kls, TAIL_QUANTILE);

    let band = |p: &[PositionMetrics]| -> Vec<bool> {
        p.iter()
            .filter(|m| m.reference_margin >= CONFIDENT_MARGIN)
            .map(|m| m.top1_agree)
            .collect()
    };
    let b4 = mcnemar(&band(&pooled_ch), &band(&pooled_c));
    let b4_outcome = if b4.n < N_MIN {
        Outcome::Unsupported
    } else {
        Outcome::from(b4.lower >= -DELTA)
    };

    Rules {
        b1,
        b1_outcome,
        b2_mean_d_upper: bounds.mean_upper,
        b2_epsilon,
        b2_outcome: Outcome::from(bounds.mean_upper <= b2_epsilon),
        b3_p99_d_upper: bounds.p99_upper,
        b3_bound,
        b3_outcome: Outcome::from(bounds.p99_upper <= b3_bound),
        b4,
        b4_outcome,
    }
}

impl Rules {
    fn outcomes(&self) -> [Outcome; 4] {
        [
            self.b1_outcome,
            self.b2_outcome,
            self.b3_outcome,
            self.b4_outcome,
        ]
    }

    /// The tri-state verdict once every guard has passed: any adjudicated
    /// failure is NOT ACCEPTABLE; missing support is UNINFORMATIVE.
    pub(super) fn verdict(&self) -> Verdict {
        let o = self.outcomes();
        if o.contains(&Outcome::Fails) {
            Verdict::NotAcceptable
        } else if o.contains(&Outcome::Unsupported) {
            Verdict::Uninformative
        } else {
            Verdict::Acceptable
        }
    }
}

/// The pooled verdict: guard A first (a failure adjudicates nothing),
/// then the rules. Per-rung rules are reported beside it, each rung's
/// thresholds from that rung's C.
pub(super) fn adjudicate(c: &[&[PositionMetrics]], ch: &[&[PositionMetrics]]) -> Value {
    let pooled_c: Vec<PositionMetrics> = c.iter().flat_map(|s| s.iter().cloned()).collect();
    if let Err(stop) = guard_a(&pooled_c) {
        return json!({"verdict": Verdict::Uninformative, "guard_a": stop, "rules": null});
    }
    let pooled = rules(c, ch);
    let per_rung: Vec<Value> = c
        .iter()
        .zip(ch)
        .map(|(cs, hs)| {
            let r = rules(&[*cs], &[*hs]);
            json!({"verdict": r.verdict(), "rules": r})
        })
        .collect();
    json!({"verdict": pooled.verdict(), "guard_a": "passed", "rules": pooled, "per_rung": per_rung})
}
