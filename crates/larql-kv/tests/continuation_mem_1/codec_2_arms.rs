//! CONTINUATION-CODEC-2's arms over a ladder, and the pooled record.
//!
//! One orchestration serves the frozen run (the frozen ladder, real
//! containers) and the executed end-to-end control (a fixture ladder), so
//! the glue — checkpoint names, resume, pooling — is exercised before any
//! long run (W2 erratum). Order per rung: R (logits kept) → NULL (guard) →
//! C (scores, logits kept) → CH → H.

use larql_kv::CodecKvState;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::represent::measure::plan::metrics::{
    summarise, PositionMetrics,
};

use super::codec_1_c3::{decode_rows, null_guard, score};
use super::codec_2::{adjudicate, DECODE, RESUME, RUNGS};
use super::codec_2_store::{logits_base, scores_name, Checkpoints};
use super::measured::CodecResidency;
use super::*;

/// The KV bits of CH and H, as frozen.
const CODEC_BITS: u8 = 4;

/// Rungs (total positions), then RESUME resumed and DECODE scored
/// teacher-forced positions at the end of each.
pub(super) struct Ladder {
    pub rungs: Vec<usize>,
    pub resume: usize,
    pub decode: usize,
}

impl Ladder {
    pub(super) fn frozen() -> Self {
        Self {
            rungs: RUNGS.to_vec(),
            resume: RESUME,
            decode: DECODE,
        }
    }

    /// The rung's journey, its first scored position and the ids each
    /// scored position predicts.
    pub(super) fn journey<'a>(&self, ids: &'a [u32], n: usize) -> (Journey, usize, &'a [u32]) {
        let start = n - self.decode;
        let journey = Journey {
            prefill: ids[..start - self.resume].to_vec(),
            resume: ids[start - self.resume..start].to_vec(),
            decode: ids[start..n].to_vec(),
        };
        (journey, start, &ids[start + 1..=n])
    }
}

/// CODEC-1 established codec residency; here a miss is a provider or
/// instrument fault that stops the run.
fn residency_regression(arm: &str, n: usize, r: Option<CodecResidency>) -> Value {
    let r = r.unwrap_or_else(|| panic!("rung {n}: {arm} reported no codec residency"));
    assert!(
        r.holds(),
        "rung {n}: {arm} residency regression — live {} expected {} strays {} scratch {}/{}",
        r.append_born_live,
        r.expected,
        r.strays,
        r.scratch_bytes,
        r.scratch_bound
    );
    json!({"holds": true, "append_born_live": r.append_born_live, "expected": r.expected,
           "strays": r.strays, "scratch_bytes": r.scratch_bytes, "scratch_bound": r.scratch_bound})
}

/// Run every arm the store lacks. `exact` must run the full-precision
/// stack and `yard` the Q4_K pack — the caller guards both.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_arms<B: PlanBackend>(
    exact: (&Subject, &PreparedOperands),
    yard: (&Subject, &PreparedOperands),
    backend: &B,
    ids: &[u32],
    ladder: &Ladder,
    store: &Checkpoints,
) {
    for &n in &ladder.rungs {
        let (journey, start, next) = ladder.journey(ids, n);
        let timed = |arm: &str, t: std::time::Instant| {
            eprintln!("rung {n}: {arm} done in {:.0}s", t.elapsed().as_secs_f64());
            t.elapsed().as_secs_f64()
        };
        let r = store.read_logits(&logits_base(n, "r")).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (r, _) = decode_rows(exact.0, exact.1, backend, RowKvState::default(), &journey);
            store.write_logits(&logits_base(n, "r"), &r);
            timed("R", t);
            r
        });
        match store.read::<Value>(&scores_name(n, "null")) {
            Some(null) if null["passed"] != true => {
                panic!("rung {n}: recorded NULL stop: {}", null["stop"])
            }
            Some(_) => {}
            None => {
                let t = std::time::Instant::now();
                let (null, _) =
                    decode_rows(exact.0, exact.1, backend, WindowKvState::new(), &journey);
                let verdict = null_guard(&r, &null);
                store.write(
                    &scores_name(n, "null"),
                    &json!({"passed": verdict.is_ok(),
                    "stop": verdict.as_ref().err(), "seconds": timed("NULL", t)}),
                );
                if let Err(stop) = verdict {
                    panic!("rung {n}: {stop}");
                }
            }
        }
        let c = store.read_logits(&logits_base(n, "c")).unwrap_or_else(|| {
            let t = std::time::Instant::now();
            let (c, _) = decode_rows(yard.0, yard.1, backend, RowKvState::default(), &journey);
            store.write(
                &scores_name(n, "c"),
                &json!({"arm": "c", "n": n, "seconds": timed("C", t),
                "positions": score(&r, &c, next, n, start)}),
            );
            store.write_logits(&logits_base(n, "c"), &c);
            c
        });
        if store.read::<Value>(&scores_name(n, "ch")).is_none() {
            let t = std::time::Instant::now();
            let (ch, res) = decode_rows(
                yard.0,
                yard.1,
                backend,
                CodecKvState::new(CODEC_BITS),
                &journey,
            );
            let residency = residency_regression("CH", n, res);
            store.write(
                &scores_name(n, "ch"),
                &json!({"arm": "ch", "n": n, "seconds": timed("CH", t),
                "residency": residency, "positions": score(&r, &ch, next, n, start),
                "positions_vs_c": score(&c, &ch, next, n, start)}),
            );
        }
        if store.read::<Value>(&scores_name(n, "h")).is_none() {
            let t = std::time::Instant::now();
            let (h, res) = decode_rows(
                exact.0,
                exact.1,
                backend,
                CodecKvState::new(CODEC_BITS),
                &journey,
            );
            let residency = residency_regression("H", n, res);
            store.write(
                &scores_name(n, "h"),
                &json!({"arm": "h", "n": n, "seconds": timed("H", t),
                "residency": residency, "positions": score(&r, &h, next, n, start)}),
            );
        }
    }
}

/// One scored arm's positions per rung, from its scores files.
pub(super) fn strata(
    store: &Checkpoints,
    ladder: &Ladder,
    arm: &str,
    key: &str,
) -> Vec<Vec<PositionMetrics>> {
    ladder
        .rungs
        .iter()
        .map(|&n| {
            let v: Value = store
                .read(&scores_name(n, arm))
                .unwrap_or_else(|| panic!("{} missing", scores_name(n, arm)));
            serde_json::from_value(v[key].clone())
                .unwrap_or_else(|e| panic!("{}[{key}]: {e}", scores_name(n, arm)))
        })
        .collect()
}

/// C's positions re-scored from kept R and C logits under `base` — the
/// same score() the run applies at C's arm.
pub(super) fn rescored_c(
    store: &Checkpoints,
    ladder: &Ladder,
    ids: &[u32],
    base: fn(usize, &str) -> String,
) -> Vec<Vec<PositionMetrics>> {
    ladder
        .rungs
        .iter()
        .map(|&n| {
            let (_, start, next) = ladder.journey(ids, n);
            let r = store.read_logits(&base(n, "r")).expect("R's kept logits");
            let c = store.read_logits(&base(n, "c")).expect("C's kept logits");
            score(&r, &c, next, n, start)
        })
        .collect()
}

fn refs(v: &[Vec<PositionMetrics>]) -> Vec<&[PositionMetrics]> {
    v.iter().map(Vec::as_slice).collect()
}

/// The pooled record: the frozen verdict of CH against C, the reported
/// comparisons, and per-rung guards, residency and wall time.
pub(super) fn record(
    programme: &str,
    stamp: &Value,
    store: &Checkpoints,
    ladder: &Ladder,
    c: &[Vec<PositionMetrics>],
) -> Value {
    let (ch, h, c_ch) = (
        strata(store, ladder, "ch", "positions"),
        strata(store, ladder, "h", "positions"),
        strata(store, ladder, "ch", "positions_vs_c"),
    );
    let summary = |v: &[Vec<PositionMetrics>]| summarise(&v.concat()).unwrap();
    let rungs: Vec<Value> = ladder
        .rungs
        .iter()
        .map(|&n| {
            let arm = |a: &str| store.read::<Value>(&scores_name(n, a)).unwrap_or(Value::Null);
            let (ch, h) = (arm("ch"), arm("h"));
            json!({"n": n, "null": arm("null"),
                "residency": {"ch": ch.get("residency"), "h": h.get("residency")},
                "seconds": {"c": arm("c").get("seconds"), "ch": ch.get("seconds"), "h": h.get("seconds")}})
        })
        .collect();
    json!({
        "programme": programme,
        "authorities": stamp,
        "verdict_ch": adjudicate(&refs(c), &refs(&ch)),
        "reported": {
            "r_c": summary(c), "r_ch_end_to_end": summary(&ch),
            "r_h_isolated_codec": summary(&h), "c_ch_kl": summary(&c_ch),
        },
        "rungs": rungs,
    })
}
