//! **The teacher-forced quality runner** — two arms through the real
//! mixed Metal stack, loaded from the source VINDEX3 container itself:
//!
//! ```text
//! baseline    every routed layer   -> source expert_bank   (BF16, Table)
//! candidate   the overlay's layers -> compiled banks       (Identity,
//!             each at its map's encoding — Q8_0, Q6_K, or a composed
//!             map holding several layers at different encodings)
//!             everything else      -> the SAME source stores
//! ```
//!
//! The changed variable is exactly the candidate's compiled scope:
//! source BF16 bytes vs the persistent bytes the compiler sealed.
//! Router, norms, attention, shared experts, state initialisation and
//! the teacher-forced token prefix are identical by construction — and
//! asserted, not assumed, per compiled layer.
//!
//! **The verdict semantics follow scale.** Below `positions_min` (4096)
//! this is a DIAGNOSTIC: the gate must refuse on positions however
//! flattering the numbers, and the harness asserts that refusal. At
//! 8192 (`LARQL_Q2A_SEQUENCES=256`) the verdict IS the product — frozen
//! `kimi-logit-v3` decides, the report is written BEFORE any verdict
//! assertion, and a FAIL is a result, not a harness failure.
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.aligned.vindex3 \
//! LARQL_KIMI_Q6_CANDIDATE=/tmp/kimi-q80-l25.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=/tmp/kimi_quality_bank \
//! LARQL_Q2A_SEQUENCES=8 LARQL_Q2A_LABEL=q80-l25 \
//!   cargo test -p larql-vindex --features gpu --release --lib q2a_teacher_forced -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use larql_compute::backend::ComputeBackend;
use larql_compute_metal::trait_impl::kimi_layer::ExecutionTrace;
use larql_compute_metal::MetalBackend;

use crate::format::vindex3::opplan::exec::kimi_source::{CandidateOverlay, KimiSourceModel};
use crate::format::vindex3::opplan::exec::stack::{LayerSpec, LayerState};
use crate::format::vindex3::opplan::exec::stack_metal::{DeviceLayer, HybridStack};
use crate::format::vindex3::represent::bank::{PositionObservation, Top1Change, TopKChange};
use crate::format::vindex3::represent::measure::outcome::{ExecutionFailure, VerifiedFacts};
use crate::format::vindex3::represent::measure::TeacherForcedRequest;
use crate::format::vindex3::represent::quality::QualityBank;
use crate::format::vindex3::represent::quality::QualityGate;

/// The stores an arm may bind, by the id the loader registers them
/// under. Named, because an attribution refusal that compared string
/// literals would be one typo away from never firing.
const SOURCE_EXPERT_BANK: &str = "kimi-source-expert-bank";
const CANDIDATE_BANK: &str = "kimi-candidate-bank";
const SOURCE_DECODER_STACK: &str = "kimi-source-decoder-stack";

/// **What a completed measurement produced, and what it verified.**
///
/// The verdict travels as a recorded fact rather than as authority:
/// this says what the gate decided, and promotion remains the
/// optimiser's to derive. Execution establishes *experiment X produced
/// observation Y*, never *therefore Y is preferred*.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeasurementReceipt {
    /// The request as performed, so a receipt names its own experiment.
    pub request: TeacherForcedRequest,
    /// The gate actually evaluated — checked against the request.
    pub gate: QualityGate,
    /// Every validity condition this run checked.
    pub verified: VerifiedFacts,
    pub bank: QualityBank,
    pub min_covered_mass: f64,
    pub wall_seconds: f64,
    /// Where the closure report was written, before any verdict was
    /// drawn: a refused run still leaves its evidence on disk.
    pub report_path: String,
    pub verdict_passed: bool,
    pub verdict_failures: Vec<String>,
}

impl MeasurementReceipt {
    /// Whether this reading may enter the evidence system.
    ///
    /// NOT whether it passed. A refused candidate that was measured
    /// correctly is admissible evidence of a refusal; a passing
    /// candidate whose run skipped a validity check is not evidence of
    /// anything.
    pub fn qualifies(&self) -> bool {
        self.verified.complete()
    }
}

pub use crate::format::vindex3::represent::measure::{
    BANK_ENV, CANDIDATE_ENV, RECORD_ENV, SOURCE_ENV,
};

mod run;
pub use run::*;
/// Sequences the null arm re-runs — enough positions for the all-zero
/// claim to cover real routing variety, cheap enough not to double the
/// run.
const NULL_SEQUENCES: usize = 4;
/// Baseline top-N kept per position. KL is exact on the mass these
/// carry; the minimum covered mass is reported so a too-flat position
/// announces itself.
///
/// 128 covered only 31 % of the baseline's mass at the worst position
/// of the first run — a teacher-forced sequence's FIRST position has no
/// context, so its distribution is close to flat over 163,840 ids and a
/// short truncation sees almost none of it. Widening costs nothing
/// measurable (the rank sort already runs for argmax and top-10) and
/// buys a KL that is exact on most of the distribution instead of a
/// third of it.
const TOP_N: usize = 2048;
/// Directory the closure reports are written to; the OS temp dir when unset.
pub const REPORT_DIR_ENV: &str = "LARQL_MEASURE_REPORT_DIR";

/// Where the machine-readable closure reports are written, one per
/// labelled run. The file name is the one historical reports carry.
fn report_path(label: &str) -> String {
    let dir = env_dir(REPORT_DIR_ENV).unwrap_or_else(std::env::temp_dir);
    dir.join(format!("kimi_{label}_report.json"))
        .to_string_lossy()
        .into_owned()
}

pub fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

fn logsumexp(v: &[f32]) -> f32 {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are never NaN"))
        .expect("non-empty")
        .0
}

fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .expect("logits are never NaN")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// One position's paired measurement, from both arms' full logit
/// vectors, traced routes, and the router's own scores.
#[allow(clippy::too_many_arguments)]
pub fn observation(
    seq: usize,
    pos: usize,
    baseline: &[f32],
    base_trace: &ExecutionTrace,
    candidate: &[f32],
    cand_trace: &ExecutionTrace,
) -> PositionObservation {
    let top_ids = top_k_ids(baseline, TOP_N);
    PositionObservation {
        sequence: seq as u32,
        position: pos as u32,
        baseline_logits: top_ids.iter().map(|&i| baseline[i]).collect(),
        candidate_logits: top_ids.iter().map(|&i| candidate[i]).collect(),
        top_ids: top_ids.iter().map(|&i| i as u32).collect(),
        baseline_logsumexp: logsumexp(baseline),
        candidate_logsumexp: logsumexp(candidate),
        baseline_argmax: argmax(baseline) as u32,
        candidate_argmax: argmax(candidate) as u32,
        baseline_top10: top_k_ids(baseline, 10).iter().map(|&i| i as u32).collect(),
        candidate_top10: top_k_ids(candidate, 10).iter().map(|&i| i as u32).collect(),
        // Every changed route WEIGHED — how close the decision was and
        // how much mixture mass moved — from the router's own selection
        // scores, so a near-tie swap and an overturned decision are
        // distinguishable rather than both being "one flip".
        route_changes: PositionObservation::weigh_route_changes(
            &base_trace.routes,
            &cand_trace.routes,
            &base_trace.selection_scores,
            &base_trace.combine_weights,
            &cand_trace.combine_weights,
        ),
        // The top-10 change WEIGHED, recorded only where the ordering
        // actually moved: a position that did not move says nothing
        // about how close the ones that did were.
        top10_change: weigh_top_k(baseline, candidate, 10),
        // The argmax flip weighed, when the winner changed.
        top1_change: weigh_top_1(baseline, candidate),
        baseline_routes: base_trace.routes.clone(),
        candidate_routes: cand_trace.routes.clone(),
    }
}

/// **What a top-k reordering actually was**, or `None` if the ordering
/// did not change.
///
/// Four facts, because the count answers none of them: how close the
/// boundary was, what the CANDIDATE did to that same pair, how much
/// probability mass moved, and how far anything travelled in rank.
fn weigh_top_k(baseline: &[f32], candidate: &[f32], k: usize) -> Option<TopKChange> {
    let (b_top, c_top) = (top_k_ids(baseline, k), top_k_ids(candidate, k));
    if b_top == c_top {
        return None;
    }
    // The boundary the baseline drew, and what the candidate did to
    // exactly those two ids. Comparing the two is the measurement a
    // worst-case `max|dlogit|` over the whole vocabulary cannot give.
    let ranked = top_k_ids(baseline, k + 1);
    let (boundary_margin, candidate_margin_same_ids) = if ranked.len() == k + 1 {
        let (lo, hi) = (ranked[k - 1], ranked[k]);
        (baseline[lo] - baseline[hi], candidate[lo] - candidate[hi])
    } else {
        (f32::NAN, f32::NAN)
    };

    // Half the L1 between the arms' top-k mass, each normalised over
    // its own k — the top-k analogue of the routed-mixture distance.
    let softmax_over = |logits: &[f32], ids: &[usize]| -> Vec<(usize, f32)> {
        let m = ids
            .iter()
            .map(|i| logits[*i])
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = ids.iter().map(|i| (logits[*i] - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        ids.iter().zip(exps).map(|(i, e)| (*i, e / sum)).collect()
    };
    let (bp, cp) = (
        softmax_over(baseline, &b_top),
        softmax_over(candidate, &c_top),
    );
    let mass_of = |v: &[(usize, f32)], id: usize| {
        v.iter()
            .find(|(i, _)| *i == id)
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    };
    let mut union: Vec<usize> = b_top.iter().chain(&c_top).copied().collect();
    union.sort_unstable();
    union.dedup();
    let mass_displaced = 0.5
        * union
            .iter()
            .map(|id| (mass_of(&bp, *id) - mass_of(&cp, *id)).abs())
            .sum::<f32>();

    // How far anything travelled. An id present in one arm's top-k and
    // absent from the other is located in the other's FULL ordering, so
    // an outsider arriving from rank 400 is not scored as a 1-place
    // move.
    let rank_in = |ordering: &[usize], id: usize, full: &[f32]| -> usize {
        ordering
            .iter()
            .position(|x| *x == id)
            .unwrap_or_else(|| full.iter().filter(|v| **v > full[id]).count())
    };
    let max_rank_displacement = union
        .iter()
        .map(|id| {
            let b = rank_in(&b_top, *id, baseline);
            let c = rank_in(&c_top, *id, candidate);
            b.abs_diff(c) as u32
        })
        .max()
        .unwrap_or(0);

    Some(TopKChange {
        boundary_margin,
        candidate_margin_same_ids,
        mass_displaced,
        max_rank_displacement,
    })
}

/// **What an argmax flip actually was**, or `None` if the winner did
/// not change.
///
/// Distinguishes a coin-flip between near-equal candidates from an
/// overturned confident choice — two events the flip count scores
/// identically, and the strictest criterion in the contract.
fn weigh_top_1(baseline: &[f32], candidate: &[f32]) -> Option<Top1Change> {
    let (b_win, c_win) = (argmax(baseline), argmax(candidate));
    if b_win == c_win {
        return None;
    }
    // Exact probabilities over the FULL vocabulary, so the mass is the
    // model's own belief rather than a renormalised truncation.
    let lse = logsumexp(baseline);
    let p = |id: usize| (baseline[id] - lse).exp();
    Some(Top1Change {
        // What the baseline held its winner by, over the pair that
        // actually swapped.
        boundary_margin: baseline[b_win] - baseline[c_win],
        // And what the candidate holds the reversed choice by.
        candidate_margin_same_ids: candidate[c_win] - candidate[b_win],
        // The probability the baseline gave up by switching.
        mass_displaced: p(b_win) - p(c_win),
    })
}

/// Per-sequence embedding rows from the exported bank.
pub fn sequence_embeddings(
    dir: &Path,
    seq: usize,
    positions: usize,
    hidden: usize,
) -> Result<Vec<Vec<f32>>, ExecutionFailure> {
    let path = dir.join(format!("seq_{seq}.f32"));
    let unreadable = |detail: String| ExecutionFailure::ArtifactUnreadable {
        what: format!("corpus sequence {seq}"),
        path: path.display().to_string(),
        detail,
    };
    let bytes = std::fs::read(&path).map_err(|e| unreadable(e.to_string()))?;
    if bytes.len() != positions * hidden * 4 {
        return Err(unreadable(format!(
            "holds {} bytes, not the {positions}x{hidden} f32 rows the manifest declares",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(hidden * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

/// Build one arm: every layer on device, the head riding the last
/// epoch. `overlay` present = the candidate arm.
pub fn build_stack<'a>(
    metal: &MetalBackend,
    model: &KimiSourceModel,
    overlay: Option<&CandidateOverlay>,
) -> Result<HybridStack<'a>, ExecutionFailure> {
    let n = model.geometry.num_layers;
    let mut device: Vec<Option<DeviceLayer>> = Vec::with_capacity(n);
    for i in 0..n {
        device.push(Some(model.device_layer(metal, i, overlay).map_err(
            |e| ExecutionFailure::LayerIncomplete {
                layer: i,
                detail: e.to_string(),
            },
        )?));
    }
    // Register the per-stack owned attention banks (the mmap-backed
    // stores are registered once by the caller).
    for d in device.iter().flatten() {
        for bank in d.attention_banks() {
            metal.register_weight_region(bank);
        }
    }
    let host: Vec<Option<(LayerSpec<'a>, LayerState)>> = (0..n).map(|_| None).collect();
    let mut stack = HybridStack::new(device, host);
    let head = model
        .head()
        .map_err(|e| ExecutionFailure::ArtifactUnreadable {
            what: "output head".into(),
            path: String::new(),
            detail: e.to_string(),
        })?;
    if !stack.attach_head(head) {
        return Err(ExecutionFailure::HeadDidNotAttach);
    }
    Ok(stack)
}

/// Run `positions` teacher-forced steps of one sequence through one
/// arm, returning each position's full logits and its execution trace.
pub fn run_sequence(
    metal: &MetalBackend,
    stack: &mut HybridStack<'_>,
    rows: &[Vec<f32>],
    hidden: usize,
) -> Result<Vec<(Vec<f32>, ExecutionTrace)>, ExecutionFailure> {
    stack
        .reset_states()
        .map_err(|e| ExecutionFailure::StepRefused {
            sequence: 0,
            position: 0,
            detail: format!("the stack did not reset: {e}"),
        })?;
    let mut out = Vec::with_capacity(rows.len());
    for (position, row) in rows.iter().enumerate() {
        let mut trace = ExecutionTrace {
            // The router's own selection scores and combine weights, so
            // a routing change can be WEIGHED rather than counted.
            // ~27 KB a token, read after the chain's single wait.
            want_selection_scores: true,
            ..ExecutionTrace::default()
        };
        let (logits, _traces, _t) = stack
            .forward_traced(metal, row, hidden, Some(&mut trace))
            .map_err(|e| ExecutionFailure::StepRefused {
                sequence: 0,
                position,
                detail: e.to_string(),
            })?;
        out.push((logits, trace));
    }
    Ok(out)
}
