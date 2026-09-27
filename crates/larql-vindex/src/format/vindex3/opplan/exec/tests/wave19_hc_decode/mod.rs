//! **Wave 19a — the decode traversal carries the hyper-connection bundle,
//! and an intermediate-state witness proves it.**
//!
//! The claim, kept to what the freeze allows: on a hyper-connected
//! component the decode step's residual is a bundle; each site reduces it
//! to the `[hidden]` vector the ordinary operator consumes and expands
//! the operator's output back; the observer sees state that cannot exist
//! on the single-stream path. Built while the public refusal at
//! `PreparedOperands::load` still stood (every image was then prepared
//! through a test-only seam); the lift that followed removed the seam,
//! and the public loader prepares what the witness proved.
//!
//! # The assertions (the freeze's A1–A6, decode half)
//!
//! ```text
//! A1  foreign      at layer 0's attention site, fed the ORACLE's own
//!                  state, the tapped split and reduced vector equal the
//!                  oracle's stage outputs, all three positions
//! A2  structural   bundle_out == expand(b, x_in, split), and it is NOT
//!                  x_in + b
//! A3  branch in    the attention-input tap == pre_attention(reduced)
//! A4  FFN site     b_ffn == the FFN recomputed from the reduced vector
//!                  (hybrid: this pins the router's raw residual)
//! A5  width        every record is streams x hidden, one per site per
//!                  layer per position
//! A6  headless     Full on the headless variant refuses at preparation
//!                  with the head reason; LayerRange runs; HeadBearing
//!                  Full reduces through the head and produces logits
//! ```
//!
//! # The controls (mutants a–d)
//!
//! Each perturbs the real traversal or the real composition, and each
//! must be caught by the assertion the freeze names — a witness that
//! could not fail would prove nothing. The unmutated traversal passes
//! every assertion first, so "caught" is a difference, not a default.

use crate::format::vindex3::opplan::exec::decode::{DecodeSession, StepRun};
use crate::format::vindex3::opplan::exec::execute_text;
use crate::format::vindex3::opplan::exec::hyper_connection::{
    expand_streams, head_reduce, Bundle, Mutation, SinkhornSplit,
};
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    HcSite, HcSiteRecord, InputSite, NoopObserver, StepEvent, StepObserver,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;

use super::wave19_hc_substrate::{
    self as substrate, Oracle, Substrate, Variant, HIDDEN, LAYERS, MIX_ROWS, NORM_EPS, POSITIONS,
    STREAMS, VOCAB,
};

/// The wave-17 stage tolerance: f32 transcription noise between torch's
/// vectorised accumulation and scalar loops, at values around 20.
pub(super) const TOLERANCE: f32 = 5e-5;
/// The smallest disagreement a control must produce to count as one —
/// well above [`TOLERANCE`], so "differs" cannot be transcription noise.
pub(super) const CONTROL_FLOOR: f32 = 1e-4;
/// Sites per layer.
pub(super) const SITES: usize = 2;

/// One site's record, owned.
pub(super) struct Record {
    pub(super) layer: usize,
    pub(super) site: HcSite,
    pub(super) position: usize,
    pub(super) split: SinkhornSplit,
    pub(super) reduced: Vec<f32>,
    pub(super) branch_output: Vec<f32>,
    pub(super) bundle_out: Bundle,
}

/// The observer: records every site and the attention-input tap.
#[derive(Default)]
pub(super) struct Witness {
    pub(super) records: Vec<Record>,
    pub(super) attention_inputs: Vec<(usize, Vec<f32>)>,
    pub(super) events: Vec<StepEvent>,
}

impl StepObserver for Witness {
    fn event(&mut self, event: StepEvent) {
        self.events.push(event);
    }

    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        if site == InputSite::Attention {
            self.attention_inputs.push((layer, values.to_vec()));
        }
    }

    fn hyper_connection_site(&mut self, record: HcSiteRecord<'_>) {
        self.records.push(Record {
            layer: record.layer,
            site: record.site,
            position: record.position,
            split: record.split.clone(),
            reduced: record.reduced.to_vec(),
            branch_output: record.branch_output.to_vec(),
            bundle_out: record.bundle_out.clone(),
        });
    }
}

impl Witness {
    pub(super) fn record(&self, layer: usize, site: HcSite, position: usize) -> Option<&Record> {
        self.records
            .iter()
            .find(|r| r.layer == layer && r.site == site && r.position == position)
    }

    fn attention_input(&self, layer: usize, position: usize) -> Option<&[f32]> {
        self.attention_inputs
            .iter()
            .filter(|(l, _)| *l == layer)
            .nth(position)
            .map(|(_, v)| v.as_slice())
    }
}

/// A decode run's observations and outputs.
pub(super) struct Run {
    pub(super) witness: Witness,
    pub(super) steps: Vec<StepRun>,
}

pub(super) fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub(super) fn close(actual: &[f32], expected: &[f32], what: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{what}: {} values against {}",
            actual.len(),
            expected.len()
        ));
    }
    let diff = max_abs_diff(actual, expected);
    if diff <= TOLERANCE {
        Ok(())
    } else {
        Err(format!("{what}: max |diff| {diff:e} exceeds {TOLERANCE:e}"))
    }
}

pub(super) fn store(sub: &Substrate) -> OperandStore {
    OperandStore::open(sub.container.path(), &sub.inspection).unwrap()
}

/// Prepare through the public loader.
pub(super) fn prepare(sub: &Substrate, slice: ExecutionSlice) -> Result<PreparedOperands, String> {
    let store = store(sub);
    PreparedOperands::load(&sub.plan, &store, &ReferenceBackend::new(), slice)
        .map_err(|e| e.to_string())
}

/// The loader's refusal text, for the negative preparations.
fn prepare_err(sub: &Substrate, slice: ExecutionSlice) -> String {
    match prepare(sub, slice) {
        Ok(_) => panic!("preparation succeeded where it must refuse"),
        Err(err) => err,
    }
}

/// The whole layer range of the substrate: the headless shape.
pub(super) fn layer_range() -> ExecutionSlice {
    ExecutionSlice::LayerRange {
        start: 0,
        end: LAYERS,
    }
}

/// Run the oracle's three positions through the stack, each entering
/// layer 0 as the oracle's own bundle, under `mutation`.
pub(super) fn run_from_oracle(sub: &Substrate, slice: ExecutionSlice, mutation: Mutation) -> Run {
    let ops = prepare(sub, slice).expect("the loader prepares the substrate");
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let oracle = Oracle::load();
    let mut witness = Witness::default();
    let mut steps = Vec::new();
    for position in 0..POSITIONS {
        let step = session
            .step_from_bundle(oracle.input(position), &mut witness, mutation)
            .expect("the step runs");
        steps.push(step);
    }
    Run { witness, steps }
}

/// Run `tokens` through a whole-stack image from the embedding, under
/// `mutation`.
pub(super) fn run_from_tokens(sub: &Substrate, tokens: &[u32], mutation: Mutation) -> Run {
    let ops = prepare(sub, ExecutionSlice::Full).expect("the loader prepares the substrate");
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let mut steps = Vec::new();
    for &token in tokens {
        steps.push(
            session
                .step_mutated(token, &mut witness, mutation)
                .expect("the step runs"),
        );
    }
    Run { witness, steps }
}

/// A1. The foreign reference: at layer 0's attention site, fed the
/// oracle's state, the split and the reduced vector are the oracle's.
pub(super) fn a1_foreign_stages(witness: &Witness, oracle: &Oracle) -> Result<(), String> {
    for position in 0..POSITIONS {
        a1_at(witness, oracle, position)?;
    }
    Ok(())
}

/// A1 at one position — so a defect that is invisible at position 0
/// and visible after it can be named as such.
pub(super) fn a1_at(witness: &Witness, oracle: &Oracle, position: usize) -> Result<(), String> {
    let record = witness
        .record(0, HcSite::Attention, position)
        .ok_or_else(|| format!("A1: no record at layer 0 attention, position {position}"))?;
    close(
        &record.split.pre,
        &oracle.stage("sinkhorn_pre", position),
        &format!("A1 position {position} pre"),
    )?;
    close(
        &record.split.post,
        &oracle.stage("sinkhorn_post", position),
        &format!("A1 position {position} post"),
    )?;
    close(
        &record.split.comb,
        &oracle.stage("sinkhorn_comb", position),
        &format!("A1 position {position} comb"),
    )?;
    close(
        &record.reduced,
        &oracle.stage("reduced", position),
        &format!("A1 position {position} reduced"),
    )
}

/// The bundle that entered each record's site, chained by (layer,
/// site, position) rather than by emission order — the decode path
/// emits one position through every site, the batch path every position
/// through one site, and the chain is the same: the oracle's state
/// enters layer 0's attention site, the attention site's output enters
/// the FFN site, and the FFN site's output enters the next layer.
fn entering_bundles<'a>(
    witness: &'a Witness,
    oracle: &Oracle,
) -> Result<Vec<(&'a Record, Bundle)>, String> {
    let mut out = Vec::new();
    for record in &witness.records {
        let entering = match record.site {
            HcSite::Attention if record.layer == 0 => oracle.input(record.position),
            HcSite::Attention => witness
                .record(record.layer - 1, HcSite::Ffn, record.position)
                .ok_or_else(|| {
                    format!(
                        "A2: no FFN record before layer {} position {}",
                        record.layer, record.position
                    )
                })?
                .bundle_out
                .clone(),
            HcSite::Ffn => witness
                .record(record.layer, HcSite::Attention, record.position)
                .ok_or_else(|| {
                    format!(
                        "A2: no attention record at layer {} position {}",
                        record.layer, record.position
                    )
                })?
                .bundle_out
                .clone(),
        };
        out.push((record, entering));
    }
    Ok(out)
}

/// A2. Stage five ran, and it is not a residual add.
pub(super) fn a2_expansion(witness: &Witness, oracle: &Oracle) -> Result<(), String> {
    for (record, x) in entering_bundles(witness, oracle)? {
        let what = format!(
            "A2 layer {} {:?} position {}",
            record.layer, record.site, record.position
        );
        let expected = expand_streams(
            &record.branch_output,
            x.as_flat(),
            &record.split,
            STREAMS,
            HIDDEN,
        );
        close(record.bundle_out.as_flat(), &expected, &what)?;
        // The single-stream form: every stream gets `x + b`.
        let added: Vec<f32> = (0..STREAMS)
            .flat_map(|j| {
                x.stream(j)
                    .iter()
                    .zip(&record.branch_output)
                    .map(|(r, b)| r + b)
                    .collect::<Vec<_>>()
            })
            .collect();
        let diff = max_abs_diff(record.bundle_out.as_flat(), &added);
        if diff < CONTROL_FLOOR {
            return Err(format!(
                "{what}: the bundle is within {diff:e} of a residual add"
            ));
        }
    }
    Ok(())
}

/// A3. The attention input is the pre-norm of the REDUCED vector.
pub(super) fn a3_branch_input(witness: &Witness, ops: &PreparedOperands) -> Result<(), String> {
    let backend = ReferenceBackend::new();
    for layer in 0..LAYERS {
        let norm = ops.layers()[layer]
            .pre_attention
            .as_ref()
            .ok_or("A3: the substrate has a pre-attention norm")?;
        for position in 0..POSITIONS {
            let record = witness
                .record(layer, HcSite::Attention, position)
                .ok_or_else(|| format!("A3: no record at layer {layer} position {position}"))?;
            let tapped = witness
                .attention_input(layer, position)
                .ok_or_else(|| format!("A3: no attention-input tap at layer {layer}"))?;
            close(
                tapped,
                &norm.apply(&backend, &record.reduced),
                &format!("A3 layer {layer} position {position}"),
            )?;
        }
    }
    Ok(())
}

/// A4. The FFN site's branch output is the FFN recomputed from the
/// reduced vector — residual argument included.
pub(super) fn a4_ffn_branch(
    witness: &Witness,
    sub: &Substrate,
    ops: &PreparedOperands,
) -> Result<(), String> {
    let backend = ReferenceBackend::new();
    for layer in 0..LAYERS {
        let prepared = &ops.layers()[layer];
        let plan_layer = &sub.plan.layers[layer];
        let (ffn, ffn_op) = match (&prepared.ffn, &plan_layer.ffn) {
            (Some(ffn), Some(op)) => (ffn, op),
            _ => return Err(format!("A4: layer {layer} carries no FFN")),
        };
        for position in 0..POSITIONS {
            let record = witness
                .record(layer, HcSite::Ffn, position)
                .ok_or_else(|| format!("A4: no FFN record at layer {layer} position {position}"))?;
            let v = &record.reduced;
            let normed = match &prepared.pre_ffn {
                Some(norm) => norm.apply(&backend, v),
                None => v.clone(),
            };
            let out = ffn
                .apply_from_residual(ffn_op, &backend, v, &normed, HIDDEN)
                .map_err(|e| e.to_string())?;
            let mut out = match &prepared.post_ffn {
                Some(norm) => norm.apply(&backend, &out),
                None => out,
            };
            crate::format::vindex3::opplan::exec::scale_residual_delta(
                plan_layer.residual_scale,
                &mut out,
            );
            close(
                &record.branch_output,
                &out,
                &format!("A4 layer {layer} position {position}"),
            )?;
        }
    }
    Ok(())
}

/// A5. One record per site per layer per position, every one a
/// `streams x hidden` bundle with `[hidden]` vectors beside it.
pub(super) fn a5_width(witness: &Witness) -> Result<(), String> {
    let expected = LAYERS * SITES * POSITIONS;
    if witness.records.len() != expected {
        return Err(format!(
            "A5: {} records, expected {expected}",
            witness.records.len()
        ));
    }
    for record in &witness.records {
        let what = format!(
            "A5 layer {} {:?} position {}",
            record.layer, record.site, record.position
        );
        if record.bundle_out.streams() != STREAMS || record.bundle_out.hidden() != HIDDEN {
            return Err(format!(
                "{what}: bundle is {} x {}",
                record.bundle_out.streams(),
                record.bundle_out.hidden()
            ));
        }
        if record.reduced.len() != HIDDEN || record.branch_output.len() != HIDDEN {
            return Err(format!("{what}: the branch vectors are not [hidden]"));
        }
        if record.split.pre.len() != STREAMS
            || record.split.post.len() != STREAMS
            || record.split.comb.len() != STREAMS * STREAMS
        {
            return Err(format!("{what}: the split is not [streams]/[streams^2]"));
        }
    }
    Ok(())
}

/// The whole decode chain on a headless, layer-range run.
fn witness_headless(mutation: Mutation) -> Vec<(&'static str, Result<(), String>)> {
    let sub = substrate::build(Variant::Headless);
    let oracle = Oracle::load();
    let run = run_from_oracle(&sub, layer_range(), mutation);
    let ops = prepare(&sub, layer_range()).unwrap();
    vec![
        ("A1", a1_foreign_stages(&run.witness, &oracle)),
        ("A2", a2_expansion(&run.witness, &oracle)),
        ("A3", a3_branch_input(&run.witness, &ops)),
        ("A4", a4_ffn_branch(&run.witness, &sub, &ops)),
        ("A5", a5_width(&run.witness)),
    ]
}

/// The hybrid chain: A4 on the estate whose router reads the residual.
fn witness_hybrid(mutation: Mutation) -> Vec<(&'static str, Result<(), String>)> {
    let sub = substrate::build(Variant::Hybrid);
    let oracle = Oracle::load();
    let run = run_from_oracle(&sub, layer_range(), mutation);
    let ops = prepare(&sub, layer_range()).unwrap();
    vec![
        ("A1", a1_foreign_stages(&run.witness, &oracle)),
        ("A2", a2_expansion(&run.witness, &oracle)),
        ("A4", a4_ffn_branch(&run.witness, &sub, &ops)),
        ("A5", a5_width(&run.witness)),
    ]
}

pub(super) fn assert_all_hold(results: &[(&str, Result<(), String>)]) {
    for (name, result) in results {
        assert!(result.is_ok(), "{name} failed: {:?}", result);
    }
}

pub(super) fn assert_caught_by(results: &[(&str, Result<(), String>)], named: &[&str]) {
    for name in named {
        let (_, result) = results
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("{name} is not in the chain"));
        assert!(result.is_err(), "{name} did not catch the mutant");
    }
}

fn store_carries_hc(sub: &Substrate) -> bool {
    let store = store(sub);
    PreparedOperands::load(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap()
    .carries_hyper_connection()
}

//
// Closure never produces these plans; the loader still refuses them by
// name rather than trusting the builder, because an image whose sites,
// topology and head disagree would run a wrong model fluently.

fn head_bearing_plan() -> (Substrate, OperandStore) {
    let sub = substrate::build(Variant::HeadBearing);
    let store = store(&sub);
    (sub, store)
}

fn load_err(
    plan: &crate::format::vindex3::opplan::ComponentOpPlan,
    store: &OperandStore,
) -> String {
    match PreparedOperands::load(plan, store, &ReferenceBackend::new(), ExecutionSlice::Full) {
        Ok(_) => panic!("preparation succeeded where it must refuse"),
        Err(err) => err.to_string(),
    }
}

mod the_positive_witness;
mod the_refusals_unchanged_and_new;
