//! Planes, layer traces and the final state of a traversal.

use super::super::ComponentOpPlan;
use crate::error::VindexError;
use backend::PlanBackend;
use continuation_registry::BoxedContinuation;
use hyper_connection::{Bundle, Mutation, SinkhornSplit};
use kv::KvState;
use observe::{CarrierTransition, HcSite, SublayerSite};
use operands::OperandSource;
use prepared::{ExecutionSlice, PreparedOperands};

#[allow(unused_imports)]
use super::*;

/// One plane of the traversal — the residual at a layer boundary, one
/// entry per position — typed by the component's residual topology
/// (wave 19b).
///
/// The batch carrier is `[positions, streams, hidden]` on a
/// hyper-connected component and `[positions, hidden]` everywhere else,
/// and the two are different types rather than one wider row: the
/// stream count and the hidden width are different semantic dimensions,
/// and a consumer that reads a plane must say which it expects. A
/// `[hidden]` row never comes out of a bundle here; it comes out of a
/// site's reduction or the head's, inside the traversal.
#[derive(Debug, Clone, PartialEq)]
pub enum Plane {
    /// One `[hidden]` row per position — every component before
    /// hyper-connections. Serialises and persists exactly as it always
    /// has.
    Rows(Vec<Vec<f32>>),
    /// One bundle per position — a hyper-connected component's residual.
    Bundles(Vec<Bundle>),
    /// One residual HISTORY per position — an attention-residual
    /// component's carrier (K3-ATTNRES-1 2b).
    ///
    /// The invariant this type exists to hold: **batching may vectorise
    /// the branch computation; it may not merge, share, reorder or
    /// reinterpret residual history.** Each position's snapshots are its
    /// own, taken at the same layers but from its own entering states,
    /// and a plane that collapsed them to one shared history would still
    /// produce plausible numbers at every site.
    Histories(Vec<attention_residual::History>),
}

impl Plane {
    /// How many positions the plane holds.
    pub fn positions(&self) -> usize {
        match self {
            Self::Rows(rows) => rows.len(),
            Self::Bundles(bundles) => bundles.len(),
            Self::Histories(histories) => histories.len(),
        }
    }

    /// The `[hidden]` rows, refused on a bundle plane. For consumers that
    /// can only mean one thing by a plane — persistence, comparison
    /// against a single-stream reference.
    pub fn try_rows(&self) -> Result<&[Vec<f32>], VindexError> {
        match self {
            Self::Rows(rows) => Ok(rows),
            Self::Bundles(bundles) => Err(VindexError::Parse(format!(
                "this plane holds {} bundles of {} streams, not [hidden] rows; the component is \
                 hyper-connected and the reader must say what a bundle means to it",
                bundles.len(),
                bundles.first().map_or(0, Bundle::streams)
            ))),
            Self::Histories(histories) => Err(VindexError::Parse(format!(
                "this plane holds {} residual histories, not [hidden] rows; the component \
                 declares the attention-residual topology and the reader must say what a \
                 prefix-plus-snapshots state means to it",
                histories.len()
            ))),
        }
    }

    /// [`Self::try_rows`] for callers on a single-stream component, where
    /// a bundle is a programming error rather than a model.
    pub fn rows(&self) -> &[Vec<f32>] {
        self.try_rows().unwrap_or_else(|e| panic!("{e}"))
    }

    /// The bundles, `None` on a row plane.
    pub fn bundles(&self) -> Option<&[Bundle]> {
        match self {
            Self::Rows(_) | Self::Histories(_) => None,
            Self::Bundles(bundles) => Some(bundles),
        }
    }

    /// The residual histories, `None` on every other plane.
    pub fn histories(&self) -> Option<&[attention_residual::History]> {
        match self {
            Self::Rows(_) | Self::Bundles(_) => None,
            Self::Histories(histories) => Some(histories),
        }
    }
}

/// Per-layer hidden-state taps, mirroring the production hook points.
#[derive(Debug)]
pub struct LayerTrace {
    /// The residual after the attention sublayer's update, per position.
    pub post_attention: Plane,
    /// The FFN's NORMED input (pre-FFN norm applied), per position —
    /// the vector the layer's gates multiply. This is the residual
    /// statistic V2's walk-FFN trace captures, and therefore the tap
    /// mutation capture must use: a gate built from anything else
    /// fires against a different vector than it was aimed at. A
    /// `[hidden]` row on every topology: it is the ordinary operator's
    /// input, which a site has already reduced.
    pub ffn_input: Vec<Vec<f32>>,
    /// The residual after the FFN sublayer's update, per position.
    pub post_layer: Plane,
}

/// What the stack's exit produced: the `[hidden]` vector the final norm
/// and head read, or — on a layer-range image of a hyper-connected
/// component, which has no exit — the bundle after the last layer.
#[derive(Debug, Clone, PartialEq)]
pub enum FinalState {
    Hidden(Vec<f32>),
    Bundle(Bundle),
    /// The last position's residual history, on a layer-range image of
    /// an attention-residual component — its output IS the state it
    /// hands on, and there is no `[hidden]` exit without the exit
    /// reduction a whole-stack image runs.
    History(attention_residual::History),
}

impl FinalState {
    /// The final `[hidden]` vector, refused when the run ended on a
    /// bundle.
    pub fn try_hidden(&self) -> Result<&[f32], VindexError> {
        match self {
            Self::Hidden(h) => Ok(h),
            Self::Bundle(b) => Err(VindexError::Parse(format!(
                "the run ended on a bundle of {} streams — a layer-range image of a \
                 hyper-connected component has no [hidden] exit",
                b.streams()
            ))),
            Self::History(h) => Err(VindexError::Parse(format!(
                "the run ended on a residual history of {} snapshot(s) — a layer-range image \
                 of an attention-residual component has no [hidden] exit until the exit \
                 reduction runs",
                h.snapshot_count()
            ))),
        }
    }

    /// [`Self::try_hidden`] for callers on a single-stream component.
    pub fn hidden(&self) -> &[f32] {
        self.try_hidden().unwrap_or_else(|e| panic!("{e}"))
    }
}

/// The full execution record of one component over one token sequence.
#[derive(Debug)]
pub struct ExecutionTrace {
    /// The residual *entering* layer 0, per position — everything the
    /// embedding op produced and nothing else.
    ///
    /// Captured because a layer-by-layer comparison needs somewhere to
    /// stand before layer 0: if the two sides already disagree here, no
    /// per-layer margin below means what it appears to mean. It is the
    /// same tap `scripts/dump_layers_hf.py` takes with a pre-hook on
    /// layer 0.
    pub embedded: Plane,
    pub layers: Vec<LayerTrace>,
    /// Plan indices of the layers that actually ran, in order.
    ///
    /// A reduced-depth run has to be able to prove it executed the
    /// prefix it asked for rather than silently falling back to the
    /// whole stack — and a full run has to be able to prove the reverse.
    /// `layers` alone cannot say that: a count is not an identity.
    pub executed_layers: Vec<usize>,
    /// What the last position left the stack as.
    pub exit: FinalState,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

impl ExecutionTrace {
    /// Final-normed hidden state of the last position (single-stream
    /// callers; see [`FinalState::hidden`]).
    pub fn final_hidden(&self) -> &[f32] {
        self.exit.hidden()
    }
}

/// One hyper-connection site's intermediate state for every position of
/// the batch (wave 19b) — the batch form of
/// [`observe::HcSiteRecord`], indexed by position.
#[derive(Debug, Clone, Copy)]
pub struct HcSitePlane<'a> {
    pub layer: usize,
    pub site: HcSite,
    pub splits: &'a [SinkhornSplit],
    pub reduced: &'a [Vec<f32>],
    pub branch_outputs: &'a [Vec<f32>],
    pub bundles_out: &'a [Bundle],
}

/// One attention-residual site across every position (K3-ATTNRES-1 2b) —
/// the batch counterpart of
/// [`AttnResSiteRecord`](super::super::observe::AttnResSiteRecord).
///
/// Everything here is per position and in position order, and that is
/// the point: a witness that could not tell the positions apart would
/// pass a traversal that shared one history between them.
#[derive(Debug)]
pub struct AttnResSitePlane<'a> {
    pub layer: usize,
    pub site: HcSite,
    /// Each position's distribution over its OWN candidates, and the
    /// vector it mixed to.
    pub reductions: &'a [attention_residual::Reduction],
    /// Each position's prefix as the site was ENTERED — before any
    /// boundary event of this layer reset it, which is the point the
    /// decode record captures too. Recording it after the reset would
    /// make the two paths describe different moments and A7 would be
    /// comparing a batch fact against a decode fact of another name.
    pub prefixes_before: &'a [Vec<f32>],
    /// Each position's snapshot count as the site was entered.
    ///
    /// Per position rather than one scalar even though the schedule
    /// makes them equal: a witness handed the count of position 0 could
    /// not tell a traversal whose positions had drifted apart in DEPTH
    /// from one whose positions agree, and the equality is a property to
    /// be checked rather than a shape to be assumed.
    pub snapshot_counts_before: &'a [usize],
    pub branch_outputs: &'a [Vec<f32>],
    /// Each position's history after its own write.
    pub histories_out: &'a [attention_residual::History],
}

/// One block-boundary event across every position — the batch
/// counterpart of
/// [`AttnResBoundaryRecord`](super::super::observe::AttnResBoundaryRecord).
#[derive(Debug)]
pub struct AttnResBoundaryPlane<'a> {
    pub layer: usize,
    pub snapshots_before: usize,
    pub snapshots_after: usize,
    /// The vector appended at each position — its OWN entering prefix
    /// state, never a shared one.
    pub values: &'a [Vec<f32>],
    /// Each position's entering prefix, so a witness can ASSERT that the
    /// appended vector is that prefix rather than trusting the caller
    /// passed the right one. Two of the rung's controls perturb exactly
    /// the difference between these two fields.
    pub entering_prefixes: &'a [Vec<f32>],
}

/// A plane handed to the caller the moment it exists, so a long run can
/// persist progress incrementally instead of holding 52 layers of hidden
/// state until the end.
#[derive(Debug)]
pub enum PlaneEvent<'a> {
    /// The residual entering layer 0 — plane 000. Not emitted when a
    /// [`ResumePoint`] skips the embedding.
    Embedded(&'a Plane),
    /// One completed layer's taps, in layer order.
    Layer { index: usize, trace: LayerTrace },
    /// One hyper-connection site's state, every position, emitted after
    /// the site's update and before the layer's own plane. Only a
    /// hyper-connected component emits it.
    HyperConnectionSite(HcSitePlane<'a>),
    /// One attention-residual site's state at every position, emitted
    /// after that site's per-position updates. Only a component that
    /// declares the topology emits it, and only where the reference
    /// reduces — layer 0's attention site emits nothing.
    AttentionResidualSite(AttnResSitePlane<'a>),
    /// One block-boundary event across every position, emitted between
    /// the attention site's reduction and the attention branch — the
    /// third contract point of a site under this topology.
    AttentionResidualBoundary(AttnResBoundaryPlane<'a>),
    /// One carrier transition at one position (RESIDUAL-BUS-1 T1), named
    /// with decode's [`CarrierTransition`](observe::CarrierTransition) and
    /// fired at the same point in the traversal. Filtered to one
    /// position, the batch stream of transitions is decode's.
    Transition {
        layer: usize,
        position: usize,
        transition: CarrierTransition,
    },
    /// One single-stream carrier write at every position, borrowed the
    /// moment the add lands (RESIDUAL-BUS-1). The batch counterpart of
    /// decode's [`StepObserver::carrier_write`](observe::StepObserver::carrier_write):
    /// the same write, at the same site, carrying the same values.
    CarrierWrite(CarrierWritePlane<'a>),
}

/// One single-stream carrier write across the batch (RESIDUAL-BUS-1).
///
/// Row `i` is position `i`'s write and matches decode's
/// [`CarrierWriteRecord`](observe::CarrierWriteRecord) at that position
/// bit for bit: `deltas[i]` is the branch output as added, after the
/// sublayer's post-norm and residual-delta scale, and `after[i]` is the
/// carrier once the add has landed. Both are borrowed where they already
/// are, so an unsubscribed traversal copies nothing for this event.
/// `before` is not carried; a consumer chains, as decode's does.
#[derive(Debug, Clone, Copy)]
pub struct CarrierWritePlane<'a> {
    pub layer: usize,
    pub site: SublayerSite,
    pub deltas: &'a [Vec<f32>],
    pub after: &'a [Vec<f32>],
    /// The per-layer scalar applied to the whole carrier after this write
    /// (Gemma 4 `layer_scalar`), on the FFN site of a component that
    /// declares one. `after` is pre-scale, as on decode's record (V3-OBS-1
    /// C5).
    pub layer_scale: Option<f32>,
}

/// Where an interrupted execution restarts.
///
/// The residual leaving layer `next_layer - 1` (plane `next_layer`) is
/// exactly the state entering `next_layer`, so a persisted plane resumes
/// the run bit-identically — no separate checkpoint format exists, and
/// none should: two formats could disagree. On a hyper-connected
/// component the plane is bundles, and a row plane is refused.
#[derive(Debug)]
pub struct ResumePoint {
    /// Index of the first layer still to execute.
    pub next_layer: usize,
    /// The residual entering that layer, one entry per position.
    pub hidden: Plane,
}

/// What execution produces beyond the streamed planes.
#[derive(Debug)]
pub struct FinalOutput {
    /// What the last position left the stack as.
    pub exit: FinalState,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

impl FinalOutput {
    /// Final-normed hidden state of the last position (single-stream
    /// callers; see [`FinalState::hidden`]).
    pub fn final_hidden(&self) -> &[f32] {
        self.exit.hidden()
    }
}

/// Execute a text-component plan on the reference backend.
///
/// The semantic anchor: naive f32, sharing no arithmetic with
/// `larql-compute`.
pub fn execute_text<'s>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
) -> Result<ExecutionTrace, VindexError> {
    // The oracle by request, from the shipped registry — not by
    // privileged construction (LOWERING-PLUGIN-1, L3).
    execute_plan_via(
        plan,
        store.into(),
        tokens,
        &lowering::LoweringRegistry::shipped(),
        &lowering::LoweringIdentity::reference(),
    )
}

/// Execute a text-component plan over `tokens` on `backend`, tracing
/// every layer.
///
/// The backend is a parameter, not a branch: nothing below reads its
/// identity, and swapping it must not change which operations run.
pub fn execute_plan<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
) -> Result<ExecutionTrace, VindexError> {
    execute_slice(plan, store, tokens, backend, ExecutionSlice::Full)
}

/// [`execute_plan`] on the provider `provider` names in `lowerings` — the
/// registry-carried path (LOWERING-PLUGIN-1, L2).
///
/// The registry is the caller's value and the identity is the caller's
/// choice; a provider the registry does not hold is refused here, by
/// identity and naming every provider it does hold, before any operand
/// is read. Nothing below constructs a provider the caller did not
/// register.
pub fn execute_plan_via<'s>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    lowerings: &lowering::LoweringRegistry,
    provider: &lowering::LoweringIdentity,
) -> Result<ExecutionTrace, VindexError> {
    let backend = lowerings.provider(provider)?;
    execute_plan(plan, store, tokens, backend)
}

/// [`execute_plan`] over a chosen [`ExecutionSlice`].
///
/// `execute_plan` is this with [`ExecutionSlice::Full`], so the two can
/// never disagree about what a whole model means.
pub fn execute_slice<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    slice: ExecutionSlice,
) -> Result<ExecutionTrace, VindexError> {
    execute_slice_over(plan, store.into(), tokens, backend, slice, None)
}

/// [`execute_slice`] over continuation state the caller selected — the
/// form a stack with state beyond softmax attention requires. `state`
/// must be fresh (position 0).
///
/// **Consumes the provider.** A one-shot traversal owns its continuation
/// state for exactly one pass and leaves nothing to continue: it never
/// advances the position, and a resumed pass writes no rows for the
/// layers it skipped. Taking the provider by value makes continuing from
/// it unrepresentable (RESIDUAL-BUS-2). The caller still chooses WHICH
/// provider (CONTINUATION-PLUGIN-1 C3); a stateful session owns its own.
pub fn execute_slice_in<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    slice: ExecutionSlice,
    mut state: BoxedContinuation,
) -> Result<ExecutionTrace, VindexError> {
    execute_slice_over(
        plan,
        store.into(),
        tokens,
        backend,
        slice,
        Some(&mut *state),
    )
}

pub(super) fn execute_slice_over<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    tokens: &[u32],
    backend: &B,
    slice: ExecutionSlice,
    state: Option<&mut dyn KvState>,
) -> Result<ExecutionTrace, VindexError> {
    let mut embedded = Plane::Rows(Vec::new());
    let mut layers = Vec::with_capacity(plan.layers.len());
    let mut executed_layers = Vec::with_capacity(plan.layers.len());
    let ops = PreparedOperands::load(plan, store, backend, slice)?;
    let mut sink = |event: PlaneEvent| {
        match event {
            PlaneEvent::Embedded(plane) => embedded = plane.clone(),
            PlaneEvent::Layer { index, trace } => {
                layers.push(trace);
                executed_layers.push(index);
            }
            // The trace is the layer boundaries; a site's state and a
            // boundary event are for the streaming sink, which is where
            // the witness reads them.
            PlaneEvent::HyperConnectionSite(_)
            | PlaneEvent::AttentionResidualSite(_)
            | PlaneEvent::AttentionResidualBoundary(_)
            | PlaneEvent::CarrierWrite(_)
            | PlaneEvent::Transition { .. } => {}
        }
        Ok(())
    };
    let out = execute_prepared_streaming_with(
        plan,
        &ops,
        tokens,
        backend,
        None,
        &mut sink,
        state,
        Mutation::None,
    )?;
    Ok(ExecutionTrace {
        embedded,
        layers,
        executed_layers,
        exit: out.exit,
        logits: out.logits,
    })
}

/// Streaming form of [`execute_plan`]: one traversal, planes delivered
/// through `sink` as they complete, with an optional [`ResumePoint`] to
/// restart an interrupted run.
///
/// [`execute_plan`] is a wrapper over this function, so the two can
/// never disagree about what the program means — there is exactly one
/// traversal in this module.
pub fn execute_plan_streaming<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<FinalOutput, VindexError> {
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    execute_prepared_streaming(plan, &ops, tokens, backend, resume, sink)
}

/// [`execute_plan_streaming`] over continuation state the caller
/// selected, and consumed (see [`execute_slice_in`]).
pub fn execute_plan_streaming_in<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    state: BoxedContinuation,
) -> Result<FinalOutput, VindexError> {
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    execute_prepared_streaming_in(plan, &ops, tokens, backend, resume, sink, state)
}
