//! Streaming entry points over a prepared plan.

use super::super::ComponentOpPlan;
use crate::error::VindexError;
use backend::PlanBackend;
use hyper_connection::Mutation;
use kv::KvState;
use operands::OperandSource;
use prepared::{ExecutionSlice, PreparedOperands};

#[allow(unused_imports)]
use super::*;

/// [`execute_plan_streaming`] over operands the caller already
/// prepared. One-shot callers keep the source-taking form above, which
/// prepares and discards; a server prepares once and calls this.
pub fn execute_prepared_streaming<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<FinalOutput, VindexError> {
    execute_prepared_streaming_with(
        plan,
        ops,
        tokens,
        backend,
        resume,
        sink,
        None,
        Mutation::None,
    )
}

/// [`execute_prepared_streaming`] over continuation state the caller
/// selected (see [`execute_slice_in`]).
pub fn execute_prepared_streaming_in<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    state: &mut dyn KvState,
) -> Result<FinalOutput, VindexError> {
    execute_prepared_streaming_with(
        plan,
        ops,
        tokens,
        backend,
        resume,
        sink,
        Some(state),
        Mutation::None,
    )
}

/// [`execute_prepared_streaming`] under a deliberate defect — the
/// wave-19b negative controls on the batch path. Test-only: production
/// has exactly one way in, and it passes [`Mutation::None`].
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn execute_prepared_streaming_mutated<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mutation: Mutation,
    state: Option<&mut dyn KvState>,
) -> Result<FinalOutput, VindexError> {
    execute_prepared_streaming_with(plan, ops, tokens, backend, resume, sink, state, mutation)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_prepared_streaming_with<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    state: Option<&mut dyn KvState>,
    mutation: Mutation,
) -> Result<FinalOutput, VindexError> {
    // A pin whose provider has gone or changed invalidates the image;
    // nothing here falls back to another realization. The registry is
    // the image's own — the store's — never a built-in default.
    ops.ensure_stack_ready()?;
    ops.ensure_providers_in(ops.registry())?;
    // And the pin's OTHER authority: the provider executing these pins
    // is the provider that decided them (LOWERING-PLUGIN-1, L4).
    if matches!(ops.slice(), prepared::ExecutionSlice::Endpoints) {
        return Err(VindexError::Parse(
            "endpoints-only operands require a distributed coordinator".into(),
        ));
    }
    ops.ensure_lowered_by(backend)?;
    // A one-shot forward starts a sequence, so any state it is given must
    // be fresh; it then runs over that state exactly as prefill would.
    if let Some(state) = state {
        if state.position() != 0 {
            return Err(VindexError::Parse(format!(
                "a one-shot forward starts a sequence, but its continuation state is already \
                 at position {}; resuming is prefill's job",
                state.position()
            )));
        }
        state.prepare_continuation(
            &continuation::plan_continuation_geometry(plan).map_err(VindexError::Parse)?,
        )?;
        return traverse(
            plan,
            ops,
            tokens,
            backend,
            resume,
            sink,
            Some(state),
            mutation,
        );
    }
    // Without state, only a wholly-softmax stack can run: `None` keeps its
    // behaviour exactly, including not materialising KV rows a caller
    // never asked for. A stack with a recurrence cannot run without
    // durable buffers, and the executor does not choose who holds them —
    // that is the caller's selection (CONTINUATION-PLUGIN-1, C3). It
    // refuses rather than manufacturing a default provider.
    if let Some(layer) = first_stateful_layer(plan) {
        return Err(VindexError::Parse(format!(
            "layer {layer} keeps continuation state beyond softmax attention; a one-shot \
             forward over this stack needs a continuation provider the caller selected — use \
             the `_in` form of this call"
        )));
    }
    traverse(
        plan,
        ops,
        tokens,
        backend,
        resume,
        sink,
        None::<&mut dyn KvState>,
        mutation,
    )
}

/// Whether a one-shot forward over `plan` needs continuation state the
/// caller selected — the `_in` forms of the execute calls. True when any
/// layer's attention is not softmax (a recurrence, a conv history, a
/// latent cache): the one rule the executor applies, exposed so a caller
/// asks it rather than re-deriving it.
pub fn requires_continuation(plan: &ComponentOpPlan) -> bool {
    first_stateful_layer(plan).is_some()
}

pub(super) fn first_stateful_layer(plan: &ComponentOpPlan) -> Option<usize> {
    plan.layers
        .iter()
        .position(|l| l.attention.softmax().is_none())
}

/// Batch prefill (VI3-INF-3): the batch traversal over `tokens`,
/// populating the **caller's** continuation state — the same provider
/// a [`decode::DecodeSession`] then resumes via
/// [`with_kv_state`](decode::DecodeSession::with_kv_state). There is
/// no batch-state → decode-state translation and the executor never
/// manufactures a state implementation: continuation state belongs to
/// the caller for its entire lifetime, and execution modes merely
/// consume and update it.
///
/// The provider is `prepare`d with the plan's geometry, appended one
/// conditioned K/V row pair per layer per position (all positions for
/// layer 0, then layer 1 — the opposite interleaving to the decode
/// step's), and its logical position advanced past `tokens`. A
/// provider already holding state is *extended*: positions continue
/// from `kv.position()`, so a long prompt can prefill in chunks.
///
/// Returns the last position's final-normed hidden state and logits,
/// so generation can sample the first continuation token without an
/// extra step.
pub fn prefill_plan<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    kv: &mut dyn KvState,
) -> Result<FinalOutput, VindexError> {
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    prefill_prepared(plan, &ops, tokens, backend, kv)
}

/// [`prefill_plan`] over operands the caller already prepared.
///
/// This is the serve path's prefill: the point of preparing a model
/// once is that batch prefill and the decode session that follows it
/// read the *same* resident operands, instead of each materialising the
/// model for itself.
pub fn prefill_prepared<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    kv: &mut dyn KvState,
) -> Result<FinalOutput, VindexError> {
    ops.ensure_stack_ready()?;
    if matches!(ops.slice(), prepared::ExecutionSlice::Endpoints) {
        return Err(VindexError::Parse(
            "endpoints-only operands require a distributed coordinator".into(),
        ));
    }
    // The FULL geometry, KV and recurrent alike. `plan_kv_geometry` is
    // the KV-only adapter and refuses a hybrid plan outright, which was
    // the right answer while nothing could execute one.
    kv.prepare_continuation(
        &continuation::plan_continuation_geometry(plan).map_err(VindexError::Parse)?,
    )?;
    let base = kv.position();
    let out = traverse(
        plan,
        ops,
        tokens,
        backend,
        None,
        &mut |_| Ok(()),
        Some(&mut *kv),
        Mutation::None,
    )?;
    kv.set_position(base + tokens.len());
    Ok(out)
}
