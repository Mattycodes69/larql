//! The reduction is the SHARED one
//! What has NOT lifted

use super::*;

/// **The batch path reduces through `attention_residual::reduce`, over
/// the state the witness says it had.**
///
/// Rebuilds each site's history from the recorded prefix and the
/// recorded boundary values — the witness's own events, not the
/// executor's internals — calls the shared reduction directly, and
/// requires bit equality.
///
/// What it rules out: a batched re-implementation of the reduction that
/// agrees with the scalar one on this fixture but is a second source of
/// truth; and a `snapshot_count_before` that is a plausible number
/// rather than the size of the set actually read, since a wrong count
/// makes the rebuilt history the wrong length and the probabilities the
/// wrong width.
#[test]
fn every_batch_reduction_is_the_shared_reduction_over_the_recorded_state() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);
    let oracle = Oracle::load();

    let mut checked = 0;
    for site in run.witness.sites() {
        let snapshots = run
            .witness
            .snapshots_for(site.layer, site.site, site.position);
        assert_eq!(
            snapshots.len(),
            site.snapshot_count_before,
            "layer {} {:?} position {}: the recorded snapshot count does not match the \
             events that produced it",
            site.layer,
            site.site,
            site.position
        );
        let mut history = History::new(site.prefix_before.clone());
        for snapshot in snapshots {
            history.push_snapshot(snapshot);
        }
        assert_eq!(history.candidate_count(), site.candidate_count);

        let (norm, proj) = oracle.site_pair(site.layer, site.site);
        let reduction = attention_residual::reduce(
            &history,
            attention_residual::SitePair {
                norm: &norm,
                proj: &proj,
            },
            NORM_EPS,
            Mutation::None,
        )
        .expect("the shared reduction runs");
        assert_eq!(
            reduction.probs, site.probs,
            "layer {} {:?} position {}: probabilities differ from the shared reduction",
            site.layer, site.site, site.position
        );
        assert_eq!(
            reduction.mixed, site.mixed,
            "layer {} {:?} position {}: mixed vector differs from the shared reduction",
            site.layer, site.site, site.position
        );
        checked += 1;
    }
    // 13 reducing sites per position, the schedule the oracle spells.
    assert_eq!(checked, 13 * POSITIONS, "sites rebuilt");
}

/// **The batch exit reduces the LAST position's own history, and its
/// distribution is the oracle's width.**
///
/// The batch path reduces once at the exit, for the last position, and
/// emits no record for it — so the exit's probabilities and mixed vector
/// are not directly observable the way a site's are. They are recovered
/// here instead of being left unchecked: the last position's final state
/// is rebuilt from the witness's own events — its last FFN site's
/// `prefix_after`, plus every boundary value it recorded — the shared
/// reduction is run over it with the SHIPPED exit pair, and the result
/// is required to equal what the traversal actually produced.
///
/// What this pins that the output comparison alone does not:
///
/// - the exit read the LAST position's history, not another position's.
///   Rebuilding from position 2's recorded events and matching is what
///   says so; a run that had reduced position 0's state would produce a
///   vector this rebuild does not predict.
/// - the exit reduced over four candidates — three snapshots plus the
///   prefix — which is the oracle's own count for this geometry.
/// - the exit's distribution lies inside the oracle's band, like every
///   other reduction in the run.
/// - the final norm is applied AFTER the reduction, which is A5's
///   ordering claim, since the rebuilt mixed vector matches only under
///   that order.
#[test]
fn the_batch_exit_reduces_the_last_positions_own_history() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let oracle = Oracle::load();
    let run = batch(&sub, &TOKENS, Mutation::None);
    let last = POSITIONS - 1;

    // The state the last position left the stack in, from the witness.
    let final_site = run
        .witness
        .site(LAYERS - 1, HcSite::Ffn, last)
        .expect("the last layer's mlp site");
    let mut history = History::new(final_site.prefix_after.clone());
    for boundary in run.witness.boundaries() {
        if boundary.position == last {
            history.push_snapshot(boundary.value.clone());
        }
    }

    let expected_candidates = oracle.count("/exit/candidate_count");
    assert_eq!(
        history.candidate_count(),
        expected_candidates,
        "the exit must reduce over the oracle's candidate count"
    );

    let exit = ops
        .attention_residual_exit()
        .expect("a whole-stack image ships the exit pair");
    let reduction =
        attention_residual::reduce(&history, exit.pair(), exit.norm_eps(), Mutation::None)
            .expect("the shared reduction runs at the exit");
    assert_eq!(
        reduction.probs.len(),
        expected_candidates,
        "the exit distribution's width"
    );
    for &p in &reduction.probs {
        assert!(
            (MIN_PROB..=MAX_PROB).contains(&p),
            "exit probability {p:e} outside the oracle's band"
        );
    }

    let normed = match ops.final_norm() {
        Some(norm) => norm.apply(&backend, &reduction.mixed),
        None => reduction.mixed.clone(),
    };
    assert_eq!(
        run.exit, normed,
        "the batch exit is not the shared reduction over the last position's own history,          under the final norm"
    );

    // And the rebuild is not vacuous: another position's history must
    // NOT predict the exit. Without this the assertion above would pass
    // on a traversal that reduced any position, as long as the rebuild
    // used the same one.
    let other_site = run
        .witness
        .site(LAYERS - 1, HcSite::Ffn, 0)
        .expect("position 0's last mlp site");
    let mut other = History::new(other_site.prefix_after.clone());
    for boundary in run.witness.boundaries() {
        if boundary.position == 0 {
            other.push_snapshot(boundary.value.clone());
        }
    }
    let other_reduction =
        attention_residual::reduce(&other, exit.pair(), exit.norm_eps(), Mutation::None).unwrap();
    assert_ne!(
        other_reduction.mixed, reduction.mixed,
        "position 0's history predicts the same exit as the last position's — the          rebuild cannot tell the positions apart and this test proves nothing"
    );
}

/// **Every position enters the stack as its OWN first prefix, with an
/// EMPTY history — and a reader that asks a history plane for rows is
/// refused by name.**
///
/// The entry condition is where every later divergence between positions
/// begins: nothing is replicated, nothing is shared, and no snapshot
/// exists until the first boundary event takes one. A traversal that
/// entered on one shared state would satisfy every count in this file
/// and none of its values.
///
/// The refusal is decision 1's other half. `try_rows` is what a caller
/// wanting `[positions, hidden]` reaches for — the CLI's layer dump is
/// exactly that caller — and on this topology it must say what the plane
/// holds instead of flattening a prefix-plus-snapshots state into a file
/// whose format nothing could read back.
#[test]
fn every_position_enters_as_its_own_prefix_with_an_empty_history() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut embedded: Option<Plane> = None;
    execute_prepared_streaming_mutated(
        &sub.plan,
        &ops,
        &TOKENS,
        &backend,
        None,
        &mut |event| {
            if let PlaneEvent::Embedded(plane) = event {
                embedded = Some(plane.clone());
            }
            Ok(())
        },
        Mutation::None,
        None,
    )
    .expect("the batch traversal runs");

    let plane = embedded.expect("the traversal emits its embedding");
    assert_eq!(plane.positions(), POSITIONS);
    assert!(
        plane.bundles().is_none(),
        "an attention-residual plane is not a bundle plane"
    );
    let histories = plane
        .histories()
        .expect("an attention-residual component enters on a history plane");
    assert_eq!(histories.len(), POSITIONS);
    for (position, history) in histories.iter().enumerate() {
        assert_eq!(
            history.snapshot_count(),
            0,
            "position {position} entered with a snapshot it could not have taken"
        );
        assert_eq!(
            history.candidate_count(),
            1,
            "position {position} enters as one candidate: its own prefix"
        );
        assert_eq!(history.hidden(), HIDDEN);
    }
    // ...and they are three DIFFERENT prefixes, which is the whole
    // premise of the separation precondition, measured at its source.
    for (left, right) in pairs() {
        let gap = max_abs_diff(
            histories[left].prefix().expect("an entering prefix"),
            histories[right].prefix().expect("an entering prefix"),
        );
        assert!(
            gap > SEPARATION,
            "positions {left} and {right} entered on the same state ({gap:e})"
        );
    }

    let refusal = match plane.try_rows() {
        Ok(_) => panic!("a history plane must not answer as [hidden] rows"),
        Err(err) => err.to_string(),
    };
    assert!(refusal.contains("residual histories"), "{refusal}");
    assert!(refusal.contains("attention-residual"), "{refusal}");
}

/// **A resume point carrying a history plane is refused by name.**
///
/// Carrying a typed state through a traversal and reconstructing it from
/// an external representation are different capabilities. Nothing reads
/// a serialised prefix-plus-snapshots state back, and inventing a format
/// for one nothing consumes would be addressability without execution in
/// a new place. 2b builds the first and refuses the second.
#[test]
fn a_resume_point_carrying_a_history_plane_is_refused() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut witness = Witness::default();
    let resume = ResumePoint {
        next_layer: 1,
        hidden: Plane::Histories(
            (0..POSITIONS)
                .map(|p| History::new(vec![p as f32 + 1.0; HIDDEN]))
                .collect(),
        ),
    };
    let refusal = match collect(
        &sub,
        &ops,
        &TOKENS,
        Some(resume),
        Mutation::None,
        &backend,
        &mut witness,
    ) {
        Ok(_) => panic!("a history resume point must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(refusal.contains("resume point"), "{refusal}");
    assert!(refusal.contains("not supported"), "{refusal}");
    assert!(
        witness.events.is_empty(),
        "the refusal must come before any observation"
    );
}

/// **A resume point carrying ROWS is refused on an attention-residual
/// component** (RESIDUAL-BUS-2 reconnaissance §6).
///
/// The resume check matched on the hyper-connection topology alone, which
/// is `None` here, so rows of the right width were accepted and every site
/// ran as a single stream: no reduction, no boundary event, no exit
/// reduction. The carrier form must match the component's declared
/// topology, or nothing may run.
#[test]
fn a_resume_point_carrying_rows_is_refused_on_an_attention_residual_component() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut witness = Witness::default();
    let resume = ResumePoint {
        next_layer: 1,
        hidden: Plane::Rows(
            (0..POSITIONS)
                .map(|p| vec![p as f32 + 1.0; HIDDEN])
                .collect(),
        ),
    };
    let refusal = match collect(
        &sub,
        &ops,
        &TOKENS,
        Some(resume),
        Mutation::None,
        &backend,
        &mut witness,
    ) {
        Ok(_) => panic!("a rows resume point on an attention-residual component must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(refusal.contains("attention-residual"), "{refusal}");
    assert!(refusal.contains("resume point"), "{refusal}");
    assert!(
        witness.events.is_empty(),
        "the refusal must come before any observation"
    );
}
