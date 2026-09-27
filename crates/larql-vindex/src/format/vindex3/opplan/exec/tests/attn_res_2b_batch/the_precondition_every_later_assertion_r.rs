//! The precondition every later assertion rests on
//! The foreign checks: the band and the schedule
//! A7: batch against decode

use super::*;

/// **The three positions carry measurably different state, everywhere.**
///
/// Runs FIRST in intent and is cited by every control below. If the
/// positions were not separated, the swap would exchange near-identical
/// histories, the broadcast would replace a state with its own twin, and
/// all three controls would report green while rejecting nothing — the
/// exact failure the oracle demonstrated on itself with a saturated
/// softmax, in its positional form.
///
/// Measures the MINIMUM pairwise separation over every site and every
/// boundary event of a clean run and requires it to exceed a fixed
/// multiple of the comparison tolerance. The margin is reported on
/// failure so a fixture that drifts toward degeneracy says how far it
/// drifted.
#[test]
fn the_three_positions_are_separated_before_any_control_is_scored() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);

    let mut worst = f32::INFINITY;
    let mut where_worst = String::new();
    let mut compared = 0;
    // Every PAIR, not every position against position 0: two positions
    // that had collapsed into each other while both differing from the
    // first would be invisible to a star comparison, and that is exactly
    // the degeneracy this test exists to exclude.
    for (left, right) in pairs() {
        for site in run.witness.sites() {
            if site.position != left {
                continue;
            }
            let peer = run
                .witness
                .site(site.layer, site.site, right)
                .unwrap_or_else(|| panic!("no record at layer {} position {right}", site.layer));
            let gap = max_abs_diff(&site.mixed, &peer.mixed);
            if gap < worst {
                worst = gap;
                where_worst = format!(
                    "layer {} {:?} positions {left}v{right}",
                    site.layer, site.site
                );
            }
            compared += 1;
        }
        for boundary in run.witness.boundaries() {
            if boundary.position != left {
                continue;
            }
            let peer = run
                .witness
                .boundaries()
                .into_iter()
                .find(|b| b.layer == boundary.layer && b.position == right)
                .expect("every position takes the same boundary events");
            let gap = max_abs_diff(&boundary.value, &peer.value);
            if gap < worst {
                worst = gap;
                where_worst = format!("boundary {} positions {left}v{right}", boundary.layer);
            }
            compared += 1;
        }
    }

    assert!(compared > 0, "the separation precondition compared nothing");
    assert!(
        worst > SEPARATION,
        "separation: the closest two positions differ by only {worst:e} at {where_worst}, \
         under the required {SEPARATION:e}. Every positional control in this file is \
         void at this separation."
    );
}

/// **A6 — the band.** No record's distribution may leave the oracle's
/// measured non-saturated interval.
///
/// The freeze's hard rule: a run outside the band is an INSTRUMENT
/// FAILURE and proves nothing either way, so this is scored before any
/// value comparison rather than alongside them.
#[test]
fn every_batch_distribution_lies_inside_the_oracles_measured_band() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);
    let sites = run.witness.sites();
    assert!(!sites.is_empty(), "the run produced no distributions");
    for site in sites {
        for &p in &site.probs {
            assert!(
                (MIN_PROB..=MAX_PROB).contains(&p),
                "layer {} {:?} position {}: probability {p:e} outside the oracle's band \
                 [{MIN_PROB:e}, {MAX_PROB}] — a saturated softmax makes every candidate-set \
                 control invisible and this run is not evidence",
                site.layer,
                site.site,
                site.position
            );
        }
    }
}

/// **A4 — the schedule, per position, against the oracle.**
///
/// The oracle's export is the authority for which layers reduce, over
/// how many candidates, and where the boundary events fall. Every
/// position must walk that schedule independently: a traversal that
/// snapshotted at one position and not another, or that let two
/// positions drift apart in DEPTH, is caught here even though its
/// numbers would all be plausible.
///
/// # Emission order is not operation order, and the difference is real
///
/// At a boundary layer the reference reduces the attention site over the
/// OLD snapshot set, then takes the snapshot, then runs the branch. But
/// a site's record cannot be emitted until its branch has run — it
/// carries `prefix_after` — so the boundary record is emitted BEFORE the
/// attention record whose reduction preceded it. Both traversals do this
/// identically, and it is intrinsic rather than incidental: a record
/// completed at entry could not report what the site left behind.
///
/// So the expected sequence below is in EMISSION order, and the
/// operation order is pinned separately and explicitly by
/// `the_boundary_falls_between_the_two_sites_at_every_position`, which
/// reads it off the counts — the mechanism the freeze names. Ordering by
/// position in the stream alone would assert a convention; ordering by
/// the counts asserts the topology.
///
/// This is also where the rung's two NUMERICALLY INERT properties are
/// proven on the batch path — layer 0 emitting no attention record at
/// all, and every layer emitting an FFN record. The oracle measured both
/// at a divergence of exactly 0.0; if this file ever reports either
/// caught by a value comparison, that comparison is broken.
#[test]
fn every_position_walks_the_oracles_schedule() {
    let sub = substrate();
    let oracle = Oracle::load();
    let run = batch(&sub, &TOKENS, Mutation::None);

    let mut expected: Vec<String> = Vec::new();
    for layer in 0..LAYERS {
        // The boundary marker sits here, ahead of the attention entry,
        // for the emission reason above — never because the event
        // precedes the reduction, which it does not.
        if oracle.ran(&format!("/witness/{layer}/snapshot_event/taken")) {
            let before = oracle.count(&format!("/witness/{layer}/snapshots_before"));
            let after = oracle.count(&format!("/witness/{layer}/snapshots_after"));
            expected.push(format!("layer {layer} boundary {before}->{after}"));
        }
        if oracle.ran(&format!("/witness/{layer}/attention_site/ran")) {
            let n = oracle.count(&format!("/witness/{layer}/attention_site/candidate_count"));
            expected.push(format!("layer {layer} attn over {n}"));
        }
        assert!(
            oracle.ran(&format!("/witness/{layer}/mlp_site/ran")),
            "the oracle's mlp site is unconditional; layer {layer} says otherwise"
        );
        let n = oracle.count(&format!("/witness/{layer}/mlp_site/candidate_count"));
        expected.push(format!("layer {layer} mlp over {n}"));
    }
    // The schedule the oracle read out of the reference: layer 0 has no
    // attention entry at all, and no entry anywhere reduces over one.
    assert!(
        !expected.contains(&"layer 0 attn over 1".to_string())
            && !expected.iter().any(|e| e.ends_with(" over 1")),
        "no site in the reference's schedule reduces over one candidate: {expected:?}"
    );

    for position in 0..POSITIONS {
        let actual: Vec<String> = run
            .witness
            .at(position)
            .iter()
            .filter(|e| e.layer() < LAYERS)
            .map(|e| match e {
                Event::Site(s) => {
                    let name = match s.site {
                        HcSite::Attention => "attn",
                        HcSite::Ffn => "mlp",
                    };
                    format!("layer {} {name} over {}", s.layer, s.candidate_count)
                }
                Event::Boundary(b) => format!(
                    "layer {} boundary {}->{}",
                    b.layer, b.snapshots_before, b.snapshots_after
                ),
            })
            .collect();

        assert_eq!(
            actual, expected,
            "position {position} does not walk the oracle's schedule"
        );
    }
}

/// **The ordering claim of decision 2, read off the counts, per
/// position.**
///
/// At a boundary layer the attention site must have reduced over the set
/// the event had NOT yet extended, and the FFN site over the one it had.
/// The counts say so unambiguously whatever order the records were
/// emitted in, which is why this is the assertion that carries the claim
/// and the sequence check above is not.
///
/// Also pins the snapshot's identity: the value appended is the ENTERING
/// prefix state, per position, and not the mixed vector or the
/// post-attention prefix — the two controls the oracle scored at 9.11e-01
/// and 1.72e+00.
#[test]
fn the_boundary_falls_between_the_two_sites_at_every_position() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);

    for position in 0..POSITIONS {
        let boundaries: Vec<_> = run
            .witness
            .boundaries()
            .into_iter()
            .filter(|b| b.position == position)
            .collect();
        assert_eq!(
            boundaries.iter().map(|b| b.layer).collect::<Vec<_>>(),
            vec![0, 3, 6],
            "position {position}: a boundary at every layer where layer % block == 0"
        );
        for boundary in boundaries {
            assert_eq!(
                boundary.snapshots_after,
                boundary.snapshots_before + 1,
                "position {position} layer {}: one snapshot per event",
                boundary.layer
            );
            assert_eq!(
                boundary.value, boundary.entering_prefix,
                "position {position} layer {}: the event snapshots the ENTERING state",
                boundary.layer
            );
            // The attention site read the set BEFORE the event...
            if let Some(attention) = run
                .witness
                .site(boundary.layer, HcSite::Attention, position)
            {
                assert_eq!(
                    attention.snapshot_count_before, boundary.snapshots_before,
                    "position {position} layer {}: the attention site read the extended set",
                    boundary.layer
                );
            }
            // ...and the FFN site read the set after it.
            let ffn = run
                .witness
                .site(boundary.layer, HcSite::Ffn, position)
                .expect("the mlp site is unconditional");
            assert_eq!(
                ffn.snapshot_count_before, boundary.snapshots_after,
                "position {position} layer {}: the mlp site read the un-extended set",
                boundary.layer
            );
        }
    }
}

/// **A7 — the batch and decode traversals agree to the bit, per
/// position.**
///
/// Exact equality, not a tolerance: the two paths run the same reduction
/// over the same state in the same order, so any difference is a
/// difference in what they did rather than in how they rounded.
///
/// Compared per position because the two paths interleave positions
/// differently — decode is position-major, batch layer-major — and the
/// ordering claim of the topology is a claim about one position's
/// sequence.
#[test]
fn the_batch_traversal_equals_the_decode_traversal_at_every_position() {
    let sub = substrate();
    let batched = batch(&sub, &TOKENS, Mutation::None);
    let stepped = decode(&sub, &TOKENS, Mutation::None);
    assert_a7(&batched.witness, &stepped.witness).expect("A7");

    // The exit. The batch path reduces at the exit for the LAST
    // position only — as every other topology's batch path does — so the
    // exit is compared as an OUTPUT rather than as a record.
    //
    // Compared at both taps, because they answer different questions.
    // The decode tap is the reduced vector before the final norm and the
    // batch one is after it, so the norm is applied to the decode tap
    // here: equality then says the two paths ran the same reduction AND
    // that the norm comes after it on both, which is A5's ordering claim.
    // The logits are compared as well, so nothing between the reduction
    // and the head is exempt.
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let normed = match ops.final_norm() {
        Some(norm) => norm.apply(&backend, &stepped.exit),
        None => stepped.exit.clone(),
    };
    assert_eq!(
        bits(&batched.exit),
        bits(&normed),
        "A7: the batch exit differs from the decode exit under the final norm"
    );
    assert_eq!(
        batched.exit.len(),
        HIDDEN,
        "the exit reduction produced the wrong width"
    );
    let batch_logits = batched.logits.expect("the batch path produces logits");
    let decode_logits = stepped.logits.expect("the decode path produces logits");
    assert_eq!(
        bits(&batch_logits),
        bits(&decode_logits),
        "A7: the batch and decode logits differ at the last position"
    );

    // And the decode arm really did run its exit reduction once per
    // step, so the equality above is an equality of two exits rather
    // than of two paths that both skipped one.
    let exit_records = stepped
        .witness
        .events
        .iter()
        .filter(|e| e.layer() == LAYERS)
        .count();
    assert_eq!(
        exit_records, POSITIONS,
        "each decode step must reduce once at the exit"
    );
}

/// A7 is bitwise, not value equality, on BOTH event kinds: a site record
/// or a boundary record whose only difference is the sign of a zero is a
/// disagreement. Derived `==` would pass either.
#[test]
fn a7_catches_a_signed_zero_in_a_site_or_a_boundary_that_value_equality_would_pass() {
    let sub = substrate();
    let batched = batch(&sub, &TOKENS, Mutation::None).witness;
    let stepped = decode(&sub, &TOKENS, Mutation::None).witness;
    assert_a7(&batched, &stepped).expect("the unaltered pair agrees");
    for boundary in [false, true] {
        let index = batched
            .events
            .iter()
            .position(|e| matches!(e, Event::Boundary(_)) == boundary)
            .expect("the witness carries both event kinds");
        let (mut ours, mut theirs) = (batched.clone(), stepped.clone());
        let theirs_index = theirs
            .events
            .iter()
            .position(|e| {
                e.position() == ours.events[index].position()
                    && e.layer() == ours.events[index].layer()
                    && matches!(e, Event::Boundary(_)) == boundary
            })
            .unwrap();
        let (a, b) = match (&mut ours.events[index], &mut theirs.events[theirs_index]) {
            (Event::Site(a), Event::Site(b)) => (&mut a.mixed[0], &mut b.mixed[0]),
            (Event::Boundary(a), Event::Boundary(b)) => (&mut a.value[0], &mut b.value[0]),
            _ => unreachable!(),
        };
        (*a, *b) = (-0.0, 0.0);
        assert_eq!(
            ours.events[index], theirs.events[theirs_index],
            "== calls them equal"
        );
        assert!(
            assert_a7(&ours, &theirs).is_err(),
            "A7 must not (boundary: {boundary})"
        );
    }
}
