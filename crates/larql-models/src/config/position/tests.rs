use super::*;

#[test]
fn zero_is_the_nope_sentinel_at_the_boundary() {
    assert_eq!(
        PositionPolicy::from_declared_theta(0.0),
        PositionPolicy::None
    );
    assert_eq!(
        PositionPolicy::from_declared_theta(500000.0),
        PositionPolicy::Rope { theta: 500000.0 }
    );
}

#[test]
fn nope_layers_have_no_theta_to_offer() {
    assert_eq!(PositionPolicy::None.rope_theta(), None);
    assert_eq!(
        PositionPolicy::Rope { theta: 10000.0 }.rope_theta(),
        Some(10000.0)
    );
}

#[test]
fn serialises_tagged() {
    assert_eq!(
        serde_json::to_string(&PositionPolicy::None).unwrap(),
        "{\"kind\":\"none\"}"
    );
    assert_eq!(
        serde_json::to_string(&PositionPolicy::Rope { theta: 500000.0 }).unwrap(),
        "{\"kind\":\"rope\",\"theta\":500000.0}"
    );
}

/// Gemma 3's global layers: `{rope_type: linear, factor: 8.0}` at
/// the family's global base. The tag is the HF spelling so a
/// container written by this build is read back as the same policy.
#[test]
fn linear_serialises_tagged_and_answers_its_own_accessors() {
    let policy = PositionPolicy::Linear {
        theta: 1e6,
        factor: 8.0,
    };
    assert_eq!(
        serde_json::to_string(&policy).unwrap(),
        "{\"kind\":\"linear\",\"theta\":1000000.0,\"factor\":8.0}"
    );
    assert_eq!(policy.rope_theta(), Some(1e6));
    assert_eq!(policy.linear(), Some(8.0));
    assert_eq!(
        policy.declared_rope_type(),
        Some(super::super::rope_types::ROPE_TYPE_LINEAR)
    );
    // A full rotary: no fraction, and neither of the other blocks.
    assert_eq!(policy.rotary_fraction(), None);
    assert_eq!(policy.yarn(), None);
    assert_eq!(policy.llama3(), None);
    // And the plain policy answers nothing for the divisor — the
    // sliding layers of the same checkpoint must not inherit it.
    assert_eq!(PositionPolicy::Rope { theta: 1e4 }.linear(), None);
}

/// Composition at the declared-theta boundary: a plain rotary takes
/// the linear divisor; a NoPE sentinel does not grow one.
#[test]
fn linear_composes_with_a_declared_theta_but_not_with_nope() {
    let scaling = DeclaredRopeScaling::Linear { factor: 4.0 };
    assert_eq!(
        PositionPolicy::from_declared_theta_with_scaling(5e5, scaling),
        PositionPolicy::Linear {
            theta: 5e5,
            factor: 4.0
        }
    );
    assert_eq!(
        PositionPolicy::from_declared_theta_with_scaling(NOPE_THETA_SENTINEL, scaling),
        PositionPolicy::None
    );
}

#[test]
fn round_trips() {
    for policy in [
        PositionPolicy::None,
        PositionPolicy::Rope { theta: 1e6 },
        PositionPolicy::Linear {
            theta: 1e6,
            factor: 8.0,
        },
        PositionPolicy::Yarn {
            theta: 150000.0,
            scaling: gpt_oss_yarn(),
        },
    ] {
        let json = serde_json::to_string(&policy).unwrap();
        let back: PositionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, policy);
    }
}

fn gpt_oss_yarn() -> YarnRopeScaling {
    YarnRopeScaling {
        factor: 32.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 4096.0,
        truncate: false,
        mscale: None,
        mscale_all_dim: None,
    }
}

#[test]
fn a_yarn_block_attaches_only_to_a_rotating_layer() {
    let scaling = gpt_oss_yarn();
    assert_eq!(
        PositionPolicy::from_declared_theta_with_scaling(
            150000.0,
            DeclaredRopeScaling::Yarn(scaling)
        ),
        PositionPolicy::Yarn {
            theta: 150000.0,
            scaling
        }
    );
    // The NoPE sentinel wins over a checkpoint-wide YaRN block.
    assert_eq!(
        PositionPolicy::from_declared_theta_with_scaling(0.0, DeclaredRopeScaling::Yarn(scaling)),
        PositionPolicy::None
    );
    // No block: plain rotary, exactly as `from_declared_theta`.
    assert_eq!(
        PositionPolicy::from_declared_theta_with_scaling(150000.0, DeclaredRopeScaling::None),
        PositionPolicy::Rope { theta: 150000.0 }
    );
}

#[test]
fn scaled_rotary_still_offers_its_theta_and_only_it_offers_yarn() {
    let scaling = gpt_oss_yarn();
    let yarn = PositionPolicy::Yarn {
        theta: 150000.0,
        scaling,
    };
    assert_eq!(yarn.rope_theta(), Some(150000.0));
    assert_eq!(yarn.yarn(), Some(scaling));
    assert_eq!(PositionPolicy::Rope { theta: 1e4 }.yarn(), None);
    assert_eq!(PositionPolicy::None.yarn(), None);
}

fn gemma4_partial() -> PositionPolicy {
    PositionPolicy::PartialRope {
        theta: 1_000_000.0,
        rotary_fraction: 0.25,
        basis: RotaryFrequencyBasis::HeadWidth,
    }
}

/// A partial rotary offers its theta and its fraction, answers to
/// HF's `proportional` spelling only on the head-width basis, and
/// carries no YaRN block; the plain-partial basis is the default class.
#[test]
fn a_partial_rotary_answers_for_its_fraction_and_class() {
    let proportional = gemma4_partial();
    assert_eq!(proportional.rope_theta(), Some(1_000_000.0));
    assert_eq!(proportional.rotary_fraction(), Some(0.25));
    assert_eq!(proportional.declared_rope_type(), Some("proportional"));
    assert_eq!(proportional.yarn(), None);
    assert!(proportional.is_rotary());
    let plain_partial = PositionPolicy::PartialRope {
        theta: 10_000.0,
        rotary_fraction: 0.5,
        basis: RotaryFrequencyBasis::RotaryWidth,
    };
    assert_eq!(plain_partial.declared_rope_type(), None);
    assert_eq!(plain_partial.rotary_fraction(), Some(0.5));
    // Full rotary, YaRN and NoPE declare no fraction; YaRN answers `yarn`.
    assert_eq!(PositionPolicy::Rope { theta: 1e4 }.rotary_fraction(), None);
    assert_eq!(PositionPolicy::None.rotary_fraction(), None);
    assert_eq!(
        PositionPolicy::Rope { theta: 1e4 }.declared_rope_type(),
        None
    );
    assert_eq!(PositionPolicy::None.declared_rope_type(), None);
    let yarn = PositionPolicy::Yarn {
        theta: 150000.0,
        scaling: gpt_oss_yarn(),
    };
    assert_eq!(yarn.declared_rope_type(), Some("yarn"));
    assert_eq!(yarn.rotary_fraction(), None);
}

/// The partial rotary round-trips with its basis tagged.
#[test]
fn a_partial_rotary_round_trips_with_its_basis() {
    let json = serde_json::to_string(&gemma4_partial()).unwrap();
    assert!(json.contains("\"kind\":\"partial_rope\""), "{json}");
    assert!(json.contains("\"basis\":\"head_width\""), "{json}");
    let back: PositionPolicy = serde_json::from_str(&json).unwrap();
    assert_eq!(back, gemma4_partial());
}

#[test]
fn rotary_means_plain_or_scaled_but_not_nope() {
    assert!(PositionPolicy::Rope { theta: 1e4 }.is_rotary());
    assert!(PositionPolicy::Yarn {
        theta: 1e4,
        scaling: gpt_oss_yarn()
    }
    .is_rotary());
    assert!(!PositionPolicy::None.is_rotary());
}

/// **The M-RoPE axis table is the operator, not a label.**
///
/// `section` counts FREQUENCY slots (`rotary_dim / 2`), and the two
/// layouts place the same counts differently: contiguous blocks
/// (`TTT…HHH…WWW…`) versus HF's interleaving (`THWTHW…`). A
/// consumer that built one while the checkpoint declared the other
/// would rotate the right dimensions by the wrong axis's position.
#[test]
fn the_mrope_axis_table_places_each_axis_where_the_layout_says() {
    // Contiguous: 4 T slots, then 3 H, then 2 W, over 10 freqs.
    let blocked = mrope_axis_table([4, 3, 2], false, 10);
    assert_eq!(blocked, vec![0, 0, 0, 0, 1, 1, 1, 2, 2, 0]);

    // Interleaved: H at 1, 4, 7…, W at 2, 5, 8…, everything else T.
    let woven = mrope_axis_table([4, 3, 2], true, 10);
    assert_eq!(woven, vec![0, 1, 2, 0, 1, 2, 0, 1, 0, 0]);

    // The two are genuinely different placements of one section.
    assert_ne!(blocked, woven);

    // Slots past what `section` accounts for stay on T, matching
    // HF's "overwrite the first dimension" construction.
    assert!(mrope_axis_table([1, 1, 1], false, 8)[3..]
        .iter()
        .all(|a| *a == 0));

    // A section wider than the frequency count truncates rather
    // than writing past the table.
    assert_eq!(mrope_axis_table([9, 9, 9], false, 4).len(), 4);
    assert_eq!(mrope_axis_table([9, 9, 9], true, 4).len(), 4);
    // Qwen3.8's real geometry: [11, 11, 10] over 32 frequencies.
    let real = mrope_axis_table([11, 11, 10], false, 32);
    assert_eq!(real.len(), 32);
    assert_eq!(real.iter().filter(|a| **a == 0).count(), 11);
    assert_eq!(real.iter().filter(|a| **a == 1).count(), 11);
    assert_eq!(real.iter().filter(|a| **a == 2).count(), 10);
}

/// Each accessor answers for the variants that carry the fact and
/// `None` for the rest — never an implied default, which is the
/// whole reason these are `Option`.
#[test]
fn the_accessors_answer_only_where_the_fact_exists() {
    let rope = PositionPolicy::Rope { theta: 10000.0 };
    let partial_rot = PositionPolicy::PartialRope {
        theta: 10000.0,
        rotary_fraction: 0.25,
        basis: RotaryFrequencyBasis::RotaryWidth,
    };
    let partial_head = PositionPolicy::PartialRope {
        theta: 10000.0,
        rotary_fraction: 0.5,
        basis: RotaryFrequencyBasis::HeadWidth,
    };
    let mrope_rot = PositionPolicy::MRope {
        theta: 10000.0,
        rotary_fraction: 0.25,
        basis: RotaryFrequencyBasis::RotaryWidth,
        section: [11, 11, 10],
        interleaved: false,
    };
    let mrope_head = PositionPolicy::MRope {
        theta: 10000.0,
        rotary_fraction: 0.25,
        basis: RotaryFrequencyBasis::HeadWidth,
        section: [8, 8, 8],
        interleaved: true,
    };
    let nope = PositionPolicy::None;

    // rotary_fraction: only the two partial-width policies have one.
    assert_eq!(partial_rot.rotary_fraction(), Some(0.25));
    assert_eq!(mrope_rot.rotary_fraction(), Some(0.25));
    assert_eq!(rope.rotary_fraction(), None);
    assert_eq!(nope.rotary_fraction(), None);

    // mrope: the axis split, and None for everything that declares
    // no axis split — never an implied single axis.
    assert_eq!(mrope_rot.mrope(), Some(([11, 11, 10], false)));
    assert_eq!(mrope_head.mrope(), Some(([8, 8, 8], true)));
    assert_eq!(partial_rot.mrope(), None);
    assert_eq!(rope.mrope(), None);
    assert_eq!(nope.mrope(), None);

    // declared_rope_type: the HF spelling, and only when it is not
    // the default class. M-RoPE's own spelling lives in
    // `mrope_section`, so a rotary-width M-RoPE answers None.
    assert_eq!(partial_head.declared_rope_type(), Some("proportional"));
    assert_eq!(mrope_head.declared_rope_type(), Some("proportional"));
    assert_eq!(partial_rot.declared_rope_type(), None);
    assert_eq!(mrope_rot.declared_rope_type(), None);
    assert_eq!(rope.declared_rope_type(), None);
    assert_eq!(nope.declared_rope_type(), None);

    // is_rotary: everything but NoPE.
    for p in [rope, partial_rot, partial_head, mrope_rot, mrope_head] {
        assert!(p.is_rotary(), "{p:?} rotates");
    }
    assert!(!nope.is_rotary());
}

#[test]
fn a_llama3_block_resolves_to_its_own_policy_and_never_to_yarn() {
    let scaling = Llama3RopeScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192.0,
    };
    let policy = PositionPolicy::from_declared_theta_with_scaling(
        500000.0,
        DeclaredRopeScaling::Llama3(scaling),
    );
    assert_eq!(
        policy,
        PositionPolicy::Llama3 {
            theta: 500000.0,
            scaling
        }
    );
    assert_eq!(policy.llama3(), Some(scaling));
    assert_eq!(policy.rope_theta(), Some(500000.0));
    assert_eq!(policy.declared_rope_type(), Some("llama3"));

    // The hazard this guards: llama3 folded into Yarn would apply
    // YaRN's attention amplitude — a rescale of every logit at every
    // position — which no Llama checkpoint declares. The accessor
    // must not answer for a family that has no such block.
    assert_eq!(policy.yarn(), None);
    // ...and the reverse, so neither accessor drifts into the other.
    let yarn = YarnRopeScaling {
        factor: 32.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 4096.0,
        truncate: true,
        mscale: None,
        mscale_all_dim: None,
    };
    let yarn_policy =
        PositionPolicy::from_declared_theta_with_scaling(150000.0, DeclaredRopeScaling::Yarn(yarn));
    assert_eq!(yarn_policy.llama3(), None);
    assert_eq!(yarn_policy.yarn(), Some(yarn));
}

#[test]
fn a_nope_layer_stays_nope_under_a_llama3_block() {
    // The composition guard: a checkpoint-wide scaling block says how
    // a ROTATING layer rotates. A layer scheduled not to rotate must
    // not acquire a rotation from it.
    let scaling = Llama3RopeScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192.0,
    };
    assert_eq!(
        PositionPolicy::from_declared_theta_with_scaling(0.0, DeclaredRopeScaling::Llama3(scaling)),
        PositionPolicy::None
    );
}
