use super::*;

/// An encoder whose output does not fill the slot the layout reserved
/// is refused — the bytes would land inside a neighbouring expert.
#[test]
fn an_encoding_that_does_not_fill_its_slot_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (mut fake, tensors) = Fake::kimi_layer(1, 0.5);
    // The SOURCE holds fewer values than the tensor's declared shape,
    // so the encoder produces a short block while the layout still
    // reserves a full one.
    let short = tensors[0].name.clone();
    fake.bytes.insert(short.clone(), vec![0u8; 512]);
    let mut idx = index(map_for(vec![Exception {
        projection: None,
        layers: Some((1, 1)),
        encoding: Some("Q6_K".into()),
    }]));
    let err = compile_expert_bank(
        &fake,
        &tensors[..1],
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("a short encoding must be refused");
    let text = format!("{err}");
    assert!(
        text.contains(&short) && text.contains("layout reserved"),
        "{text}"
    );
}

/// An index that cannot be written is reported rather than swallowed —
/// a compilation that believes it checkpointed and did not would resume
/// from nothing.
#[test]
fn an_unwritable_index_path_is_reported() {
    let dir = tempfile::tempdir().expect("tmp");
    // A directory where the index file should be: the write fails, and
    // the failure must surface.
    let path = dir.path().join("index.json");
    std::fs::create_dir(&path).expect("occupy the path with a directory");
    let idx = index(map_for(vec![]));
    assert!(
        write_index_atomically(&idx, &path).is_err(),
        "a failed index write must be reported"
    );
}

/// **A composed map holds several layers in ONE candidate, each at its
/// own encoding, placed disjointly.** L2 at Q8_0 beside L3 at Q6_K:
/// every L2 seal lands inside [0, q8_layer_bytes), every L3 seal at
/// q8_layer_bytes onward, no overlaps, and the ledger's total is the
/// sum of the two encodings' own extents — nothing shared, nothing
/// invented.
#[test]
fn a_composed_map_compiles_two_layers_disjointly_each_at_its_own_encoding() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (fake2, t2) = Fake::kimi_layer(2, 0.3);
    let (fake3, t3) = Fake::kimi_layer(3, 1.7);
    let mut bytes = fake2.bytes;
    bytes.extend(fake3.bytes);
    let fake = Fake { bytes };
    let tensors: Vec<SourceTensor> = t2.into_iter().chain(t3).collect();

    let map = PrecisionMap {
        name: "composed-l2q80-l3q6k".into(),
        encoding: "Q8_0".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((2, 2)),
                encoding: Some("Q8_0".into()),
            },
            Exception {
                projection: None,
                layers: Some((3, 3)),
                encoding: Some("Q6_K".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    };
    let mut idx = index(map);
    assert_eq!(
        idx.can_represent_as,
        vec!["Q8_0".to_string(), "Q6_K".to_string()],
        "a composed candidate states every encoding it can represent"
    );
    let outcome = compile_expert_bank(
        &fake,
        &tensors,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect("composed compile");
    assert_eq!(outcome.sealed, 2 * 3 * EXPERTS as usize);
    assert!(idx.ledger.overlaps().is_empty(), "layers must not collide");

    // Per-matrix bytes from the format's own arithmetic.
    let q8_matrix = LayerBankLayout::matrix_bytes("Q8_0", INTER, HIDDEN).expect("q8");
    let q6_matrix = LayerBankLayout::matrix_bytes("Q6_K", INTER, HIDDEN).expect("q6");
    let q8_layer = 3 * q8_matrix * u64::from(EXPERTS);
    let q6_layer = 3 * q6_matrix * u64::from(EXPERTS);
    assert_eq!(idx.ledger.compiled_bytes(), q8_layer + q6_layer);

    for seal in idx.ledger.sealed.values() {
        let layer: u32 = seal.tensor.split('.').next().unwrap().parse().unwrap();
        let end = seal.target_offset + seal.target_len;
        match layer {
            2 => {
                assert_eq!(seal.encoding, "Q8_0", "{}", seal.tensor);
                assert!(end <= q8_layer, "{} spills past its layer", seal.tensor);
            }
            3 => {
                assert_eq!(seal.encoding, "Q6_K", "{}", seal.tensor);
                assert!(
                    seal.target_offset >= q8_layer && end <= q8_layer + q6_layer,
                    "{} is outside its layer's band",
                    seal.tensor
                );
            }
            other => panic!("unexpected layer {other}"),
        }
    }
}

/// **A rerun handed a SUBSET of the composed scope must refuse, never
/// silently relocate.** The placement derives layer bases from the
/// handed scope; re-running with only layer 3's tensors would put its
/// bank at base 0 — on top of layer 2's bytes. The seal-offset guard
/// refuses by name instead.
#[test]
fn a_rerun_over_a_subset_of_the_composed_scope_is_refused() {
    let dir = tempfile::tempdir().expect("tmp");
    let out = dir.path().join("bank.bin");
    let (fake2, t2) = Fake::kimi_layer(2, 0.3);
    let (fake3, t3) = Fake::kimi_layer(3, 1.7);
    let mut bytes = fake2.bytes;
    bytes.extend(fake3.bytes);
    let fake = Fake { bytes };
    let both: Vec<SourceTensor> = t2.into_iter().chain(t3.iter().cloned()).collect();

    let map = PrecisionMap {
        name: "composed-subset-guard".into(),
        encoding: "Q8_0".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((2, 3)),
                encoding: Some("Q8_0".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    };
    let mut idx = index(map);
    compile_expert_bank(
        &fake,
        &both,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect("full-scope compile");

    let err = compile_expert_bank(
        &fake,
        &t3,
        &opts("target.expert_bank", EXPERTS, &out),
        &mut idx,
        &mut |_| {},
    )
    .expect_err("a subset scope recomputes every base and must refuse");
    let text = format!("{err}");
    assert!(
        text.contains("placement") && text.contains("full layer scope"),
        "the refusal must explain the scope mismatch: {text}"
    );
}

/// `CandidatePlacement` itself: single-layer placement is base 0
/// (byte-compatible with every existing candidate), an unmapped layer
/// and a per-projection-mixed layer are refused by name.
#[test]
fn placement_bases_and_refusals() {
    let single = PrecisionMap {
        name: "one".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![],
    };
    let p = CandidatePlacement::resolve(&single, Role::ExpertWeight, &[7], EXPERTS, HIDDEN, INTER)
        .expect("single layer places");
    assert_eq!(p.layer_base(7).expect("base"), 0, "single layer = base 0");
    assert!(p.layer_base(8).is_err(), "an unplaced layer is refused");

    // Two layers, one encoding: second base = first extent.
    let p2 =
        CandidatePlacement::resolve(&single, Role::ExpertWeight, &[7, 9], EXPERTS, HIDDEN, INTER)
            .expect("two layers place");
    let extent = p2.layout(7).expect("layout").layer_bytes().expect("bytes");
    assert_eq!(p2.layer_base(9).expect("base"), extent);

    // A layer the map does not compile cannot be placed.
    let scoped = PrecisionMap {
        name: "scoped".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: None,
                layers: Some((7, 7)),
                encoding: Some("Q6_K".into()),
            },
            Exception {
                projection: None,
                layers: None,
                encoding: None,
            },
        ],
    };
    let err =
        CandidatePlacement::resolve(&scoped, Role::ExpertWeight, &[8], EXPERTS, HIDDEN, INTER)
            .expect_err("an out-of-scope layer must refuse");
    assert!(format!("{err}").contains("compiles none"), "{err}");

    // Mixed encodings WITHIN one layer are not placeable: identity
    // addressing carries one stride.
    let mixed = PrecisionMap {
        name: "mixed".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions: vec![
            Exception {
                projection: Some("w1".into()),
                layers: Some((7, 7)),
                encoding: Some("Q8_0".into()),
            },
            Exception {
                projection: None,
                layers: Some((7, 7)),
                encoding: Some("Q6_K".into()),
            },
        ],
    };
    let err = CandidatePlacement::resolve(&mixed, Role::ExpertWeight, &[7], EXPERTS, HIDDEN, INTER)
        .expect_err("per-projection mixing within a layer must refuse");
    assert!(format!("{err}").contains("ONE encoding"), "{err}");
}

#[test]
fn completed_bank_authority_is_read_from_disk_and_rejects_changed_bytes() {
    use super::super::super::candidate_authority::{read_candidate, CandidateAuthorityRefusal};
    let dir = tempfile::tempdir().unwrap();
    let bank = dir.path().join("bank.bin");
    {
        let (source, tensors) = Fake::kimi_layer(1, 0.3);
        let mut idx = index(map_for(vec![]));
        compile_expert_bank(
            &source,
            &tensors,
            &opts("target.expert_bank", EXPERTS, &bank),
            &mut idx,
            &mut |_| {},
        )
        .unwrap();
        write_index_atomically(&idx, &dir.path().join("index.json")).unwrap();
    }
    let state = read_candidate(dir.path()).unwrap();
    assert_eq!(state.decisions().compiled(), (EXPERTS * 3) as usize);
    let mut bytes = std::fs::read(&bank).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&bank, bytes).unwrap();
    assert!(matches!(
        read_candidate(dir.path()),
        Err(CandidateAuthorityRefusal::Binding { .. })
    ));
}
