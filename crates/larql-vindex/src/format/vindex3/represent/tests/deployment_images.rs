//! Deployment images

use super::*;

#[test]
fn a_deployment_image_drops_superseded_sources_and_keeps_protected_ones() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = deployment_of(&tmp);
    let index = index_of(&out);

    // Compiled objects: only the pack.
    for c in &report.compiled_objects {
        let reps: Vec<&RepresentationEntry> = index
            .representations
            .values()
            .filter(|e| e.object == c.object)
            .collect();
        assert_eq!(reps.len(), 1, "{} should carry only its pack", c.object);
        assert_eq!(reps[0].encoding, DTYPE_NVFP4);
    }

    // Protected objects still travel — an image missing its BF16 embedding
    // is smaller and does not execute.
    for p in &report.preserved_objects {
        let e = index
            .representations
            .values()
            .find(|e| e.object == p.object)
            .unwrap_or_else(|| panic!("{} must travel with the image", p.object));
        assert_ne!(e.encoding, DTYPE_NVFP4);
        assert!(
            out.join(&e.segment).is_file(),
            "{} bytes are present",
            p.object
        );
    }

    // Every declared segment resolves; nothing points at absent bytes.
    for (id, e) in &index.representations {
        assert!(out.join(&e.segment).is_file(), "{id} -> {}", e.segment);
        let key = e.segment.trim_end_matches(".bin");
        assert!(index.segments.contains_key(key), "{key} undeclared");
    }
}

#[test]
fn a_deployment_image_says_it_is_derived_and_names_its_source() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = deployment_of(&tmp);
    let index = index_of(&out);

    // "Executable, not re-compilable" has to be a claim the artifact makes,
    // not something an operator infers from what is missing.
    assert_eq!(index.authority, ContainerAuthority::Derived);
    assert!(index.derived_from_model.is_some());
    for e in index.representations.values() {
        if e.encoding == DTYPE_NVFP4 {
            assert!(
                e.source_representation_digest.is_some(),
                "a derived pack names the bytes it derives from"
            );
        }
    }
}

#[test]
fn the_graph_of_a_deployment_image_declares_only_present_bytes() {
    // The store binds `representations.first()`. If the graph still listed
    // the dropped canonical encoding, every reader would resolve to a
    // segment that is not in the image.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = deployment_of(&tmp);
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();
    let index = index_of(&out);
    let compiled: BTreeSet<&str> = report
        .compiled_objects
        .iter()
        .map(|c| c.object.as_str())
        .collect();

    for object in &graph.objects {
        if compiled.contains(object.id.as_str()) {
            assert_eq!(object.representations.len(), 1, "{}", object.id);
            assert_eq!(object.representations[0].encoding, DTYPE_NVFP4);
        }
        for r in &object.representations {
            let id = format!("{}@{}", object.id, r.encoding);
            if let Some(e) = index.representations.get(&id) {
                assert!(out.join(&e.segment).is_file(), "{id}");
            }
        }
    }
}

#[test]
fn an_archival_container_is_unaffected_by_the_deployment_switch() {
    // The default must stay archival: canonical bytes present, authority
    // unchanged, so nobody loses their source by omitting a flag.
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let index = index_of(&out);
    assert_eq!(index.authority, ContainerAuthority::Canonical);
    assert!(index.derived_from_model.is_none());
    for (id, e) in &index_of(&src).representations {
        assert!(index.representations.contains_key(id), "{id} was dropped");
        assert!(out.join(&e.segment).is_file());
    }
}

#[test]
fn a_protected_projection_is_carried_not_compiled() {
    // The R1 mechanism: an eligible role, held back anyway, so a precision
    // map can be expressed and then measured.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("protected.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("v_proj");
    let report = compile_representation(&src, &out, &spec).unwrap();

    let index = index_of(&out);
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == DTYPE_NVFP4)
        .unwrap();
    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    for t in &header.tensors {
        if t.name.contains("v_proj") {
            assert_ne!(t.dtype, DTYPE_NVFP4, "{} was compiled anyway", t.name);
        }
    }
    assert!(header.tensors.iter().any(|t| t.dtype == DTYPE_NVFP4));

    // A protected tensor is bigger than a compiled one, so the pack must
    // grow relative to R0 — the byte cost of the protection is the thing
    // being traded against fidelity.
    let r0_out = tmp.path().join("r0.vindex3");
    let r0 = compile_representation(&src, &r0_out, &RepresentSpec::nvfp4()).unwrap();
    assert!(
        report.compiled_objects[0].compiled_bytes > r0.compiled_objects[0].compiled_bytes,
        "protection costs bytes"
    );
}

#[test]
fn a_protected_depth_range_is_carried_not_compiled() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("early.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().layers(0, 0);
    compile_representation(&src, &out, &spec).unwrap();

    let index = index_of(&out);
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == DTYPE_NVFP4)
        .unwrap();
    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    for t in &header.tensors {
        if t.name.starts_with("0.") {
            assert_ne!(t.dtype, DTYPE_NVFP4, "{} is in a protected range", t.name);
        }
    }
    assert!(
        header.tensors.iter().any(|t| t.dtype == DTYPE_NVFP4),
        "later layers still compile"
    );
}

#[test]
fn a_protection_that_decides_nothing_is_refused_before_anything_is_written() {
    // `v-proj` protects no tensor: before the map check this compiled
    // every v_proj and recorded a map claiming they were held back.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let typo = tmp.path().join("typo.vindex3");
    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("v-proj");
    let err = compile_representation(&src, &typo, &spec)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`v-proj -> source` matches no eligible tensor"),
        "{err}"
    );
    assert!(
        !typo.exists(),
        "a refused map must not leave an output behind"
    );

    // A protection wholly inside an earlier one is dead, and says which.
    let dead = tmp.path().join("dead.vindex3");
    spec.protect = policy::Protections::default()
        .projection("v_proj")
        .projection_in("v_proj", 0, 0);
    let err = compile_representation(&src, &dead, &spec)
        .unwrap_err()
        .to_string();
    assert!(err.contains("decided first by exception(s) [0]"), "{err}");
    assert!(!dead.exists());
}

/// The precision map is authority, and both arms must run the SAME
/// program.
///
/// R0 could not see this: its map protects nothing in the decoder stack,
/// so "quantise everything" and "reproduce the map" coincide. The moment a
/// map is mixed they diverge, and the guarantee that stored and transient
/// differ *only* in whether compiled bytes already existed would quietly
/// stop being true.
///
/// The failure this pins is specific: a transient oracle that read
/// `--backend metal-nvfp4-*` as permission to quantise a tensor the map
/// deliberately kept at BF16 would be measuring a different model than the
/// one that was compiled.
#[test]
fn a_mixed_precision_map_runs_identically_on_both_arms() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("mixed.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    // q_proj protected, everything else eligible: a genuinely mixed map.
    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("q_proj");
    compile_representation(&src, &out, &spec).unwrap();

    let index = index_of(&out);
    let pack = index
        .representations
        .values()
        .find(|e| e.encoding == DTYPE_NVFP4)
        .expect("a pack exists");
    let (header, _) = read_segment_header(&out.join(&pack.segment)).unwrap();

    let open = |source| {
        let insp = inspect_container(&out, false).unwrap();
        OperandStore::open_for(&out, &insp, Some(DTYPE_NVFP4), source).unwrap()
    };
    let stored = open(RepresentationSource::Stored);
    let transient = open(RepresentationSource::Transient);

    let mut protected = 0usize;
    let mut compiled = 0usize;

    for t in &header.tensors {
        if t.shape.len() != 2 {
            continue;
        }
        let op = OperandRef {
            object: pack.object.clone(),
            tensor: t.name.clone(),
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
        };
        let a = load_weight((&stored).into(), &op, WeightFormat::Nvfp4).unwrap();
        let b = load_weight((&transient).into(), &op, WeightFormat::Nvfp4).unwrap();

        match (&a, &b) {
            (LoadedWeight::F16(x), LoadedWeight::F16(y)) => {
                // The map protected it: BOTH arms must keep it float, and
                // the bytes must match.
                assert!(t.name.contains("q_proj"), "{} unexpectedly float", t.name);
                assert_eq!(
                    &x.as_slice()[..x.logical_len()],
                    &y.as_slice()[..y.logical_len()],
                    "{}: protected tensor differs between arms",
                    t.name
                );
                protected += 1;
            }
            (
                LoadedWeight::Nvfp4 {
                    packed: p1,
                    scales: s1,
                    tensor_scale: t1,
                    ..
                },
                LoadedWeight::Nvfp4 {
                    packed: p2,
                    scales: s2,
                    tensor_scale: t2,
                    ..
                },
            ) => {
                assert!(!t.name.contains("q_proj"), "{} should be protected", t.name);
                assert_eq!(
                    &p1.as_slice()[..p1.logical_len()],
                    &p2.as_slice()[..p2.logical_len()],
                    "{}: codes differ",
                    t.name
                );
                assert_eq!(
                    &s1.as_slice()[..s1.logical_len()],
                    &s2.as_slice()[..s2.logical_len()],
                    "{}: scales differ",
                    t.name
                );
                assert_eq!(
                    t1.to_bits(),
                    t2.to_bits(),
                    "{}: tensor scale differs",
                    t.name
                );
                compiled += 1;
            }
            _ => panic!("{}: arms bound different formats", t.name),
        }
    }

    assert!(
        protected > 0,
        "the fixture must exercise a protected tensor"
    );
    assert!(compiled > 0, "and a compiled one");

    // Manufacture differs by exactly the compiled tensors, and by nothing
    // else — which is the whole content of "the arms differ only in
    // whether the bytes already existed".
    assert_eq!(stored.runtime_quantised(), 0);
    assert_eq!(
        transient.runtime_quantised() as usize,
        compiled,
        "transient manufactured something other than the map's compiled set"
    );
    assert_eq!(stored.bound_at_stored_precision() as usize, protected);
    assert_eq!(transient.bound_at_stored_precision() as usize, protected);
}

#[test]
fn the_container_declares_the_program_that_produced_it() {
    // Authority, not description: a reader can ask *why* a tensor is BF16
    // without having to observe that some pack stored it that way.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("mapped.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.protect = policy::Protections::default().projection("v_proj");
    compile_representation(&src, &out, &spec).unwrap();

    let program = index_of(&out)
        .precision_map
        .expect("the container states its program");
    assert_eq!(program.name, "r1-protect-v_proj");
    assert_eq!(program.encoding, DTYPE_NVFP4);
    assert!(program.roles.iter().any(|r| r == "decoder-linear"));

    use crate::format::vindex3::represent::map::Precision;
    assert_eq!(
        program.resolve(policy::Role::DecoderLinear, "0.self_attn.v_proj.weight"),
        Precision::Source
    );
    assert_eq!(
        program.resolve(policy::Role::DecoderLinear, "0.self_attn.k_proj.weight"),
        Precision::Compiled(DTYPE_NVFP4)
    );
    // And the map is a policy, not a transcript: it does not grow with the
    // model it was compiled against.
    assert!(program.exceptions.len() <= 2);
}

#[test]
fn a_pack_that_disagrees_with_the_declared_program_is_refused() {
    // The check `stored` owes. Without it a container could declare one
    // program and execute another, and the declaration would be decoration.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = compiled_pair(&tmp);

    // Claim a program that protects a projection the pack actually compiled.
    let mut index = index_of(&out);
    let mut program = index.precision_map.clone().unwrap();
    program.name = "claims-q-protected".into();
    program.exceptions = policy::Protections::default()
        .projection("q_proj")
        .as_exceptions();
    index.precision_map = Some(program);
    std::fs::write(
        out.join(INDEX_JSON),
        serde_json::to_string_pretty(&index).unwrap(),
    )
    .unwrap();

    let inspection = inspect_container(&out, false).unwrap();
    let store = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .unwrap();
    let (object, tensor, dtype, shape) = a_compiled_tensor(&out);
    let err = match load_weight(
        (&store).into(),
        &OperandRef {
            object,
            tensor,
            dtype,
            shape,
        },
        WeightFormat::Nvfp4,
    ) {
        Ok(_) => panic!("a non-conforming pack was executed"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("does not permit"), "{err}");
    assert!(
        err.contains("claims-q-protected"),
        "the refusal names the program: {err}"
    );
}

/// The serialised name and [`Role::name`] are one vocabulary.
///
/// A record persisted by an evidence gate and a role named on a CLI flag
/// have to resolve to the same authority key; two spellings of one role
/// would let a precision map silently fail to govern the tensors an
/// experiment measured.
#[test]
fn the_serde_form_is_the_role_name() {
    for r in super::super::policy::Role::ALL {
        let json = serde_json::to_string(r).expect("role serialises");
        assert_eq!(
            json,
            format!("\"{}\"", r.name()),
            "{r} serialises differently from its name"
        );
        let back: super::super::policy::Role = serde_json::from_str(&json).expect("round trips");
        assert_eq!(back, *r);
        assert_eq!(super::super::policy::Role::parse(r.name()), Some(*r));
    }
}
