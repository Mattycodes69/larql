//! Directory/segment coherence: what inspection is FOR

use super::*;

/// THE G3 gate: encode the pair, then reconstruct the system solely from
/// the container — components, topology (incl. NoPE), the edge — with a
/// coherent directory and no source access.
#[test]
fn encode_then_inspect_reconstructs_the_system_without_the_source() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    let outcome = encode_system(&named, out.path()).unwrap();
    assert!(outcome.representations >= 7);
    assert!(outcome.total_payload_bytes > 0);

    // Sources gone: move nothing, read nothing — inspect uses only `out`.
    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(
        inspection.is_coherent(),
        "defects: {:?}",
        inspection.defects
    );

    let target = inspection
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    assert_eq!(target.num_layers, 8);
    assert_eq!(target.hidden_size, 64);
    assert_eq!(target.sliding_layers, Some(6));
    assert_eq!(target.full_layers, Some(2));
    assert_eq!(target.nope_layers, Some(2));
    assert_eq!(target.window, Some(16));

    let draft = inspection
        .components
        .iter()
        .find(|c| c.id == "draft")
        .unwrap();
    assert_eq!(draft.num_layers, 2);

    assert_eq!(inspection.graph.edges.len(), 1);
    let edge = &inspection.graph.edges[0];
    assert_eq!(edge.producer_layers, vec![1, 3, 5]);
    assert_eq!(edge.block_size, Some(4));
    assert_eq!(edge.consumer_object, "draft.feature_projector");
}

/// Payload bytes survive the trip exactly: read a tensor back out of its
/// segment via the header table and compare with the deterministic source
/// pattern.
#[test]
fn payload_bytes_round_trip_exactly() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    // The drafter's projector: encoder.fc.weight is 192*64 BF16 = 24576
    // bytes at source offsets [8320, 32896).
    let segment_path = out
        .path()
        .join(SEGMENTS_DIR)
        .join("draft.feature_projector.bin");
    let (header, payload_start) = read_segment_header(&segment_path).unwrap();
    let entry = header
        .tensors
        .iter()
        .find(|t| t.name == "fc.weight")
        .expect("object-relative name for the fusion tensor");
    assert_eq!(entry.shape, vec![192, 64]);

    let mut file = std::fs::File::open(&segment_path).unwrap();
    file.seek(SeekFrom::Start(payload_start + entry.offset))
        .unwrap();
    let mut encoded = vec![0u8; entry.len as usize];
    file.read_exact(&mut encoded).unwrap();

    // Source pattern over the whole shard payload region, sliced at the
    // tensor's declared source offsets.
    let source_slice = &payload_pattern(32896)[8320..32896];
    assert_eq!(encoded, source_slice, "payload bytes differ from source");
}

/// Object-relative names carry no artifact-global prefixes.
#[test]
fn segment_tensor_names_are_object_relative() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let (header, _) = read_segment_header(
        &out.path()
            .join(SEGMENTS_DIR)
            .join("target.decoder_stack.bin"),
    )
    .unwrap();
    for tensor in &header.tensors {
        assert!(
            !tensor.name.starts_with("model."),
            "artifact-global name leaked: {}",
            tensor.name
        );
    }
    // Multi-binding object: names stay unique after prefix stripping.
    let (header, _) = read_segment_header(
        &out.path()
            .join(SEGMENTS_DIR)
            .join("draft.feature_projector.bin"),
    )
    .unwrap();
    let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["fc.weight", "output_norm_enc.weight"]);
}

/// Encoding is deterministic: same inputs, same hashes.
#[test]
fn encode_is_deterministic() {
    let (_a, _b, named) = glimmer_system();
    let out1 = tempfile::tempdir().unwrap();
    let out2 = tempfile::tempdir().unwrap();
    encode_system(&named, out1.path()).unwrap();
    encode_system(&named, out2.path()).unwrap();
    let read = |p: &std::path::Path| {
        std::fs::read_to_string(p.join(crate::format::filenames::INDEX_JSON)).unwrap()
    };
    assert_eq!(read(out1.path()), read(out2.path()));
}

/// An inadmissible plan is refused before a single byte is written.
#[test]
fn inadmissible_plan_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory
        .config_keys
        .push(larql_models::inventory::ConfigKeyFact {
            path: "some_future_field_nobody_reviewed".to_string(),
            value: serde_json::json!(42),
            status: larql_models::inventory::KeyStatus::Unconsumed,
        });
    let named = vec![("llama-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    let err = encode_system(&named, out.path()).unwrap_err();
    assert!(err.to_string().contains("inadmissible"), "{err}");
    assert!(
        !out.path()
            .join(crate::format::filenames::INDEX_JSON)
            .exists(),
        "a refused encode must not leave an index behind"
    );
}

/// `encode_graph` runs the same validation as the planned path: a graph
/// a caller built (or damaged) itself is refused BEFORE any bytes are
/// written, with the defect named.
#[test]
fn a_graph_that_fails_validation_is_refused_before_encode() {
    use crate::format::vindex3::encode::encode_graph;
    use crate::format::vindex3::graph::build_from_inventories;

    let (_a, _b, named) = glimmer_system();
    let mut graph = build_from_inventories(&named).graph;
    let duplicate = graph.objects[0].clone();
    graph.objects.push(duplicate);
    let out = tempfile::tempdir().unwrap();
    let err = encode_graph(&graph, &named, out.path()).unwrap_err();
    assert!(
        err.to_string().contains("failed validation"),
        "the refusal must say the graph is at fault: {err}"
    );
}

/// An object stripped of every representation has nothing to encode —
/// graph validation does not police representations (that is encode's
/// concern), so encode itself must refuse and name the object.
#[test]
fn an_object_with_no_representation_is_refused_by_name() {
    use crate::format::vindex3::encode::encode_graph;
    use crate::format::vindex3::graph::build_from_inventories;

    let (_a, _b, named) = glimmer_system();
    let mut graph = build_from_inventories(&named).graph;
    let victim = graph.objects[0].id.clone();
    graph.objects[0].representations.clear();
    let out = tempfile::tempdir().unwrap();
    let err = encode_graph(&graph, &named, out.path()).unwrap_err();
    assert!(
        err.to_string().contains("carries no representation") && err.to_string().contains(&victim),
        "the refusal must name the empty object: {err}"
    );
}

/// A corrupted segment byte is caught by `inspect --verify` — with no
/// source access.
#[test]
fn verify_catches_a_flipped_payload_byte() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    let victim = out.path().join(SEGMENTS_DIR).join("target.embedding.bin");
    let mut bytes = std::fs::read(&victim).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&victim, bytes).unwrap();

    let clean = inspect_container(out.path(), false).unwrap();
    assert!(
        clean.is_coherent(),
        "structure still coherent without verify"
    );
    let verified = inspect_container(out.path(), true).unwrap();
    assert!(!verified.is_coherent());
    assert!(verified
        .defects
        .iter()
        .any(|d| format!("{d:?}").contains("target.embedding")));
}

/// The graph manifest is written verbatim and reloads as the same graph.
#[test]
fn graph_manifest_round_trips() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.path().join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();
    assert!(graph.validate().is_empty());
    assert_eq!(graph.edges.len(), 1);
}

/// A known dense model encodes and reconstructs the same way — the path is
/// generic, not Glimmer-shaped.
#[test]
fn known_dense_encodes_and_inspects() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system_unenforced(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(inspection.is_coherent(), "{:?}", inspection.defects);
    assert!(inspection.graph.edges.is_empty());
    assert!(inspection
        .index
        .representations
        .keys()
        .any(|k| k.starts_with("target.embedding@")));
}

/// A header length past the sanity bound is refused, not allocated.
///
/// The guard exists because the length prefix is the first thing read from
/// an untrusted file: without it a corrupt or hostile prefix becomes a
/// multi-gigabyte `vec![0; header_len]` before anything has been validated.
/// The refusal must name the claimed size, so a reader can tell a corrupt
/// file from a merely unsupported one.
#[test]
fn a_header_length_past_the_bound_is_refused_before_allocating() {
    use std::io::Write;
    const ABSURD_HEADER_LEN: u64 = 512 * 1024 * 1024;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.segment");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&ABSURD_HEADER_LEN.to_le_bytes()).unwrap();
    // Deliberately no header body: the guard must fire on the prefix alone,
    // before any attempt to read that many bytes.
    f.flush().unwrap();
    drop(f);

    let err = read_segment_header(&path).unwrap_err().to_string();
    assert!(
        err.contains(&ABSURD_HEADER_LEN.to_string()),
        "refusal must name the claimed length: {err}"
    );
    assert!(err.contains("corrupt"), "{err}");
}

/// A header length inside the bound but with no body behind it is a short
/// read, not the corruption refusal — the two failures are different and a
/// reader must not be told the wrong one.
#[test]
fn a_truncated_header_is_a_short_read_not_a_corruption_claim() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.segment");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&64u64.to_le_bytes()).unwrap();
    f.write_all(b"not sixty-four bytes").unwrap();
    f.flush().unwrap();
    drop(f);

    let err = read_segment_header(&path).unwrap_err().to_string();
    assert!(
        !err.contains("corrupt"),
        "a short body is not the over-long-header failure: {err}"
    );
}

/// An object whose bindings select nothing is a disagreement between the
/// graph and the inventory, and it must be named as such.
///
/// Silently planning zero tensors would write a well-formed segment holding
/// no bytes — a container that passes every structural check and cannot
/// serve the object it claims to carry.
#[test]
fn an_object_matching_no_source_tensors_refuses() {
    use crate::format::vindex3::encode::plan_object_tensors;
    use crate::format::vindex3::graph::object::{
        LogicalObject, ObjectKind, Representation, SourceBinding,
    };
    use std::collections::BTreeMap;

    let dir = tempfile::tempdir().unwrap();
    let inventory = known_dense(dir.path());
    let mut inventories = BTreeMap::new();
    inventories.insert("only-artifact", &inventory);

    let object = LogicalObject {
        id: "target.embedding".to_string(),
        component: "target".to_string(),
        kind: ObjectKind::Embedding,
        source_bindings: vec![SourceBinding {
            artifact: "only-artifact".to_string(),
            // A prefix no tensor in the inventory carries.
            tensor_prefix: "no.such.prefix".to_string(),
            tensors: 0,
            bytes: 0,
        }],
        representations: Vec::<Representation>::new(),
    };

    let err = match plan_object_tensors(&object, &inventories, std::slice::from_ref(&object)) {
        Err(e) => e.to_string(),
        Ok(planned) => panic!(
            "planned {} tensors from a prefix nothing matches",
            planned.len()
        ),
    };
    assert!(err.contains("target.embedding"), "{err}");
    assert!(
        err.contains("bindings and inventory disagree"),
        "the refusal must say which two things disagree: {err}"
    );
}

/// `binding_owner` matches a binding's prefix only at a segment boundary.
///
/// A plain `starts_with` would make `model.layers_extra.0` resolve to the
/// binding for `model.layers`, quietly filing one object's tensors under
/// another. The boundary check is the thing being pinned, so the negative
/// case has to be a name that *shares a textual prefix* and is still not a
/// match — a name that merely differs would pass either way.
#[test]
fn binding_owner_matches_on_segment_boundaries_not_text_prefixes() {
    use crate::format::vindex3::encode::binding_owner;
    use crate::format::vindex3::graph::object::{
        LogicalObject, ObjectKind, Representation, SourceBinding,
    };

    let object = LogicalObject {
        id: "target.stack".to_string(),
        component: "target".to_string(),
        kind: ObjectKind::DecoderStack,
        source_bindings: vec![SourceBinding {
            artifact: "art".to_string(),
            tensor_prefix: "model.layers".to_string(),
            tensors: 1,
            bytes: 1,
        }],
        representations: Vec::<Representation>::new(),
    };

    assert_eq!(binding_owner(&object, "model.layers"), Some("art"));
    assert_eq!(binding_owner(&object, "model.layers.0.attn"), Some("art"));
    assert_eq!(binding_owner(&object, "model.layers_extra.0"), None);
    assert_eq!(binding_owner(&object, "other.tensor"), None);
}

#[test]
fn a_directory_entry_naming_an_unknown_object_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        index["representations"][&id]["object"] = serde_json::json!("target.no_such_object");
    });
    assert!(
        defects
            .iter()
            .any(|d| d.contains("references unknown object")
                && d.contains("target.no_such_object")),
        "{defects:?}"
    );
}

#[test]
fn a_directory_tensor_count_disagreeing_with_the_segment_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        index["representations"][&id]["tensor_count"] = serde_json::json!(9_999);
    });
    assert!(
        defects
            .iter()
            .any(|d| d.contains("tensors, directory says 9999")),
        "{defects:?}"
    );
}

#[test]
fn a_directory_payload_size_disagreeing_with_the_file_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        index["representations"][&id]["payload_bytes"] = serde_json::json!(1);
    });
    assert!(
        defects
            .iter()
            .any(|d| d.contains("bytes, expected") && d.contains("payload")),
        "{defects:?}"
    );
}

/// A clean container reports no defects and a complete execution surface —
/// the control the three tampering tests above are measured against. Without
/// it, "defect found" could just mean the fixture never inspects clean.
#[test]
fn an_untampered_container_inspects_clean_and_complete() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("only-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system_unenforced(&named, out.path()).unwrap();

    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(inspection.is_coherent(), "{:?}", inspection.defects);
    assert!(
        inspection.execution_completeness().is_empty(),
        "{:?}",
        inspection.execution_completeness()
    );
}
