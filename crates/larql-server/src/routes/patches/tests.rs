use super::*;

fn request(url: &str) -> ApplyPatchRequest {
    ApplyPatchRequest {
        url: Some(url.into()),
        patch: None,
    }
}

fn sources(dir: &std::path::Path) -> PatchSources {
    PatchSources {
        dir: Some(dir.to_path_buf()),
        allow_hf: false,
    }
}

#[test]
fn a_patch_inside_the_patch_dir_loads_by_url() {
    let dir = tempfile::tempdir().unwrap();
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "m".into(),
        base_checksum: None,
        created_at: "2026-09-26".into(),
        description: None,
        author: None,
        tags: Vec::new(),
        operations: Vec::new(),
    };
    patch.save(&dir.path().join("p.vlp")).unwrap();
    let (loaded, name) = resolve_patch(&sources(dir.path()), &request("p.vlp")).unwrap();
    assert_eq!(loaded.base_model, "m");
    assert_eq!(name, "p.vlp");
}

#[test]
fn an_unparseable_patch_is_a_bad_request_without_the_parser_detail() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("bad.vlp"), "not json").unwrap();
    let Err(err) = resolve_patch(&sources(dir.path()), &request("bad.vlp")) else {
        panic!("an unparseable patch is refused");
    };
    let msg = err.to_string();
    assert!(msg.contains("could not be loaded"), "{msg}");
    assert!(
        !msg.contains("expected"),
        "parser detail must not reach the client: {msg}"
    );
}

#[test]
fn hf_references_are_refused_unless_allowed() {
    let Err(err) = resolve_patch(&PatchSources::default(), &request("hf://someone/patch")) else {
        panic!("hf:// is off by default");
    };
    assert!(err.to_string().contains("--allow-hf-patches"), "{err}");
}
