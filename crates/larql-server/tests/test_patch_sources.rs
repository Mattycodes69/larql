//! `POST /v1/patches {"url": …}` loads only from where the operator said.

use larql_server::routes::patches::PatchSources;

fn scratch(name: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("larql-patch-sources-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn by_default_no_path_or_hub_reference_is_loadable() {
    let sources = PatchSources::default();
    assert!(sources.resolve("/etc/passwd").is_err());
    assert!(sources.resolve("relative.vlp").is_err());
    assert!(sources.resolve("hf://someone/repo").is_err());
}

#[test]
fn a_path_inside_the_patch_dir_resolves() {
    let dir = scratch("inside");
    std::fs::write(dir.join("p.vlp"), "{}").expect("write patch");
    let sources = PatchSources {
        dir: Some(dir.clone()),
        allow_hf: false,
    };
    let got = sources.resolve("p.vlp").expect("inside the patch dir");
    assert_eq!(got, dir.canonicalize().unwrap().join("p.vlp"));
}

#[test]
fn escaping_the_patch_dir_is_refused_like_a_missing_file() {
    let root = scratch("escape");
    let dir = root.join("patches");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(root.join("secret.vlp"), "{}").unwrap();
    let sources = PatchSources {
        dir: Some(dir),
        allow_hf: false,
    };
    let escape = sources.resolve("../secret.vlp").unwrap_err().to_string();
    let absolute = sources
        .resolve(root.join("secret.vlp").to_str().unwrap())
        .unwrap_err()
        .to_string();
    let missing = sources.resolve("nope.vlp").unwrap_err().to_string();
    // The same shape of message for all three: nothing to probe with.
    for msg in [&escape, &absolute, &missing] {
        assert!(msg.contains("is not available"), "{msg}");
    }
}
