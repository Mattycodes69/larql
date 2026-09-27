//! `shard_loader::download_and_load_shard` — the Mode B receiver.
//!
//! Covers the verification contract (H9): a present content hash is always
//! checked, a missing one is refused unless `--allow-unverified-shards`
//! (`UnverifiedShards::Allow`), and a value that is not a SHA-256 at all is
//! refused either way. Also the path-safety of `model_id` and the atomic,
//! self-cleaning unpack.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use larql_server::shard_archive::SHARD_SHA256_HEX_LEN;
use larql_server::shard_loader::{
    download_and_load_shard, shard_dest_path, ShardFetch, ShardLoaded, UnverifiedShards,
    MAX_MODEL_ID_LEN,
};

const MODEL: &str = "gemma-test";
const MANIFEST: &[u8] = b"{\"hello\":\"world\"}";
const LAYER: [u8; 4] = [1, 2, 3, 4];
/// An origin nobody listens on: any refusal that must happen BEFORE the
/// network is proven by pointing here (a fetch would fail differently).
const DEAD_ORIGIN: &str = "http://127.0.0.1:9";

fn build_tar(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut tar = tar::Builder::new(&mut buf);
        for (name, content) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, *content).unwrap();
        }
        tar.finish().unwrap();
    }
    buf
}

fn good_tar() -> Vec<u8> {
    build_tar(&[("index.json", MANIFEST), ("layer-0.bin", &LAYER)])
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A donor serving `body` with `status` on `/v1/shard/...`, counting hits.
struct Donor {
    origin: String,
    hits: Arc<AtomicUsize>,
}

async fn spawn_donor(status: StatusCode, body: Vec<u8>) -> Donor {
    #[derive(Clone)]
    struct Served {
        status: StatusCode,
        body: Arc<Vec<u8>>,
        hits: Arc<AtomicUsize>,
    }
    async fn serve(State(s): State<Served>, Path((_m, _r)): Path<(String, String)>) -> Response {
        s.hits.fetch_add(1, Ordering::SeqCst);
        Response::builder()
            .status(s.status)
            .header(header::CONTENT_TYPE, "application/x-tar")
            .body(Body::from(s.body.as_ref().clone()))
            .unwrap()
    }
    let hits = Arc::new(AtomicUsize::new(0));
    let served = Served {
        status,
        body: Arc::new(body),
        hits: hits.clone(),
    };
    let app = Router::new()
        .route("/v1/shard/{model_id}/{range}", get(serve))
        .with_state(served);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Donor {
        origin: format!("http://{addr}"),
        hits,
    }
}

fn fetch<'a>(origin: &'a str, model_id: &'a str, expected: &'a str) -> ShardFetch<'a> {
    ShardFetch {
        origin_url: origin,
        model_id,
        layer_start: 0,
        layer_end: 5,
        expected_sha256: expected,
    }
}

fn store_str(tmp: &TempDir) -> String {
    tmp.path().to_str().unwrap().to_string()
}

/// No shard, no half-unpacked tmp dir, no leftover tar.
fn assert_nothing_landed(tmp: &TempDir) {
    let model_dir = tmp.path().join(MODEL);
    assert!(!model_dir.join("layers-0-5").exists(), "shard unpacked");
    assert!(!model_dir.join(".tmp-layers-0-5").exists(), "tmp dir left");
    assert!(
        !model_dir.join(".tmp-layers-0-5.tar").exists(),
        "download scratch left"
    );
}

#[tokio::test]
async fn verified_download_unpacks_atomically_and_is_idempotent() {
    let tar = good_tar();
    let digest = sha256_hex(&tar);
    let donor = spawn_donor(StatusCode::OK, tar).await;
    let tmp = TempDir::new().unwrap();
    let store = store_str(&tmp);

    let loaded = download_and_load_shard(
        fetch(&donor.origin, MODEL, &digest),
        &store,
        UnverifiedShards::Refuse,
    )
    .await
    .expect("matching content hash must load");
    assert_eq!(loaded, ShardLoaded::Verified(digest.clone()));
    assert_eq!(loaded.verified_sha256(), digest);

    let dest = shard_dest_path(&store, MODEL, 0, 5).expect("safe id");
    assert_eq!(std::fs::read(dest.join("index.json")).unwrap(), MANIFEST);
    assert_eq!(std::fs::read(dest.join("layer-0.bin")).unwrap(), LAYER);
    assert!(!tmp.path().join(MODEL).join(".tmp-layers-0-5").exists());
    assert!(!tmp.path().join(MODEL).join(".tmp-layers-0-5.tar").exists());

    // A second call finds the shard and does not re-download or re-verify.
    let again = download_and_load_shard(
        fetch(&donor.origin, MODEL, &digest),
        &store,
        UnverifiedShards::Refuse,
    )
    .await
    .expect("re-load must be idempotent");
    assert_eq!(again, ShardLoaded::AlreadyPresent);
    assert_eq!(again.verified_sha256(), "");
    assert_eq!(donor.hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn uppercase_digest_is_accepted() {
    let tar = good_tar();
    let digest = sha256_hex(&tar).to_ascii_uppercase();
    let donor = spawn_donor(StatusCode::OK, tar).await;
    let tmp = TempDir::new().unwrap();
    let loaded = download_and_load_shard(
        fetch(&donor.origin, MODEL, &digest),
        &store_str(&tmp),
        UnverifiedShards::Refuse,
    )
    .await
    .expect("hex case is not content");
    assert_eq!(loaded.verified_sha256(), digest.to_ascii_lowercase());
}

#[tokio::test]
async fn mismatched_content_hash_is_refused_and_cleaned_up() {
    let donor = spawn_donor(StatusCode::OK, good_tar()).await;
    let tmp = TempDir::new().unwrap();
    let wrong = "f".repeat(SHARD_SHA256_HEX_LEN);
    // Even the permissive policy checks a hash that is present.
    let err = download_and_load_shard(
        fetch(&donor.origin, MODEL, &wrong),
        &store_str(&tmp),
        UnverifiedShards::Allow,
    )
    .await
    .expect_err("mismatch must be refused");
    assert!(err.to_string().contains("hash mismatch"), "got: {err}");
    assert_nothing_landed(&tmp);
}

#[tokio::test]
async fn missing_content_hash_is_refused_without_the_flag() {
    let tmp = TempDir::new().unwrap();
    for placeholder in ["", "0000000000000000", &"0".repeat(SHARD_SHA256_HEX_LEN)] {
        let err = download_and_load_shard(
            fetch(DEAD_ORIGIN, MODEL, placeholder),
            &store_str(&tmp),
            UnverifiedShards::Refuse,
        )
        .await
        .expect_err("a placeholder hash must be refused");
        assert!(
            err.to_string().contains("--allow-unverified-shards"),
            "refusal must name the opt-in, got: {err}"
        );
    }
    assert_nothing_landed(&tmp);
}

#[tokio::test]
async fn missing_content_hash_is_accepted_with_the_flag() {
    let donor = spawn_donor(StatusCode::OK, good_tar()).await;
    let tmp = TempDir::new().unwrap();
    let loaded = download_and_load_shard(
        fetch(&donor.origin, MODEL, ""),
        &store_str(&tmp),
        UnverifiedShards::from_flag(true),
    )
    .await
    .expect("opted-in unverified load");
    assert_eq!(loaded, ShardLoaded::Unverified);
    assert_eq!(loaded.verified_sha256(), "", "must not claim a hash");
    let dest = shard_dest_path(&store_str(&tmp), MODEL, 0, 5).unwrap();
    assert_eq!(std::fs::read(dest.join("index.json")).unwrap(), MANIFEST);
}

/// The H9 bug shape: a 16-hex identity hash in the content-hash field. It is
/// not a placeholder, so the opt-in does not cover it.
#[tokio::test]
async fn identity_hash_in_content_field_is_always_refused() {
    let tmp = TempDir::new().unwrap();
    for policy in [UnverifiedShards::Refuse, UnverifiedShards::Allow] {
        let err = download_and_load_shard(
            fetch(DEAD_ORIGIN, MODEL, "0123456789abcdef"),
            &store_str(&tmp),
            policy,
        )
        .await
        .expect_err("an identity hash is not a content hash");
        assert!(err.to_string().contains("not a SHA-256"), "got: {err}");
    }
}

#[tokio::test]
async fn http_error_from_origin_is_refused() {
    let donor = spawn_donor(StatusCode::NOT_FOUND, Vec::new()).await;
    let tmp = TempDir::new().unwrap();
    let err = download_and_load_shard(
        fetch(&donor.origin, MODEL, &sha256_hex(b"")),
        &store_str(&tmp),
        UnverifiedShards::Refuse,
    )
    .await
    .expect_err("404 must fail");
    assert!(err.to_string().contains("HTTP 404"), "got: {err}");
    assert_nothing_landed(&tmp);
}

#[tokio::test]
async fn verified_but_corrupt_archive_fails_unpack_and_cleans_up() {
    // A truncated archive with a garbage tail; announcing its true hash
    // proves the failure is the unpack, not the digest.
    let mut body = good_tar();
    body.truncate(body.len() / 2);
    body.extend_from_slice(b"not a tar trailer");
    let digest = sha256_hex(&body);
    let donor = spawn_donor(StatusCode::OK, body).await;
    let tmp = TempDir::new().unwrap();
    let result = download_and_load_shard(
        fetch(&donor.origin, MODEL, &digest),
        &store_str(&tmp),
        UnverifiedShards::Refuse,
    )
    .await;
    assert!(result.is_err(), "truncated archive must not load");
    assert_nothing_landed(&tmp);
}

/// `model_id` is remote input from the router's AssignMsg. These are the
/// shapes that let a peer choose where bytes land on this disk.
#[tokio::test]
async fn traversing_model_ids_are_refused() {
    let too_long = "a".repeat(MAX_MODEL_ID_LEN + 1);
    let tmp = TempDir::new().unwrap();
    for hostile in [
        "../../../../etc/cron.d",
        "..",
        ".",
        "a/../../b",
        "foo/bar",
        "/absolute",
        "back\\slash",
        "trailing/",
        "nul\0byte",
        "",
        too_long.as_str(),
    ] {
        assert!(
            shard_dest_path("/store", hostile, 0, 1).is_none(),
            "built a destination path for hostile model_id {hostile:?}"
        );
        assert!(
            download_and_load_shard(
                fetch(DEAD_ORIGIN, hostile, ""),
                &store_str(&tmp),
                UnverifiedShards::Allow,
            )
            .await
            .is_err(),
            "downloaded for hostile model_id {hostile:?}"
        );
    }
}

/// The negative control: a validator that refused EVERYTHING would pass the
/// test above and silently break Mode B.
#[test]
fn real_model_ids_stay_inside_the_store() {
    let longest = "m".repeat(MAX_MODEL_ID_LEN);
    for good in [
        "gemma3-4b-q4k",
        "gpt-oss-20b.vindex3",
        "Muse_Glimmer-30B",
        "a",
        longest.as_str(),
    ] {
        let path = shard_dest_path("/store", good, 0, 1).expect("path for a real id");
        assert!(path.starts_with("/store"), "{good:?} escaped the store");
        assert!(path.ends_with(format!("{good}/layers-0-1")));
    }
}

#[test]
fn unverified_policy_defaults_to_refuse() {
    assert_eq!(UnverifiedShards::default(), UnverifiedShards::Refuse);
    assert_eq!(UnverifiedShards::from_flag(false), UnverifiedShards::Refuse);
    assert_eq!(UnverifiedShards::from_flag(true), UnverifiedShards::Allow);
}
