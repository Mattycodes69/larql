//! End-to-end Mode B smoke tests.
//!
//! Scenario: a router, donors covering a layer range, and a spare that joins
//! as available. When the router asks the spare to replicate a range that a
//! live donor still covers, it must:
//!   1. Resolve the surviving replica's `listen_url` as origin
//!   2. Send `AssignMsg` carrying the donor's CONTENT hash
//!      (`AnnounceMsg.shard_sha256`), never its identity hash (`vindex_hash`)
//!   3. Let the spare's `shard_loader` stream the tar from
//!      `GET /v1/shard/{model_id}/{start}-{end}`, verify its SHA-256 and
//!      unpack it locally
//!   4. Register the spare as serving after `ReadyMsg`
//!
//! The donor stub serves the same deterministic archive
//! (`shard_archive::write_shard_tar`) the real `/v1/shard` route does, and
//! announces its `shard_content_sha256` exactly as the real announce does.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use parking_lot::RwLock;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use larql_router::grid::service::GridServiceImpl;
use larql_router::grid::GridState;
use larql_router_protocol::{
    grid_service_server::GridServiceServer, AnnounceMsg, AssignMsg, AvailableMsg,
    GridServiceClient, ReadyMsg, RouterMessage, RouterPayload, ServerMessage, ServerPayload,
};
use larql_server::announce::{try_once_available, AvailableConfig};
use larql_server::shard_archive::{shard_content_sha256, write_shard_tar, SHARD_SHA256_HEX_LEN};
use larql_server::shard_loader::{
    download_and_load_shard, ShardFetch, ShardLoaded, UnverifiedShards,
};
use tonic::transport::Server;

const MODEL: &str = "test-model";
const DONOR_INDEX: &[u8] = b"{\"shard\":\"donor\"}";
const DONOR_LAYER: [u8; 3] = [0x11, 0x22, 0x33];
const SPARE_RAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DONOR_RAM_BYTES: u64 = 1024 * 1024 * 1024;
/// How long a Mode B handshake may take before the test calls it hung.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(3);
/// Settle time for a gRPC message to reach the router's state.
const SETTLE: Duration = Duration::from_millis(200);
/// Time a refusing spare is given to (wrongly) finish a download.
const REFUSAL_WINDOW: Duration = Duration::from_millis(500);
/// Time for a gRPC server or a production spare loop to come up.
const STARTUP: Duration = Duration::from_millis(300);

type SpareTask = tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>;
type RouterStream = tonic::Streaming<RouterMessage>;

/// A donor's on-disk vindex directory plus the content hash it announces.
struct DonorDir {
    dir: TempDir,
    sha256: String,
}

fn donor_dir() -> DonorDir {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("index.json"), DONOR_INDEX).unwrap();
    std::fs::create_dir(dir.path().join("layers")).unwrap();
    std::fs::write(dir.path().join("layers").join("layer-0.bin"), DONOR_LAYER).unwrap();
    let sha256 = shard_content_sha256(dir.path()).unwrap();
    DonorDir { dir, sha256 }
}

/// Serve `dir` over `/v1/shard` with the production archive encoding.
async fn spawn_shard_donor(dir: PathBuf) -> std::net::SocketAddr {
    async fn handler(
        State(dir): State<Arc<PathBuf>>,
        Path((_model, _range)): Path<(String, String)>,
    ) -> Response {
        let tar = write_shard_tar(&dir, Vec::new()).unwrap();
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-tar")
            .body(Body::from(tar))
            .unwrap()
    }
    let app = Router::new()
        .route("/v1/shard/{model_id}/{range}", get(handler))
        .with_state(Arc::new(dir));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

async fn spawn_router() -> (std::net::SocketAddr, Arc<RwLock<GridState>>) {
    let state = Arc::new(RwLock::new(GridState::default()));
    let svc = GridServiceImpl::new(state.clone());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stream = tokio_stream::wrappers::TcpListenerStream::new(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(GridServiceServer::new(svc))
            .serve_with_incoming(stream)
            .await
            .unwrap();
    });
    tokio::time::sleep(STARTUP).await;
    (addr, state)
}

/// Announce a donor. `vindex_hash` is its identity, `shard_sha256` its
/// content hash (empty = cannot vouch).
async fn announce_client(
    router_addr: std::net::SocketAddr,
    listen_url: String,
    layers: (u32, u32),
    vindex_hash: &str,
    shard_sha256: &str,
) -> (mpsc::Sender<ServerMessage>, RouterStream) {
    let mut client = GridServiceClient::connect(format!("http://{router_addr}"))
        .await
        .unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let inbound = client
        .join(ReceiverStream::new(rx))
        .await
        .unwrap()
        .into_inner();

    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: MODEL.into(),
            layer_start: layers.0,
            layer_end: layers.1,
            ram_bytes: DONOR_RAM_BYTES,
            listen_url,
            vindex_hash: vindex_hash.to_string(),
            shard_sha256: shard_sha256.to_string(),
            expert_start: 0,
            expert_end: 0,
            serves_openai: false,
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

/// Join as a manual (non-production) spare and advertise capacity.
async fn manual_spare(
    router_addr: std::net::SocketAddr,
    store_path: &str,
) -> (mpsc::Sender<ServerMessage>, RouterStream) {
    let mut client = GridServiceClient::connect(format!("http://{router_addr}"))
        .await
        .unwrap();
    let (tx, rx) = mpsc::channel::<ServerMessage>(32);
    let inbound = client
        .join(ReceiverStream::new(rx))
        .await
        .unwrap()
        .into_inner();
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Available(AvailableMsg {
            ram_bytes: SPARE_RAM_BYTES,
            disk_bytes: 0,
            store_path: store_path.into(),
        })),
    })
    .await
    .unwrap();
    (tx, inbound)
}

async fn next_assign(inbound: &mut RouterStream) -> AssignMsg {
    tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
        loop {
            match inbound.next().await {
                Some(Ok(rm)) => {
                    if let Some(RouterPayload::Assign(a)) = rm.payload {
                        return a;
                    }
                }
                Some(Err(e)) => panic!("spare stream error: {e}"),
                None => panic!("spare stream closed before AssignMsg"),
            }
        }
    })
    .await
    .expect("AssignMsg should arrive")
}

fn serving_urls(state: &RwLock<GridState>) -> Vec<String> {
    state
        .read()
        .status_response()
        .servers
        .iter()
        .map(|s| s.listen_url.clone())
        .collect()
}

/// The content hash the router holds for the server at `listen_url`.
fn router_content_hash(state: &RwLock<GridState>, listen_url: &str) -> String {
    state
        .read()
        .servers()
        .map(|(_, e)| e)
        .find(|e| e.listen_url == listen_url)
        .map(|e| e.shard_sha256.clone())
        .unwrap_or_else(|| panic!("{listen_url} must be registered as serving"))
}

fn shard_dir(store: &TempDir) -> PathBuf {
    store.path().join(MODEL).join("layers-0-4")
}

/// Everything a production-loop scenario keeps alive.
struct Scenario {
    store: TempDir,
    state: Arc<RwLock<GridState>>,
    spare: SpareTask,
    donor: DonorDir,
    _donor_stream: (mpsc::Sender<ServerMessage>, RouterStream),
}

/// Spawn a donor announcing `announced_sha256` (`None` = its real content
/// hash), a spare running the production Mode B loop under `policy`, and
/// assign the spare layers 0-4.
async fn assign_spare_from_donor(
    announced_sha256: Option<&str>,
    policy: UnverifiedShards,
    spare_url: &str,
) -> Scenario {
    let donor = donor_dir();
    let donor_http = spawn_shard_donor(donor.dir.path().to_path_buf()).await;
    let (router_addr, state) = spawn_router().await;
    let announced = announced_sha256.unwrap_or(&donor.sha256).to_string();
    let donor_stream = announce_client(
        router_addr,
        format!("http://{donor_http}"),
        (0, 4),
        "identity",
        &announced,
    )
    .await;
    tokio::time::sleep(SETTLE).await;

    let store = TempDir::new().unwrap();
    let cfg = AvailableConfig {
        join_url: format!("http://{router_addr}"),
        listen_url: spare_url.into(),
        ram_bytes: SPARE_RAM_BYTES,
        disk_bytes: 0,
        store_path: store.path().to_string_lossy().to_string(),
        grid_key: None,
        quic_cert_fingerprint: None,
        unverified_shards: policy,
    };
    let spare = tokio::spawn(async move { try_once_available(&cfg).await });
    tokio::time::sleep(STARTUP).await;
    assert!(
        state.write().try_assign_gap(MODEL, 0, 4, 0, 0, 0),
        "try_assign_gap should succeed: live origin exists + spare is available"
    );
    Scenario {
        store,
        state,
        spare,
        donor,
        _donor_stream: donor_stream,
    }
}

async fn expect_handshake_ok(spare: SpareTask) {
    tokio::time::timeout(HANDSHAKE_TIMEOUT, spare)
        .await
        .expect("try_once_available must complete")
        .expect("task must not panic")
        .expect("Mode B handshake should succeed");
}

#[tokio::test]
async fn mode_b_full_vertical_handoff() {
    let donor = donor_dir();
    let donor_http = spawn_shard_donor(donor.dir.path().to_path_buf()).await;
    let donor_listen_url = format!("http://{donor_http}");
    let (router_addr, state) = spawn_router().await;

    // Two replicas of layers 0-4 with DIFFERENT identity hashes but the same
    // bytes on disk — so the same content hash.
    let _donor_a = announce_client(
        router_addr,
        donor_listen_url.clone(),
        (0, 4),
        "hash-A",
        &donor.sha256,
    )
    .await;
    let _donor_b = announce_client(
        router_addr,
        donor_listen_url.clone(),
        (0, 4),
        "hash-B",
        &donor.sha256,
    )
    .await;
    tokio::time::sleep(SETTLE).await;
    assert_eq!(serving_urls(&state).len(), 2, "two donors must register");

    let store = TempDir::new().unwrap();
    let store_path = store.path().to_string_lossy().to_string();
    let (spare_tx, mut spare_inbound) = manual_spare(router_addr, &store_path).await;
    tokio::time::sleep(SETTLE).await;

    // Drive the assignment through the GridState API the rebalancer uses;
    // the wire path (Available → Assign → download → Ready) is the same.
    assert!(
        state.write().try_assign_gap(MODEL, 0, 4, 0, 0, 0),
        "try_assign_gap must succeed when a live replica exists as origin"
    );

    let assign = next_assign(&mut spare_inbound).await;
    assert_eq!(assign.model_id, MODEL);
    assert_eq!((assign.layer_start, assign.layer_end), (0, 4));
    assert_eq!(assign.origin_url, donor_listen_url);
    // H9: the CONTENT hash travels, never either donor's identity hash.
    assert_eq!(assign.shard_sha256, donor.sha256);

    let fetch = ShardFetch {
        origin_url: &assign.origin_url,
        model_id: &assign.model_id,
        layer_start: assign.layer_start,
        layer_end: assign.layer_end,
        expected_sha256: &assign.shard_sha256,
    };
    let loaded = download_and_load_shard(fetch, &store_path, UnverifiedShards::Refuse)
        .await
        .expect("a real donor's content hash must verify");
    assert_eq!(loaded, ShardLoaded::Verified(donor.sha256.clone()));

    let dest = shard_dir(&store);
    assert_eq!(std::fs::read(dest.join("index.json")).unwrap(), DONOR_INDEX);
    assert_eq!(
        std::fs::read(dest.join("layers").join("layer-0.bin")).unwrap(),
        DONOR_LAYER
    );

    spare_tx
        .send(ServerMessage {
            payload: Some(ServerPayload::Ready(ReadyMsg {
                model_id: assign.model_id.clone(),
                layer_start: assign.layer_start,
                layer_end: assign.layer_end,
                listen_url: "http://spare:9999".into(),
                expert_start: 0,
                expert_end: 0,
                shard_sha256: loaded.verified_sha256().to_string(),
            })),
        })
        .await
        .unwrap();
    tokio::time::sleep(SETTLE).await;
    assert_eq!(
        router_content_hash(&state, "http://spare:9999"),
        donor.sha256
    );
}

/// Drives the whole Mode B round-trip through `announce::try_once_available`
/// — the loop the daemon runs — against a donor announcing its real content
/// hash: Available → Assign → verified download → Ready → Ack.
#[tokio::test]
async fn mode_b_try_once_available_drives_full_handshake() {
    let spare_url = "http://spare-via-try-once:9999";
    let s = assign_spare_from_donor(None, UnverifiedShards::Refuse, spare_url).await;
    expect_handshake_ok(s.spare).await;

    assert_eq!(
        std::fs::read(shard_dir(&s.store).join("index.json")).unwrap(),
        DONOR_INDEX
    );
    // The router learned the spare's verified content hash from ReadyMsg,
    // so the spare is itself a vouched-for origin now.
    assert_eq!(router_content_hash(&s.state, spare_url), s.donor.sha256);
}

/// A donor that announced no content hash cannot seed a spare that did not
/// opt in: the spare refuses before downloading anything.
#[tokio::test]
async fn mode_b_spare_refuses_origin_without_content_hash() {
    let spare_url = "http://spare-refusing:9999";
    let s = assign_spare_from_donor(Some(""), UnverifiedShards::Refuse, spare_url).await;
    tokio::time::sleep(REFUSAL_WINDOW).await;

    assert!(
        !shard_dir(&s.store).exists(),
        "an unverified shard was unpacked without --allow-unverified-shards"
    );
    assert!(
        !serving_urls(&s.state).contains(&spare_url.to_string()),
        "a refusing spare must not register as serving"
    );
    assert!(!s.spare.is_finished(), "a refusal is not an Ack");
    s.spare.abort();
}

/// With `--allow-unverified-shards` the same assignment loads, and the
/// spare reports no verified content hash back to the router.
#[tokio::test]
async fn mode_b_spare_accepts_unhashed_origin_when_allowed() {
    let spare_url = "http://spare-permissive:9999";
    let s = assign_spare_from_donor(Some(""), UnverifiedShards::Allow, spare_url).await;
    expect_handshake_ok(s.spare).await;

    assert_eq!(
        std::fs::read(shard_dir(&s.store).join("index.json")).unwrap(),
        DONOR_INDEX
    );
    assert!(
        router_content_hash(&s.state, spare_url).is_empty(),
        "an unverified load must not claim a content hash"
    );
}

/// A donor whose announced content hash does not match its bytes is refused
/// even by a spare that allows unverified shards: a present hash is always
/// checked.
#[tokio::test]
async fn mode_b_spare_refuses_mismatched_content_hash() {
    let wrong = "f".repeat(SHARD_SHA256_HEX_LEN);
    let spare_url = "http://spare-mismatch:9999";
    let s = assign_spare_from_donor(Some(&wrong), UnverifiedShards::Allow, spare_url).await;
    assert_ne!(s.donor.sha256, wrong);
    tokio::time::sleep(REFUSAL_WINDOW).await;

    assert!(!shard_dir(&s.store).exists(), "mismatched shard unpacked");
    assert!(!serving_urls(&s.state).contains(&spare_url.to_string()));
    s.spare.abort();
}

#[tokio::test]
async fn no_assign_when_gap_has_no_surviving_origin() {
    let (router_addr, state) = spawn_router().await;

    // Single donor for layers 10-14 — no replicas.
    let (donor_tx, donor_inbound) = announce_client(
        router_addr,
        "http://donor:8080".into(),
        (10, 14),
        "hash-X",
        "",
    )
    .await;

    let store = TempDir::new().unwrap();
    let (_spare_tx, mut spare_inbound) =
        manual_spare(router_addr, &store.path().to_string_lossy()).await;
    tokio::time::sleep(SETTLE).await;

    // Kill the only donor — no live origin for layers 10-14.
    drop(donor_tx);
    drop(donor_inbound);
    tokio::time::sleep(SETTLE).await;

    // Even if the gap were detected, find_origin_for would return None:
    // the spare must not receive an AssignMsg.
    let result = tokio::time::timeout(REFUSAL_WINDOW, spare_inbound.next()).await;
    assert!(
        result.is_err(),
        "spare must not receive AssignMsg without a live origin: got {result:?}"
    );
    assert_eq!(state.read().status_response().servers.len(), 0);
}
