//! The JSON wire, bearer auth and every wire's profiled path: records reach
//! the provider-call observer with worker timing when the worker sends it,
//! and a failure is recorded as incomplete rather than dropped.

use super::*;
use crate::profile_testing::Capture;
use axum::http::{HeaderMap, HeaderName, StatusCode};

const TOKEN: &str = "grid-secret";
const HIDDEN: usize = 4;
/// Worker-reported handler time the stubs send back.
const WORKER_HANDLER_NS: u64 = 1;

/// JSON stub modes.
const JSON_TIMED: usize = 0;
const JSON_UNTIMED: usize = 1;
const JSON_ERROR: usize = 2;
const JSON_REBOUND: usize = 3;

/// Stream stub modes.
const STREAM_OK: usize = 0;
const STREAM_WRONG_SEQUENCE: usize = 1;
const STREAM_NO_TIMING: usize = 2;
const STREAM_OVERSIZE_TIMING: usize = 3;

fn timing_json() -> String {
    serde_json::to_string(&WorkerTiming {
        handler_ns: WORKER_HANDLER_NS,
        ..Default::default()
    })
    .unwrap()
}

fn profile_header() -> HeaderName {
    HeaderName::from_static(PROFILE_HEADER)
}

async fn authorized_binding(headers: HeaderMap) -> axum::response::Response {
    let expected = format!("Bearer {TOKEN}");
    if headers
        .get(axum::http::header::AUTHORIZATION)
        .is_none_or(|v| v != expected.as_str())
    {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Json(binding()).into_response()
}

async fn json_forward(
    State(mode): State<Arc<AtomicUsize>>,
    Json(request): Json<Request>,
) -> axum::response::Response {
    let mut response = Response {
        binding: request.binding,
        layer: request.layer,
        row: request.row,
    };
    match mode.load(Ordering::SeqCst) {
        JSON_TIMED => ([(profile_header(), timing_json())], Json(response)).into_response(),
        JSON_ERROR => StatusCode::SERVICE_UNAVAILABLE.into_response(),
        JSON_REBOUND => {
            response.binding.program.artifact = "b".repeat(64);
            Json(response).into_response()
        }
        _ => Json(response).into_response(),
    }
}

async fn timed_binary_forward(bytes: Bytes) -> axum::response::Response {
    let frame = binary::decode(&bytes, Direction::Request, HIDDEN).unwrap();
    let response = binary::encode(
        Direction::Response,
        frame.handle,
        frame.sequence,
        frame.layer as usize,
        &frame.row,
    )
    .unwrap();
    (
        [
            (
                axum::http::header::CONTENT_TYPE,
                binary::CONTENT_TYPE.to_string(),
            ),
            (profile_header(), timing_json()),
        ],
        response,
    )
        .into_response()
}

async fn timed_stream_upgrade(
    State(mode): State<Arc<AtomicUsize>>,
    ws: axum::extract::ws::WebSocketUpgrade,
) -> axum::response::Response {
    ws.protocols([binary::STREAM_PROTOCOL])
        .on_upgrade(move |mut socket| async move {
            use axum::extract::ws::Message;
            let Some(Ok(Message::Text(options))) = socket.recv().await else {
                return;
            };
            let options: binary::StreamOptions = serde_json::from_str(&options).unwrap();
            while let Some(Ok(Message::Binary(bytes))) = socket.recv().await {
                let frame = binary::decode(&bytes, Direction::Request, HIDDEN).unwrap();
                let response = binary::encode(
                    Direction::Response,
                    frame.handle,
                    frame.sequence,
                    frame.layer as usize,
                    &frame.row,
                )
                .unwrap();
                if options.profile {
                    let mode = mode.load(Ordering::SeqCst);
                    let sequence = if mode == STREAM_WRONG_SEQUENCE {
                        frame.sequence + 1
                    } else {
                        frame.sequence
                    };
                    let timing = serde_json::to_string(&binary::StreamTiming {
                        sequence,
                        timing: WorkerTiming {
                            handler_ns: WORKER_HANDLER_NS,
                            ..Default::default()
                        },
                    })
                    .unwrap();
                    let control = match mode {
                        STREAM_NO_TIMING => Message::Binary(response.clone().into()),
                        STREAM_OVERSIZE_TIMING => {
                            Message::Text(" ".repeat(binary::CONTROL_LIMIT + 1).into())
                        }
                        _ => Message::Text(timing.into()),
                    };
                    if socket.send(control).await.is_err() {
                        return;
                    }
                }
                if socket.send(Message::Binary(response.into())).await.is_err() {
                    return;
                }
            }
        })
}

/// Accepts the upgrade without echoing the subprotocol.
async fn bare_upgrade(ws: axum::extract::ws::WebSocketUpgrade) -> axum::response::Response {
    ws.on_upgrade(|_socket| async {})
}

async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, server)
}

#[tokio::test]
async fn json_wire_authenticates_binding_and_records_profiled_calls() {
    let mode = Arc::new(AtomicUsize::new(JSON_UNTIMED));
    let app = Router::new()
        .route(PATH, get(authorized_binding).post(json_forward))
        .with_state(mode.clone());
    let (url, server) = serve(app).await;
    tokio::task::spawn_blocking(move || {
        let urls = vec![url];
        assert!(HttpFfnShards::connect(&urls, None).is_err(), "no bearer");
        assert!(HttpFfnShards::connect(&urls, Some("bad\ntoken")).is_err());
        let client = HttpFfnShards::connect(&urls, Some(TOKEN)).unwrap();
        assert_eq!(client.bindings(), vec![binding()]);
        let row = [0.5f32; HIDDEN];

        assert_eq!(client.forward_bound(0, 0, &row, &binding()).unwrap(), row);
        assert!(client.forward(1, 0, &row).is_err(), "unknown shard");
        mode.store(JSON_REBOUND, Ordering::SeqCst);
        let err = client.forward_bound(0, 0, &row, &binding()).unwrap_err();
        assert!(err.contains("changed binding or layer"), "got: {err}");

        let capture = Capture::start();
        mode.store(JSON_TIMED, Ordering::SeqCst);
        assert_eq!(client.forward(0, 0, &row).unwrap().row, row);
        mode.store(JSON_UNTIMED, Ordering::SeqCst);
        client.forward(0, 0, &row).unwrap();
        mode.store(JSON_ERROR, Ordering::SeqCst);
        let err = client.forward(0, 0, &row).unwrap_err();
        assert!(err.contains("dense FFN HTTP"), "got: {err}");
        let trace = capture.finish_provider_calls();

        assert_eq!(trace.len(), 3);
        assert_eq!(trace[0]["complete"], true);
        assert_eq!(trace[0]["worker"]["handler_ns"], WORKER_HANDLER_NS);
        assert!(trace[1].get("worker").is_none(), "never invent worker time");
        assert_eq!(trace[2]["complete"], false);
        assert_eq!(
            trace[2]["http_status"],
            StatusCode::SERVICE_UNAVAILABLE.as_u16()
        );
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn binary_wire_records_worker_timing_when_profiled() {
    let mode = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(PATH, get(|| async { Json(binding()) }))
        .route(binary::OPEN_PATH, post(open))
        .route(binary::PATH, post(timed_binary_forward))
        .with_state(mode);
    let (url, server) = serve(app).await;
    tokio::task::spawn_blocking(move || {
        let client = HttpFfnShards::connect_binary(&[url], None).unwrap();
        let row = [0.25f32; HIDDEN];
        // The trait's `forward` on a binary client answers through the frame.
        let response = client.forward(0, 0, &row).unwrap();
        assert_eq!((response.layer, response.row), (0, row.to_vec()));

        let capture = Capture::start();
        client.forward_bound(0, 0, &row, &binding()).unwrap();
        let err = client
            .forward_bound(0, 0, &row[..HIDDEN - 1], &binding())
            .unwrap_err();
        assert!(err.contains("width mismatch"), "got: {err}");
        let trace = capture.finish_provider_calls();
        assert_eq!(trace.len(), 1, "a refused width never reaches the wire");
        assert_eq!(trace[0]["wire"], "binary-f32-v1");
        assert_eq!(trace[0]["complete"], true);
        assert_eq!(trace[0]["worker"]["handler_ns"], WORKER_HANDLER_NS);
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn stream_wire_profiles_and_refuses_bad_timing_frames() {
    let mode = Arc::new(AtomicUsize::new(STREAM_OK));
    let app = Router::new()
        .route(PATH, get(authorized_binding))
        .route(binary::OPEN_PATH, post(open))
        .route(binary::STREAM_PATH, get(timed_stream_upgrade))
        .with_state(mode.clone());
    let (url, server) = serve(app).await;
    tokio::task::spawn_blocking(move || {
        let urls = vec![url];
        let row = [1.5f32; HIDDEN];

        let client = HttpFfnShards::connect_stream(&urls, Some(TOKEN)).unwrap();
        let capture = Capture::start();
        assert_eq!(client.forward_bound(0, 0, &row, &binding()).unwrap(), row);
        assert_eq!(client.forward_bound(0, 0, &row, &binding()).unwrap(), row);
        let trace = capture.finish_provider_calls();
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0]["wire"], "stream-f32-v1");
        assert!(
            trace[0].get("stream_setup_bytes").is_some(),
            "first call negotiates"
        );
        assert!(trace[1].get("stream_setup_bytes").is_none());
        assert_eq!(trace[1]["worker"]["handler_ns"], WORKER_HANDLER_NS);
        // The profiling mode was fixed by the first call on this stream.
        let err = client.forward_bound(0, 0, &row, &binding()).unwrap_err();
        assert!(err.contains("cannot change"), "got: {err}");

        // At this width the socket's own message cap (the larger of one frame
        // and `CONTROL_LIMIT`) refuses an oversize timing before the client's
        // explicit check can.
        for (fault, expect) in [
            (STREAM_WRONG_SEQUENCE, "sequence mismatch"),
            (STREAM_NO_TIMING, "timing missing"),
            (STREAM_OVERSIZE_TIMING, "too long"),
        ] {
            mode.store(STREAM_OK, Ordering::SeqCst);
            let client = HttpFfnShards::connect_stream(&urls, Some(TOKEN)).unwrap();
            mode.store(fault, Ordering::SeqCst);
            let capture = Capture::start();
            let err = client.forward_bound(0, 0, &row, &binding()).unwrap_err();
            assert!(err.contains(expect), "fault {fault}: {err}");
            assert_eq!(capture.finish_provider_calls()[0]["complete"], false);
        }
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn stream_refuses_a_server_that_drops_the_subprotocol() {
    let mode = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(PATH, get(|| async { Json(binding()) }))
        .route(binary::OPEN_PATH, post(open))
        .route(binary::STREAM_PATH, get(bare_upgrade))
        .with_state(mode);
    let (url, server) = serve(app).await;
    // `localhost` can resolve to an address the stub is not bound on (::1)
    // before 127.0.0.1; the stream dialler must fall through to the next.
    let url = url.replace("127.0.0.1", "localhost");
    tokio::task::spawn_blocking(move || {
        let err = HttpFfnShards::connect_stream(&[url], None)
            .err()
            .expect("subprotocol mismatch must refuse");
        assert!(err.contains("subprotocol"), "got: {err}");
    })
    .await
    .unwrap();
    server.abort();
}
