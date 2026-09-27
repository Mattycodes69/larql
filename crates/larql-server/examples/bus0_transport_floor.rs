//! RESIDUAL-BUS-0: where does a remote carrier round trip spend its time
//! when no model runs at all?
//!
//! PROFILE-1 and WIRE-2 measured the remote-FFN transport remainder
//! (round trip minus worker handler) at 85-232 us per call on loopback.
//! WIRE-2 showed that replacing HTTP with a persistent WebSocket removes
//! about 40% of it. This isolates the rest into layers, using fixed payload
//! byte counts and a no-op transform, one layer per arm:
//!
//! ```text
//! T0   persistent std TcpStream, blocking, dedicated server thread
//! T1   T0 + the server thread hands each frame to a compute thread and waits
//! T2   tokio accept/read + spawn_blocking hop (the server's scheduling shape)
//! T3   axum route + reqwest::blocking client (the current expert wire shape)
//! T3s  T3 dispatched from a per-call std::thread::scope spawn (the grid's shape)
//! ```
//!
//! Frozen protocol: `docs/residual-bus-0.md`. The arm differences are the
//! quantities; this example makes no claim about any model.
//!
//! Usage:
//!   cargo run --release -p larql-server --example bus0_transport_floor \
//!       -- <out.json> [calls] [warmup] [tokio_workers]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::serve::ListenerExt;

/// The expert wire's content type, so T3 checks the header the real route checks.
const CONTENT_TYPE: &str = larql_router_protocol::vindex3_experts::CONTENT_TYPE;
const ROUTE: &str = "/bus0";
const LOOPBACK: &str = "127.0.0.1:0";
/// Length prefix of the T0-T2 frames: one little-endian u32.
const PREFIX_BYTES: usize = 4;
/// Filler byte for payloads, so no page of either buffer is all zero.
const FILL: u8 = 0xA5;
const DEFAULT_CALLS: usize = 10_000;
const DEFAULT_WARMUP: usize = 1_000;
/// PROFILE-1's worker processes ran two Tokio workers.
const DEFAULT_TOKIO_WORKERS: usize = 2;
const CLIENT_TIMEOUT: Duration = Duration::from_secs(60);
const PERCENTILES: [f64; 4] = [0.50, 0.90, 0.99, 0.999];

/// Request/response body bytes per call, from the recorded wire formulas.
struct Shape {
    name: &'static str,
    request: usize,
    response: usize,
}

/// PROFILE-1 one-worker VEX1/VEY1 (hidden 2880, 4 experts); WIRE-2 VFF1/VFR1
/// for Qwen3 0.6B (hidden 1024) and Gemma 3 4B (hidden 2560).
const SHAPES: [Shape; 3] = [
    Shape {
        name: "gptoss-experts",
        request: 11_576,
        response: 46_136,
    },
    Shape {
        name: "qwen3-dense",
        request: 4_132,
        response: 4_132,
    },
    Shape {
        name: "gemma3-dense",
        request: 10_276,
        response: 10_276,
    },
];

#[derive(Clone, Copy, Debug)]
enum Arm {
    T0,
    T1,
    T2,
    T3,
    T3s,
}

const ARMS: [Arm; 5] = [Arm::T0, Arm::T1, Arm::T2, Arm::T3, Arm::T3s];

fn read_frame(stream: &mut TcpStream, buf: &mut Vec<u8>) -> std::io::Result<()> {
    let mut len = [0u8; PREFIX_BYTES];
    stream.read_exact(&mut len)?;
    buf.resize(u32::from_le_bytes(len) as usize, 0);
    stream.read_exact(buf)
}

fn write_frame(stream: &mut TcpStream, body: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(body.len() as u32).to_le_bytes())?;
    stream.write_all(body)
}

/// T0/T1 server: one dedicated blocking thread serving one connection.
/// With `handoff`, each frame crosses to a compute thread and back.
fn spawn_blocking_server(response: usize, handoff: bool) -> SocketAddr {
    let listener = TcpListener::bind(LOOPBACK).unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.set_nodelay(true).unwrap();
        // Every arm allocates its reply per call, as the real handler does.
        let compute = handoff.then(|| {
            let (to_compute, from_io) = mpsc::channel::<Vec<u8>>();
            let (to_io, from_compute) = mpsc::channel::<(Vec<u8>, Vec<u8>)>();
            std::thread::spawn(move || {
                for request in from_io {
                    to_io.send((request, vec![FILL; response])).unwrap();
                }
            });
            (to_compute, from_compute)
        });
        let mut buf = Vec::new();
        while read_frame(&mut stream, &mut buf).is_ok() {
            let sent = match &compute {
                Some((to_compute, from_compute)) => {
                    to_compute.send(std::mem::take(&mut buf)).unwrap();
                    let (request, out) = from_compute.recv().unwrap();
                    buf = request;
                    write_frame(&mut stream, &out)
                }
                None => write_frame(&mut stream, &vec![FILL; response]),
            };
            if sent.is_err() {
                break;
            }
        }
    });
    addr
}

fn tokio_runtime(workers: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .unwrap()
}

/// T2 server: async read, `spawn_blocking` no-op transform, async write.
fn spawn_tokio_server(rt: &tokio::runtime::Runtime, response: usize) -> SocketAddr {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = rt
        .block_on(tokio::net::TcpListener::bind(LOOPBACK))
        .unwrap();
    let addr = listener.local_addr().unwrap();
    rt.spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        stream.set_nodelay(true).unwrap();
        let mut len = [0u8; PREFIX_BYTES];
        let mut buf = Vec::new();
        while stream.read_exact(&mut len).await.is_ok() {
            buf.resize(u32::from_le_bytes(len) as usize, 0);
            if stream.read_exact(&mut buf).await.is_err() {
                break;
            }
            let request = std::mem::take(&mut buf);
            let (request, out) =
                tokio::task::spawn_blocking(move || (request, vec![FILL; response]))
                    .await
                    .unwrap();
            buf = request;
            let prefix = (out.len() as u32).to_le_bytes();
            if stream.write_all(&prefix).await.is_err() || stream.write_all(&out).await.is_err() {
                break;
            }
        }
    });
    addr
}

/// T3 server: the expert route's shape. It checks the content type, reads
/// the whole body as `Bytes`, runs a `spawn_blocking` no-op and answers with
/// the wire content type, on a listener that sets `TCP_NODELAY` per
/// accepted connection.
fn spawn_axum_server(rt: &tokio::runtime::Runtime, response: usize) -> SocketAddr {
    let handler = move |headers: HeaderMap, body: Bytes| async move {
        if headers
            .get(header::CONTENT_TYPE)
            .is_none_or(|h| h != CONTENT_TYPE)
        {
            return StatusCode::BAD_REQUEST.into_response();
        }
        let out = tokio::task::spawn_blocking(move || {
            drop(body);
            vec![FILL; response]
        })
        .await
        .unwrap();
        ([(header::CONTENT_TYPE, CONTENT_TYPE)], out).into_response()
    };
    let app = axum::Router::new().route(ROUTE, axum::routing::post(handler));
    let listener = rt
        .block_on(tokio::net::TcpListener::bind(LOOPBACK))
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let listener = listener.tap_io(|stream| stream.set_nodelay(true).unwrap());
    rt.spawn(async move { axum::serve(listener, app).await.unwrap() });
    addr
}

/// The `HttpExpertShards` client configuration, without authentication.
fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(CLIENT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .pool_idle_timeout(None)
        .pool_max_idle_per_host(1)
        .build()
        .unwrap()
}

fn http_call(client: &reqwest::blocking::Client, url: &str, body: &[u8], response: usize) {
    let reply = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, CONTENT_TYPE)
        .body(body.to_vec())
        .send()
        .and_then(|r| r.error_for_status())
        .unwrap();
    assert_eq!(reply.bytes().unwrap().len(), response);
}

/// Times `warmup + calls` round trips of one arm on one shape. Returns the
/// measured per-call nanoseconds. The payload copy into the request body
/// (T3) sits inside the timed region, as it does in the real client.
fn run_arm(arm: Arm, shape: &Shape, calls: usize, warmup: usize, workers: usize) -> Vec<u64> {
    let request = vec![FILL; shape.request];
    let mut samples = Vec::with_capacity(calls);
    let mut record = |i: usize, started: Instant| {
        if i >= warmup {
            samples.push(started.elapsed().as_nanos() as u64);
        }
    };
    match arm {
        Arm::T0 | Arm::T1 | Arm::T2 => {
            let rt = matches!(arm, Arm::T2).then(|| tokio_runtime(workers));
            let addr = match &rt {
                Some(rt) => spawn_tokio_server(rt, shape.response),
                None => spawn_blocking_server(shape.response, matches!(arm, Arm::T1)),
            };
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.set_nodelay(true).unwrap();
            let mut buf = Vec::new();
            for i in 0..warmup + calls {
                let started = Instant::now();
                write_frame(&mut stream, &request).unwrap();
                read_frame(&mut stream, &mut buf).unwrap();
                record(i, started);
                assert_eq!(buf.len(), shape.response);
            }
        }
        Arm::T3 | Arm::T3s => {
            let rt = tokio_runtime(workers);
            let addr = spawn_axum_server(&rt, shape.response);
            let url = format!("http://{addr}{ROUTE}");
            let client = http_client();
            for i in 0..warmup + calls {
                let started = Instant::now();
                if matches!(arm, Arm::T3s) {
                    std::thread::scope(|scope| {
                        scope.spawn(|| http_call(&client, &url, &request, shape.response));
                    });
                } else {
                    http_call(&client, &url, &request, shape.response);
                }
                record(i, started);
            }
        }
    }
    samples
}

fn summary(mut ns: Vec<u64>) -> serde_json::Value {
    ns.sort_unstable();
    let at = |q: f64| ns[((ns.len() - 1) as f64 * q).round() as usize] as f64 / 1e3;
    let mean = ns.iter().sum::<u64>() as f64 / ns.len() as f64 / 1e3;
    let mut out = serde_json::json!({ "calls": ns.len(), "mean_us": mean });
    for q in PERCENTILES {
        out[format!("p{}_us", q * 100.0)] = at(q).into();
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args
        .next()
        .expect("usage: <out.json> [calls] [warmup] [tokio_workers]");
    let calls: usize = args.next().map_or(DEFAULT_CALLS, |a| a.parse().unwrap());
    let warmup: usize = args.next().map_or(DEFAULT_WARMUP, |a| a.parse().unwrap());
    let workers: usize = args
        .next()
        .map_or(DEFAULT_TOKIO_WORKERS, |a| a.parse().unwrap());

    // Two passes, the second in reverse arm order, so slow drift shows up
    // as disagreement between an arm's passes rather than as an arm effect.
    let mut rows = Vec::new();
    for pass in 0..2 {
        let order: Vec<Arm> = if pass == 0 {
            ARMS.to_vec()
        } else {
            ARMS.iter().rev().copied().collect()
        };
        for shape in &SHAPES {
            for &arm in &order {
                let s = summary(run_arm(arm, shape, calls, warmup, workers));
                println!(
                    "pass {pass} {:<15} {:<4} p50 {:>8.1} us  p99 {:>8.1} us  mean {:>8.1} us",
                    shape.name,
                    format!("{arm:?}"),
                    s["p50_us"].as_f64().unwrap(),
                    s["p99_us"].as_f64().unwrap(),
                    s["mean_us"].as_f64().unwrap(),
                );
                rows.push(serde_json::json!({
                    "pass": pass,
                    "shape": shape.name,
                    "request_bytes": shape.request,
                    "response_bytes": shape.response,
                    "arm": format!("{arm:?}"),
                    "summary": s,
                }));
            }
        }
    }
    let record = serde_json::json!({
        "protocol": "docs/residual-bus-0.md",
        "calls": calls,
        "warmup": warmup,
        "tokio_workers": workers,
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "started_unix_s": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        "rows": rows,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&record).unwrap()).unwrap();
    println!("wrote {out}");
}
