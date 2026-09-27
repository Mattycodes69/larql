//! `HttpLayerShards` — the blocking V3 layer-prefix transport: bearer auth,
//! binding discovery, exact row forwarding and error surfacing.

use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};

use larql_router::vindex3::HttpLayerShards;
use larql_router_protocol::vindex3::{Binding, Request, Response, PATH, SCHEMA};
use larql_router_protocol::vindex3_transport::ShardTransport;

const TOKEN: &str = "grid-secret";
const HIDDEN: usize = 4;
/// Row value the stub refuses with a 500, to exercise error surfacing.
const POISON: f32 = -1.0;

fn binding() -> Binding {
    Binding {
        schema: SCHEMA,
        artifact: "a".repeat(64),
        backend: "cpu".into(),
        lowering: "cpu-production/v1".into(),
        start: 0,
        end: 1,
        layers: 2,
        hidden: HIDDEN,
    }
}

fn authorized(headers: &HeaderMap) -> bool {
    let expected = format!("Bearer {TOKEN}");
    headers
        .get(axum::http::header::AUTHORIZATION)
        .is_some_and(|v| v == expected.as_str())
}

async fn get_binding(headers: HeaderMap) -> axum::response::Response {
    if !authorized(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Json(binding()).into_response()
}

async fn forward(headers: HeaderMap, Json(request): Json<Request>) -> axum::response::Response {
    if !authorized(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if request.rows.iter().flatten().any(|&x| x == POISON) {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }
    Json(Response {
        binding: request.binding,
        rows: request
            .rows
            .into_iter()
            .map(|r| r.into_iter().map(|x| x * 2.0).collect())
            .collect(),
    })
    .into_response()
}

#[tokio::test]
async fn layer_shards_forward_rows_under_bearer_auth() {
    let app = Router::new().route(PATH, get(get_binding).post(forward));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    // A trailing slash on the root must not double up in the route.
    let url = format!("http://{}/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    tokio::task::spawn_blocking(move || {
        let urls = vec![url];
        assert!(HttpLayerShards::connect(&urls, None).is_err(), "no bearer");
        assert!(HttpLayerShards::connect(&urls, Some("bad\ntoken")).is_err());

        let shards = HttpLayerShards::connect(&urls, Some(TOKEN)).unwrap();
        assert_eq!(shards.bindings(), vec![binding()]);

        let response = shards.forward(0, vec![vec![1.5; HIDDEN]]).unwrap();
        assert_eq!(response.binding, binding());
        assert_eq!(response.rows, vec![vec![3.0; HIDDEN]]);

        let err = shards.forward(1, vec![vec![1.0; HIDDEN]]).unwrap_err();
        assert!(err.contains("unknown V3 shard"), "got: {err}");
        assert!(shards.forward(0, vec![vec![POISON; HIDDEN]]).is_err());
    })
    .await
    .unwrap();
    server.abort();
}
