//! V3 dense-FFN workers over HTTP preserve local continuation.

use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v3_dense_ffn_workers_over_http_preserve_local_continuation() {
    use larql_inference::vindex3::{
        dense_ffn::{prepare_coordinator, DenseFfnSession},
        LogitsSession,
    };
    use larql_router::vindex3_ffn::HttpFfnShards;
    use larql_router_protocol::vindex3_ffn::{Binding, PATH};
    use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
    let container = v3_container();
    let mut urls = Vec::new();
    let mut servers = Vec::new();
    assert!(load_artifact(
        container.path().to_str().unwrap(),
        LoadVindexOptions {
            ffn_only: true,
            ..Default::default()
        }
    )
    .is_err());
    for layer in 0..2 {
        let LoadedArtifact::V3(model) = load_artifact(
            container.path().to_str().unwrap(),
            LoadVindexOptions {
                ffn_only: true,
                layer_range: Some(
                    larql_server::bootstrap::parse_layer_range(&format!("{layer}-{layer}"))
                        .unwrap(),
                ),
                ..Default::default()
            },
        )
        .unwrap() else {
            panic!("V3")
        };
        let ops = model.runtime.operands();
        assert_eq!(
            ops.slice(),
            &ExecutionSlice::DenseFfns {
                start: layer,
                end: layer + 1
            }
        );
        assert!(!ops.has_output());
        let census = ops.residency_census();
        assert_eq!(census.attention.total(), 0);
        assert_eq!(census.embedding.total(), 0);
        assert_eq!(census.glue.total(), 0);
        assert!(census.ffn.total() > 0);
        let state = v3_state(container.path());
        state.model_set.write().unwrap().v3_models = vec![Arc::new(*model)];
        let app = larql_server::routes::single_model_router(state);
        let denied = common::post_json(
            app.clone(),
            "/v1/completions",
            serde_json::json!({"prompt":PROMPT,"max_tokens":1}),
        )
        .await;
        assert!(!denied.status().is_success());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        urls.push(format!("http://{}", listener.local_addr().unwrap()));
        servers.push(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap()
        }));
    }
    let client = reqwest::Client::new();
    let binding: Binding = client
        .get(format!("{}{PATH}", urls[0]))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let good =
        serde_json::json!({"binding":binding,"layer":0,"row":vec![0.3;binding.program.hidden]});
    let a: serde_json::Value = client
        .post(format!("{}{PATH}", urls[0]))
        .json(&good)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let b: serde_json::Value = client
        .post(format!("{}{PATH}", urls[0]))
        .json(&good)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(a, b);
    let mut wrong = binding.clone();
    wrong.program.artifact = "0".repeat(64);
    for body in [
        serde_json::json!({"binding":binding,"layer":1,"row":vec![0.3;binding.program.hidden]}),
        serde_json::json!({"binding":binding,"layer":0,"row":[0.3]}),
        serde_json::json!({"binding":wrong,"layer":0,"row":vec![0.3;binding.program.hidden]}),
    ] {
        assert_eq!(
            client
                .post(format!("{}{PATH}", urls[0]))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    use larql_router_protocol::vindex3_ffn::binary::{self, Direction};
    let opened: binary::Opened = client
        .post(format!("{}{}", urls[0], binary::OPEN_PATH))
        .json(&binding)
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(opened.binding, binding);
    let mut mismatched = binding.clone();
    mismatched.program.artifact = "0".repeat(64);
    assert_eq!(
        client
            .post(format!("{}{}", urls[0], binary::OPEN_PATH))
            .json(&mismatched)
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    let packet = binary::encode(
        Direction::Request,
        opened.handle,
        17,
        0,
        &vec![0.3; binding.program.hidden],
    )
    .unwrap();
    for _ in 0..2 {
        let response = client
            .post(format!("{}{}", urls[0], binary::PATH))
            .header(reqwest::header::CONTENT_TYPE, binary::CONTENT_TYPE)
            .body(packet.clone())
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let decoded =
            binary::decode(&response, Direction::Response, binding.program.hidden).unwrap();
        assert_eq!(decoded.sequence, 17);
        let expected: larql_router_protocol::vindex3_ffn::Response =
            serde_json::from_value(a.clone()).unwrap();
        assert_eq!(
            decoded.row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.row.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    // A fresh preparation of the very same artifact invalidates old handles.
    let LoadedArtifact::V3(reopened) = load_artifact(
        container.path().to_str().unwrap(),
        LoadVindexOptions {
            ffn_only: true,
            layer_range: Some((0, 1)),
            ..Default::default()
        },
    )
    .unwrap() else {
        panic!("V3")
    };
    assert_ne!(reopened.ffn_wire.as_ref().unwrap().handle, opened.handle);
    for mode in 0..6 {
        let mut bad = packet.clone();
        match mode {
            0 => bad[4..20].copy_from_slice(&reopened.ffn_wire.as_ref().unwrap().handle),
            1 => bad[28..32].copy_from_slice(&1u32.to_le_bytes()),
            2 => {
                bad.pop();
            }
            3 => bad.push(0),
            4 => bad[32..36].copy_from_slice(&0u32.to_le_bytes()),
            _ => bad[binary::HEADER_BYTES..binary::HEADER_BYTES + 4]
                .copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
        }
        assert_eq!(
            client
                .post(format!("{}{}", urls[0], binary::PATH))
                .header(reqwest::header::CONTENT_TYPE, binary::CONTENT_TYPE)
                .body(bad)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let path = container.path().to_path_buf();
    tokio::task::spawn_blocking(move || {
        use tungstenite::{client::IntoClientRequest, Message};
        let opened: binary::Opened = reqwest::blocking::Client::new().post(format!("{}{}", urls[0], binary::OPEN_PATH)).json(&binding).send().unwrap().json().unwrap();
        for fault in 0..7 {
            let address = urls[0].strip_prefix("http://").unwrap();
            let tcp = std::net::TcpStream::connect(address).unwrap();
            tcp.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
            let mut request = format!("ws://{address}{}", binary::STREAM_PATH).into_client_request().unwrap();
            request.headers_mut().insert("Sec-WebSocket-Protocol", binary::STREAM_PROTOCOL.parse().unwrap());
            let (mut socket, _) = tungstenite::client(request, tcp).unwrap();
            socket.send(Message::Text(r#"{"profile":false}"#.into())).unwrap();
            let mut frame = binary::encode(binary::Direction::Request, opened.handle, 1, 0, &vec![0.3;binding.program.hidden]).unwrap();
            // A valid operation first proves the stream is admitted. Duplicate
            // sequence, unlike stateless HTTP replay, is then a protocol error.
            socket.send(Message::Binary(frame.clone().into())).unwrap();
            assert!(matches!(socket.read().unwrap(), Message::Binary(_)));
            frame[20..28].copy_from_slice(&2u64.to_le_bytes());
            match fault {
                0 => frame[4] ^= 1,
                1 => frame[28..32].copy_from_slice(&1u32.to_le_bytes()),
                2 => frame[20..28].copy_from_slice(&1u64.to_le_bytes()),
                3 => frame[36..40].copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
                4 => { frame.pop(); },
                5 => frame.push(0),
                6 => frame[32..36].copy_from_slice(&0u32.to_le_bytes()),
                _ => unreachable!(),
            }
            socket.send(Message::Binary(frame.into())).unwrap();
            assert!(!matches!(socket.read(), Ok(Message::Binary(_))), "stream accepted fault {fault}");
        }
        let runtime = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new()).unwrap();
        let stream_transport = HttpFfnShards::connect_stream(&urls, None).unwrap();
        let stream_ops = prepare_coordinator(&path, runtime.plan(), runtime.operands(), runtime.backend(), stream_transport).unwrap();
        let mut stream_session = DenseFfnSession::new(runtime.plan(), &stream_ops, runtime.backend(), &row_continuation(runtime.plan())).unwrap();
        let binary_transport = HttpFfnShards::connect_binary(&urls, None).unwrap();
        let binary_ops = prepare_coordinator(&path, runtime.plan(), runtime.operands(), runtime.backend(), binary_transport).unwrap();
        let mut binary_session = DenseFfnSession::new(runtime.plan(), &binary_ops, runtime.backend(), &row_continuation(runtime.plan())).unwrap();
        let transport = HttpFfnShards::connect(&urls, None).unwrap();
        let ops = prepare_coordinator(
            &path,
            runtime.plan(),
            runtime.operands(),
            runtime.backend(),
            transport,
        )
        .unwrap();
        let mut remote = DenseFfnSession::new(runtime.plan(), &ops, runtime.backend(), &row_continuation(runtime.plan())).unwrap();
        let mut local = runtime.session(&row_continuation(runtime.plan())).unwrap();
        let mut smoke = Vec::new();
        for (position, id) in [3, 17, 28, 0, 11, 3, 17, 28, 0, 11].into_iter().enumerate() {
            use larql_inference::vindex3::dense_ffn::profile::Capture;
            let capture = Capture::start().unwrap();
            let expected = local.step(id).unwrap();
            let local_rows = capture.finish();
            assert_eq!(local_rows.len(), 1);
            assert!(local_rows[0].provider_calls.is_empty());
            let capture = Capture::start().unwrap();
            let actual = remote.step(id).unwrap();
            let rows = capture.finish();
            assert_eq!(rows.len(), 1);
            let row = &rows[0];
            assert!(row.complete);
            assert_eq!(row.position, position);
            assert_eq!(row.total_ns, row.attention_ns + row.ffn_ns + row.reentry_ns + row.other_ns);
            assert_eq!(row.provider_calls.len(), 2);
            for (layer, call) in row.provider_calls.iter().enumerate() {
                assert_eq!(call["layer"], layer);
                assert_eq!(call["complete"], true);
                assert!(call["request_bytes"].as_u64().unwrap() > 0);
                assert!(call["response_bytes"].as_u64().unwrap() > 0);
                let worker = &call["worker"];
                assert!(worker["ffn_ns"].as_u64().unwrap() <= worker["execute_ns"].as_u64().unwrap());
                assert!(worker["execute_ns"].as_u64().unwrap() <= worker["handler_ns"].as_u64().unwrap());
            }
            let capture = Capture::start().unwrap();
            let binary_logits = binary_session.step(id).unwrap();
            let binary_rows = capture.finish();
            assert_eq!(binary_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            for call in &binary_rows[0].provider_calls {
                assert_eq!(call["request_bytes"], binary::HEADER_BYTES + 4 * ops.hidden());
                assert_eq!(call["response_bytes"], call["request_bytes"]);
            }
            let capture = Capture::start().unwrap();
            let stream_logits = stream_session.step(id).unwrap();
            let stream_rows = capture.finish();
            assert_eq!(stream_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            for call in &stream_rows[0].provider_calls {
                assert_eq!(call["request_bytes"], binary::HEADER_BYTES + 4 * ops.hidden());
                assert_eq!(call["response_bytes"], call["request_bytes"]);
                assert!(call["telemetry_bytes"].as_u64().unwrap() > 0);
                assert!(call["websocket_overhead_bytes"].as_u64().unwrap() > 0);
                assert!(call["worker"]["ffn_ns"].as_u64().unwrap() <= call["worker"]["handler_ns"].as_u64().unwrap());
            }
            smoke.push(serde_json::json!({"token_id": id, "local": local_rows[0], "remote": rows[0]}));
            assert_eq!(
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        // Optional diagnostic artifact, explicitly a tiny debug fixture, not a benchmark.
        if let Some(path) = std::env::var_os("LARQL_V3_FFN_SMOKE_PROFILE") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
            writeln!(file, "{}", serde_json::json!({"schema": "larql.v3.ffn-smoke.v1", "benchmark": false, "layers": 2, "hidden": ops.hidden(), "profile": "debug synthetic HTTP loopback; no exclusivity or warmup claim"})).unwrap();
            for row in smoke { writeln!(file, "{row}").unwrap(); }
        }
    })
    .await
    .unwrap();
    for server in servers {
        server.abort();
    }
}
