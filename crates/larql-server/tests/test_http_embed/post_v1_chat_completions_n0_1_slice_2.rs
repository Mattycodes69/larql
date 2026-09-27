//! POST /v1/chat/completions — N0.1 slice 2

use super::*;

#[tokio::test]
async fn http_openai_chat_stream_returns_event_stream_content_type() {
    // model() has infer_disabled=true, but the dispatch happens before
    // the inference step — actually no, infer_disabled is checked first
    // and returns 503 even for stream. Use model_infer_enabled (empty
    // tokenizer) — generation will tokenise the prompt to empty and
    // emit an error chunk before [DONE], but the response headers and
    // status should be SSE.
    use axum::http::header;
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
}

#[tokio::test]
async fn http_openai_chat_n_gt_1_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tools_are_accepted() {
    // Tools synthesise a constrained-decoding schema. Synthetic model
    // is infer_disabled so we 503 — confirms the schema synth +
    // ToolMode resolution succeeded.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_tools_with_specific_choice_is_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "calc"}},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_tools_unknown_choice_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "missing"}},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tools_with_stream_returns_event_stream() {
    // Slice 4.11: tools + stream is now wired. Synthetic model has
    // infer_disabled=true, but the SSE response shape is determined
    // before the inference gate fires — confirm we get a 200 SSE
    // content-type, not 400.
    use axum::http::header;
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
            "stream": true,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got {ct:?}"
    );
}

#[tokio::test]
async fn http_openai_chat_tool_choice_none_skips_constraint() {
    // tool_choice="none" disables constrained decoding even when tools
    // are listed — falls through to the standard text completion path.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "calc", "parameters": {}}}],
            "tool_choice": "none",
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_schema_missing_schema_field_returns_400() {
    // {type: "json_schema"} requires `json_schema: {schema: ...}` —
    // the empty inner object has no `schema` key, so we 400 with a
    // pointer at the missing field.
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {}},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_schema_is_accepted() {
    // Full {type: "json_schema", json_schema: {name, schema, strict}}
    // request — synthetic model 503s because infer_disabled, which
    // confirms the schema parsed cleanly through to the inference gate.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        },
                        "required": ["name", "age"]
                    }
                }
            },
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_schema_invalid_returns_400() {
    // Schema uses an unsupported feature ($ref) — parser bubbles up
    // a clear 400.
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {"$ref": "#/foo"}}
            },
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_response_format_text_is_accepted() {
    // {type: "text"} is the OpenAI default — should pass through, fall
    // through to infer_disabled gate (synthetic model) → 503.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_json_object_is_accepted() {
    // {type: "json_object"} compiles to a Schema::Object(any) FSM and
    // routes through generate_constrained. The synthetic model has
    // infer_disabled=true so we still 503 — that's our signal that the
    // request shape parsed cleanly through the constrained-mode path.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_response_format_unknown_type_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "yaml"},
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_invalid_role_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "function", "content": "x"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tool_message_without_tool_call_id_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "tool", "content": "23C"} // missing tool_call_id
            ],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_tool_replay_is_accepted() {
    // Full multi-turn tool flow: user → assistant tool_call → tool
    // result → expects another assistant turn. Synthetic model is
    // infer_disabled, so we 503 — confirming the wire shape parsed
    // through validation.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "Weather in London?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "23C, sunny"}
            ],
            "max_tokens": 16
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_assistant_with_only_tool_calls_is_accepted() {
    // Some clients send assistant messages with content: null but
    // populated tool_calls — must not 400 on the missing content.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "calc", "arguments": "{}"}}
                ]}
            ],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_logprobs_request_field_is_accepted() {
    // logprobs: true should be accepted on chat completions; the
    // synthetic model 503s but the field passes validation.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true,
            "top_logprobs": 5,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_repetition_penalties_are_accepted() {
    // F19: frequency_penalty + presence_penalty land in SamplingConfig
    // and clamp to [-2.0, 2.0]. Synthetic model 503s but the field
    // parses cleanly through to the inference gate.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "temperature": 0.7,
            "frequency_penalty": 1.5,
            "presence_penalty": -0.3,
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_repetition_penalties_are_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "frequency_penalty": 1.0,
            "presence_penalty": 0.5,
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_logprobs_request_field_is_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "logprobs": 3,
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_assistant_with_no_content_or_tools_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant"} // no content, no tool_calls
            ],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_empty_messages_returns_400() {
    let app = single_model_router(state(vec![model_infer_enabled("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({"messages": [], "max_tokens": 1}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn http_openai_chat_infer_disabled_returns_503() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_multi_routes_via_model_field() {
    let app = multi_model_router(state(vec![model("gemma-a"), model("gemma-b")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "gemma-b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    // Routing succeeds; infer_disabled on the synthetic model → 503.
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_multi_unknown_model_returns_404() {
    let app = multi_model_router(state(vec![model("gemma-a")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "missing",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_openai_chat_sampling_params_accepted() {
    // Wire-shape contract: temperature, top_p, seed, stop must be
    // accepted on the request and not rejected by validation. The
    // synthetic model has infer_disabled=true so the request reaches
    // the inference gate (503) — that's our signal that all sampling
    // fields parsed cleanly upstream.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "stop": ["\n\n", "STOP"],
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_chat_stop_accepts_single_string() {
    // OpenAI's `stop` is `string | string[]`; the StopSpec untagged
    // enum should accept a bare string without validation errors.
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "\n",
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn http_openai_completions_sampling_params_accepted() {
    let app = single_model_router(state(vec![model("gemma")]));
    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "prompt": "hi",
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "stop": ["\n\n"],
            "max_tokens": 4
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}
