//! Live Ollama query: ms/token from its own `eval_count` / `eval_duration`.

const GENERATE_URL: &str = "http://localhost:11434/api/generate";
const WARMUP_BODY: &str =
    r#"{"model":"gemma3:4b","prompt":"Hi","stream":false,"options":{"num_predict":5}}"#;
const MEASURE_BODY: &str = r#"{"model":"gemma3:4b","prompt":"Explain quantum computing","stream":false,"options":{"num_predict":50}}"#;
/// `eval_duration` is reported in nanoseconds.
const NS_PER_MS: f64 = 1e6;

/// Ollama's decode ms/token, or 0.0 when it is not running or answered
/// without a usable eval count.
pub fn ollama_ms_per_token() -> f64 {
    // Warm up
    let _ = std::process::Command::new("curl")
        .args(["-s", GENERATE_URL, "-d", WARMUP_BODY])
        .output();

    let out = std::process::Command::new("curl")
        .args(["-s", GENERATE_URL, "-d", MEASURE_BODY])
        .output()
        .ok();

    let Some(o) = out else {
        return 0.0;
    };
    let text = String::from_utf8_lossy(&o.stdout);
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) else {
        return 0.0;
    };
    let ec = val["eval_count"].as_f64().unwrap_or(0.0);
    let en = val["eval_duration"].as_f64().unwrap_or(1.0);
    if ec > 0.0 {
        en / NS_PER_MS / ec
    } else {
        0.0
    }
}
