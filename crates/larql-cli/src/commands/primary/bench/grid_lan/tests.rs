use super::*;

fn ctx_from(pairs: &[(&'static str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.to_string()))
        .collect()
}

// ── substitute ──────────────────────────────────────────────────────────

#[test]
fn substitute_replaces_named_placeholders() {
    let ctx = ctx_from(&[("name", "alice"), ("count", "3")]);
    assert_eq!(
        substitute("hi {name}, take {count} steps", &ctx),
        "hi alice, take 3 steps"
    );
}

#[test]
fn substitute_leaves_unknown_placeholders_literal() {
    let ctx = ctx_from(&[("name", "alice")]);
    assert_eq!(substitute("hi {name}, {age}", &ctx), "hi alice, {age}");
}

#[test]
fn substitute_handles_double_braces_as_literal_brace() {
    let ctx = ctx_from(&[("x", "y")]);
    assert_eq!(
        substitute("{{not a placeholder}}", &ctx),
        "{not a placeholder}"
    );
}

#[test]
fn substitute_keeps_dangling_opener_verbatim() {
    let ctx = ctx_from(&[]);
    assert_eq!(substitute("oops {unfinished", &ctx), "oops {unfinished");
}

// ── command_for ─────────────────────────────────────────────────────────

fn sample_config() -> GridLanConfig {
    GridLanConfig {
        larql_bin: "./bin/larql".into(),
        defaults: Defaults {
            repeats: 3,
            tokens: 60,
            warmup: 5,
            prompt: "Hello".into(),
        },
        models: Models {
            dense: "models/dense.vindex".into(),
            moe: "models/moe.vindex".into(),
        },
        runs: vec![RunSpec {
            id: "dense-stream".into(),
            enabled: true,
            kind: "dense".into(),
            command: vec![
                "{larql_bin}".into(),
                "bench".into(),
                "{dense_model}".into(),
                "--tokens".into(),
                "{tokens}".into(),
                "--warmup".into(),
                "{warmup}".into(),
            ],
            env: BTreeMap::new(),
            vars: BTreeMap::new(),
            estimate: None,
            repeats: None,
        }],
    }
}

#[test]
fn command_for_substitutes_all_default_keys() {
    let cfg = sample_config();
    let argv = command_for(&cfg.runs[0], &cfg);
    assert_eq!(
        argv,
        vec![
            "./bin/larql".to_string(),
            "bench".into(),
            "models/dense.vindex".into(),
            "--tokens".into(),
            "60".into(),
            "--warmup".into(),
            "5".into(),
        ]
    );
}

#[test]
fn command_for_per_run_vars_override_defaults() {
    let mut cfg = sample_config();
    cfg.runs[0].vars.insert("tokens".into(), "120".into());
    let argv = command_for(&cfg.runs[0], &cfg);
    assert!(argv.iter().any(|a| a == "120"));
    assert!(!argv.iter().any(|a| a == "60"));
}

// ── parse_bench_output ──────────────────────────────────────────────────

#[test]
fn parse_bench_output_picks_up_data_rows() {
    // Matches the renderer in `bench/output.rs:format_data_row`.
    let stdout = "\
  Backend                    prefill       mean        p50      tok/s  steps  notes
  ────────────────────────────────────────────────────────────────────────────────
  metal                       125.2ms     12.34ms    11.05ms       81.0      50  ok
  cpu                          80.4ms     30.12ms    28.90ms       33.2      50  warm
";
    let parsed = parse_bench_output(stdout);
    assert_eq!(parsed.bench_rows.len(), 2);
    let row = &parsed.bench_rows[0];
    assert_eq!(row.backend, "metal");
    assert!((row.prefill_ms - 125.2).abs() < 1e-6);
    assert!((row.mean_ms - 12.34).abs() < 1e-6);
    assert!((row.p50_ms - 11.05).abs() < 1e-6);
    assert!((row.tok_per_s - 81.0).abs() < 1e-6);
    assert_eq!(row.steps, 50);
    assert_eq!(row.note, "ok");
}

#[test]
fn parse_bench_output_skips_header_and_separator() {
    let stdout = "\
  Backend                    prefill       mean        p50      tok/s  steps  notes
  ────────────────────────────────────────────────────────────────────────────────
";
    let parsed = parse_bench_output(stdout);
    assert!(parsed.bench_rows.is_empty());
}

#[test]
fn parse_bench_output_extracts_remote_stage_breakdown() {
    let stdout = "\
  Backend                    prefill       mean        p50      tok/s  steps  notes
  ────────────────────────────────────────────────────────────────────────────────
  remote-ffn                  100.0ms     12.00ms    11.50ms       83.3      50  http
    attn+norm+lmhead       3.20ms
    ffn round-trips        9.10ms
    total/tok             12.30ms
";
    let parsed = parse_bench_output(stdout);
    assert_eq!(parsed.bench_rows.len(), 1);
    // Keys mirror run.py: `+` and ` ` become `_`; other chars (here `/`) stay.
    assert!((parsed.remote_stage_ms["attn_norm_lmhead"] - 3.20).abs() < 1e-6);
    assert!((parsed.remote_stage_ms["ffn_round-trips"] - 9.10).abs() < 1e-6);
    assert!((parsed.remote_stage_ms["total/tok"] - 12.30).abs() < 1e-6);
}

#[test]
fn parse_bench_output_ignores_malformed_lines() {
    let stdout = "\
  Backend                    prefill
  not a row at all
   indented_subline 4.00ms
";
    let parsed = parse_bench_output(stdout);
    assert!(parsed.bench_rows.is_empty());
    assert!(parsed.remote_stage_ms.is_empty());
}

// ── encoded_bytes / q8k_bytes ────────────────────────────────────────────

#[test]
fn q8k_bytes_layout_matches_python_reference() {
    // hidden=2816, blocks=11 → 2816 + 11*4 + 11*8*2 = 3036
    assert_eq!(q8k_bytes(2816), 3036);
    // hidden=256 → 1 block → 256 + 4 + 16 = 276
    assert_eq!(q8k_bytes(256), 276);
}

#[test]
fn encoded_bytes_known_formats() {
    assert_eq!(encoded_bytes(2816, "f32").unwrap(), 11264);
    assert_eq!(encoded_bytes(2816, "f16").unwrap(), 5632);
    assert_eq!(encoded_bytes(2816, "q8k").unwrap(), 3036);
    assert_eq!(encoded_bytes(2816, "none").unwrap(), 0);
    assert!(encoded_bytes(2816, "bogus").is_err());
}

// ── estimate_bytes ───────────────────────────────────────────────────────

#[test]
fn estimate_bytes_dense_stream_assumes_fanout_one() {
    let est = Estimate {
        model_kind: "dense".into(),
        dispatch: "streaming".into(),
        encoding: "f32".into(),
        response_encoding: "f32".into(),
        hidden: 2816,
        layers: 60,
        shards: 2,
        active_shards: None,
    };
    let out = estimate_bytes(&est, 30).unwrap();
    // 60 layers × 1 fanout × 11264 bytes = 675840
    assert_eq!(out.upload_bytes_per_token, 675840);
    assert_eq!(out.active_shards_assumed, 1);
}

#[test]
fn estimate_bytes_moe_streaming_fanout_equals_shards() {
    let est = Estimate {
        model_kind: "moe".into(),
        dispatch: "streaming".into(),
        encoding: "f32".into(),
        response_encoding: "f32".into(),
        hidden: 2816,
        layers: 30,
        shards: 4,
        active_shards: None,
    };
    let out = estimate_bytes(&est, 30).unwrap();
    assert_eq!(out.active_shards_assumed, 4);
    // 30 × 4 × 11264 = 1351680
    assert_eq!(out.upload_bytes_per_token, 1351680);
}

#[test]
fn estimate_bytes_moe_batch_prefers_active_shards_override() {
    let est = Estimate {
        model_kind: "moe".into(),
        dispatch: "batch".into(),
        encoding: "q8k".into(),
        response_encoding: "f32".into(),
        hidden: 2816,
        layers: 30,
        shards: 4,
        active_shards: Some(2),
    };
    let out = estimate_bytes(&est, 30).unwrap();
    assert_eq!(out.active_shards_assumed, 2);
    // Upload: 30 layers × 2 fanout × q8k(2816)=3036 = 182160
    assert_eq!(out.upload_bytes_per_token, 182160);
}

// ── stats / repeat decision ─────────────────────────────────────────────

#[test]
fn mean_and_cov_basic() {
    assert!(mean(&[]).is_none());
    assert!((mean(&[1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-6);
    assert!(coefficient_of_variation(&[5.0]).is_none(), "1 sample");
    assert!(coefficient_of_variation(&[0.0, 0.0]).is_none(), "mean 0");
    // 90/100/110 → mean 100, stddev sqrt(200/3) ≈ 8.165 → cov ≈ 0.0816
    let cov = coefficient_of_variation(&[90.0, 100.0, 110.0]).unwrap();
    assert!((cov - 0.0816).abs() < 1e-3, "got {cov}");
}

#[test]
fn extra_repeats_needed_triggers_only_above_threshold() {
    // CoV ~0.08 < 0.15 → no extra repeats.
    assert_eq!(extra_repeats_needed(&[90.0, 100.0, 110.0], 0.15, 2), 0);
    // CoV ~0.45 > 0.15 → 2 more.
    assert_eq!(extra_repeats_needed(&[50.0, 100.0, 150.0], 0.15, 2), 2);
    // Fewer than 2 samples → cannot decide, default to 0.
    assert_eq!(extra_repeats_needed(&[100.0], 0.15, 2), 0);
}

// ── safe_name ───────────────────────────────────────────────────────────

#[test]
fn safe_name_collapses_unsafe_chars() {
    assert_eq!(safe_name("dense-http stream f32"), "dense-http_stream_f32");
    assert_eq!(safe_name("/leading/and trailing/"), "leading_and_trailing");
    assert_eq!(safe_name("ok.name_42"), "ok.name_42");
}

// ── selected_runs ───────────────────────────────────────────────────────

fn cfg_with_runs(specs: Vec<(&str, bool)>) -> GridLanConfig {
    GridLanConfig {
        larql_bin: "/bin".into(),
        defaults: Defaults::default(),
        models: Models::default(),
        runs: specs
            .into_iter()
            .map(|(id, enabled)| RunSpec {
                id: id.into(),
                enabled,
                kind: String::new(),
                command: vec![],
                env: BTreeMap::new(),
                vars: BTreeMap::new(),
                estimate: None,
                repeats: None,
            })
            .collect(),
    }
}

#[test]
fn selected_runs_filters_disabled_by_default() {
    let cfg = cfg_with_runs(vec![("a", true), ("b", false), ("c", true)]);
    let picked: Vec<&str> = selected_runs(&cfg, None, false)
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(picked, vec!["a", "c"]);
}

#[test]
fn selected_runs_include_disabled_returns_all() {
    let cfg = cfg_with_runs(vec![("a", true), ("b", false)]);
    let picked: Vec<&str> = selected_runs(&cfg, None, true)
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(picked, vec!["a", "b"]);
}

#[test]
fn selected_runs_only_filters_to_named_ids() {
    let cfg = cfg_with_runs(vec![("a", true), ("b", true), ("c", true)]);
    let only = vec!["b".to_string(), "c".to_string()];
    let picked: Vec<&str> = selected_runs(&cfg, Some(&only), false)
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(picked, vec!["b", "c"]);
}

// ── config deserialization smoke test ────────────────────────────────────

#[test]
fn config_deserializes_minimal_run_list() {
    let json = r#"{
            "runs": [
                {
                    "id": "minimal",
                    "command": ["./larql", "bench", "{prompt}"]
                }
            ]
        }"#;
    let cfg: GridLanConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.runs.len(), 1);
    assert_eq!(cfg.runs[0].id, "minimal");
    assert!(cfg.runs[0].enabled);
    assert_eq!(cfg.larql_bin, "./target/release/larql");
    assert_eq!(cfg.defaults.tokens, 30);
}

#[test]
fn config_round_trips_estimate_block() {
    let json = r#"{
            "runs": [{
                "id": "with-est",
                "command": [],
                "estimate": {
                    "model_kind": "moe",
                    "encoding": "f16",
                    "hidden": 2816,
                    "layers": 30
                }
            }]
        }"#;
    let cfg: GridLanConfig = serde_json::from_str(json).unwrap();
    let est = cfg.runs[0].estimate.as_ref().unwrap();
    assert_eq!(est.model_kind, "moe");
    assert_eq!(est.dispatch, "streaming"); // default
    assert_eq!(est.shards, 1); // default
    let bytes = estimate_bytes(est, 30).unwrap();
    // 30 × 1 × f16(2816)=5632 = 168960
    assert_eq!(bytes.upload_bytes_per_token, 168960);
}
