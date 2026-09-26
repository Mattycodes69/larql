use super::*;
use larql_compute::cpu_backend;
use larql_inference::{cpu_engine_backend, ModelWeights};
use ndarray::Array2;

fn all_kinds() -> Vec<EngineKind> {
    vec![
        EngineKind::Standard { window_size: None },
        EngineKind::Standard {
            window_size: Some(64),
        },
        EngineKind::NoCache,
        EngineKind::MarkovResidual { window_size: None },
        EngineKind::MarkovResidual {
            window_size: Some(32),
        },
        EngineKind::WindowedCheckpoint { window_size: 64 },
        EngineKind::TurboQuant { bits: 4 },
        EngineKind::TurboQuant { bits: 3 },
        EngineKind::Apollo {
            injection_layer: 30,
            inject_coefficient: 10.0,
            top_k: 8,
            bos_token_id: None,
        },
    ]
}

#[test]
fn all_engines_memory_zero_before_prefill() {
    for kind in all_kinds() {
        let engine = kind.clone().build(cpu_engine_backend());
        assert_eq!(
            engine.memory_bytes(),
            0,
            "{} should have 0 memory before prefill",
            kind.display_name()
        );
    }
}

#[test]
fn all_engines_have_valid_name() {
    let expected = [
        "standard",
        "standard",
        "no-cache",
        "markov-rs",
        "markov-rs",
        "windowed-checkpoint",
        "turbo-quant",
        "turbo-quant",
        "apollo",
    ];
    for (kind, expected_name) in all_kinds().into_iter().zip(expected.iter()) {
        let engine = kind.build(cpu_engine_backend());
        assert_eq!(engine.name(), *expected_name);
    }
}

#[test]
fn all_engines_info_has_nonempty_fields() {
    for kind in all_kinds() {
        let name = kind.display_name();
        let engine = kind.build(cpu_engine_backend());
        let info = engine.info();
        assert!(!info.name.is_empty(), "{name}: empty name");
        assert!(!info.backend.is_empty(), "{name}: empty backend");
    }
}

#[test]
fn all_engines_window_tokens_zero_before_prefill() {
    for kind in all_kinds() {
        let engine = kind.clone().build(cpu_engine_backend());
        assert_eq!(
            engine.window_tokens(),
            0,
            "{} window_tokens should be 0 before prefill",
            kind.display_name()
        );
    }
}

#[test]
fn all_engines_cold_bytes_zero_before_prefill() {
    for kind in all_kinds() {
        let engine = kind.clone().build(cpu_engine_backend());
        assert_eq!(
            engine.cold_bytes(),
            0,
            "{} cold_bytes should be 0 before prefill",
            kind.display_name()
        );
    }
}

#[test]
fn all_engines_stage_summary_none_before_decode() {
    for kind in all_kinds() {
        let engine = kind
            .clone()
            .build_with_profiling(cpu_engine_backend(), true);
        assert!(
            engine.stage_summary().is_none(),
            "{} stage_summary should be None before decode",
            kind.display_name()
        );
    }
}

#[test]
fn from_name_unknown_param_ignored_defaults_apply() {
    match EngineKind::from_name("unlimited-context:unknown=42") {
        Some(EngineKind::WindowedCheckpoint { window_size: 512 }) => {}
        other => panic!("unknown param should use default, got {other:?}"),
    }
}

#[test]
fn supported_names_every_entry_parses_back_via_from_name() {
    // Every name `supported_names()` advertises must be a name the
    // parser actually accepts. Catches the failure mode where
    // someone adds a variant to one side without the other.
    for name in EngineKind::supported_names() {
        let kind = EngineKind::from_name(name)
            .unwrap_or_else(|| panic!("supported_names lists {name:?} but from_name rejected it"));
        assert_eq!(
            kind.display_name(),
            *name,
            "supported_names entry {name:?} parses to a different display_name"
        );
    }
}

#[test]
fn supported_names_covers_every_engine_kind_variant() {
    // Build one of every variant via the parser, collect their
    // canonical names, and verify supported_names() lists each.
    // This is the test the doc comment refers to; adding a new
    // EngineKind variant without adding a supported_names entry
    // makes this test fail with a useful diff.
    let one_of_each: Vec<&'static str> = [
        "standard",
        "no-cache",
        "markov-rs",
        "markov-rs-codec",
        "unlimited-context",
        "turbo-quant",
        "apollo",
        "boundary-kv",
        "boundary-per-layer",
        "semantic-promotion",
    ]
    .iter()
    .map(|s| {
        EngineKind::from_name(s)
            .unwrap_or_else(|| panic!("test fixture {s:?} failed to parse"))
            .display_name()
    })
    .collect();
    for name in &one_of_each {
        assert!(
            EngineKind::supported_names().contains(name),
            "EngineKind variant with display_name {name:?} is missing from supported_names"
        );
    }
    assert_eq!(
        EngineKind::supported_names().len(),
        one_of_each.len(),
        "supported_names and the variant set are out of sync"
    );
}

/// The criterion bench must cover every engine that can be benched
/// on the synthetic fixture. It previously listed 7 of 9 — the three
/// engines this PR touches most (`markov-rs-codec`, `boundary-kv`,
/// `boundary-per-layer`) had no microbenchmark at all, and nothing
/// failed when they were added. Now an engine is either in
/// `bench_specs` or explicitly in `bench_excluded_names` with a
/// reason; there is no third, silent option.
#[test]
fn bench_specs_cover_every_benchable_engine() {
    let excluded: Vec<&str> = EngineKind::bench_excluded_names()
        .iter()
        .map(|(n, _)| *n)
        .collect();

    let benched: Vec<&'static str> = EngineKind::bench_specs()
        .iter()
        .map(|s| {
            EngineKind::from_name(s)
                .unwrap_or_else(|| panic!("bench_specs entry {s:?} no longer parses"))
                .display_name()
        })
        .collect();

    for name in EngineKind::supported_names() {
        if excluded.contains(name) {
            assert!(
                !benched.contains(name),
                "{name:?} is listed as excluded but also appears in bench_specs"
            );
            continue;
        }
        assert!(
            benched.contains(name),
            "engine {name:?} has no criterion bench arm — add a spec to \
             EngineKind::bench_specs, or name it in bench_excluded_names \
             with the reason it cannot be benched"
        );
    }
}

#[test]
fn bench_excluded_names_carry_a_reason_and_are_real_engines() {
    for (name, reason) in EngineKind::bench_excluded_names() {
        assert!(
            EngineKind::supported_names().contains(name),
            "bench_excluded_names lists {name:?}, which is not a supported engine"
        );
        assert!(
            reason.len() > 20,
            "exclusion of {name:?} needs a real reason, got {reason:?}"
        );
    }
}

#[test]
fn from_name_all_engines_parseable() {
    let specs = [
        ("standard", "standard"),
        ("standard:window=128", "standard"),
        ("markov-bounded", "standard"),
        ("no-cache", "no-cache"),
        ("none", "no-cache"),
        ("markov-rs", "markov-rs"),
        ("windowed-checkpoint", "windowed-checkpoint"),
        // Pre-rename spellings still parse, and normalise to the new
        // canonical name rather than echoing themselves back.
        ("unlimited-context", "windowed-checkpoint"),
        ("unlimited", "windowed-checkpoint"),
        ("turbo-quant", "turbo-quant"),
        ("tq3", "turbo-quant"),
        ("apollo", "apollo"),
        ("semantic-promotion", "semantic-promotion"),
        ("semantic-promotion:base=markov-rs", "semantic-promotion"),
    ];
    for (spec, expected_display) in specs {
        let kind =
            EngineKind::from_name(spec).unwrap_or_else(|| panic!("{spec:?} failed to parse"));
        assert_eq!(
            kind.display_name(),
            expected_display,
            "{spec} parsed to wrong display_name"
        );
    }
}

#[test]
fn semantic_promotion_defaults_to_an_unbounded_standard_base_in_observe_mode() {
    let kind = EngineKind::from_name("semantic-promotion").unwrap();
    let EngineKind::SemanticPromotion { base, mode } = kind else {
        panic!("expected a SemanticPromotion variant");
    };
    assert!(matches!(*base, EngineKind::Standard { window_size: None }));
    assert_eq!(mode, semantic_promotion::PromotionMode::Observe);
}

#[test]
fn semantic_promotion_nests_a_parameterised_base_spec() {
    // The outer split takes only the first colon, so the base spec
    // keeps its own `:key=value` tail.
    let kind = EngineKind::from_name("semantic-promotion:base=standard:window=512").unwrap();
    let EngineKind::SemanticPromotion { base, .. } = kind else {
        panic!("expected a SemanticPromotion variant");
    };
    assert!(matches!(
        *base,
        EngineKind::Standard {
            window_size: Some(512)
        }
    ));
}

#[test]
fn semantic_promotion_rejects_an_unknown_base_or_mode() {
    assert!(EngineKind::from_name("semantic-promotion:base=nonsuch").is_none());
    assert!(EngineKind::from_name("semantic-promotion:mode=delete-everything").is_none());
}

#[test]
fn semantic_promotion_builds_and_wraps_its_base() {
    let engine = EngineKind::from_name("semantic-promotion")
        .unwrap()
        .build(larql_inference::cpu_engine_backend());
    assert_eq!(engine.name(), "semantic-promotion(standard)");
    assert!(engine.is_kv());
}

#[test]
fn semantic_promotion_refuses_to_build_an_enforcing_mode() {
    // No base engine implements the masking or snapshot hooks yet,
    // so the enforcing modes must not construct — the build arm
    // surfaces that as a panic rather than downgrading to Observe.
    let kind = EngineKind::from_name("semantic-promotion:mode=enforce").unwrap();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kind.build(larql_inference::cpu_engine_backend())
    }));
    assert!(built.is_err(), "enforcing mode must not construct today");
}

/// Synthetic engine that does not override `prefill_quant` /
/// `decode_step_quant`. Exercises the default trait methods that route to
/// the f32 fallback — every shipped engine overrides these, so without
/// this fixture they sit at 0% line coverage.
struct DefaultMethodsEngine {
    /// Counts calls to `prefill` to confirm the q4k → prefill fallback
    /// path actually dispatches through the f32 method.
    prefill_calls: usize,
    decode_calls: usize,
}

impl KvEngine for DefaultMethodsEngine {
    fn name(&self) -> &str {
        "default-methods-test"
    }
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: self.name().into(),
            description: "test fixture".into(),
            backend: "cpu".into(),
            config: String::new(),
        }
    }
    fn prefill(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn larql_inference::ffn::FfnBackend,
        _token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        self.prefill_calls += 1;
        Ok(Array2::zeros((1, 4)))
    }
    fn decode_step(
        &mut self,
        _weights: &ModelWeights,
        _ffn: &dyn larql_inference::ffn::FfnBackend,
        _token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_calls += 1;
        Ok(Array2::zeros((1, 4)))
    }
    fn memory_bytes(&self) -> usize {
        0
    }
}

#[test]
fn default_q4k_methods_fallback_to_f32() {
    use larql_inference::ffn::WeightFfn;
    let weights = larql_inference::test_utils::make_test_weights();
    let index = larql_inference::test_utils::make_test_vindex(&weights);
    let backend = cpu_backend();
    let ffn = WeightFfn { weights: &weights };
    let mut engine = DefaultMethodsEngine {
        prefill_calls: 0,
        decode_calls: 0,
    };

    // Build a separate &mut binding for the `prefill_quant` call.
    let weights_for_q4k = larql_inference::test_utils::make_test_weights();
    let out = engine.prefill_quant(&weights_for_q4k, &ffn, &index, &[1, 2, 3], &*backend);
    assert!(out.is_ok());
    assert_eq!(
        engine.prefill_calls, 1,
        "default prefill_quant must call prefill"
    );

    let out = engine.decode_step_quant(&weights_for_q4k, &ffn, &index, 4, &*backend);
    assert!(out.is_ok());
    assert_eq!(
        engine.decode_calls, 1,
        "default decode_step_quant must call decode_step"
    );
}

#[test]
fn default_window_tokens_and_cold_bytes_are_zero() {
    // Both have default impls returning 0; exercises the trait defaults
    // for an engine that doesn't override them.
    let engine = DefaultMethodsEngine {
        prefill_calls: 0,
        decode_calls: 0,
    };
    assert_eq!(engine.window_tokens(), 0);
    assert_eq!(engine.cold_bytes(), 0);
    assert!(engine.stage_summary().is_none());
    assert_eq!(engine.name(), "default-methods-test");
}

// ── split_specs ──────────────────────────────────────────────────────────

#[test]
fn split_specs_legacy_comma_for_simple_engines() {
    // No colons → no param-comma ambiguity. Comma is the legacy separator.
    let v = EngineKind::split_specs("standard,markov-rs,no-cache");
    assert_eq!(v, vec!["standard", "markov-rs", "no-cache"]);
}

#[test]
fn split_specs_legacy_comma_with_single_param_each() {
    // Single param per engine is unambiguous under comma split: each
    // engine's `name:key=value` doesn't contain a comma.
    let v = EngineKind::split_specs("standard:window=512,markov-rs:window=256");
    assert_eq!(v, vec!["standard:window=512", "markov-rs:window=256"]);
}

#[test]
fn split_specs_semicolon_separator_for_multi_param_engines() {
    // Multi-param engines need ';' as the list separator to avoid
    // colliding with their param commas.
    let v = EngineKind::split_specs(
        "boundary-kv:chunk_tokens=64,sequence_id=demo;markov-rs:window=256",
    );
    assert_eq!(
        v,
        vec![
            "boundary-kv:chunk_tokens=64,sequence_id=demo",
            "markov-rs:window=256",
        ]
    );
}

#[test]
fn split_specs_trims_whitespace() {
    let v = EngineKind::split_specs(" standard , markov-rs ");
    assert_eq!(v, vec!["standard", "markov-rs"]);
}

#[test]
fn split_specs_drops_empty_entries() {
    let v = EngineKind::split_specs(",,standard,,markov-rs,");
    assert_eq!(v, vec!["standard", "markov-rs"]);
}

#[test]
fn split_specs_semicolon_drops_empties_and_trims() {
    let v = EngineKind::split_specs(" ; standard ;; markov-rs ; ");
    assert_eq!(v, vec!["standard", "markov-rs"]);
}

#[test]
fn split_specs_single_engine_returns_one_entry() {
    assert_eq!(EngineKind::split_specs("standard"), vec!["standard"]);
    assert_eq!(
        EngineKind::split_specs("boundary-kv:chunk_tokens=64,sequence_id=demo"),
        vec!["boundary-kv:chunk_tokens=64,sequence_id=demo"]
    );
}

#[test]
fn split_specs_empty_returns_empty_vec() {
    assert!(EngineKind::split_specs("").is_empty());
    assert!(EngineKind::split_specs(" ").is_empty());
    assert!(EngineKind::split_specs(",,,").is_empty());
    assert!(EngineKind::split_specs(";;;").is_empty());
}

#[test]
fn split_specs_round_trips_with_from_name() {
    // Each split entry must round-trip through EngineKind::from_name.
    let input = "standard;markov-rs:window=512;boundary-kv:chunk_tokens=64,sequence_id=demo";
    let specs = EngineKind::split_specs(input);
    for s in &specs {
        assert!(
            EngineKind::from_name(s).is_some(),
            "spec {s:?} should parse"
        );
    }
}
