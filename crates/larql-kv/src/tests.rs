use super::*;

#[test]
fn engine_kind_from_name_roundtrip() {
    for name in &[
        "markov-rs",
        "markov_rs",
        "markov-residual",
        "markov_residual",
    ] {
        assert!(
            matches!(
                EngineKind::from_name(name),
                Some(EngineKind::MarkovResidual { .. })
            ),
            "failed to parse {name:?}"
        );
    }
    for name in &[
        "windowed-checkpoint",
        "windowed_checkpoint",
        "unlimited",
        "unlimited-context",
        "unlimited_context",
    ] {
        assert!(
            matches!(
                EngineKind::from_name(name),
                Some(EngineKind::WindowedCheckpoint { .. })
            ),
            "failed to parse {name:?}"
        );
    }
    assert!(EngineKind::from_name("unknown").is_none());
    assert!(EngineKind::from_name("").is_none());
}

#[test]
fn engine_kind_from_name_with_params() {
    match EngineKind::from_name("standard") {
        Some(EngineKind::Standard { window_size: None }) => {}
        other => panic!("expected Standard{{window=None}}, got {other:?}"),
    }
    match EngineKind::from_name("standard:window=512") {
        Some(EngineKind::Standard {
            window_size: Some(512),
        }) => {}
        other => panic!("expected Standard{{window=512}}, got {other:?}"),
    }
    match EngineKind::from_name("markov-bounded:window=256") {
        // Legacy flag → Standard{Some(N)}.
        Some(EngineKind::Standard {
            window_size: Some(256),
        }) => {}
        other => panic!("expected Standard{{window=256}}, got {other:?}"),
    }
    match EngineKind::from_name("no-cache") {
        Some(EngineKind::NoCache) => {}
        other => panic!("expected NoCache, got {other:?}"),
    }
    match EngineKind::from_name("none") {
        Some(EngineKind::NoCache) => {}
        other => panic!("expected NoCache from 'none', got {other:?}"),
    }
    match EngineKind::from_name("markov-rs:window=1024") {
        Some(EngineKind::MarkovResidual {
            window_size: Some(1024),
            ..
        }) => {}
        other => panic!("expected MarkovResidual{{window=1024}}, got {other:?}"),
    }
    match EngineKind::from_name("unlimited-context:window=256") {
        Some(EngineKind::WindowedCheckpoint { window_size: 256 }) => {}
        other => panic!("expected WindowedCheckpoint{{window=256}}, got {other:?}"),
    }
    match EngineKind::from_name("turbo-quant:bits=3") {
        Some(EngineKind::TurboQuant { bits: 3 }) => {}
        other => panic!("expected TurboQuant{{bits=3}}, got {other:?}"),
    }
    match EngineKind::from_name("apollo:layer=25,coef=8.0,top_k=12") {
        Some(EngineKind::Apollo {
            injection_layer: 25,
            top_k: 12,
            // No `bos=` param → None: never a hardcoded model default.
            bos_token_id: None,
            ..
        }) => {}
        other => panic!("expected Apollo{{layer=25,top_k=12,bos=None}}, got {other:?}"),
    }
    match EngineKind::from_name("apollo:layer=25,bos=2") {
        Some(EngineKind::Apollo {
            injection_layer: 25,
            bos_token_id: Some(2),
            ..
        }) => {}
        other => panic!("expected Apollo{{layer=25,bos=Some(2)}}, got {other:?}"),
    }
    match EngineKind::from_name("markov-rs:unknown=999") {
        Some(EngineKind::MarkovResidual {
            window_size: None, ..
        }) => {}
        other => panic!("expected MarkovResidual{{window=None}}, got {other:?}"),
    }
}

// ── BoundaryKv parsing ───────────────────────────────────────────────

#[test]
fn engine_kind_from_name_boundary_kv_aliases() {
    for name in &["boundary-kv", "boundary_kv", "boundary"] {
        assert!(
            matches!(
                EngineKind::from_name(name),
                Some(EngineKind::BoundaryKv { .. })
            ),
            "failed to parse {name:?}"
        );
    }
}

#[test]
fn engine_kind_from_name_boundary_kv_with_params() {
    match EngineKind::from_name("boundary-kv:chunk_tokens=64,sequence_id=demo") {
        Some(EngineKind::BoundaryKv {
            chunk_tokens: 64,
            sequence_id,
            window_size: None,
        }) => assert_eq!(sequence_id, "demo"),
        other => panic!("expected BoundaryKv with custom params, got {other:?}"),
    }
}

#[test]
fn engine_kind_from_name_boundary_kv_defaults() {
    match EngineKind::from_name("boundary-kv") {
        Some(EngineKind::BoundaryKv {
            chunk_tokens: 512,
            sequence_id,
            window_size: None,
        }) => assert_eq!(sequence_id, "default"),
        other => panic!("expected default BoundaryKv, got {other:?}"),
    }
}

#[test]
fn engine_kind_from_name_boundary_kv_with_window() {
    match EngineKind::from_name("boundary-kv:window=128,chunk_tokens=32") {
        Some(EngineKind::BoundaryKv {
            window_size: Some(128),
            chunk_tokens: 32,
            ..
        }) => {}
        other => panic!("expected BoundaryKv{{window=128,chunk=32}}, got {other:?}"),
    }
}

// ── BoundaryPerLayer parsing ─────────────────────────────────────────

#[test]
fn a_malformed_value_rejects_the_spec_instead_of_defaulting() {
    // `window=abc` must not mean "unbounded", nor `bits=260` wrap to 4.
    for spec in [
        "standard:window=abc",
        "markov-rs:window=-1",
        "turbo-quant:bits=260",
        "unlimited-context:window=big",
        "boundary-per-layer:layers=x",
        "apollo:coef=strong",
    ] {
        assert!(
            EngineKind::from_name(spec).is_none(),
            "{spec} must be refused"
        );
    }
    assert!(matches!(
        EngineKind::from_name("turbo-quant:bits=3"),
        Some(EngineKind::TurboQuant { bits: 3 })
    ));
}

#[test]
fn engine_kind_from_name_boundary_per_layer_aliases() {
    for name in &["boundary-per-layer", "boundary_per_layer", "boundary-pl"] {
        assert!(
            matches!(
                EngineKind::from_name(name),
                Some(EngineKind::BoundaryPerLayer { .. })
            ),
            "failed to parse {name:?}"
        );
    }
}

#[test]
fn engine_kind_from_name_boundary_per_layer_adopts_the_model_depth_by_default() {
    match EngineKind::from_name("boundary-per-layer") {
        Some(EngineKind::BoundaryPerLayer {
            window_size: None,
            num_layers: None,
        }) => {}
        other => panic!("expected BoundaryPerLayer{{layers=model}}, got {other:?}"),
    }
}

#[test]
fn engine_kind_from_name_boundary_per_layer_with_window_and_layers() {
    match EngineKind::from_name("boundary-per-layer:window=256,layers=12") {
        Some(EngineKind::BoundaryPerLayer {
            window_size: Some(256),
            num_layers: Some(12),
        }) => {}
        other => panic!("expected BoundaryPerLayer{{window=256,layers=12}}, got {other:?}"),
    }
}

// ── MarkovResidualCodec parsing ──────────────────────────────────────

#[test]
fn engine_kind_from_name_markov_rs_codec_aliases() {
    for name in &[
        "markov-rs-codec",
        "markov_rs_codec",
        "markov-residual-codec",
        "markov_residual_codec",
    ] {
        assert!(
            matches!(
                EngineKind::from_name(name),
                Some(EngineKind::MarkovResidualCodec {
                    codec: markov_residual_codec::ColdResidualCodec::Bf16,
                    ..
                })
            ),
            "failed to parse {name:?}"
        );
    }
}

#[test]
fn engine_kind_from_name_markov_rs_codec_with_window() {
    match EngineKind::from_name("markov-rs-codec:window=256") {
        Some(EngineKind::MarkovResidualCodec {
            window_size: Some(256),
            codec: markov_residual_codec::ColdResidualCodec::Bf16,
            ..
        }) => {}
        other => panic!("expected MarkovResidualCodec{{window=256,codec=Bf16}}, got {other:?}"),
    }
}

// ── display_name for new variants ────────────────────────────────────

#[test]
fn engine_kind_display_name_covers_new_variants() {
    let kinds = [
        EngineKind::BoundaryKv {
            window_size: None,
            chunk_tokens: 512,
            sequence_id: "x".into(),
        },
        EngineKind::MarkovResidualCodec {
            window_size: None,
            codec: markov_residual_codec::ColdResidualCodec::Bf16,
        },
    ];
    let expected = ["boundary-kv", "markov-rs-codec"];
    for (k, name) in kinds.into_iter().zip(expected) {
        assert_eq!(k.display_name(), name);
    }
}

// ── build() for new variants ─────────────────────────────────────────

#[test]
fn engine_kind_build_boundary_kv_returns_engine() {
    let kind = EngineKind::BoundaryKv {
        window_size: None,
        chunk_tokens: 16,
        sequence_id: "test".into(),
    };
    let engine = kind.build(larql_inference::cpu_engine_backend());
    assert_eq!(engine.name(), "boundary-kv");
}

#[test]
fn engine_kind_build_markov_rs_codec_returns_engine() {
    let kind = EngineKind::MarkovResidualCodec {
        window_size: Some(32),
        codec: markov_residual_codec::ColdResidualCodec::Bf16,
    };
    let engine = kind.build(larql_inference::cpu_engine_backend());
    assert_eq!(engine.name(), "markov-rs-codec");
}

// ── split_specs edge: first piece doesn't parse ───────────────────────

#[test]
fn split_specs_first_piece_unparseable_is_preserved() {
    // The first comma piece doesn't match a known engine name. The
    // splitter keeps it so the caller can surface a parse error rather
    // than silently dropping it (lines 239-242).
    let v = EngineKind::split_specs("garbage_engine,standard");
    // garbage_engine doesn't parse → kept as the first spec; then
    // 'standard' parses → becomes second spec.
    assert_eq!(v, vec!["garbage_engine", "standard"]);
    // The caller's `from_name` will then fail on "garbage_engine".
    assert!(EngineKind::from_name(&v[0]).is_none());
    assert!(EngineKind::from_name(&v[1]).is_some());
}

#[test]
fn engine_info_summary_with_config() {
    let info = EngineInfo {
        name: "markov-rs".into(),
        description: "residual KV".into(),
        backend: "cpu".into(),
        config: "window=512".into(),
    };
    let s = info.summary();
    assert!(s.contains("markov-rs"));
    assert!(s.contains("cpu"));
    assert!(s.contains("window=512"));
}

#[test]
fn engine_info_summary_no_config() {
    let info = EngineInfo {
        name: "test".into(),
        description: "desc".into(),
        backend: "metal".into(),
        config: String::new(),
    };
    let s = info.summary();
    assert!(!s.contains("()"));
}
