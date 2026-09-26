//! Issue #199: `--engine` was accepted and silently ignored on the CPU
//! Q4K generation path, so every engine driven through `larql run`
//! exercised the same code. Anyone A/B-ing engines that way would have
//! compared the default against itself.

use super::{engine_unsupported_on_uncached_path, validate_engine_spec};

const UNKNOWN: &str = "not-an-engine";

#[test]
fn no_engine_named_is_fine() {
    assert!(validate_engine_spec(None).is_ok());
}

#[test]
fn every_supported_name_validates() {
    // Guards the pairing between the validator and the help text it
    // prints: a name advertised as supported must actually parse.
    for name in larql_kv::EngineKind::supported_names() {
        assert!(
            validate_engine_spec(Some(name)).is_ok(),
            "{name} is advertised as supported but does not validate"
        );
    }
}

#[test]
fn pre_rename_aliases_still_validate() {
    // Scripts and baselines predating the WindowedCheckpoint rename must
    // keep working.
    for alias in ["unlimited", "windowed-checkpoint", "unlimited_context"] {
        assert!(validate_engine_spec(Some(alias)).is_ok(), "{alias}");
    }
}

#[test]
fn a_parameterised_spec_validates() {
    assert!(validate_engine_spec(Some("windowed-checkpoint:window=64")).is_ok());
}

#[test]
fn an_unknown_spec_is_rejected_and_says_what_is_supported() {
    let err = validate_engine_spec(Some(UNKNOWN)).expect_err("must reject");
    assert!(
        err.contains(UNKNOWN),
        "the message must name the spec: {err}"
    );
    for name in larql_kv::EngineKind::supported_names() {
        assert!(err.contains(name), "message omits {name}: {err}");
    }
}

#[test]
fn the_uncached_path_refusal_names_the_spec_and_a_way_forward() {
    // A refusal that does not say what to do instead is a dead end; the
    // whole point is that the caller stops getting silent default
    // behaviour and learns where the engines are actually comparable.
    let msg = engine_unsupported_on_uncached_path("markov-rs");
    assert!(msg.contains("markov-rs"), "{msg}");
    assert!(msg.contains("--metal"), "{msg}");
    assert!(msg.contains("larql bench"), "{msg}");
}
