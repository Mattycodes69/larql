use super::*;
use crate::{Cli, Commands};
use clap::Parser;

fn forwarded(argv: &[&str]) -> Vec<String> {
    let cli = Cli::try_parse_from(["larql", "serve"].iter().chain(argv)).unwrap();
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve")
    };
    serve_command_args(&args).unwrap()
}

/// Server flags reach the server exactly as typed, including ones the
/// old mirror never declared and pairs it could not express.
#[test]
fn flags_are_forwarded_verbatim() {
    let argv = [
        "--dir",
        "/models",
        "--hnsw",
        "--warmup-hnsw",
        "--v3-backend",
        "metal",
        "--session-ttl-secs=60",
        "--port",
        "9000",
    ];
    assert_eq!(forwarded(&argv), argv);
}

/// Nothing is added: the server's defaults apply, not copies of them.
#[test]
fn no_arguments_forward_nothing() {
    assert!(forwarded(&[]).is_empty());
}

/// `--help` and `-h` go to the server, which owns the flag list.
#[test]
fn help_is_forwarded_to_the_server() {
    assert_eq!(forwarded(&["--help"]), ["--help"]);
    assert_eq!(forwarded(&["-h"]), ["-h"]);
}

/// A leading non-flag argument is the model, and it is resolved.
#[test]
fn a_leading_model_is_resolved_and_the_rest_forwarded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_str().unwrap();
    let out = forwarded(&[path, "--port", "9000"]);
    assert_eq!(out[1..], ["--port", "9000"]);
    assert_eq!(
        out[0],
        super::super::serve_resolve::resolve_serve_target(path).unwrap()
    );
}

/// A resolution failure is an error, not a silently unresolved name.
#[test]
fn an_unresolvable_model_is_an_error() {
    let cli = Cli::try_parse_from([
        "larql",
        "serve",
        "definitely-not-a-cached-vindex-serve-test",
    ])
    .unwrap();
    let Commands::Serve(args) = cli.command else {
        panic!("expected serve")
    };
    assert!(serve_command_args(&args).is_err());
}

/// Only a leading argument is treated as the model; later non-flag
/// arguments are flag values and are never resolved.
#[test]
fn only_the_first_argument_can_be_the_model() {
    let argv = ["--layers", "0-14", "--join", "grpc://router:50052"];
    assert_eq!(forwarded(&argv), argv);
}
