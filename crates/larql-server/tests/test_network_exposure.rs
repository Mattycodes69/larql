//! The server binds loopback by default and refuses to start
//! unauthenticated on a network address unless explicitly told to.

use clap::Parser;
use larql_server::bootstrap::{Cli, DEFAULT_HOST};

fn cli(args: &[&str]) -> Cli {
    let mut argv = vec!["larql-server"];
    argv.extend_from_slice(args);
    Cli::try_parse_from(argv).expect("arguments parse")
}

#[test]
fn the_default_bind_is_loopback_and_needs_no_key() {
    let c = cli(&[]);
    assert_eq!(c.host, DEFAULT_HOST);
    assert!(c.check_network_exposure().is_ok());
}

#[test]
fn an_unauthenticated_network_bind_is_refused() {
    for host in ["0.0.0.0", "::", "192.168.1.10", "myhost.local"] {
        let err = cli(&["--host", host]).check_network_exposure().unwrap_err();
        assert!(
            err.to_string().contains("--insecure-public"),
            "{host}: {err}"
        );
    }
}

#[test]
fn a_network_bind_is_allowed_with_a_key_or_an_explicit_opt_in() {
    assert!(cli(&["--host", "0.0.0.0", "--api-key", "k"])
        .check_network_exposure()
        .is_ok());
    assert!(cli(&["--host", "0.0.0.0", "--insecure-public"])
        .check_network_exposure()
        .is_ok());
    assert!(cli(&["--host", "::1"]).check_network_exposure().is_ok());
}

#[test]
fn the_read_only_public_explorer_profile_may_bind_publicly() {
    assert!(cli(&["--host", "0.0.0.0", "--public-explorer"])
        .check_network_exposure()
        .is_ok());
}
