//! `--api-key` guards the gRPC port exactly as it guards HTTP.

use larql_server::auth::{bearer_matches, grpc_interceptor};

fn call(auth: Option<&str>) -> tonic::Request<()> {
    let mut request = tonic::Request::new(());
    if let Some(value) = auth {
        request
            .metadata_mut()
            .insert("authorization", value.parse().expect("ascii metadata"));
    }
    request
}

#[test]
fn without_a_configured_key_every_call_passes() {
    let mut intercept = grpc_interceptor(None);
    assert!(intercept(call(None)).is_ok());
}

#[test]
fn with_a_key_only_the_matching_bearer_passes() {
    let mut intercept = grpc_interceptor(Some("sk-test".into()));
    assert!(intercept(call(Some("Bearer sk-test"))).is_ok());
    for bad in [
        None,
        Some("Bearer wrong"),
        Some("sk-test"),
        Some("Basic sk-test"),
    ] {
        let status = intercept(call(bad)).unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated, "{bad:?}");
    }
}

#[test]
fn bearer_matching_needs_the_scheme_and_the_exact_token() {
    assert!(bearer_matches(Some("Bearer k"), "k"));
    assert!(!bearer_matches(Some("Bearer k2"), "k"));
    assert!(!bearer_matches(Some("k"), "k"));
    assert!(!bearer_matches(None, "k"));
}
