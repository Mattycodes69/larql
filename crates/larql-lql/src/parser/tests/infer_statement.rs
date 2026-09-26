//! INFER STATEMENT

use super::*;

#[test]
fn parse_infer_minimal() {
    let stmt = parse(r#"INFER "The capital of France is" TOP 5;"#).unwrap();
    match stmt {
        Statement::Infer {
            prompt,
            top,
            compare,
            route,
            generate,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert_eq!(top, Some(5));
            assert!(!compare);
            assert!(route.is_none());
            assert!(generate.is_none());
        }
        _ => panic!("expected Infer"),
    }
}

#[test]
fn parse_infer_with_compare() {
    let stmt = parse(r#"INFER "test prompt" TOP 3 COMPARE;"#).unwrap();
    match stmt {
        Statement::Infer {
            prompt,
            top,
            compare,
            route,
            generate,
        } => {
            let _ = generate;
            assert_eq!(prompt, "test prompt");
            assert_eq!(top, Some(3));
            assert!(compare);
            assert!(route.is_none());
        }
        _ => panic!("expected Infer"),
    }
}

#[test]
fn parse_infer_route_verify() {
    let stmt = parse(r#"INFER "The capital of Atlantis is" ROUTE VERIFY TOP 3;"#).unwrap();
    match stmt {
        Statement::Infer { route, .. } => {
            let r = route.expect("ROUTE VERIFY → Some(route)");
            assert!(!r.fallback);
            assert!(r.topk.is_none());
        }
        _ => panic!("expected Infer"),
    }
}

#[test]
fn parse_infer_route_verify_fallback_topk() {
    let stmt = parse(r#"INFER "The capital of Persia is" ROUTE VERIFY FALLBACK TOPK 8;"#).unwrap();
    match stmt {
        Statement::Infer { route, .. } => {
            let r = route.expect("ROUTE VERIFY FALLBACK → Some(route)");
            assert!(r.fallback);
            assert_eq!(r.topk, Some(8));
        }
        _ => panic!("expected Infer"),
    }
}

#[test]
fn parse_infer_route_verify_exit() {
    // ROUTE VERIFY [TOPK n] EXIT → early-exit flag set, fallback off.
    let stmt =
        parse(r#"INFER "The capital of France is" ROUTE VERIFY TOPK 5 EXIT TOP 3;"#).unwrap();
    match stmt {
        Statement::Infer { route, .. } => {
            let r = route.expect("ROUTE VERIFY … EXIT → Some(route)");
            assert!(r.exit);
            assert!(!r.fallback);
            assert_eq!(r.topk, Some(5));
        }
        _ => panic!("expected Infer"),
    }
}

#[test]
fn parse_infer_route_verify_no_exit_defaults_false() {
    let stmt = parse(r#"INFER "x" ROUTE VERIFY;"#).unwrap();
    match stmt {
        Statement::Infer { route, .. } => assert!(!route.expect("route").exit),
        _ => panic!("expected Infer"),
    }
}

#[test]
fn parse_infer_route_requires_verify() {
    // ROUTE must be followed by VERIFY.
    assert!(parse(r#"INFER "x" ROUTE FALLBACK;"#).is_err());
}

#[test]
fn parse_infer_no_top() {
    let stmt = parse(r#"INFER "test";"#).unwrap();
    match stmt {
        Statement::Infer { top, .. } => assert!(top.is_none()),
        _ => panic!("expected Infer"),
    }
}
