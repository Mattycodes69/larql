//! RATELIMIT MIDDLEWARE

use super::*;

#[tokio::test]
async fn rate_limit_blocks_when_exhausted() {
    // 1/sec → first request with trusted X-Forwarded-For passes, second is rejected.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp1 = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK, "first request should pass");

    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp2 = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "second request should be rate-limited"
    );
}

#[tokio::test]
async fn rate_limit_health_exempt() {
    // Even with a 1/sec limiter exhausted, /v1/health is exempt.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    // Exhaust the limiter for 127.0.0.1 via X-Forwarded-For.
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp1 = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // Verify exhausted on /v1/stats.
    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp2 = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);

    // Health check is exempt — should still pass.
    let app3 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp3 = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/health")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp3.status(),
        StatusCode::OK,
        "/v1/health should be exempt from rate limiting"
    );
}

#[tokio::test]
async fn rate_limit_forwarded_for_header_used_as_ip_when_trusted() {
    // X-Forwarded-For: 10.0.0.1 → uses that IP, different from 10.0.0.2.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let proxy_addr: SocketAddr = "192.0.2.10:443".parse().unwrap();

    // Exhaust 10.0.0.1 bucket.
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let _ = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.1")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // 10.0.0.1 is now blocked.
    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp_blocked = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.1")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_blocked.status(), StatusCode::TOO_MANY_REQUESTS);

    // 10.0.0.2 has its own bucket — should pass.
    let app3 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp_other = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.2")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp_other.status(),
        StatusCode::OK,
        "different IP should have its own bucket"
    );
}

#[tokio::test]
async fn rate_limit_forwarded_for_header_ignored_by_default() {
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.3"] {
        let app = router_with_limiter(Arc::clone(&rl));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .header("x-forwarded-for", ip)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn rate_limit_no_ip_passes_through() {
    // No X-Forwarded-For and no ConnectInfo → middleware has no IP to check.
    // Per the implementation: if ip is None, the check is skipped entirely.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    // Make multiple requests with no IP info — all should pass (no IP → no rate limit applied).
    for _ in 0..3 {
        let app = router_with_limiter(Arc::clone(&rl));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Without an IP, rate_limit_middleware skips the check and passes through.
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "no IP → should pass through even beyond limit"
        );
    }
}
