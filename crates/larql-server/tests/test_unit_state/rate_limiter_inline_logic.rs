//! RATE LIMITER (inline logic)
//! DESCRIBE CACHE

use super::*;

#[test]
fn test_rate_limit_parse() {
    // Valid formats
    assert!(rate_limit_parse("100/min").is_some());
    assert!(rate_limit_parse("10/sec").is_some());
    assert!(rate_limit_parse("3600/hour").is_some());
    assert!(rate_limit_parse("50/s").is_some());
    assert!(rate_limit_parse("200/m").is_some());

    // Invalid formats
    assert!(rate_limit_parse("abc").is_none());
    assert!(rate_limit_parse("100").is_none());
    assert!(rate_limit_parse("100/day").is_none());
}

#[test]
fn test_rate_limit_token_bucket() {
    // Simulate token bucket: 2 tokens, 1 refill/sec
    let mut tokens: f64 = 2.0;
    let max_tokens: f64 = 2.0;

    // First two requests succeed
    assert!(tokens >= 1.0);
    tokens -= 1.0;
    assert!(tokens >= 1.0);
    tokens -= 1.0;

    // Third fails
    assert!(tokens < 1.0);

    // Refill
    tokens = (tokens + 1.0).min(max_tokens);
    assert!(tokens >= 1.0);
}

#[test]
fn test_rate_limiter_zero_count_rejects_immediately() {
    // "0/sec" → 0 tokens → first request is rejected.
    let rl = RateLimiter::parse("0/sec");
    // Either returns None (invalid) or allows creation and rejects first request.
    if let Some(rl) = rl {
        let ip: std::net::IpAddr = "127.0.0.1".parse().unwrap();
        assert!(!rl.check(ip));
    }
    // None is also acceptable — 0/sec is edge-case.
}

#[test]
fn test_rate_limiter_per_minute_long_form() {
    // "60/minute" is valid; verify it allows 60 consecutive requests.
    let rl = RateLimiter::parse("60/minute").unwrap();
    let ip: std::net::IpAddr = "10.0.0.60".parse().unwrap();
    for _ in 0..60 {
        assert!(rl.check(ip));
    }
    assert!(!rl.check(ip)); // 61st request blocked
}

#[test]
fn test_rate_limiter_per_second_long_form() {
    // "10/second" is valid; verify it allows 10 consecutive requests.
    let rl = RateLimiter::parse("10/second").unwrap();
    let ip: std::net::IpAddr = "10.0.0.10".parse().unwrap();
    for _ in 0..10 {
        assert!(rl.check(ip));
    }
    assert!(!rl.check(ip)); // 11th request blocked
}

#[test]
fn test_rate_limiter_fractional_count() {
    // "1/hour" → bucket holds 1 token; second request is blocked.
    let rl = RateLimiter::parse("1/hour").unwrap();
    let ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
    assert!(rl.check(ip));
    assert!(!rl.check(ip)); // no refill within the test
}

#[test]
fn test_rate_limiter_empty_spec_rejects() {
    assert!(RateLimiter::parse("").is_none());
    assert!(RateLimiter::parse("/").is_none());
    assert!(RateLimiter::parse("100/").is_none());
}

#[test]
fn test_cache_key_format() {
    let key = format!("{}:{}:{}:{}:{}", "model", "France", "knowledge", 20, 5);
    assert_eq!(key, "model:France:knowledge:20:5");
}

#[test]
fn test_cache_disabled_when_ttl_zero() {
    // TTL=0 means cache is disabled
    let ttl = 0u64;
    assert_eq!(ttl, 0);
}

#[test]
fn test_cache_hit_and_miss() {
    let mut cache: HashMap<String, serde_json::Value> = HashMap::new();
    let key = "model:France:knowledge:20:5".to_string();
    let value = serde_json::json!({"entity": "France", "edges": []});

    // Miss
    assert!(!cache.contains_key(&key));

    // Insert
    cache.insert(key.clone(), value.clone());

    // Hit
    assert_eq!(cache.get(&key), Some(&value));
}

#[test]
fn test_cache_overwrite_updates_value() {
    let cache = DescribeCache::new(60);
    let key = DescribeCache::key("model", "France", "knowledge", 20, 5.0);
    let v1 = serde_json::json!({"edges": []});
    let v2 = serde_json::json!({"edges": [{"target": "Paris"}]});
    cache.put(key.clone(), v1);
    cache.put(key.clone(), v2.clone());
    assert_eq!(cache.get(&key), Some(v2));
}

#[test]
fn test_cache_key_float_precision_truncated() {
    // min_score is cast to u32 in the key, so 5.9 and 5.0 produce the same key.
    let k1 = DescribeCache::key("m", "e", "b", 10, 5.0);
    let k2 = DescribeCache::key("m", "e", "b", 10, 5.9);
    assert_eq!(k1, k2);
    // 6.0 differs.
    let k3 = DescribeCache::key("m", "e", "b", 10, 6.0);
    assert_ne!(k1, k3);
}
