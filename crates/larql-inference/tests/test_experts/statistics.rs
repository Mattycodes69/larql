//! statistics
//! geometry
//! trig (radians)
//! string_ops
//! hash
//! logic
//! finance
//! element
//! http_status
//! isbn

use super::*;

#[test]
fn statistics_mean() {
    assert_approx(
        "statistics",
        "mean",
        json!({"values": [1,2,3,4,5]}),
        3.0,
        1e-12,
    );
}

#[test]
fn statistics_median_odd() {
    assert_approx(
        "statistics",
        "median",
        json!({"values": [1,3,5,7,9]}),
        5.0,
        1e-12,
    );
}

#[test]
fn statistics_median_even() {
    assert_approx(
        "statistics",
        "median",
        json!({"values": [1,2,3,4]}),
        2.5,
        1e-12,
    );
}

#[test]
fn statistics_mode() {
    assert_eq_expert(
        "statistics",
        "mode",
        json!({"values": [1,2,2,3,3,3]}),
        json!([3.0]),
    );
}

#[test]
fn statistics_min() {
    assert_approx(
        "statistics",
        "min",
        json!({"values": [4,2,9,1,7]}),
        1.0,
        1e-12,
    );
}

#[test]
fn statistics_max() {
    assert_approx(
        "statistics",
        "max",
        json!({"values": [4,2,9,1,7]}),
        9.0,
        1e-12,
    );
}

#[test]
fn statistics_sort() {
    assert_eq_expert(
        "statistics",
        "sort",
        json!({"values": [5,2,8,1]}),
        json!([1.0, 2.0, 5.0, 8.0]),
    );
}

#[test]
fn statistics_count() {
    assert_eq_expert(
        "statistics",
        "count",
        json!({"values": [1,2,3,4,5]}),
        json!(5),
    );
}

#[test]
fn statistics_stddev() {
    // Population stddev of [2,4,4,4,5,5,7,9] is exactly 2.
    assert_approx(
        "statistics",
        "stddev",
        json!({"values": [2,4,4,4,5,5,7,9]}),
        2.0,
        1e-12,
    );
}

#[test]
fn geometry_circle_area() {
    assert_approx(
        "geometry",
        "circle_area",
        json!({"r": 10}),
        std::f64::consts::PI * 100.0,
        1e-9,
    );
}

#[test]
fn geometry_sphere_volume() {
    assert_approx(
        "geometry",
        "sphere_volume",
        json!({"r": 5}),
        4.0 / 3.0 * std::f64::consts::PI * 125.0,
        1e-9,
    );
}

#[test]
fn geometry_triangle_area() {
    assert_approx(
        "geometry",
        "triangle_area_bh",
        json!({"base": 10, "height": 6}),
        30.0,
        1e-12,
    );
}

#[test]
fn geometry_rectangle_perimeter() {
    assert_approx(
        "geometry",
        "rectangle_perimeter",
        json!({"l": 5, "w": 8}),
        26.0,
        1e-12,
    );
}

#[test]
fn geometry_hypotenuse() {
    assert_approx(
        "geometry",
        "hypotenuse",
        json!({"a": 3, "b": 4}),
        5.0,
        1e-12,
    );
}

#[test]
fn trig_sin_pi_6() {
    assert_approx(
        "trig",
        "sin",
        json!({"x": std::f64::consts::FRAC_PI_6}),
        0.5,
        1e-12,
    );
}

#[test]
fn trig_cos_zero() {
    assert_approx("trig", "cos", json!({"x": 0}), 1.0, 1e-12);
}

#[test]
fn trig_tan_pi_4() {
    assert_approx(
        "trig",
        "tan",
        json!({"x": std::f64::consts::FRAC_PI_4}),
        1.0,
        1e-12,
    );
}

#[test]
fn trig_asin_half() {
    assert_approx(
        "trig",
        "asin",
        json!({"x": 0.5}),
        std::f64::consts::FRAC_PI_6,
        1e-12,
    );
}

#[test]
fn trig_deg_to_rad() {
    assert_approx(
        "trig",
        "deg_to_rad",
        json!({"deg": 90}),
        std::f64::consts::FRAC_PI_2,
        1e-12,
    );
}

#[test]
fn string_ops_reverse() {
    assert_eq_expert(
        "string_ops",
        "reverse",
        json!({"s": "hello"}),
        json!("olleh"),
    );
}

#[test]
fn string_ops_palindrome_true() {
    assert_eq_expert(
        "string_ops",
        "is_palindrome",
        json!({"s": "racecar"}),
        json!(true),
    );
}

#[test]
fn string_ops_palindrome_false() {
    assert_eq_expert(
        "string_ops",
        "is_palindrome",
        json!({"s": "hello"}),
        json!(false),
    );
}

#[test]
fn string_ops_anagram_true() {
    assert_eq_expert(
        "string_ops",
        "is_anagram",
        json!({"a": "listen", "b": "silent"}),
        json!(true),
    );
}

#[test]
fn string_ops_anagram_false() {
    assert_eq_expert(
        "string_ops",
        "is_anagram",
        json!({"a": "hello", "b": "world"}),
        json!(false),
    );
}

#[test]
fn string_ops_caesar() {
    assert_eq_expert(
        "string_ops",
        "caesar",
        json!({"s": "abc", "shift": 1}),
        json!("bcd"),
    );
}

#[test]
fn string_ops_uppercase() {
    assert_eq_expert(
        "string_ops",
        "uppercase",
        json!({"s": "hello"}),
        json!("HELLO"),
    );
}

#[test]
fn hash_base64_encode() {
    assert_eq_expert(
        "hash",
        "base64_encode",
        json!({"s": "hello"}),
        json!("aGVsbG8="),
    );
}

#[test]
fn hash_base64_decode() {
    assert_eq_expert(
        "hash",
        "base64_decode",
        json!({"s": "aGVsbG8="}),
        json!("hello"),
    );
}

#[test]
fn hash_hex_encode() {
    assert_eq_expert(
        "hash",
        "hex_encode",
        json!({"s": "test"}),
        json!("74657374"),
    );
}

#[test]
fn hash_url_encode() {
    assert_eq_expert(
        "hash",
        "url_encode",
        json!({"s": "hello world"}),
        json!("hello%20world"),
    );
}

#[test]
fn hash_fnv() {
    if let Some(v) = call("hash", "fnv1a_32", json!({"s": "key"})) {
        assert!(v.as_str().unwrap_or("").starts_with("0x"));
    }
}

#[test]
fn logic_eval_and() {
    assert_eq_expert(
        "logic",
        "eval",
        json!({"expr": "A AND B", "assignments": {"A": true, "B": false}}),
        json!(false),
    );
}

#[test]
fn logic_tautology() {
    assert_eq_expert(
        "logic",
        "classify",
        json!({"expr": "A OR NOT A"}),
        json!("tautology"),
    );
}

#[test]
fn logic_contradiction() {
    assert_eq_expert(
        "logic",
        "classify",
        json!({"expr": "A AND NOT A"}),
        json!("contradiction"),
    );
}

#[test]
fn logic_contingent() {
    assert_eq_expert(
        "logic",
        "classify",
        json!({"expr": "A OR B"}),
        json!("contingent"),
    );
}

#[test]
fn logic_truth_table_rows() {
    if let Some(v) = call("logic", "truth_table", json!({"expr": "A AND B"})) {
        let rows = v
            .get("rows")
            .and_then(|r| r.as_array())
            .expect("rows array");
        assert_eq!(rows.len(), 4);
    }
}

#[test]
fn finance_future_value() {
    assert_approx(
        "finance",
        "future_value",
        json!({"pv": 1000, "rate_pct": 5, "years": 10}),
        1628.89,
        1.0,
    );
}

#[test]
fn finance_compound_interest() {
    assert_approx(
        "finance",
        "compound_interest",
        json!({"principal": 1000, "rate_pct": 10, "years": 1}),
        100.0,
        1e-9,
    );
}

#[test]
fn finance_kelly() {
    assert_approx("finance", "kelly", json!({"p": 0.6, "b": 2}), 0.4, 1e-9);
}

#[test]
fn finance_roi() {
    assert_approx(
        "finance",
        "roi",
        json!({"gain": 120, "cost": 100}),
        0.20,
        1e-9,
    );
}

#[test]
fn finance_npv() {
    // -1000 + 400/1.1 + 400/1.1² + 400/1.1³ ≈ -5.26
    if let Some(v) = call(
        "finance",
        "npv",
        json!({"cash_flows": [-1000, 400, 400, 400], "discount_pct": 10}),
    ) {
        let got = v.as_f64().expect("number");
        assert!((got + 5.26).abs() < 1.0, "got {got}");
    }
}

#[test]
fn element_atomic_number() {
    assert_field(
        "element",
        "by_name",
        json!({"name": "oxygen"}),
        "z",
        json!(8),
    );
}

#[test]
fn element_symbol() {
    assert_field(
        "element",
        "by_name",
        json!({"name": "carbon"}),
        "symbol",
        json!("C"),
    );
}

#[test]
fn element_name_by_number() {
    assert_field(
        "element",
        "by_number",
        json!({"z": 79}),
        "name",
        json!("gold"),
    );
}

#[test]
fn element_mass() {
    if let Some(v) = call("element", "by_name", json!({"name": "hydrogen"})) {
        let mass = v.get("mass").and_then(|m| m.as_f64()).expect("mass");
        assert!((mass - 1.008).abs() < 1e-3);
    }
}

#[test]
fn http_status_404() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 404}),
        "reason",
        json!("Not Found"),
    );
}

#[test]
fn http_status_200() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 200}),
        "reason",
        json!("OK"),
    );
}

#[test]
fn http_status_500() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 500}),
        "reason",
        json!("Internal Server Error"),
    );
}

#[test]
fn http_status_301() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 301}),
        "reason",
        json!("Moved Permanently"),
    );
}

#[test]
fn http_status_403_category() {
    assert_field(
        "http_status",
        "lookup",
        json!({"code": 403}),
        "category",
        json!("4xx"),
    );
}

#[test]
fn http_status_unknown() {
    // 999 is not a real code — expert declines.
    assert!(call("http_status", "lookup", json!({"code": 999})).is_none());
}

#[test]
fn isbn_valid_13() {
    assert_field(
        "isbn",
        "validate",
        json!({"isbn": "978-0-596-52068-7"}),
        "valid",
        json!(true),
    );
}

#[test]
fn isbn_valid_10() {
    assert_field(
        "isbn",
        "validate",
        json!({"isbn": "0-306-40615-2"}),
        "valid",
        json!(true),
    );
}

#[test]
fn isbn_invalid() {
    assert_field(
        "isbn",
        "validate",
        json!({"isbn": "978-0-000-00000-0"}),
        "valid",
        json!(false),
    );
}
