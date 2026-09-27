//! arithmetic

use super::*;

#[test]
fn arithmetic_add() {
    assert_eq_expert("arithmetic", "add", json!({"a": 12, "b": 34}), json!(46.0));
}

#[test]
fn arithmetic_subtract() {
    assert_eq_expert("arithmetic", "sub", json!({"a": 100, "b": 37}), json!(63.0));
}

#[test]
fn arithmetic_multiply() {
    assert_eq_expert("arithmetic", "mul", json!({"a": 7, "b": 8}), json!(56.0));
}

#[test]
fn arithmetic_divide() {
    assert_eq_expert("arithmetic", "div", json!({"a": 144, "b": 12}), json!(12.0));
}

#[test]
fn arithmetic_divide_by_zero() {
    assert_eq_expert("arithmetic", "div", json!({"a": 1, "b": 0}), Value::Null);
}

#[test]
fn arithmetic_power() {
    assert_eq_expert("arithmetic", "pow", json!({"a": 2, "b": 10}), json!(1024.0));
}

#[test]
fn arithmetic_mod() {
    assert_eq_expert("arithmetic", "mod", json!({"a": 17, "b": 5}), json!(2));
}

#[test]
fn arithmetic_prime_true() {
    assert_eq_expert("arithmetic", "is_prime", json!({"n": 17}), json!(true));
}

#[test]
fn arithmetic_prime_false() {
    assert_eq_expert("arithmetic", "is_prime", json!({"n": 15}), json!(false));
}

#[test]
fn arithmetic_gcd() {
    assert_eq_expert("arithmetic", "gcd", json!({"a": 48, "b": 18}), json!(6));
}

#[test]
fn arithmetic_lcm() {
    assert_eq_expert("arithmetic", "lcm", json!({"a": 4, "b": 6}), json!(12));
}

#[test]
fn arithmetic_factorial() {
    assert_eq_expert("arithmetic", "factorial", json!({"n": 5}), json!(120));
}

#[test]
fn arithmetic_binary() {
    assert_eq_expert(
        "arithmetic",
        "to_base",
        json!({"n": 255, "base": 2}),
        json!("11111111"),
    );
}

#[test]
fn arithmetic_hex() {
    assert_eq_expert(
        "arithmetic",
        "to_base",
        json!({"n": 255, "base": 16}),
        json!("FF"),
    );
}

#[test]
fn arithmetic_roman_from() {
    assert_eq_expert("arithmetic", "from_roman", json!({"s": "XIV"}), json!(14));
}

#[test]
fn arithmetic_roman_to() {
    assert_eq_expert("arithmetic", "to_roman", json!({"n": 42}), json!("XLII"));
}

#[test]
fn arithmetic_percent_of() {
    assert_eq_expert(
        "arithmetic",
        "percent_of",
        json!({"pct": 20, "n": 150}),
        json!(30.0),
    );
}

#[test]
fn arithmetic_unknown_op() {
    // Unknown op should return None (expert declines).
    assert!(call("arithmetic", "flibbertigibbet", json!({})).is_none());
}
