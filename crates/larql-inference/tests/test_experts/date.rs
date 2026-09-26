//! date
//! unit

use super::*;

#[test]
fn date_days_between() {
    assert_eq_expert(
        "date",
        "days_between",
        json!({"from": {"year": 2023, "month": 3, "day": 15}, "to": {"year": 2023, "month": 3, "day": 20}}),
        json!(5),
    );
}

#[test]
fn date_days_between_year() {
    assert_eq_expert(
        "date",
        "days_between",
        json!({"from": {"year": 2023, "month": 1, "day": 1}, "to": {"year": 2024, "month": 1, "day": 1}}),
        json!(365),
    );
}

#[test]
fn date_day_of_week_wednesday() {
    // 25 December 2024 was a Wednesday (ISO index 3).
    assert_eq_expert(
        "date",
        "day_of_week",
        json!({"date": {"year": 2024, "month": 12, "day": 25}}),
        json!(3),
    );
}

#[test]
fn date_add_days() {
    assert_eq_expert(
        "date",
        "add_days",
        json!({"date": {"year": 2025, "month": 1, "day": 1}, "days": 10}),
        json!({"year": 2025, "month": 1, "day": 11}),
    );
}

#[test]
fn date_subtract_days() {
    assert_eq_expert(
        "date",
        "add_days",
        json!({"date": {"year": 2023, "month": 3, "day": 10}, "days": -5}),
        json!({"year": 2023, "month": 3, "day": 5}),
    );
}

#[test]
fn date_leap_year_true() {
    assert_eq_expert("date", "is_leap_year", json!({"year": 2024}), json!(true));
}

#[test]
fn date_leap_year_false() {
    assert_eq_expert("date", "is_leap_year", json!({"year": 2023}), json!(false));
}

#[test]
fn date_days_in_feb_leap() {
    assert_eq_expert(
        "date",
        "days_in_month",
        json!({"year": 2024, "month": 2}),
        json!(29),
    );
}

#[test]
fn date_days_in_feb_normal() {
    assert_eq_expert(
        "date",
        "days_in_month",
        json!({"year": 2023, "month": 2}),
        json!(28),
    );
}

#[test]
fn date_weeks_between() {
    assert_eq_expert(
        "date",
        "weeks_between",
        json!({"from": {"year": 2024, "month": 1, "day": 1}, "to": {"year": 2025, "month": 1, "day": 1}}),
        json!(52),
    );
}

#[test]
fn unit_km_to_m() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 5, "from": "km", "to": "m"}),
        5000.0,
        1e-6,
    );
}

#[test]
fn unit_miles_to_km() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 10, "from": "mi", "to": "km"}),
        16.0934,
        1e-3,
    );
}

#[test]
fn unit_kg_to_lbs() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 70, "from": "kg", "to": "lb"}),
        154.32,
        0.5,
    );
}

#[test]
fn unit_celsius_to_fahrenheit() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 100, "from": "C", "to": "F"}),
        212.0,
        1e-6,
    );
}

#[test]
fn unit_inches_to_cm() {
    assert_approx(
        "unit",
        "convert",
        json!({"value": 12, "from": "in", "to": "cm"}),
        30.48,
        1e-6,
    );
}

#[test]
fn unit_incompatible_groups() {
    // length to mass => explicit null, not None (expert does handle the op).
    assert_eq_expert(
        "unit",
        "convert",
        json!({"value": 1, "from": "km", "to": "kg"}),
        Value::Null,
    );
}
