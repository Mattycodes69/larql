use super::cmp_score_desc;
use std::cmp::Ordering;

#[test]
fn cmp_score_desc_orders_descending() {
    assert_eq!(cmp_score_desc(1.0, 2.0), Ordering::Greater);
    assert_eq!(cmp_score_desc(2.0, 1.0), Ordering::Less);
    assert_eq!(cmp_score_desc(1.0, 1.0), Ordering::Equal);
}

#[test]
fn cmp_score_desc_treats_nan_as_equal() {
    assert_eq!(cmp_score_desc(f32::NAN, 1.0), Ordering::Equal);
    assert_eq!(cmp_score_desc(1.0, f32::NAN), Ordering::Equal);
    assert_eq!(cmp_score_desc(f32::NAN, f32::NAN), Ordering::Equal);
}

#[test]
fn sort_with_nan_does_not_panic() {
    // Reproduces the REV1 hazard: a single NaN in the score vector
    // would have panicked the gRPC worker via `partial_cmp().unwrap()`.
    // After the fix, sort completes; we don't claim a total order
    // (NaN-as-Equal breaks strict weak ordering, so the finite values
    // around NaNs may not be globally descending), only that the call
    // is safe and preserves length and finite/NaN counts.
    let mut scores = [3.0f32, f32::NAN, 1.0, 5.0, f32::NAN, 2.0];
    scores.sort_by(|a, b| cmp_score_desc(*a, *b));

    assert_eq!(scores.len(), 6);
    assert_eq!(scores.iter().filter(|s| s.is_nan()).count(), 2);
    let finite: Vec<f32> = scores.iter().copied().filter(|s| !s.is_nan()).collect();
    assert_eq!(finite.len(), 4);
    assert!(finite.contains(&5.0) && finite.contains(&3.0));
}

#[test]
fn sort_descending_when_all_finite() {
    let mut scores = [3.0f32, 1.0, 5.0, 2.0, 4.0];
    scores.sort_by(|a, b| cmp_score_desc(*a, *b));
    assert_eq!(scores, [5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn sort_with_infinities_is_well_defined() {
    let mut scores = [1.0f32, f32::INFINITY, -1.0, f32::NEG_INFINITY, 0.0];
    scores.sort_by(|a, b| cmp_score_desc(*a, *b));
    assert_eq!(scores, [f32::INFINITY, 1.0, 0.0, -1.0, f32::NEG_INFINITY]);
}
