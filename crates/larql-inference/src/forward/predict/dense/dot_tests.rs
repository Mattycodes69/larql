use super::*;

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}

#[test]
fn f32_dot_matches_scalar_on_aligned_length() {
    // 2560 = Gemma 3 4B hidden — clean multiple of 16.
    let a: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.013).sin()).collect();
    let b: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.021).cos()).collect();
    let s = scalar_dot(&a, &b);
    let g = f32_dot(&a, &b);
    // Pairwise-summed NEON ordering vs left-to-right scalar — allow
    // small relative drift.
    let rel = ((s - g).abs() / s.abs().max(1e-6)) as f64;
    assert!(rel < 1e-4, "scalar={s} neon={g}");
}

#[test]
fn f32_dot_handles_unaligned_tail() {
    // 23 is not a multiple of 16 — exercises the scalar tail.
    let a: Vec<f32> = (0..23).map(|i| (i + 1) as f32).collect();
    let b: Vec<f32> = (0..23).map(|i| (i as f32 * 0.5) + 1.0).collect();
    let s = scalar_dot(&a, &b);
    let g = f32_dot(&a, &b);
    assert!((s - g).abs() < 1e-4, "scalar={s} neon={g}");
}

#[test]
fn f32_dot_empty_returns_zero() {
    assert_eq!(f32_dot(&[], &[]), 0.0);
}
