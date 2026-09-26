//! attn_fused (QK-norm + RoPE + append + attend)

use super::*;

#[test]
fn attn_fused_sink_rescales_each_head_uniformly() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5u32, 2u32, 2u32, 8u32);
    let plain = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, None);
    let sinked = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, Some(&SINKS));

    for head in 0..num_q as usize {
        let r = per_head_ratio(&plain, &sinked, head, head_dim as usize);
        assert!(
            r > 0.0 && r < 1.0,
            "head {head}: a sink must strictly shrink the output, got factor {r}"
        );
    }
}

#[test]
fn attn_fused_larger_sink_diverts_more_mass() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5u32, 2u32, 2u32, 8u32);
    let plain = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, None);
    let small = run_attn_fused(
        &device,
        &lib,
        t_len,
        num_q,
        num_kv,
        head_dim,
        Some(&[0.0, 0.0]),
    );
    let large = run_attn_fused(
        &device,
        &lib,
        t_len,
        num_q,
        num_kv,
        head_dim,
        Some(&[4.0, 4.0]),
    );

    for head in 0..num_q as usize {
        let hd = head_dim as usize;
        let r_small = per_head_ratio(&plain, &small, head, hd);
        let r_large = per_head_ratio(&plain, &large, head, hd);
        assert!(
            r_large < r_small,
            "head {head}: a larger sink must divert more mass ({r_large} vs {r_small})"
        );
    }
}

#[test]
fn attn_fused_applies_each_heads_own_sink() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5u32, 2u32, 2u32, 8u32);
    let plain = run_attn_fused(&device, &lib, t_len, num_q, num_kv, head_dim, None);
    // Head 0 negligible, head 1 dominant.
    let mixed = run_attn_fused(
        &device,
        &lib,
        t_len,
        num_q,
        num_kv,
        head_dim,
        Some(&[-40.0, 40.0]),
    );

    let hd = head_dim as usize;
    let r0 = per_head_ratio(&plain, &mixed, 0, hd);
    assert!(
        (r0 - 1.0).abs() < 1e-3,
        "head 0's negligible sink should leave it unchanged, got factor {r0}"
    );
    let head1: f32 = mixed[hd..].iter().map(|x| x.abs()).sum();
    assert!(
        head1 < 1e-3,
        "head 1's dominant sink should zero it, got magnitude {head1}"
    );
}

#[test]
fn attn_fused_without_sinks_is_unchanged_by_the_new_bindings() {
    // Two no-sink runs must agree exactly: the added bindings must not
    // perturb the default path.
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let a = run_attn_fused(&device, &lib, 5, 2, 2, 8, None);
    let b = run_attn_fused(&device, &lib, 5, 2, 2, 8, Some(&[-1e30, -1e30]));
    let diff = max_diff(&a, &b);
    assert!(
        diff < MAX_DIFF,
        "an unreachably-negative sink must match the no-sink path: max diff {diff}"
    );
}
