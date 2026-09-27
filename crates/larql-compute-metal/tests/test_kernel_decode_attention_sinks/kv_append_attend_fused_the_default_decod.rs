//! kv_append_attend_fused (the default decode kernel)

use super::*;

#[test]
fn kv_append_attend_fused_with_sinks_matches_cpu_reference() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, ref_k, ref_v) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let gpu = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &ck,
        &cv,
        &nk,
        &nv,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        Some(&SINKS),
        0.0,
    );
    let cpu = cpu_decode_reference(
        &q,
        &ref_k,
        &ref_v,
        t_len,
        num_q,
        num_kv,
        head_dim,
        scale,
        Some(&SINKS),
        0.0,
    );

    let diff = max_diff(&cpu, &gpu);
    assert!(
        diff < MAX_DIFF,
        "kv_append_attend_fused with sinks: max diff {diff}\nCPU: {:?}\nGPU: {:?}",
        &cpu[..8],
        &gpu[..8]
    );
}

#[test]
fn kv_append_attend_fused_without_sinks_is_unchanged() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, ref_k, ref_v) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let gpu = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &ck,
        &cv,
        &nk,
        &nv,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        None,
        0.0,
    );
    let cpu = cpu_decode_reference(
        &q, &ref_k, &ref_v, t_len, num_q, num_kv, head_dim, scale, None, 0.0,
    );

    let diff = max_diff(&cpu, &gpu);
    assert!(
        diff < MAX_DIFF,
        "kv_append_attend_fused without sinks regressed: max diff {diff}"
    );
}

#[test]
fn kv_append_attend_fused_sinks_divert_attention_mass() {
    // A sink that dominates the logits must shrink the output toward
    // zero. Guards against the kernel accepting the binding but never
    // applying it — which the parity test alone could not distinguish
    // if both sides were wrong in the same way.
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, _, _) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let run = |sinks: Option<&[f32]>| {
        run_kv_append_attend_fused(
            &device,
            &lib,
            &q,
            &ck,
            &cv,
            &nk,
            &nv,
            t_len as u32,
            num_q as u32,
            num_kv as u32,
            head_dim as u32,
            scale,
            sinks,
            0.0,
        )
    };
    let plain = run(None);
    let dominated = run(Some(&[40.0, 40.0]));

    let plain_mag: f32 = plain.iter().map(|x| x.abs()).sum();
    let dom_mag: f32 = dominated.iter().map(|x| x.abs()).sum();
    assert!(
        plain_mag > 1e-3,
        "degenerate fixture — no-sink output is already ~zero"
    );
    assert!(
        dom_mag < plain_mag * 0.01,
        "a dominant sink must absorb nearly all mass: {dom_mag} vs {plain_mag}"
    );
}

#[test]
fn kv_append_attend_fused_applies_each_heads_own_sink() {
    // Head 0 gets a negligible sink, head 1 a dominant one. Only head 1
    // should collapse — fails if the kernel indexes `sinks` wrongly or
    // reads a single value for every head.
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (5usize, 2usize, 2usize, 8usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let (q, ck, cv, nk, nv, _, _) = decode_fixture(t_len, num_q, num_kv, head_dim);

    let out = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &ck,
        &cv,
        &nk,
        &nv,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        Some(&[-40.0, 40.0]),
        0.0,
    );
    let head0: f32 = out[..head_dim].iter().map(|x| x.abs()).sum();
    let head1: f32 = out[head_dim..].iter().map(|x| x.abs()).sum();
    assert!(head0 > 1e-3, "head 0 should be unaffected, got {head0}");
    assert!(
        head1 < 1e-3,
        "head 1's dominant sink should zero it, got {head1}"
    );
}
