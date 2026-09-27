//! kv_attention (the non-fused fallback) — slice 3 (audit F7/F8)

use super::*;

#[test]
fn kv_attention_fallback_with_sinks_matches_cpu_reference() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (7usize, 2usize, 1usize, 32usize);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (q, _ck, _cv, _nk, _nv, ref_k, ref_v) = decode_fixture(t_len, num_q, num_kv, head_dim);
    for kernel in ["kv_attention", "kv_attention_long"] {
        let gpu = run_kv_attention(
            &device,
            &lib,
            kernel,
            &q,
            &ref_k,
            &ref_v,
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
        let d = max_diff(&cpu, &gpu);
        assert!(d < MAX_DIFF, "{kernel} sinks parity: max_diff={d:.3e}");
    }
}

#[test]
fn kv_attention_fallback_with_softcap_matches_cpu_reference() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (7usize, 2usize, 1usize, 32usize);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let softcap = 0.5f32; // small cap so capping is strongly discriminating
    let (q, _ck, _cv, _nk, _nv, ref_k, ref_v) = decode_fixture(t_len, num_q, num_kv, head_dim);
    for kernel in ["kv_attention", "kv_attention_long"] {
        let gpu = run_kv_attention(
            &device,
            &lib,
            kernel,
            &q,
            &ref_k,
            &ref_v,
            t_len as u32,
            num_q as u32,
            num_kv as u32,
            head_dim as u32,
            scale,
            None,
            softcap,
        );
        let cpu = cpu_decode_reference(
            &q, &ref_k, &ref_v, t_len, num_q, num_kv, head_dim, scale, None, softcap,
        );
        let d = max_diff(&cpu, &gpu);
        assert!(d < MAX_DIFF, "{kernel} softcap parity: max_diff={d:.3e}");
        // Control: the instrument must fail on a known-different input —
        // capped output must differ from uncapped, or this test could
        // pass with the cap silently dropped on both sides.
        let uncapped = run_kv_attention(
            &device,
            &lib,
            kernel,
            &q,
            &ref_k,
            &ref_v,
            t_len as u32,
            num_q as u32,
            num_kv as u32,
            head_dim as u32,
            scale,
            None,
            0.0,
        );
        assert!(
            max_diff(&uncapped, &gpu) > 1e-3,
            "{kernel}: softcap=0.5 output identical to uncapped — cap not applied"
        );
    }
}

#[test]
fn kv_append_attend_fused_with_softcap_matches_cpu_reference() {
    let Some((device, lib)) = device_and_lib() else {
        return;
    };
    let (t_len, num_q, num_kv, head_dim) = (6usize, 2usize, 1usize, 32usize);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let softcap = 0.5f32;
    let (q, cache_k, cache_v, new_k, new_v, ref_k, ref_v) =
        decode_fixture(t_len, num_q, num_kv, head_dim);
    let gpu = run_kv_append_attend_fused(
        &device,
        &lib,
        &q,
        &cache_k,
        &cache_v,
        &new_k,
        &new_v,
        t_len as u32,
        num_q as u32,
        num_kv as u32,
        head_dim as u32,
        scale,
        Some(&SINKS),
        softcap,
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
        softcap,
    );
    let d = max_diff(&cpu, &gpu);
    assert!(d < MAX_DIFF, "fused softcap+sinks parity: max_diff={d:.3e}");
}
