use super::*;

#[test]
fn profile_total_ms_sums_buckets() {
    let p = ProfileTimings {
        attn_ms: 1.5,
        gate_up_ms: 2.5,
        down_ms: 1.0,
        ..Default::default()
    };
    assert!((p.total_ms() - 5.0).abs() < 1e-9);
}

#[test]
fn profile_format_summary_handles_zero_total() {
    let p = ProfileTimings::default();
    let s = p.format_summary(34);
    // No NaN-percent panics, total prints as 0.00.
    assert!(s.contains("total=0.00ms"));
    assert!(s.contains("34 layers"));
}

#[test]
fn profile_format_summary_includes_per_layer_average() {
    let p = ProfileTimings {
        attn_ms: 6.0,
        gate_up_ms: 3.0,
        down_ms: 1.0,
        ..Default::default()
    };
    let s = p.format_summary(10);
    assert!(s.contains("total=10.00ms"));
    assert!(s.contains("1.000ms/layer"));
}

/// `format_summary(0)` takes the `else` branch (per_layer = 0.0).
#[test]
fn profile_format_summary_zero_layers_uses_zero_per_layer() {
    let p = ProfileTimings {
        attn_ms: 1.0,
        gate_up_ms: 1.0,
        down_ms: 1.0,
        ..Default::default()
    };
    let s = p.format_summary(0);
    assert!(s.contains("0 layers"));
    assert!(s.contains("0.000ms/layer"));
}

// ── DecodeStateDump ────────────────────────────────────────────

#[test]
fn state_dump_with_capacity_preallocates_three_slots() {
    let dump = DecodeStateDump::with_capacity(8);
    assert_eq!(dump.h_in_per_layer.capacity(), 8);
    assert_eq!(dump.k_new_per_layer.capacity(), 8);
    assert_eq!(dump.v_new_per_layer.capacity(), 8);
    // The buffers are empty until the backend fills them.
    assert!(dump.h_in_per_layer.is_empty());
}

#[test]
fn state_dump_is_complete_for_requires_every_layer_populated() {
    let mut dump = DecodeStateDump::with_capacity(2);
    assert!(!dump.is_complete_for(2));
    dump.h_in_per_layer.push(vec![0.0; 4]);
    dump.k_new_per_layer.push(vec![0.0; 4]);
    dump.v_new_per_layer.push(vec![0.0; 4]);
    assert!(!dump.is_complete_for(2)); // only 1 layer
    dump.h_in_per_layer.push(vec![0.0; 4]);
    dump.k_new_per_layer.push(vec![0.0; 4]);
    dump.v_new_per_layer.push(vec![0.0; 4]);
    assert!(dump.is_complete_for(2));
}

#[test]
fn state_dump_is_complete_under_full_matches_is_complete_for() {
    let mut dump = DecodeStateDump::with_capacity(1);
    dump.h_in_per_layer.push(vec![0.0; 4]);
    // Under Full: K/V also required.
    assert!(!dump.is_complete_under(1, StateDumpMask::Full));
    dump.k_new_per_layer.push(vec![0.0; 4]);
    dump.v_new_per_layer.push(vec![0.0; 4]);
    assert!(dump.is_complete_under(1, StateDumpMask::Full));
}

#[test]
fn state_dump_is_complete_under_h_only_ignores_kv() {
    let mut dump = DecodeStateDump::with_capacity(1);
    dump.h_in_per_layer.push(vec![0.0; 4]);
    // K/V never populated, but HOnly is satisfied by h_in alone.
    assert!(dump.is_complete_under(1, StateDumpMask::HOnly));
    // But still not Full.
    assert!(!dump.is_complete_under(1, StateDumpMask::Full));
}

#[test]
fn state_dump_is_complete_under_none_is_trivially_true() {
    let dump = DecodeStateDump::default();
    // Nothing populated; None mask doesn't require anything.
    assert!(dump.is_complete_under(8, StateDumpMask::None));
}

#[test]
fn state_dump_mask_default_is_full() {
    assert_eq!(StateDumpMask::default(), StateDumpMask::Full);
}

// ── DecodeBackend trait defaults ──────────────────────────────────
//
// A minimal stub that uses every default. Covers the trait's
// default bodies — `None` returns, no-op writes, the delegating
// defaults (e.g. `full_pipeline_q4_with_head_replacement` →
// `full_pipeline_q4`).

struct StubDecode;
impl DecodeBackend for StubDecode {}

fn stub_layers() -> Vec<crate::FullPipelineLayer<'static>> {
    vec![crate::FullPipelineLayer::default()]
}

#[test]
fn default_full_pipeline_q4_returns_none() {
    let b = StubDecode;
    let layers = stub_layers();
    let r = b.full_pipeline_q4(&layers, &[0.0; 4], 4, 4, 1, false, 0.0);
    assert!(r.is_none());
}

#[test]
fn default_full_pipeline_q4_with_head_replacement_delegates_to_no_intervention() {
    let b = StubDecode;
    let layers = stub_layers();
    // Default delegates to `full_pipeline_q4` (which is `None`).
    let r = b.full_pipeline_q4_with_head_replacement(
        &layers, &[0.0; 4], 4, 4, 1, false, 0.0, 0, 0, &[0.0; 2],
    );
    assert!(r.is_none());
}

#[test]
fn default_multi_layer_q4_ffn_returns_none() {
    let b = StubDecode;
    let r = b.multi_layer_q4_ffn(&[], &[0.0; 4], 4, 4);
    assert!(r.is_none());
}

#[test]
fn default_has_kv_cache_returns_false() {
    let b = StubDecode;
    assert!(!b.has_kv_cache());
}

#[test]
fn default_kv_cache_helpers_are_no_ops() {
    let b = StubDecode;
    // None of these should panic; nothing observable changes.
    b.populate_kv_layer(0, &[0.0; 4], &[0.0; 4], 1, 2, 2);
    b.reset_kv_cache();
    b.truncate_kv_cache(0);
    b.preallocate_kv_cache_per_layer(&[(2, 4)], 16);
    assert_eq!(b.kv_cache_len(), 0);
}

/// The capacity-aware variant falls back to the uniform allocation
/// using the *largest* requested capacity. Taking the max rather than
/// the min matters: a backend that ignores per-layer capacity must
/// over-allocate, never under-allocate — too small is an overrun,
/// too large is only waste.
#[test]
fn default_capacity_preallocate_falls_back_to_the_largest_capacity() {
    let b = StubDecode;
    // Mixed sliding/global capacities, as a real model produces.
    b.preallocate_kv_cache_per_layer_with_capacity(&[(2, 4), (2, 4)], &[2048, 4096]);
    // Empty input must not panic on the `max` of an empty iterator.
    b.preallocate_kv_cache_per_layer_with_capacity(&[], &[]);
    assert_eq!(b.kv_cache_len(), 0);
}

#[test]
fn default_decode_token_returns_none() {
    let b = StubDecode;
    let layers = stub_layers();
    let r = b.decode_token(&layers, &[0.0; 4], 4, 4);
    assert!(r.is_none());
}

#[test]
fn default_decode_token_with_state_dump_delegates_to_decode_token() {
    let b = StubDecode;
    let layers = stub_layers();
    let mut dump = DecodeStateDump::default();
    let r = b.decode_token_with_state_dump(&layers, &[0.0; 4], 4, 4, Some(&mut dump));
    assert!(r.is_none());
    // Default doesn't populate the dump.
    assert!(dump.h_in_per_layer.is_empty());
}

#[test]
fn default_decode_token_with_state_dump_masked_delegates_through() {
    let b = StubDecode;
    let layers = stub_layers();
    for mask in [
        StateDumpMask::Full,
        StateDumpMask::HOnly,
        StateDumpMask::None,
    ] {
        let r = b.decode_token_with_state_dump_masked(&layers, &[0.0; 4], 4, 4, None, mask);
        assert!(r.is_none(), "mask {mask:?} should produce None");
    }
}

#[test]
fn default_decode_token_with_moe_ignores_hook() {
    let b = StubDecode;
    let layers = stub_layers();
    let mut hook_called = 0;
    let mut moe_fn = |_l: usize, _h: &[f32]| {
        hook_called += 1;
        vec![0.0; 4]
    };
    let r = b.decode_token_with_moe(&layers, &[0.0; 4], 4, 4, &mut moe_fn);
    assert!(r.is_none());
    // Default delegates to `decode_token` (None) before reaching layers,
    // so the MoE hook is never called.
    assert_eq!(hook_called, 0);
}

#[test]
fn default_decode_token_q4k_moe_returns_none() {
    let b = StubDecode;
    let layers = stub_layers();
    let get_expert = |_l: usize, _e: usize| -> Option<(&[u8], &[u8])> { None };
    let r = b.decode_token_q4k_moe(&layers, &[0.0; 4], 4, 4, 1e-6, &get_expert);
    assert!(r.is_none());
}

#[test]
fn default_decode_token_with_moe_split_combines_pair() {
    let b = StubDecode;
    let layers = stub_layers();
    let mut fire = |_l: usize, _h: &[f32]| {};
    let mut collect = |_l: usize| vec![0.0; 4];
    let r = b.decode_token_with_moe_split(&layers, &[0.0; 4], 4, 4, &mut fire, &mut collect);
    assert!(r.is_none());
}

#[test]
fn default_decode_token_split_profile_returns_zero_timings() {
    let b = StubDecode;
    let layers = stub_layers();
    let (r, attn, gu, dn) = b.decode_token_split_profile(&layers, &[0.0; 4], 4, 4);
    assert!(r.is_none());
    assert_eq!(attn, 0.0);
    assert_eq!(gu, 0.0);
    assert_eq!(dn, 0.0);
}

#[test]
fn default_prefill_kquant_returns_none() {
    let b = StubDecode;
    let layers = stub_layers();
    let r = b.prefill_kquant(&layers, &[0.0; 4], 4, 4, 1, false, 0.0);
    assert!(r.is_none());
}

#[test]
fn default_full_pipeline_kquant_capture_pre_wo_returns_none() {
    let b = StubDecode;
    let layers = stub_layers();
    let r = b.full_pipeline_kquant_capture_pre_wo(&layers, &[0.0; 4], 4, 4, 1, false, 0.0, 0, 0);
    assert!(r.is_none());
}

#[test]
fn default_prefill_kquant_with_head_replacement_delegates_to_no_intervention() {
    let b = StubDecode;
    let layers = stub_layers();
    let r = b.prefill_kquant_with_head_replacement(
        &layers, &[0.0; 4], 4, 4, 1, false, 0.0, 0, 0, &[0.0; 2],
    );
    // Default delegates to `prefill_kquant` (None).
    assert!(r.is_none());
}
