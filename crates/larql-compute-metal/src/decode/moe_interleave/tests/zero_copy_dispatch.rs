use super::*;

/// Every `try_inline_zero_copy_moe` precondition satisfied: pure-MoE
/// layer (no dense FFN branch, via `FullPipelineLayer::default()`'s
/// empty `up`/`down` weights), identity-combine routing policy
/// (`top_k_softmax`'s `post_expert_norm: None`, `layer_scalar: 0.0`,
/// no combined-output norm), no diagnostic captures, and every
/// expert's bytes pre-registered as a zero-copy region. This is the
/// merged-CB fast path `handle_moe_interleave` takes when the
/// backend's expert scratch is live — never reached by the
/// staged-path tests in `moe_dispatch.rs`/the integration suite,
/// which all use the hybrid (dense+MoE) or default routing-policy
/// shape instead.
#[test]
fn try_inline_zero_copy_moe_encodes_experts_and_combine_on_registered_region() {
    let m = backend();
    let hidden = 256usize;
    let inter = 256usize;
    let top_k = 2usize;
    let num_experts = 4usize;

    let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

    // Lay every expert out contiguously in one page-aligned anonymous
    // mmap, exactly the production `register_weight_region` contract.
    let total: usize = expert_gu
        .iter()
        .zip(expert_down.iter())
        .map(|(g, d)| g.len() + d.len())
        .sum();
    let mut region = memmap2::MmapMut::map_anon(total).expect("anon mmap");
    let mut offsets = Vec::with_capacity(num_experts);
    let mut cursor = 0usize;
    for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
        region[cursor..cursor + g.len()].copy_from_slice(g);
        let g_off = cursor;
        cursor += g.len();
        region[cursor..cursor + d.len()].copy_from_slice(d);
        offsets.push((g_off, g.len(), cursor, d.len()));
        cursor += d.len();
    }
    let region = region.make_read_only().expect("read-only mmap");
    assert!(
        m.bufs.register_region(&region[..]),
        "page-aligned anon mmap must register"
    );

    // `moe.experts_gate_up`/`experts_down` MUST be slices into the
    // registered `region`, not the original `expert_gu`/`expert_down`
    // vectors those bytes were copied from — those still live at a
    // different, unregistered address. Passing the originals here was
    // the actual bug this test spent several CI round-trips finding:
    // every precondition matched, but `resolve_selected_experts`
    // still failed because `moe`'s own byte slices didn't point into
    // the region `register_region` was called on, so `resolve_region`
    // correctly reported no match for either selected expert.
    let experts_gate_up: Vec<&[u8]> = offsets
        .iter()
        .map(|&(g_off, g_len, _, _)| &region[g_off..g_off + g_len])
        .collect();
    let experts_down: Vec<&[u8]> = offsets
        .iter()
        .map(|&(_, _, d_off, d_len)| &region[d_off..d_off + d_len])
        .collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0003).sin() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
    let moe = MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        // `top_k_softmax`, NOT the crate default (`top_k_renorm_scaled`):
        // the default's `post_expert_norm: RmsNorm` fails this
        // function's identity-combine precondition outright.
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q4_K,
        router_proj: &router_w,
        router_scale: &router_scale,
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: &pre_norm_w,
        post_ffn1_norm: &pre_norm_w,
        post_experts_norm: &pre_norm_w,
        num_experts,
        top_k,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
    };

    let scratch = MoeScratch::new_public(&m, top_k, hidden, inter);
    // `FullPipelineLayer::default()` has empty `up`/`down` weights
    // (`has_dense_ffn() == false`), `layer_scalar: 0.0`,
    // `moe_combined_output_norm: false`, `ffn_is_remote: false` —
    // every non-MoE precondition this function checks.
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter,
        inter_padded: inter,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);

    let h_post_attn_data = synth(hidden, 0.9);
    let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
    let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
    // Unused by this precondition/path combination — one shared dummy
    // buffer is enough for every field `try_inline_zero_copy_moe`
    // never reads.
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &h_post_attn_buf,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &new_h_buf,
    };

    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    // `try_inline_zero_copy_moe` REPLACES `*enc`/`*cmd` in place on the
    // fast-path hit — it assumes the caller already ended/committed
    // the incoming encoder (exactly what `handle_moe_interleave` does
    // right before calling it). Skipping this crashes the whole test
    // binary: Metal fatally asserts on dropping a command encoder
    // that was never `end_encoding()`'d.
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:213",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(
        took_zero_copy_path,
        "every precondition was satisfied; the merged-CB fast path must fire"
    );
    assert!(!encoder_ended);

    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:232",
    )
    .expect("command buffer completed");

    let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite combine output"
    );
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "combine wrote an all-zero buffer — vacuous dispatch"
    );
}

/// The test above puts every expert in ONE registered region, so
/// `encode_experts_zero_copy`'s `single_base` check is always true and
/// only the grouped-kernel arms run. Registering each expert in its
/// OWN region instead forces `single_base` to false for both the
/// gate/up and down dispatches regardless of which two experts the
/// router selects, driving the per-expert (non-grouped) fused Q4_K
/// kernel and per-expert down-matvec fallback arms — the other half
/// of that function's dispatch-shape branching. Also sets non-empty
/// `experts_gate_up_bias`/`experts_down_bias` to drive the bias-staging
/// block here and the `has_bias` combine arm in
/// `encode_experts_and_combine_zero_copy`, neither of which the
/// bias-free test above reaches. Non-empty biases force
/// `gate_rule: ClampedGlu` too — `biased_gated_servable` requires
/// either ClampedGlu or both bias arrays empty, since a `Gated`
/// layer with expert biases has no kernel — which additionally
/// covers the ClampedGlu activation arm the first test never takes.
#[test]
fn try_inline_zero_copy_moe_uses_non_grouped_dispatch_across_separate_regions() {
    let m = backend();
    let hidden = 256usize;
    let inter = 256usize;
    let top_k = 2usize;
    let num_experts = 4usize;

    let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

    // One page-aligned anonymous mmap PER expert — `resolve_region`
    // returns the same Metal buffer for any two sub-slices of the same
    // registered region, so this is what actually forces
    // `single_base` to observe distinct base buffers.
    let mut regions = Vec::with_capacity(num_experts);
    for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
        let mut region = memmap2::MmapMut::map_anon(g.len() + d.len()).expect("anon mmap");
        region[..g.len()].copy_from_slice(g);
        region[g.len()..].copy_from_slice(d);
        let region = region.make_read_only().expect("read-only mmap");
        assert!(
            m.bufs.register_region(&region[..]),
            "page-aligned anon mmap must register"
        );
        regions.push(region);
    }
    let experts_gate_up: Vec<&[u8]> = regions
        .iter()
        .zip(expert_gu.iter())
        .map(|(region, g)| &region[..g.len()])
        .collect();
    let experts_down: Vec<&[u8]> = regions
        .iter()
        .zip(expert_gu.iter())
        .map(|(region, g)| &region[g.len()..])
        .collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0003).sin() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
    // Non-empty so `expert_mlp(..).gate_up_bias`/`down_bias` are
    // non-empty too — `ExpertMlp::expert_mlp` slices these per-expert
    // at strides `2 * inter` and `hidden` respectively.
    let experts_gate_up_bias = vec![0.1f32; num_experts * 2 * inter];
    let experts_down_bias = vec![0.05f32; num_experts * hidden];
    let moe = MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q4_K,
        router_proj: &router_w,
        router_scale: &router_scale,
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: &pre_norm_w,
        post_ffn1_norm: &pre_norm_w,
        post_experts_norm: &pre_norm_w,
        num_experts,
        top_k,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &experts_gate_up_bias,
        experts_down_bias: &experts_down_bias,
        // `biased_gated_servable` requires EITHER ClampedGlu OR both
        // bias arrays empty — "a Gated layer with expert biases has
        // no kernel" (see try_inline_zero_copy_moe's own comment).
        // Non-empty biases with `Gated` here made the function bail
        // at that check on the first attempt; this is also the
        // combination that drives the ClampedGlu activation arm
        // (limit/alpha values match tests/test_moe_clamped_glu_q6k.rs).
        gate_rule: MoeGateRule::ClampedGlu {
            limit: 7.0,
            alpha: 1.702,
        },
    };

    let scratch = MoeScratch::new_public(&m, top_k, hidden, inter);
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter,
        inter_padded: inter,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);

    let h_post_attn_data = synth(hidden, 0.4);
    let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
    let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &h_post_attn_buf,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &new_h_buf,
    };

    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:388",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(
        took_zero_copy_path,
        "every precondition was satisfied; the merged-CB fast path must fire"
    );
    assert!(!encoder_ended);

    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:407",
    )
    .expect("command buffer completed");

    let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite combine output"
    );
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "combine wrote an all-zero buffer — vacuous dispatch"
    );
}

/// Both tests above use `expert_data_format: QuantFormat::Q4_K` — the
/// two Q6_K arms in `encode_experts_zero_copy` (grouped and
/// non-grouped matvec) are entirely untested by either. Single shared
/// region (`single_base` true) drives the Q6_K grouped kernel arm,
/// same shape `tests/test_kernel_moe_expert_dispatch.rs`'s
/// `zero_copy_grouped_q6k_dispatch_matches_staged_path` already
/// proves numerically — this test only needs the fast path to fire
/// and produce a non-vacuous result, not bit-exact parity.
#[test]
fn try_inline_zero_copy_moe_uses_q6k_grouped_dispatch() {
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;

    let m = backend();
    let hidden = 256usize;
    let inter = 256usize;
    let top_k = 2usize;
    let num_experts = 4usize;

    let mut expert_gu: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_down: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let gate = synth(inter * hidden, 0.21 + e as f32 * 0.13);
        let up = synth(inter * hidden, 0.51 + e as f32 * 0.17);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        expert_gu.push(quantize_q6_k(&gu));
        let raw_down = synth(hidden * inter, 0.83 + e as f32 * 0.07);
        let down_padded = pad_rows_to_256(&raw_down, hidden, inter);
        expert_down.push(quantize_q6_k(&down_padded));
    }

    let total: usize = expert_gu
        .iter()
        .zip(expert_down.iter())
        .map(|(g, d)| g.len() + d.len())
        .sum();
    let mut region = memmap2::MmapMut::map_anon(total).expect("anon mmap");
    let mut offsets = Vec::with_capacity(num_experts);
    let mut cursor = 0usize;
    for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
        region[cursor..cursor + g.len()].copy_from_slice(g);
        let g_off = cursor;
        cursor += g.len();
        region[cursor..cursor + d.len()].copy_from_slice(d);
        offsets.push((g_off, g.len(), cursor, d.len()));
        cursor += d.len();
    }
    let region = region.make_read_only().expect("read-only mmap");
    assert!(
        m.bufs.register_region(&region[..]),
        "page-aligned anon mmap must register"
    );
    let experts_gate_up: Vec<&[u8]> = offsets
        .iter()
        .map(|&(g_off, g_len, _, _)| &region[g_off..g_off + g_len])
        .collect();
    let experts_down: Vec<&[u8]> = offsets
        .iter()
        .map(|&(_, _, d_off, d_len)| &region[d_off..d_off + d_len])
        .collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0004).cos() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
    let moe = MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q6_K,
        router_proj: &router_w,
        router_scale: &router_scale,
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: &pre_norm_w,
        post_ffn1_norm: &pre_norm_w,
        post_experts_norm: &pre_norm_w,
        num_experts,
        top_k,
        intermediate_size: inter,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
    };

    let scratch = MoeScratch::new_public_with_format(
        &m,
        top_k,
        hidden,
        inter,
        QuantFormat::Q6_K,
        moe.gate_up_cols(hidden),
    );
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter,
        inter_padded: inter,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);

    let h_post_attn_data = synth(hidden, 0.6);
    let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
    let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &h_post_attn_buf,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &new_h_buf,
    };

    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:564",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(
        took_zero_copy_path,
        "every precondition was satisfied; the merged-CB fast path must fire"
    );
    assert!(!encoder_ended);

    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:583",
    )
    .expect("command buffer completed");

    let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite combine output"
    );
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "combine wrote an all-zero buffer — vacuous dispatch"
    );
}
