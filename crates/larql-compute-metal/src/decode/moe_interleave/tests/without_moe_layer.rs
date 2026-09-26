use super::*;

/// `layer.moe.is_none()` is the first precondition check — must
/// bail out before touching the command buffer/encoder at all.
#[test]
fn try_inline_zero_copy_moe_returns_false_without_moe_layer() {
    let m = backend();
    // `MoeScratch::new` debug-asserts `weight_cols.is_multiple_of(block)`
    // (Q4_K block = 256 elements) unconditionally, before this test's
    // early-return path is ever reached — must be a block multiple even
    // though the actual dispatch never runs.
    let hidden = 256usize;
    let layer = FullPipelineLayer {
        moe: None,
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter: hidden,
        inter_padded: hidden,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let scratch = MoeScratch::new_public(&m, 1, hidden, hidden);
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);
    let h_post_attn_data = vec![0.0f32; hidden];
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &dummy,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &dummy,
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
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:651",
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
    assert!(!took_zero_copy_path);
    // Early-return arms never touch `*encoder_ended` — it must come
    // back exactly as the caller left it (already ended, per the
    // real `handle_moe_interleave` calling convention above), not
    // flipped to `false` as the success path would.
    assert!(
        encoder_ended,
        "must leave caller state untouched on bail-out"
    );
}
