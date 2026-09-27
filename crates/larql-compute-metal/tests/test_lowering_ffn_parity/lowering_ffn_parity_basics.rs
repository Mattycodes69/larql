use super::*;

#[test]
fn lowered_ffn_matches_the_cpu_program_and_reads_its_judged_facts() {
    let gpu = larql_compute_metal::MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let h = deterministic(HIDDEN, 1);
    let norm_w = deterministic(HIDDEN, 2);
    let gate_f = deterministic(INTER * HIDDEN, 3);
    let up_f = deterministic(INTER * HIDDEN, 4);
    let down_f = deterministic(HIDDEN * INTER, 5);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    // The reference consumes the *quantised* weights, so the comparison
    // isolates the lowering from quantisation error — which Q2 already
    // measured separately and which would otherwise dominate here.
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();

    let post_w = deterministic(HIDDEN, 6);
    let reference = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::BeforeResidual(&post_w, POST_EPS),
    );
    let got = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        Some(&post_w),
        POST_EPS,
    );

    let m = compare(&reference, &got);
    eprintln!(
        "lowered FFN vs CPU program: max_abs {:.3e}  rel_rms {:.3e}  cosine {:.9}",
        m.max_abs, m.rel_rms, m.cosine
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "lowered FFN produced non-finite output"
    );
    assert!(
        m.rel_rms < 1e-4 && m.cosine > 0.999_999,
        "lowered FFN disagrees with its own program: rel_rms {:.3e}, cosine {:.9}",
        m.rel_rms,
        m.cosine
    );

    // ── Control 1: the centred-norm offset is read ──────────────────
    let no_offset = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        0.0,
        true,
        true,
        PostNormMode::BeforeResidual(&post_w, POST_EPS),
    );
    assert_control("centred-norm offset", &no_offset, &got, m.rel_rms);

    // ── Control 2: the activation is SiLU-GLU, not plain GLU ────────
    let plain_glu = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        false,
        true,
        PostNormMode::BeforeResidual(&post_w, POST_EPS),
    );
    assert_control("SiLU-GLU activation", &plain_glu, &got, m.rel_rms);

    // ── Control 3: the residual is applied ──────────────────────────
    let no_residual = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        false,
        PostNormMode::None,
    );
    assert_control("FFN residual", &no_residual, &got, m.rel_rms);

    // ── Control 4: the post-FFN norm exists at all ──────────────────
    let no_post = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::None,
    );
    assert_control("post-FFN norm omitted", &no_post, &got, m.rel_rms);

    // Control 5 (post-norm epsilon) is deliberately NOT asserted here.
    // At this fixture's magnitudes the branch output has mean-square
    // ~11, so `sqrt(ms + 1e-5)` and `sqrt(ms + 1e-8)` differ by ~4.4e-7
    // relative — *below* this lowering's own ~9e-7 parity residual. The
    // control would fire at 1.0x and prove nothing, which is a fact
    // about where epsilon is observable, not about the lowering.
    // `post_norm_epsilon_is_read_where_it_is_observable` tests it in the
    // regime where the distinction exists.

    // ── Control 6: normalise the branch, THEN add — not add then
    //    normalise the sum. "Post-FFN norm" reads both ways; only one
    //    is the interpreter's.
    let after = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::AfterResidual(&post_w, POST_EPS),
    );
    assert_control(
        "post-norm applied after the residual",
        &after,
        &got,
        m.rel_rms,
    );
}

/// The post-norm epsilon, tested where it is observable.
///
/// Epsilon only moves the result when it is a real fraction of the
/// branch output's mean-square. The main fixture's branch has ms ~11, so
/// 1e-5 and 1e-8 are indistinguishable there — below the lowering's own
/// error. Here the down projection is scaled down until ms ~1e-4, where
/// 1e-5 shifts the RMS by ~5% and the two epsilons are unmistakable.
///
/// GPU-vs-GPU on purpose: the question is whether the plan's epsilon is
/// *plumbed through* to the kernel, and comparing two lowered runs
/// answers exactly that without a reference in between.
/// Two-norm placement folds the residual add into the down-projection
/// write (A-5b rung 2a). Every other test here passes `Some(post_norm)`,
/// which takes the four-norm branch — so the fused path shipped
/// unexercised, and stayed that way when it grew byte offsets for the
/// packed-operand layout. That is the branch where dropping an offset
/// would compute a different matrix's rows with the residual added on
/// top: finite, plausible, wrong.
#[test]
fn two_norm_placement_folds_the_residual_into_the_down_write() {
    let gpu = larql_compute_metal::MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let h = deterministic(HIDDEN, 31);
    let norm_w = deterministic(HIDDEN, 32);
    let gate_f = deterministic(INTER * HIDDEN, 33);
    let up_f = deterministic(INTER * HIDDEN, 34);
    let down_f = deterministic(HIDDEN * INTER, 35);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    // Reference consumes the QUANTISED weights, so this isolates the
    // lowering from representation error (measured separately in Q2).
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();
    // `run_lowered` takes no activation argument — the harness fixes SiLU,
    // so the reference must too. Passing `false` here compares against a
    // GELU program and reports a kernel divergence that is really a
    // fixture mistake (rel_rms 0.56 on the first run).
    let expect = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::None,
    );

    // post_norm = None is what selects the fused branch.
    let got = run_lowered(&gpu, &h, &norm_w, &gate, &up, &down, NORM_OFFSET, None, EPS);
    let m = compare(&expect, &got);
    assert!(
        m.rel_rms < 1e-4,
        "fused-residual down projection diverged: rel_rms {}, max_abs {}",
        m.rel_rms,
        m.max_abs
    );
    // The residual really is added, not dropped: without it the output
    // would be the branch alone, which differs from h by construction.
    assert!(
        got.iter().zip(&h).any(|(o, i)| (o - i).abs() > 1e-6),
        "output equals the input residual — the FFN branch was not added"
    );
}

#[test]
fn post_norm_epsilon_is_read_where_it_is_observable() {
    let gpu = larql_compute_metal::MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let h = deterministic(HIDDEN, 1);
    let norm_w = deterministic(HIDDEN, 2);
    let post_w = deterministic(HIDDEN, 6);
    let gate_f = deterministic(INTER * HIDDEN, 3);
    let up_f = deterministic(INTER * HIDDEN, 4);
    // Scaled so the branch output's mean-square lands near 1e-4.
    let down_f: Vec<f32> = deterministic(HIDDEN * INTER, 5)
        .iter()
        .map(|v| v * 3e-3)
        .collect();

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();

    let with_post = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        Some(&post_w),
        POST_EPS,
    );
    let with_pre = run_lowered(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        Some(&post_w),
        EPS,
    );

    // Both runs include the residual, which is identical between them and
    // dwarfs the branch; compare the branch contribution alone.
    //
    // Note the branch is measured *after* the post-norm, so its own
    // mean-square is ~1 by construction whatever the down projection's
    // scale — the magnitude that decides observability is the pre-norm
    // one, which is not visible from outside the encoder. The scaling of
    // `down_f` above is what puts it in range; the assertion below is on
    // the effect, not on a proxy for it.
    let branch_post: Vec<f32> = with_post.iter().zip(&h).map(|(a, b)| a - b).collect();
    let branch_pre: Vec<f32> = with_pre.iter().zip(&h).map(|(a, b)| a - b).collect();
    let m = compare(&branch_post, &branch_pre);
    // Judged against the lowering's own parity residual, as everywhere
    // else: ~9e-7 on this fixture.
    let ratio = m.rel_rms / 8.921e-7;
    eprintln!(
        "post-norm eps 1e-8 vs 1e-5: rel_rms {:.3e} = {ratio:.0}x the parity residual",
        m.rel_rms
    );
    assert!(
        ratio > CONTROL_MARGIN,
        "the plan's post-norm epsilon must reach the kernel: swapping 1e-8 for 1e-5 \
         moved the branch only {ratio:.1}x the parity residual ({:.3e})",
        m.rel_rms
    );
}

/// Does the dense-FFN lowering REACH its SiTU-GLU arm?
///
/// `tests/test_lowering_situ.rs` already qualifies the `situ_glu` kernel
/// against the scalar authority, but it binds `bind_situ_glu` directly.
/// Nothing there — or anywhere — drives
/// `lowering::ffn::encode_gate_up_act_from_normed`'s
/// `FfnActivation::SituGlu` arm, so a lowering that ignored the plan's
/// combine and ran `geglu_silu` would keep every one of those tests
/// green. A proven kernel nobody dispatches is not an executed kernel.
///
/// The witness is an analytic identity rather than a second transcription
/// of the reference. `situ_glu` computes
///
/// ```text
/// beta * tanh(g / beta) * sigmoid(g) * u        (linear_beta = None)
/// ```
///
/// and `beta * tanh(g / beta) -> g` as `beta` grows, so at a large beta
/// with the up branch uncapped the combine IS `silu(g) * u`. That gives
/// the arm a real external reference — the same `cpu_reference` the SiLU
/// parity test uses — with only `shape.activation` changed.
///
/// Paired, because the identity alone is satisfied by a lowering that
/// never reached the arm at all: at GLM/K3-scale parameters the SiTU
/// combine must MOVE the answer, far past the parity residual. One arm
/// says the lowering computes SiTU's formula; the other says it is not
/// quietly computing SiLU.
#[test]
fn the_lowering_dispatches_situ_glu_and_binds_its_parameters() {
    let gpu = larql_compute_metal::MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let h = deterministic(HIDDEN, 11);
    let norm_w = deterministic(HIDDEN, 12);
    let gate_f = deterministic(INTER * HIDDEN, 13);
    let up_f = deterministic(INTER * HIDDEN, 14);
    let down_f = deterministic(HIDDEN * INTER, 15);

    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();

    // `beta` large enough that `beta*tanh(g/beta)` is `g` to well inside
    // f32, small enough that `g/beta` is not flushed: the normed gate
    // here is O(1), so g/beta ~ 1e-4 and the cubic term ~1e-9 relative.
    const WIDE_BETA: f32 = 1.0e4;

    let reference = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        true,
        PostNormMode::None,
    );
    let wide = run_lowered_act(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::SituGlu {
            beta: WIDE_BETA,
            linear_beta: None,
        },
    );

    assert!(
        wide.iter().all(|v| v.is_finite()),
        "the SiTU lowering produced non-finite output"
    );
    let m = compare(&reference, &wide);
    eprintln!(
        "SiTU(beta={WIDE_BETA:.0e}, no cap) vs the SiLU program: max_abs {:.3e}  \
         rel_rms {:.3e}  cosine {:.9}",
        m.max_abs, m.rel_rms, m.cosine
    );
    assert!(
        m.rel_rms < 1e-4 && m.cosine > 0.999_999,
        "at a wide beta the SiTU combine must BE silu(g)*u, so the lowering \
         disagreeing here means it bound the kernel's parameters wrongly: \
         rel_rms {:.3e}, cosine {:.9}",
        m.rel_rms,
        m.cosine
    );

    // The pairing: at real parameters the arm must move the answer, or
    // the agreement above is equally consistent with the lowering having
    // ignored `FfnActivation::SituGlu` and run `geglu_silu`.
    let capped = run_lowered_act(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::SituGlu {
            beta: 4.0,
            linear_beta: Some(25.0),
        },
    );
    assert!(
        capped.iter().all(|v| v.is_finite()),
        "the capped SiTU lowering produced non-finite output"
    );
    assert_control("SiTU softcap at K3 parameters", &capped, &wide, m.rel_rms);

    // And `linear_beta: None` is a DIFFERENT function from an infinite
    // bound, not a spelling of it — the flag the kernel binds separately.
    // Capping only the up branch, at the same beta, must move the answer
    // too, or `has_linear` is not reaching the shader.
    let up_capped = run_lowered_act(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::SituGlu {
            beta: WIDE_BETA,
            linear_beta: Some(0.5),
        },
    );
    assert_control(
        "SiTU up-branch cap (has_linear)",
        &up_capped,
        &wide,
        m.rel_rms,
    );
}

/// A plan's residual-scale op (Granite `residual_multiplier`) scales the
/// FFN branch before it joins the residual stream: `h + s * ffn(h)`.
/// The lowering once hard-coded the add at 1.0 and ran Granite silently
/// wrong; the control is that unscaled add.
#[test]
fn a_residual_scale_scales_the_ffn_branch_before_the_add() {
    let gpu = larql_compute_metal::MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    const SCALE: f32 = 0.22;
    let h = deterministic(HIDDEN, 11);
    let norm_w = deterministic(HIDDEN, 12);
    let gate_f = deterministic(INTER * HIDDEN, 13);
    let up_f = deterministic(INTER * HIDDEN, 14);
    let down_f = deterministic(HIDDEN * INTER, 15);
    let gate = nvfp4::quantize(&gate_f, INTER, HIDDEN).unwrap();
    let up = nvfp4::quantize(&up_f, INTER, HIDDEN).unwrap();
    let down = nvfp4::quantize(&down_f, HIDDEN, INTER).unwrap();
    let gate_q = nvfp4::round_trip(&gate_f, INTER, HIDDEN).unwrap();
    let up_q = nvfp4::round_trip(&up_f, INTER, HIDDEN).unwrap();
    let down_q = nvfp4::round_trip(&down_f, HIDDEN, INTER).unwrap();

    // The branch alone, then the scaled and the unscaled residual adds.
    let branch = cpu_reference(
        &h,
        &norm_w,
        &gate_q,
        &up_q,
        &down_q,
        NORM_OFFSET,
        true,
        false,
        PostNormMode::None,
    );
    let reference: Vec<f32> = h.iter().zip(&branch).map(|(a, d)| a + SCALE * d).collect();
    let unscaled: Vec<f32> = h.iter().zip(&branch).map(|(a, d)| a + d).collect();

    let got = run_lowered_scaled(
        &gpu,
        &h,
        &norm_w,
        &gate,
        &up,
        &down,
        NORM_OFFSET,
        None,
        EPS,
        FfnActivation::Silu,
        Some(SCALE),
    );
    let m = compare(&reference, &got);
    eprintln!(
        "lowered FFN, residual scale {SCALE}: max_abs {:.3e}  rel_rms {:.3e}  cosine {:.9}",
        m.max_abs, m.rel_rms, m.cosine
    );
    assert!(
        m.rel_rms < 1e-4 && m.cosine > 0.999_999,
        "lowered FFN ignores the residual scale: rel_rms {:.3e}, cosine {:.9}",
        m.rel_rms,
        m.cosine
    );
    assert_control("residual scale", &unscaled, &got, m.rel_rms);
}
