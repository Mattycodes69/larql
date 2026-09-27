use super::*;

/// **R5d's gate.** Every link in the device dependency chain, in order.
#[test]
fn the_whole_layer_on_device_matches_link_by_link() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = cpu_layer(&fx);
    let state = KdaDeviceState::zeros(&metal, fx.shape());
    let got = metal
        .kimi_decoder_layer_traced(fx.layer(&state), &fx.x)
        .expect("device layer");

    let check = |name: &str, a: &[f32], b: &[f32]| {
        let d = max_abs(a, b);
        eprintln!("[r5d] {name:>24}: max|Δ| {d:e}");
        assert!(d < TOLERANCE, "{name}: max|Δ| {d:e}");
    };
    check("input_normed", &got.input_normed, &cpu.input_normed);
    check("attention", &got.attention, &cpu.attention.output);
    check(
        "after_attention",
        &got.after_attention,
        &cpu.after_attention,
    );
    check(
        "post_attention_normed",
        &got.post_attention_normed,
        &cpu.post_attention_normed,
    );
    check("router_logits", &got.router_logits, &cpu.moe.router.logits);
    check("router_scores", &got.router_scores, &cpu.moe.router.scores);
    check(
        "router_selection_scores",
        &got.router_selection_scores,
        &cpu.moe.router.selection_scores,
    );

    // Exact ids, not a tolerance — a route is a decision, not a number.
    let want_ids: Vec<u32> = cpu
        .moe
        .router
        .selected_ids
        .iter()
        .map(|&i| i as u32)
        .collect();
    eprintln!("[r5d] {:>24}: {:?}", "selected_ids", got.selected_ids);
    assert_eq!(got.selected_ids, want_ids, "the device chose other experts");

    // The routed weights the MoE multiplied by, then the shared branch's
    // unscaled 1.0 — the ordering the combine relies on.
    check(
        "combine_weights(routed)",
        &got.combine_weights[..fx.top_k],
        &cpu.moe.router.weights,
    );
    assert_eq!(
        got.combine_weights[fx.top_k], 1.0,
        "the shared branch is summed, never scaled"
    );

    // The offset table the router wrote is the one the residency map
    // names for the experts it chose — the seam itself.
    let want_offsets: Vec<u32> = want_ids
        .iter()
        .map(|&id| fx.residency[id as usize])
        .collect();
    assert_eq!(
        got.expert_offsets, want_offsets,
        "the GPU-written offset table does not match the residency map"
    );

    for (slot, want) in cpu.moe.expert_outputs.iter().enumerate() {
        let a = &got.expert_outputs[slot * fx.hidden..(slot + 1) * fx.hidden];
        let d = max_abs(a, want);
        assert!(d < TOLERANCE, "expert slot {slot}: max|Δ| {d:e}");
    }
    let shared = &got.expert_outputs[fx.top_k * fx.hidden..(fx.top_k + 1) * fx.hidden];
    check("shared_output", shared, &cpu.moe.shared_output);
    check("layer_output", &got.output, &cpu.output);
    check("layer_output vs HF", &got.output, &fx.oracle_layer_output);
    eprintln!(
        "[r5d] one command buffer, gpu {:.3} ms, the host never saw an expert id",
        got.gpu_ms
    );
}

/// **Control.** An expert the router picks that is not resident must be
/// REFUSED, not served another expert's weights.
///
/// Without this the seam would be trusting a table it cannot check: the
/// router writes offsets, and nothing downstream can tell a wrong offset
/// from a right one.
#[test]
fn selecting_a_non_resident_expert_is_refused() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let state = KdaDeviceState::zeros(&metal, fx.shape());
    let mut broken = fx.residency.clone();
    // Evict the expert the router ranks first.
    let cpu = cpu_layer(&fx);
    broken[cpu.moe.router.selected_ids[0]] = NOT_RESIDENT;
    let mut w = fx.layer(&state);
    {
        let m = moe_mut(&mut w);
        let a = ExpertAddressing::Table(&broken);
        m.gate.addressing = a;
        m.up.addressing = a;
        m.down.addressing = a;
    }

    let refused = metal.kimi_decoder_layer(w, &fx.x);
    assert!(
        matches!(
            refused,
            Err(larql_compute_metal::trait_impl::grouped_experts::GroupedError::
                LayerRouteNotResident { layer: 0, .. })
        ),
        "a non-resident selection must be refused as such, not served: {refused:?}"
    );
    eprintln!("[r5d] control: evicting a selected expert is refused — {refused:?}");
}

/// **Control.** The two router transcriptions that still produce
/// plausible output must not pass.
///
/// Perturbing the fixture rather than the kernel: a bias that cannot
/// change selection, and weights that would come from the biased scores.
#[test]
fn the_router_controls_still_bite_on_device() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let cpu = cpu_layer(&fx);
    let state = KdaDeviceState::zeros(&metal, fx.shape());

    // Omitting the correction bias must change the selection. On this
    // fixture the consequence is a REFUSAL rather than a different
    // answer, and that is the strongest possible outcome: only the
    // biased route's experts are resident, so a changed route
    // immediately names an expert that has no address, and the device
    // guard catches it. Two things are proved at once — the bias decides
    // selection, and a route that leaves the resident set is refused
    // rather than served someone else's weights.
    let zeros = vec![0.0f32; fx.experts];
    let mut unbiased = fx.layer(&state);
    moe_mut(&mut unbiased).router_bias = &zeros;
    let refused = metal.kimi_decoder_layer(unbiased, &fx.x);
    assert!(
        refused.is_err(),
        "dropping the correction bias changed nothing — the bias must decide \
         selection on this fixture, or this control proves nothing"
    );

    // And say WHY, from the CPU router, so the refusal is not mistaken
    // for an unrelated fault.
    let unbiased_route = crate::format::vindex3::opplan::exec::kimi_router::route(
        &cpu.post_attention_normed,
        &fx.router_weight,
        &zeros,
        fx.experts,
        fx.top_k,
        fx.renormalize,
        fx.branch_scale as f64,
        crate::format::vindex3::opplan::exec::kimi_router::Mutation::None,
    );
    assert_ne!(
        unbiased_route.selected_ids, cpu.moe.router.selected_ids,
        "the correction bias must change the route on this fixture"
    );
    eprintln!(
        "[r5d] control: dropping the correction bias moves the route {:?} -> {:?}, \
         which the device refuses as non-resident",
        cpu.moe.router.selected_ids, unbiased_route.selected_ids,
    );
}

/// What a whole layer costs on each side, and how many crossings it
/// makes.
#[test]
fn report_whole_layer_cost() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let state = KdaDeviceState::zeros(&metal, fx.shape());
    let host = || {
        let t = Instant::now();
        std::hint::black_box(cpu_layer(&fx));
        t.elapsed().as_secs_f64() * 1000.0
    };
    let device = || {
        // Reset outside the timer: the recurrent state advances every
        // step, and on this fixture a drifted state eventually routes to
        // an expert that is not resident. Holding the token constant is
        // also what makes the two arms comparable — `cpu_layer` starts
        // from a zero state every call.
        state.reset();
        let t = Instant::now();
        let (out, gpu) = metal
            .kimi_decoder_layer(fx.layer(&state), &fx.x)
            .expect("device layer");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    for _ in 0..WARMUP {
        host();
        device();
    }
    let (mut h, mut dw, mut dg) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        h.push(host());
        let (w, g) = device();
        dw.push(w);
        dg.push(g);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (h.len() / 3).max(1);
    let ramp = mean(&h[..third]) / mean(&h[h.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (host_ms, dev_ms, gpu_ms) = (median(&mut h), median(&mut dw), median(&mut dg));

    // Decompose the device GPU time: the attention alone is already
    // measured at rung 5c, so what the layer adds beyond it is the
    // router, the MoE and the norms. Without this the layer's total is
    // just a number with no stage attached.
    let kda_state = KdaDeviceState::zeros(&metal, fx.shape());
    let mut kda_gpu = Vec::new();
    for i in 0..WARMUP + ITERS {
        kda_state.reset();
        let (_, g) = metal
            .kda_attention_step(
                fx.kda_device(),
                fx.shape(),
                &kda_state,
                &fx.input_norm_applied(),
            )
            .expect("attention alone");
        if i >= WARMUP {
            kda_gpu.push(g);
        }
    }
    let attn_ms = median(&mut kda_gpu);

    // And the MoE alone, through the already-measured block path with
    // host-built offsets — so "everything else" can be split into the
    // experts and the rest rather than left as a residual.
    let cpu = cpu_layer(&fx);
    // The block path binds ONE bank per projection, so this probe
    // chooses the co-located layout: routed bytes with the shared
    // payload appended. A layout an arm MAY choose — the layer under
    // test binds the shared regions independently.
    let cat = |routed: &[u8], shared: &[u8]| {
        let mut v = routed.to_vec();
        v.extend_from_slice(shared);
        v
    };
    let (cat_gate, cat_up, cat_down) = (
        cat(&fx.bank_gate, &fx.shared_gate),
        cat(&fx.bank_up, &fx.shared_up),
        cat(&fx.bank_down, &fx.shared_down),
    );
    let offsets: Vec<ExpertOffset> = cpu
        .moe
        .router
        .selected_ids
        .iter()
        .map(|&id| ExpertOffset(fx.residency[id]))
        .chain(std::iter::once(ExpertOffset(fx.bank_gate.len() as u32)))
        .collect();
    let banks = MoeFfnBanks {
        gate: ExpertBankRef {
            weights: &cat_gate,
            offsets: &offsets,
        },
        up: ExpertBankRef {
            weights: &cat_up,
            offsets: &offsets,
        },
        down: ExpertBankRef {
            weights: &cat_down,
            offsets: &offsets,
        },
        hidden: fx.hidden,
        inter: fx.inter,
    };
    let call = [MoeBlockCall {
        banks,
        x: &cpu.post_attention_normed,
    }];
    let mut moe_gpu = Vec::new();
    for i in 0..WARMUP + ITERS {
        let (_, g) = metal
            .bf16_moe_ffn_blocks(
                &call,
                larql_compute_metal::trait_impl::bf16_moe_block::BlockLowering::Separate,
            )
            .expect("moe alone");
        if i >= WARMUP {
            moe_gpu.push(g);
        }
    }
    let moe_ms = median(&mut moe_gpu);

    eprintln!("[r5d] one complete decoder layer (KDA + router + 8 routed + shared MoE)");
    eprintln!("[r5d]   host   {host_ms:.3} ms   all stages on CPU");
    eprintln!(
        "[r5d]   device {dev_ms:.3} ms   gpu-busy {gpu_ms:.3} ms, host {:.3} ms \
         over 1 crossing  [{:.2}x]",
        dev_ms - gpu_ms,
        host_ms / dev_ms,
    );
    eprintln!(
        "[r5d]   of which attention {attn_ms:.3} ms, MoE {moe_ms:.3} ms, \
         remainder {:.3} ms (router + norms + residuals)",
        gpu_ms - attn_ms - moe_ms,
    );
    eprintln!("[r5d]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}

/// **The signature: one logical expert, three physical slots, two
/// encodings, correct output.**
#[test]
fn location_and_representation_vary_independently_for_one_semantic_expert() {
    let Some((b, f)) = setup() else {
        return;
    };
    let per = f.inter * f.hidden * 2;
    let blocks = f.bank_gate.len() / per;
    let (pg, pu, pd) = perms(blocks);
    for i in 0..blocks {
        assert!(
            pg[i] != pu[i] && pu[i] != pd[i] && pg[i] != pd[i],
            "block {i} maps to {}/{}/{} — a shared coordinate would survive",
            pg[i],
            pu[i],
            pd[i]
        );
    }

    // The all-BF16 reference, unpermuted.
    let state_ref = KdaDeviceState::zeros(&b, f.shape());
    let (want, _) = b
        .kimi_decoder_layer(f.layer(&state_ref), &f.x)
        .expect("reference layer runs");

    // gate/up become Q6_K; down stays BF16. Each is permuted
    // differently. The SHARED branch follows each projection's encoding
    // from its own regions — it has no location in the routed banks to
    // permute, which is the whole point of it being a separate region.
    let (q_gate_src, q_per) = to_q6k_blocks(&f.bank_gate, per, f.inter, f.hidden);
    let (q_up_src, _) = to_q6k_blocks(&f.bank_up, per, f.inter, f.hidden);
    let (q_shared_gate, _) = to_q6k_blocks(&f.shared_gate, per, f.inter, f.hidden);
    let (q_shared_up, _) = to_q6k_blocks(&f.shared_up, per, f.inter, f.hidden);
    let gate = permute(&q_gate_src, q_per, &pg);
    let up = permute(&q_up_src, q_per, &pu);
    let down = permute(&f.bank_down, per, &pd);
    let (tg, tu, td) = (
        table(&f.residency, per, q_per, &pg),
        table(&f.residency, per, q_per, &pu),
        table(&f.residency, per, per, &pd),
    );
    let state = KdaDeviceState::zeros(&b, f.shape());
    let mut w = f.layer(&state);
    if let FfnSpec::Moe(m) = &mut w.ffn {
        m.gate = ProjectionBank {
            routed: EncodedRegion {
                bytes: &gate,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tg),
            shared: Some(EncodedRegion {
                bytes: &q_shared_gate,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.up = ProjectionBank {
            routed: EncodedRegion {
                bytes: &up,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tu),
            shared: Some(EncodedRegion {
                bytes: &q_shared_up,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.down = ProjectionBank {
            routed: EncodedRegion {
                bytes: &down,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ExpertAddressing::Table(&td),
            shared: Some(EncodedRegion {
                bytes: &f.shared_down,
                encoding: ExpertEncoding::Bf16,
            }),
        };
    }
    let (got, _) = b
        .kimi_decoder_layer(w, &f.x)
        .expect("the mixed layer must execute");

    // Every selected expert really does sit at three distinct slots.
    let ids: Vec<usize> = f.ids_order.clone();
    for &e in ids.iter().take(f.top_k) {
        let (a, c, d) = (
            tg[e] as usize / q_per,
            tu[e] as usize / q_per,
            td[e] as usize / per,
        );
        assert!(
            a != c && c != d && a != d,
            "expert {e} resolves to slots {a}/{c}/{d} — not three distinct places"
        );
    }

    // The tolerance is Q6's own, taken from this very layer: quantise
    // gate/up WITHOUT permuting and measure. Anything the permutation
    // adds beyond that is an addressing fault, not representation.
    let unpermuted = {
        let identity: Vec<usize> = (0..blocks).collect();
        let g = permute(&q_gate_src, q_per, &identity);
        let u = permute(&q_up_src, q_per, &identity);
        let (tgi, tui) = (
            table(&f.residency, per, q_per, &identity),
            table(&f.residency, per, q_per, &identity),
        );
        let st = KdaDeviceState::zeros(&b, f.shape());
        let mut w2 = f.layer(&st);
        if let FfnSpec::Moe(m) = &mut w2.ffn {
            m.gate = ProjectionBank {
                routed: EncodedRegion {
                    bytes: &g,
                    encoding: ExpertEncoding::Q6K,
                },
                addressing: ExpertAddressing::Table(&tgi),
                shared: Some(EncodedRegion {
                    bytes: &q_shared_gate,
                    encoding: ExpertEncoding::Q6K,
                }),
            };
            m.up = ProjectionBank {
                routed: EncodedRegion {
                    bytes: &u,
                    encoding: ExpertEncoding::Q6K,
                },
                addressing: ExpertAddressing::Table(&tui),
                shared: Some(EncodedRegion {
                    bytes: &q_shared_up,
                    encoding: ExpertEncoding::Q6K,
                }),
            };
            m.down.shared = Some(EncodedRegion {
                bytes: &f.shared_down,
                encoding: ExpertEncoding::Bf16,
            });
        }
        b.kimi_decoder_layer(w2, &f.x)
            .expect("unpermuted mixed runs")
            .0
    };
    let rel = |a: &[f32], c: &[f32]| {
        let se: f64 = a.iter().zip(c).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let ss: f64 = c.iter().map(|y| (*y as f64).powi(2)).sum();
        (se / ss).sqrt()
    };
    let envelope = rel(&unpermuted, &want);
    let mixed = rel(&got, &want);
    eprintln!(
        "[c4] Q6_K gate/up + BF16 down, three independent permutations: \
         rel_rms vs BF16 {mixed:.3e}; the same encodings UNPERMUTED give {envelope:.3e}"
    );
    assert!(
        envelope > 0.0,
        "quantising gate/up must move the answer, or this proves nothing"
    );
    assert!(
        mixed <= envelope * 1.05,
        "permuting the three projections independently must cost nothing beyond Q6's own \
         error: {mixed:.3e} against an envelope of {envelope:.3e}"
    );

    // The control: mismatch ONE projection's table and the answer must move.
    let state_bad = KdaDeviceState::zeros(&b, f.shape());
    let mut bad = f.layer(&state_bad);
    if let FfnSpec::Moe(m) = &mut bad.ffn {
        m.gate = ProjectionBank {
            routed: EncodedRegion {
                bytes: &gate,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tu),
            shared: Some(EncodedRegion {
                bytes: &q_shared_gate,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.up = ProjectionBank {
            routed: EncodedRegion {
                bytes: &up,
                encoding: ExpertEncoding::Q6K,
            },
            addressing: ExpertAddressing::Table(&tu),
            shared: Some(EncodedRegion {
                bytes: &q_shared_up,
                encoding: ExpertEncoding::Q6K,
            }),
        };
        m.down = ProjectionBank {
            routed: EncodedRegion {
                bytes: &down,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ExpertAddressing::Table(&td),
            shared: Some(EncodedRegion {
                bytes: &f.shared_down,
                encoding: ExpertEncoding::Bf16,
            }),
        };
    }
    let (wrong, _) = b.kimi_decoder_layer(bad, &f.x).expect("runs");
    assert!(
        rel(&wrong, &want) > envelope * 2.0,
        "giving gate another projection's table must break the answer, or the three \
         mappings are not really independent"
    );
}

/// **A projection whose bytes are not its declared encoding is refused
/// BEFORE any kernel runs.**
///
/// Q6_K bytes dispatched as BF16 would be read as roughly 2.4x more
/// data than exists — the layer must refuse rather than launch and let
/// a kernel interpret whatever follows.
///
/// This is the too-small direction, which a room check catches. The
/// opposite (BF16 bytes declared Q6_K, which are LARGER than the claim
/// and pass every room check) needs the bank's exact extent and is
/// covered by `represent::physical`'s
/// `a_bank_whose_bytes_are_not_its_declared_encoding_is_refused`.
#[test]
fn a_projection_declaring_the_wrong_encoding_is_refused_before_execution() {
    let Some((b, f)) = setup() else {
        return;
    };
    let per = f.inter * f.hidden * 2;
    let (q_gate, q_per) = to_q6k_blocks(&f.bank_gate, per, f.inter, f.hidden);
    assert!(
        q_per < per,
        "Q6_K must be smaller than bf16 for this to bite"
    );

    let ids: Vec<usize> = (0..f.bank_gate.len() / per).collect();
    let t = table(&f.residency, per, q_per, &ids);
    let state = KdaDeviceState::zeros(&b, f.shape());
    let mut w = f.layer(&state);
    if let FfnSpec::Moe(m) = &mut w.ffn {
        // Real Q6_K bytes, with the offsets they need — but claiming to
        // be BF16, which needs 2.4x the room.
        m.gate = ProjectionBank {
            routed: EncodedRegion {
                bytes: &q_gate,
                encoding: ExpertEncoding::Bf16,
            },
            addressing: ExpertAddressing::Table(&t),
            shared: Some(EncodedRegion {
                bytes: &f.shared_gate,
                encoding: ExpertEncoding::Bf16,
            }),
        };
    }
    let err = b
        .kimi_decoder_layer(w, &f.x)
        .expect_err("bytes that are not the declared encoding must be refused");
    assert!(
        matches!(err, GroupedError::OffsetOutOfRange { slot: 0, .. }),
        "expected the gate projection to be named, got {err:?}"
    );
}
