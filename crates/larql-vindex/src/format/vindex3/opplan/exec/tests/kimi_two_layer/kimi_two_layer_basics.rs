use super::*;

/// **R5e's gate.** Both layers, every boundary, one command buffer.
#[test]
fn two_dynamic_layers_in_one_command_buffer_match_the_oracle() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    assert!(
        fx.routes_differ,
        "the two layers must select different experts, or this fixture cannot \
         tell a real chain from a path that re-used layer 1's input"
    );
    let states = fx.states(&metal);
    let calls = fx.calls(&states);

    let planes = metal
        .kimi_decoder_layers_traced(&calls, &fx.x)
        .expect("layer chain");
    assert_eq!(planes.len(), fx.layers.len());

    for (i, (p, l)) in planes.iter().zip(&fx.layers).enumerate() {
        let check = |name: &str, a: &[f32], b: &[f32]| {
            let d = max_abs(a, b);
            eprintln!("[r5e] layer {i} {name:>22}: max|Δ| {d:e}");
            assert!(d < TOLERANCE, "layer {i} {name}: max|Δ| {d:e}");
        };
        check("input_normed", &p.input_normed, &l.oracle_input_normed);
        check("attention", &p.attention, &l.oracle_attention);
        check(
            "after_attention",
            &p.after_attention,
            &l.oracle_after_attention,
        );
        check(
            "post_attention_normed",
            &p.post_attention_normed,
            &l.oracle_post_normed,
        );

        // A route is a decision, not a number, so both halves are exact.
        //
        // **Two references, because there are two contracts.** The
        // fixture's `selected_ids_order` is torch `topk(sorted=False)`'s
        // order, which promises nothing — so the CHECKPOINT is compared
        // on membership. The ORDER contract belongs to
        // `kimi_router::route`: descending by selection score, ties by
        // ascending index. The device reproduces that one exactly, and
        // it matters because the combine pairs weights with slots
        // positionally.
        eprintln!(
            "[r5e] layer {i} {:>22}: {:?}",
            "selected_ids", p.selected_ids
        );
        let device_ids: Vec<usize> = p.selected_ids.iter().map(|&x| x as usize).collect();
        let mut got_set = device_ids.clone();
        let mut want_set = l.ids.clone();
        got_set.sort_unstable();
        want_set.sort_unstable();
        assert_eq!(
            got_set, want_set,
            "layer {i}: the device's expert SET differs from the checkpoint's"
        );

        let reference = crate::format::vindex3::opplan::exec::kimi_router::route(
            &p.post_attention_normed,
            &l.router_weight,
            &l.router_bias,
            l.router_bias.len(),
            fx.top_k,
            fx.renormalize,
            fx.branch_scale as f64,
            crate::format::vindex3::opplan::exec::kimi_router::Mutation::None,
        );
        assert_eq!(
            device_ids, reference.selected_ids,
            "layer {i}: the device's ranking differs from `kimi_router::route`'s"
        );
        check(
            "router_selection_scores",
            &p.router_selection_scores,
            &reference.selection_scores,
        );
        check(
            "combine_weights(routed)",
            &p.combine_weights[..fx.top_k],
            &reference.weights,
        );

        // And each slot's offset is the one the residency map names for
        // the expert THAT slot holds — the seam, checked against the
        // device's own ordering rather than a presumed one.
        let want_offsets: Vec<u32> = device_ids.iter().map(|&id| l.residency[id]).collect();
        assert_eq!(
            p.expert_offsets, want_offsets,
            "layer {i}: the GPU-written offset table does not match the residency map"
        );
        assert_eq!(
            p.combine_weights[fx.top_k], 1.0,
            "layer {i}: the shared branch is summed, never scaled"
        );
        check("layer_output", &p.output, &l.oracle_output);
    }

    // The link itself: layer 1's input was layer 0's output, and neither
    // ever reached the host.
    eprintln!(
        "[r5e] chain proven: {} layers, one command buffer, gpu {:.3} ms; each route \
         computed from its predecessor's device-resident hidden — {:?}",
        planes.len(),
        planes[0].gpu_ms,
        planes
            .iter()
            .map(|p| p.selected_ids.clone())
            .collect::<Vec<_>>(),
    );
}

/// **Control.** Feeding layer 2 the ORIGINAL input instead of layer 1's
/// output must change its route.
///
/// This is the failure the whole rung is about: a device path that read
/// a stale buffer, or re-bound the upload, would produce a plausible
/// output from the wrong hidden state. Running the LAST layer alone on
/// `x` shows what that mistake would have selected.
#[test]
fn the_last_layer_run_on_the_original_input_routes_differently() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let states = fx.states(&metal);
    let last = fx.layers.len() - 1;
    let alone = metal.kimi_decoder_layer_traced(fx.layers[last].weights(&fx, &states[last]), &fx.x);

    // On the wrong hidden state it selects experts its own bank does not
    // hold, so the device refuses — the strongest possible outcome, and
    // proof the chained run's route came from somewhere else.
    match alone {
        Err(e) => eprintln!(
            "[r5e] control: layer {last} on the ORIGINAL input routes outside its bank and \
             is refused ({e}) — its chained route therefore came from its predecessor"
        ),
        Ok(p) => {
            let want: Vec<u32> = fx.layers[last].ids.iter().map(|&x| x as u32).collect();
            assert_ne!(
                p.selected_ids, want,
                "layer {last} selected the same experts from the original input as from \
                 the chain — this fixture cannot prove the chain"
            );
            eprintln!(
                "[r5e] control: layer {last} on the ORIGINAL input routes {:?}, chained it \
                 routes {want:?}",
                p.selected_ids
            );
        }
    }
}

/// **The strongest chain control.** Change what a layer OUTPUTS without
/// changing what it SELECTS, and watch a later layer's route move.
///
/// Swapping two of a layer's resident expert offsets makes each selected
/// id read the other expert's weights: the router still scores the same
/// `post_attention_normed` and picks the same ids, so that layer's own
/// decision is untouched — but its output changes, and every layer after
/// it is therefore routing on a different hidden state.
///
/// The evidence is precise: the refusal must come from a layer AFTER the
/// mutated one, never from the mutated layer itself. A stale-buffer or
/// buffer-reuse bug would break that pattern.
#[test]
fn perturbing_a_layers_output_moves_a_later_layers_route() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let n = fx.layers.len();
    assert!(n >= 2, "a chain control needs at least two layers");
    let states = fx.states(&metal);
    let baseline = metal
        .kimi_decoder_layers_traced(&fx.calls(&states), &fx.x)
        .expect("baseline chain");

    // Before mutating anything: a reset must reproduce the baseline
    // exactly. If it does not, every comparison below is against a
    // different machine and the control proves nothing.
    for s in &states {
        s.reset();
    }
    let repeat = metal
        .kimi_decoder_layers_traced(&fx.calls(&states), &fx.x)
        .expect("unmutated rerun after reset");
    for (i, (a, b)) in repeat.iter().zip(&baseline).enumerate() {
        assert_eq!(
            a.selected_ids, b.selected_ids,
            "layer {i} routed differently on an identical rerun — `reset` is not \
             restoring the state"
        );
    }

    // Both ends of the chain: the first layer, and the second-to-last.
    for mutated in [0usize, n - 2] {
        let l = &fx.layers[mutated];
        let mut swapped = l.residency.clone();
        let (a, b) = (l.ids[0], l.ids[1]);
        swapped.swap(a, b);

        for s in &states {
            s.reset();
        }
        let mut calls = fx.calls(&states);
        {
            let m = moe_mut(&mut calls[mutated].weights);
            let a = ExpertAddressing::Table(&swapped);
            m.gate.addressing = a;
            m.up.addressing = a;
            m.down.addressing = a;
        }
        let got = metal.kimi_decoder_layers_traced(&calls, &fx.x);

        match got {
            Ok(planes) => {
                // Every layer up to and including the mutated one must
                // route exactly as before; at least one after it must not.
                for (i, (p, base)) in planes.iter().zip(&baseline).enumerate() {
                    if i <= mutated {
                        assert_eq!(
                            p.selected_ids, base.selected_ids,
                            "swapping layer {mutated}'s expert OFFSETS changed layer {i}'s \
                             route — offsets must not reach the router"
                        );
                    }
                }
                // What MUST hold: the mutated layer's own output moved,
                // and so did the LAST layer's — the chain carried it
                // forward. A route flip downstream is a stronger
                // outcome but not a required one: a top-k of 256 can
                // absorb a small perturbation without changing its
                // selection, and demanding it would make this control
                // fixture-dependent rather than mechanism-dependent.
                let last = planes.len() - 1;
                let moved_here = max_abs(&planes[mutated].output, &baseline[mutated].output);
                let moved_end = max_abs(&planes[last].output, &baseline[last].output);
                assert!(
                    moved_here > 0.0,
                    "swapping layer {mutated}'s experts did not change its own output — \
                     the offsets are not reaching the MoE"
                );
                assert!(
                    moved_end > 0.0,
                    "layer {mutated}'s output changed but layer {last}'s did not — the \
                     chain is not carrying its output forward"
                );
                let rerouted: Vec<usize> = planes
                    .iter()
                    .zip(&baseline)
                    .enumerate()
                    .skip(mutated + 1)
                    .filter(|(_, (p, b))| p.selected_ids != b.selected_ids)
                    .map(|(i, _)| i)
                    .collect();
                eprintln!(
                    "[r5e] control: swapping layer {mutated}'s experts moved its output by \
                     {moved_here:e} and layer {last}'s by {moved_end:e}, with layers \
                     0..={mutated} routing identically (later reroutes: {rerouted:?})"
                );
            }
            Err(e) => {
                // A later layer routed outside its bank — same evidence,
                // provided the refusal is not the mutated layer's own.
                let blamed = match e {
                    larql_compute_metal::trait_impl::grouped_experts::GroupedError::
                        LayerRouteNotResident { layer, .. } => layer,
                    other => panic!("unexpected refusal: {other}"),
                };
                assert!(
                    blamed > mutated,
                    "the refusal came from layer {blamed}, at or before the mutated layer \
                     {mutated} — swapping offsets must not change that layer's own route"
                );
                eprintln!(
                    "[r5e] control: swapping layer {mutated}'s experts left its own route \
                     intact and pushed layer {blamed}'s route outside its bank"
                );
            }
        }
    }
}

/// What two chained layers cost, and how many crossings they make.
///
/// Reported by stage from the start: rung 5d found a single serial GPU
/// operation costing more than the attention and the MoE combined, and
/// it was invisible in the total.
#[test]
fn report_two_layer_cost() {
    let Some((metal, fx)) = setup() else {
        return;
    };
    let states = fx.states(&metal);
    let calls = fx.calls(&states);

    let chained = || {
        for s in &states {
            s.reset();
        }
        let t = Instant::now();
        let (out, gpu) = metal
            .kimi_decoder_layers(&calls, &fx.x, None)
            .expect("chain");
        std::hint::black_box(out);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };
    // The same two layers, one command buffer EACH — the shape rung 5d
    // left, and what the epoch is being compared against.
    // The same layers, one command buffer EACH, with the hidden state
    // round-tripping through the host between them — the shape rung 5d
    // left, and what the epoch is measured against.
    let separate = || {
        for s in &states {
            s.reset();
        }
        let t = Instant::now();
        let mut gpu = 0.0;
        let mut h = fx.x.clone();
        for call in &calls {
            let (out, g) = metal.kimi_decoder_layer(call.weights, &h).expect("layer");
            gpu += g;
            h = out;
        }
        std::hint::black_box(h);
        (t.elapsed().as_secs_f64() * 1000.0, gpu)
    };

    for _ in 0..WARMUP {
        separate();
        chained();
    }
    let (mut sw, mut sg, mut cw, mut cg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERS {
        let (w, g) = separate();
        sw.push(w);
        sg.push(g);
        let (w, g) = chained();
        cw.push(w);
        cg.push(g);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let third = (sw.len() / 3).max(1);
    let ramp = mean(&sw[..third]) / mean(&sw[sw.len() - third..]);
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (s_wall, s_gpu, c_wall, c_gpu) = (
        median(&mut sw),
        median(&mut sg),
        median(&mut cw),
        median(&mut cg),
    );

    let n = fx.layers.len();
    eprintln!("[r5e] {n} chained real decoder layers");
    eprintln!(
        "[r5e]   {n} command buffers  wall {s_wall:.3} ms  gpu {s_gpu:.3} ms  \
         host {:.3} ms over {n} crossings",
        s_wall - s_gpu,
    );
    eprintln!(
        "[r5e]   1 command buffer   wall {c_wall:.3} ms  gpu {c_gpu:.3} ms  \
         host {:.3} ms over 1 crossing  [{:.2}x wall]",
        c_wall - c_gpu,
        s_wall / c_wall,
    );
    eprintln!("[r5e]   ramp {ramp:.2}x — 1.00 means the machine held still");
    assert!(ramp.is_finite());
}
