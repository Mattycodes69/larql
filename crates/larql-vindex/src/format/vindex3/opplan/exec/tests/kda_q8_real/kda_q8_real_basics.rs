use super::*;

#[test]
fn one_kda_layers_projections_at_q8_through_the_consequence_metrics() {
    let (Some(source_dir), Some(bank_dir)) = (env_dir(SOURCE_ENV), env_dir(BANK_ENV)) else {
        eprintln!("skipped: set {SOURCE_ENV} and {BANK_ENV}");
        return;
    };
    if std::env::var("LARQL_RESIDENCY_SET").is_ok() {
        panic!("unset LARQL_RESIDENCY_SET: this run must use implicit residency");
    }
    let Some(metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let layers = target_layers();
    let sequences = env_count(SEQUENCES_ENV, SEQUENCES_DEFAULT);

    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bank_dir.join("manifest.json")).expect("manifest"))
            .expect("bank manifest parses");
    let positions = manifest["positions"].as_u64().unwrap() as usize;

    let t0 = Instant::now();
    let model = KimiSourceModel::open(&source_dir).expect("source container opens");
    let g = model.geometry.clone();
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    model
        .register_stores(&metal, &moe_layers)
        .expect("stores register");
    // Optional CROSS-FAMILY composition: an expert candidate (compiled
    // banks, q2a's own overlay machinery) beside the transient KDA
    // requant. The baseline arm binds neither.
    let overlay = env_dir(CANDIDATE_ENV).map(|dir| {
        let o = CandidateOverlay::open(&dir, &source_dir, &g).expect("candidate overlay opens");
        o.register_store(&metal);
        o
    });
    let expert_layers: Vec<u32> = overlay
        .as_ref()
        .map(|o| o.compiled_layers().to_vec())
        .unwrap_or_default();
    let q8h = head_q8();
    let mla_layers = layer_list(MLA_ENV);
    let shared_layers = layer_list(SHARED_ENV);
    assert!(
        !layers.is_empty()
            || overlay.is_some()
            || q8h
            || !mla_layers.is_empty()
            || !shared_layers.is_empty(),
        "an empty scope with no overlay and no head flag measures nothing"
    );

    // ── Experimental identity, resolved ONCE and checked BEFORE any layer
    // loads. REAL-EVIDENCE-1: a stale shell measured `-mla23-26-headq8`
    // while the programme believed it was reproducing the flagship arm,
    // and every integrity check passed. The declaration decides. ──
    let scope = RuntimeScope::resolved(
        layers.clone(),
        mla_layers.clone(),
        shared_layers.clone(),
        q8h,
        raw_scope_env(),
    );
    let overlay_name = overlay.as_ref().map(|o| o.index.map.name.clone());
    let label = run_label(
        &layers,
        overlay_name.as_deref(),
        &mla_layers,
        &shared_layers,
        q8h,
    );
    let run = run_name(&label);
    let declared_path = env_dir(EXPECT_IDENTITY_ENV);
    let declared = declared_path
        .as_ref()
        .map(|p| DeclaredIdentity::load(p).unwrap_or_else(|e| panic!("[identity] {e}")));
    match &declared {
        Some(d) => {
            d.check_transition(&run, overlay_name.as_deref(), &scope)
                .unwrap_or_else(|r| panic!("[identity] REFUSED before loading any layer — {r}"));
            eprintln!(
                "[identity] stage 1 PASSED: run {run}, expert_candidate {overlay_name:?}, \
                 scope {} — matches {}",
                scope.describe(),
                declared_path.as_ref().unwrap().display()
            );
        }
        None => eprintln!(
            "[identity] UNCHECKED: {EXPECT_IDENTITY_ENV} unset; run {run}, scope {}",
            scope.describe()
        ),
    }
    let (base_layers, none) = build_layers(&metal, &model, &[], None);
    let (cand_layers, swapped) = build_layers_scoped(
        &metal,
        &model,
        &layers,
        &mla_layers,
        &shared_layers,
        overlay.as_ref(),
    );
    assert!(none.is_empty());
    assert_eq!(
        swapped.len(),
        layers.len() + mla_layers.len() + shared_layers.len(),
        "every target was re-encoded"
    );
    let bf16_bytes: usize = swapped.iter().map(|(b, _)| b).sum();
    let q8_bytes: usize = swapped.iter().map(|(_, q)| q).sum();
    assert!(
        layers.is_empty() || q8_bytes < bf16_bytes,
        "Q8_0 must be smaller than bf16 — the swap did not happen"
    );
    if let Some(d) = &declared {
        d.check_bytes(bf16_bytes, q8_bytes)
            .unwrap_or_else(|r| panic!("[identity] REFUSED before measuring — {r}"));
        eprintln!("[identity] stage 2 PASSED: bytes bf16 {bf16_bytes} q8_0 {q8_bytes}");
    }
    // Attribution BEFORE assembly: proven on the layers themselves, so
    // "identical except the targets' projections" is a checked fact,
    // not a construction argument.
    assert_arms_differ_only_at(
        &base_layers,
        &cand_layers,
        &layers,
        &mla_layers,
        &shared_layers,
        &expert_layers,
    );
    let (null_layers, _) = build_layers(&metal, &model, &[], None);
    let mut baseline = assemble(&metal, &model, base_layers);
    let mut candidate = assemble_with_head(&metal, &model, cand_layers, q8h);
    let mut null_partner = assemble(&metal, &model, null_layers);
    metal.seal_weight_regions();
    eprintln!(
        "[kda-q8] arms loaded in {:.1}s; layers {layers:?} projections {bf16_bytes} -> \
         {q8_bytes} bytes ({:.1}% of bf16); every other bank byte-equal or \
         pointer-identical, PROVEN above",
        t0.elapsed().as_secs_f64(),
        100.0 * q8_bytes as f64 / bf16_bytes as f64,
    );

    // ── Null arm: BF16 against itself must be EXACTLY zero — the same
    // instrument-integrity gate the quality runner carries. ──
    {
        let mut builder = BankBuilder::new();
        for seq in 0..NULL_SEQUENCES {
            let rows = sequence_embeddings(&bank_dir, seq, positions, g.hidden)
                .expect("the corpus sequence reads");
            let a = run_sequence(&metal, &mut baseline, &rows, g.hidden).expect("the arm runs");
            let b = run_sequence(&metal, &mut null_partner, &rows, g.hidden).expect("the arm runs");
            for (pos, ((la, ta), (lb, tb))) in a.into_iter().zip(b).enumerate() {
                builder.observe(&observation(seq, pos, &la, &ta, &lb, &tb));
            }
        }
        let null_bank = builder.finish();
        assert_eq!(
            null_bank.logits.kl_p99, 0.0,
            "null arm KL must be exactly zero"
        );
        assert_eq!(null_bank.logits.max_logit_delta, 0.0);
        assert_eq!(null_bank.logits.top1_flips, 0);
        assert_eq!(null_bank.routing.route_flips, 0);
        eprintln!(
            "[kda-q8] null arm: {} positions, everything exactly zero",
            null_bank.positions
        );
    }
    // The null partner is a THIRD full stack (~17 GB of wired
    // attention banks) needed only for the instrument check above.
    // Holding it through the measurement pushed the process against
    // the wired-collector wall at four-family scale — the mid-run
    // flat-instrument episode — so it is released here, before the
    // long run begins.
    drop(null_partner);

    let t1 = Instant::now();
    let mut builder = BankBuilder::new();

    // ── REAL-EVIDENCE-1: record THIS arm, and only this one. ──
    //
    // The null loop above is instrument validation, not samples from the
    // candidate transition, and its exact-zero assertions are its own
    // evidence. Mixing it into this stream would put non-samples in a
    // file whose whole purpose is to be resampled.
    //
    // The candidate is the overlay PLUS the runtime requant applied by
    // `build_layers_scoped`, so the manifest carries the RESOLVED scope.
    // Without it a stream is well-formed, hash-verified, and about a
    // different intervention.
    let stream_label = std::env::var(LABEL_ENV).unwrap_or_else(|_| "unlabelled-kda-q8".to_string());
    let record_dir = env_dir(RECORD_ENV).map(|root| stream_dir(&root, &stream_label));
    let mut recorder = record_dir.as_ref().map(|dir| {
        StreamWriter::create(
            dir,
            &StreamIdentity {
                source_identity: source_dir.display().to_string(),
                candidate_identity: overlay_name.clone().unwrap_or_else(|| "no-overlay".into()),
                scope: scope.clone(),
                producer: "kda_q8_real measurement arm".into(),
                protocol_identity: TEACHER_FORCED_TWO_ARM.to_string(),
                code_identity: code_identity(),
                bank_identity: bank_dir.display().to_string(),
                draw_identity: stream_label.clone(),
                sequences: sequences as u32,
                positions_per_sequence: positions as u32,
            },
        )
        .expect("the observation stream opens")
    });

    // KL by position index, across sequences — the token-distance curve.
    let mut kl_by_pos: Vec<Vec<f64>> = vec![Vec::new(); positions];
    for seq in 0..sequences {
        let rows = sequence_embeddings(&bank_dir, seq, positions, g.hidden)
            .expect("the corpus sequence reads");
        let base = run_sequence(&metal, &mut baseline, &rows, g.hidden).expect("the arm runs");
        let cand = run_sequence(&metal, &mut candidate, &rows, g.hidden).expect("the arm runs");
        for (pos, ((lb, tb), (lc, tc))) in base.into_iter().zip(cand).enumerate() {
            kl_by_pos[pos].push(full_kl(&lb, &lc));
            // The SAME instance reaches both sinks. A reconstruction
            // would be a different experiment from the one measured.
            let obs = observation(seq, pos, &lb, &tb, &lc, &tc);
            if let Some(r) = recorder.as_mut() {
                r.record(&obs).expect("the observation stream accepts");
            }
            builder.observe(&obs);
        }
    }
    let recorded = recorder.take().map(|r| {
        r.finish()
            .expect("the observation stream completes and hashes")
    });
    let min_covered = builder.min_covered_mass();
    let bank = builder.finish();
    if let Some(m) = &recorded {
        // Only the bank can say whether recording stopped early: the
        // manifest's count comes from the writer's own counter, so a
        // truncated stream is self-consistent.
        assert_eq!(
            m.observations, bank.positions,
            "recorded {} observations against {} measured positions; a partial \
             stream is not evidence about this run",
            m.observations, bank.positions
        );
        eprintln!(
            "[stream] {} observations, {} sequences, scope {:?}, sha {}",
            m.observations, m.sequences, m.scope, m.stream_sha256
        );
    }
    let curve: Vec<serde_json::Value> = kl_by_pos
        .iter()
        .map(|v| {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).expect("KL is finite"));
            let mean = s.iter().sum::<f64>() / s.len() as f64;
            serde_json::json!({"mean": mean, "max": s[s.len() - 1]})
        })
        .collect();
    // Quartile means of the curve, so decay/hold/accumulate reads off
    // one line without a plot.
    let q = positions / 4;
    let qmean = |r: std::ops::Range<usize>| {
        let vals: Vec<f64> = r.flat_map(|p| kl_by_pos[p].iter().copied()).collect();
        vals.iter().sum::<f64>() / vals.len() as f64
    };
    eprintln!(
        "[kda-q8] token-distance KL means by quartile of position 0..{positions}: \
         {:.3e} | {:.3e} | {:.3e} | {:.3e}",
        qmean(0..q),
        qmean(q..2 * q),
        qmean(2 * q..3 * q),
        qmean(3 * q..positions),
    );

    let evidence = QualityEvidence {
        gate: kimi_logit_v3(),
        bank: bank.clone(),
    };
    let verdict = evidence.verdict();
    // v3 is the strict authority every earlier cell cited; balanced-v1
    // is the frozen contract a COMPOSED map is admitted under. Both
    // travel in the artifact so a reader never has to re-derive the
    // verdict this run's claim rests on from the bank by hand.
    let balanced_gate = kimi_logit_balanced_v1();
    let balanced = balanced_gate.evaluate(&bank);
    let bank_manifest_sha256 = {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(bank_dir.join("manifest.json")).expect("bank manifest reads");
        format!("{:x}", Sha256::digest(&bytes))
    };
    let report = serde_json::json!({
        "run": run,
        "scope": "KDA projections (qkv bank + o_proj), transient requant; optional compiled expert candidate beside it",
        "runtime_scope": scope,
        "declared_identity": declared_path.as_ref().map(|p| p.display().to_string()),
        "code_identity": code_identity(),
        "source_identity": source_dir.display().to_string(),
        "stream": recorded.as_ref().map(|m| serde_json::json!({
            "dir": record_dir.as_ref().map(|d| d.display().to_string()),
            "observations": m.observations,
            "stream_sha256": m.stream_sha256,
        })),
        "expert_candidate": overlay_name,
        "gate": evidence.gate,
        "authority_report": evidence.report(),
        "verdict_passed": verdict.passed(),
        "verdict_failures": verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "balanced_gate": balanced_gate,
        "verdict_balanced_v1_passed": balanced.passed(),
        "verdict_balanced_v1_failures": balanced.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "bank_manifest_sha256": bank_manifest_sha256,
        "bank": bank,
        "positions": bank.positions,
        "min_covered_mass": min_covered,
        "bytes": {"bf16": bf16_bytes, "q8_0": q8_bytes},
        "kl_by_position": curve,
        "wall_seconds": t1.elapsed().as_secs_f64(),
    });
    let path = format!("/tmp/kimi_kda-q8-l{label}_report.json");
    let serialised = serde_json::to_vec_pretty(&report).expect("serialises");
    std::fs::write(&path, &serialised).expect("report writes");
    // The summary belongs beside the stream it summarises — /tmp does
    // not survive the night, and the replay gate needs both.
    if let Some(dir) = &record_dir {
        std::fs::write(dir.join(REPORT_BESIDE_STREAM), &serialised).expect("report writes");
    }
    eprintln!("{}", evidence.report());
    eprintln!("[kda-q8] verdict (v3): {verdict:?}");
    eprintln!("[kda-q8] verdict (balanced-v1): {balanced:?} — report at {path}");

    if bank.positions < 4096 {
        assert!(!verdict.passed(), "sub-4096 positions can never pass v3");
        assert!(verdict
            .failures
            .iter()
            .any(|(c, d)| *c == Criterion::Positions && d.contains("< 4096")));
    }
}
