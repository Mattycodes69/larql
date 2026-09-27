//! The teacher-forced two-arm measurement run.

use crate::format::vindex3::opplan::exec::kimi_source::{CandidateOverlay, KimiSourceModel};
use crate::format::vindex3::represent::bank::BankBuilder;
use crate::format::vindex3::represent::measure::outcome::{
    ExecutionFailure, Inadmissible, MeasurementRefusal, VerifiedFacts,
};
use crate::format::vindex3::represent::measure::TeacherForcedRequest;
use crate::format::vindex3::represent::measure::TEACHER_FORCED_TWO_ARM;
use crate::format::vindex3::represent::observation_stream::{
    stream_dir, StreamIdentity, StreamWriter,
};
use crate::format::vindex3::represent::physical::{ExpertEncoding, ProjectionAddressing};
use crate::format::vindex3::represent::quality::{kimi_logit_v1, kimi_logit_v2, QualityEvidence};
use larql_compute::backend::ComputeBackend;
use larql_compute_metal::MetalBackend;
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

#[allow(unused_imports)]
use super::*;

/// **Run the teacher-forced two-arm measurement.**
///
/// Every condition that made this trustworthy as a `#[cfg(test)]`
/// harness is preserved as a typed refusal — see
/// [`outcome`](super::super::outcome) for the conservation inventory naming
/// where each one went. A test runner would have turned a violated
/// `assert!` red; a caller gets `Inadmissible` instead, and can tell it
/// from an `ExecutionFailure` that is worth retrying.
pub fn measure_teacher_forced(
    request: &TeacherForcedRequest,
) -> Result<MeasurementReceipt, MeasurementRefusal> {
    // Refused before anything is loaded: an unknown gate, an unknown
    // procedure, an empty slice, a missing artifact.
    let requested_gate = request.admit().map_err(|e| {
        MeasurementRefusal::Execution(ExecutionFailure::ArtifactUnreadable {
            what: "request".into(),
            path: request.source.display().to_string(),
            detail: e.to_string(),
        })
    })?;
    let (source_dir, candidate_dir, bank_dir) = (
        request.source.clone(),
        request.candidate.clone(),
        request.quality_bank.clone(),
    );
    let sequences = || request.sequences;
    let mut verified = VerifiedFacts::default();
    // The residency SET would try to wire ~94 GB of expert bank, past
    // the wired-collector wall (~45 GB). This run uses registered
    // regions under IMPLICIT residency; refusing is better than
    // silently degrading into a measurement of the collector.
    if std::env::var("LARQL_RESIDENCY_SET").is_ok() {
        return Err(Inadmissible::ResidencyModeWouldBeMeasured {
            detail: "LARQL_RESIDENCY_SET is set: the run would wire ~94 GB of expert bank \
                     past the ~45 GB collector wall and measure the collector"
                .into(),
        }
        .into());
    }
    let Some(metal) = MetalBackend::new() else {
        return Err(ExecutionFailure::BackendUnavailable {
            detail: "MetalBackend::new() returned None — the shader library failed to build".into(),
        }
        .into());
    };

    let unreadable = |what: &str, path: &Path, detail: String| {
        MeasurementRefusal::Execution(ExecutionFailure::ArtifactUnreadable {
            what: what.into(),
            path: path.display().to_string(),
            detail,
        })
    };
    let manifest_path = bank_dir.join("manifest.json");
    let manifest: Value = std::fs::read(&manifest_path)
        .map_err(|e| unreadable("quality bank manifest", &manifest_path, e.to_string()))
        .and_then(|b| {
            serde_json::from_slice(&b)
                .map_err(|e| unreadable("quality bank manifest", &manifest_path, e.to_string()))
        })?;
    let field = |k: &str| {
        manifest[k].as_u64().map(|v| v as usize).ok_or_else(|| {
            unreadable(
                "quality bank manifest",
                &manifest_path,
                format!("`{k}` is missing or not a number"),
            )
        })
    };
    let positions_per_seq = field("positions")?;
    let bank_hidden = field("hidden")?;

    let t0 = Instant::now();
    let model = KimiSourceModel::open(&source_dir)
        .map_err(|e| unreadable("source container", &source_dir, e.to_string()))?;
    let g = model.geometry.clone();
    if g.hidden != bank_hidden {
        return Err(Inadmissible::CorpusNotForThisModel {
            corpus_hidden: bank_hidden,
            model_hidden: g.hidden,
        }
        .into());
    }
    let overlay = CandidateOverlay::open(&candidate_dir, &source_dir, &g)
        .map_err(|e| unreadable("candidate overlay", &candidate_dir, e.to_string()))?;
    if overlay.compiled_layers().is_empty() {
        return Err(Inadmissible::CandidateCompilesNothing.into());
    }
    verified.compiled_layers = overlay.compiled_layers().to_vec();
    verified.compiled_projections = overlay.compiled_projections().to_vec();
    eprintln!(
        "[q2a] source + candidate opened in {:.1}s; candidate compiles layers {:?}",
        t0.elapsed().as_secs_f64(),
        overlay.compiled_layers()
    );
    // The EARLIEST compiled layer (verify_complete sorts): the one
    // layer whose router input is untouched in BOTH arms, so its own
    // routing must be identical — the cascade/local discriminator. A
    // composed map's later compiled layers see moved hidden states and
    // may legitimately route differently.
    let overlay_layer = overlay.compiled_layers()[0] as usize;
    let compiled_set: Vec<usize> = overlay
        .compiled_layers()
        .iter()
        .map(|&l| l as usize)
        .collect();

    // ── Load both arms; the loader refuses on any missing operand. ──
    let t1 = Instant::now();
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    let registered = model
        .register_stores(&metal, &moe_layers)
        .map_err(|e| unreadable("source stores", &source_dir, e.to_string()))?;
    overlay.register_store(&metal);
    let mut baseline = build_stack(&metal, &model, None)?;
    let mut candidate = build_stack(&metal, &model, Some(&overlay))?;
    metal.seal_weight_regions();
    eprintln!(
        "[q2a] both arms loaded in {:.1}s ({registered} mmap store regions registered, \
         implicit residency)",
        t1.elapsed().as_secs_f64()
    );
    // ── Physical attribution: the intended stores are the bound ones,
    //    at EVERY compiled layer — a composed map earns the check per
    //    layer, at that layer's own encoding. Every violation below is
    //    INADMISSIBLE rather than a failure: the logits would still
    //    compute, and would be a measurement of something else.
    for &probe_layer in &compiled_set {
        let probe_b = model.device_layer(&metal, probe_layer, None).map_err(|e| {
            ExecutionFailure::LayerIncomplete {
                layer: probe_layer,
                detail: e.to_string(),
            }
        })?;
        let probe_c = model
            .device_layer(&metal, probe_layer, Some(&overlay))
            .map_err(|e| ExecutionFailure::LayerIncomplete {
                layer: probe_layer,
                detail: e.to_string(),
            })?;
        let unexpected = |arm: &str, projection: &str, want: &str, got: &str| {
            MeasurementRefusal::Inadmissible(Inadmissible::UnexpectedPhysicalRead {
                arm: arm.into(),
                layer: probe_layer,
                projection: projection.into(),
                expected_store: want.into(),
                actual_store: got.into(),
            })
        };

        // The BASELINE arm is entirely source-backed, whatever the
        // candidate's scope.
        for (name, proj) in [
            ("gate", &probe_b.bank.gate),
            ("up", &probe_b.bank.up),
            ("down", &probe_b.bank.down),
        ] {
            if proj.store_id() != SOURCE_EXPERT_BANK {
                return Err(unexpected(
                    "baseline",
                    name,
                    SOURCE_EXPERT_BANK,
                    proj.store_id(),
                ));
            }
            if proj.encoding() != ExpertEncoding::Bf16 {
                return Err(unexpected(
                    "baseline",
                    name,
                    "BF16",
                    &format!("{:?}", proj.encoding()),
                ));
            }
        }

        // The CANDIDATE arm substitutes exactly the projections the
        // overlay compiled, and leaves the rest source-backed — the
        // asymmetry a projection-scoped experiment IS.
        let compiled: Vec<&str> = overlay
            .compiled_projections()
            .iter()
            .map(String::as_str)
            .collect();
        for (name, proj, spelling) in [
            ("gate", &probe_c.bank.gate, "w1"),
            ("up", &probe_c.bank.up, "w3"),
            ("down", &probe_c.bank.down, "w2"),
        ] {
            let want_candidate = compiled.contains(&spelling);
            let (store, enc) = if want_candidate {
                (
                    CANDIDATE_BANK,
                    overlay.encoding_of(probe_layer as u32).map_err(|e| {
                        MeasurementRefusal::Inadmissible(Inadmissible::AddressingMismatch {
                            layer: probe_layer,
                            projection: name.into(),
                            detail: format!(
                                "the layer is in the compiled set and declares no encoding: {e}"
                            ),
                        })
                    })?,
                )
            } else {
                (SOURCE_EXPERT_BANK, ExpertEncoding::Bf16)
            };
            if proj.store_id() != store {
                return Err(unexpected("candidate", name, store, proj.store_id()));
            }
            if proj.encoding() != enc {
                return Err(unexpected(
                    "candidate",
                    name,
                    &format!("{enc:?}"),
                    &format!("{:?}", proj.encoding()),
                ));
            }
            // A source-backed projection of the candidate arm must be
            // the SAME BYTES as the baseline's — pointer-identical, so
            // the only difference between the arms is the compiled one.
            if !want_candidate {
                let b = match name {
                    "gate" => &probe_b.bank.gate,
                    "up" => &probe_b.bank.up,
                    _ => &probe_b.bank.down,
                };
                if proj.region.region.bytes().as_ptr() != b.region.region.bytes().as_ptr() {
                    return Err(Inadmissible::ProtectedOperandChanged {
                        layer: probe_layer,
                        projection: name.into(),
                    }
                    .into());
                }
            }
        }

        // Only a COMPILED projection is identity-addressed. A
        // projection-scoped candidate leaves the others table-addressed
        // over the source, and that asymmetry inside one layer is the
        // thing this binding exists to express.
        for (name, proj) in [
            ("gate", &probe_c.bank.gate),
            ("up", &probe_c.bank.up),
            ("down", &probe_c.bank.down),
        ] {
            let is_candidate = proj.store_id() == CANDIDATE_BANK;
            let mismatch = |detail: String| {
                MeasurementRefusal::Inadmissible(Inadmissible::AddressingMismatch {
                    layer: probe_layer,
                    projection: name.into(),
                    detail,
                })
            };
            match (&proj.addressing, is_candidate) {
                (ProjectionAddressing::Identity { experts, .. }, true) => {
                    if *experts != g.experts {
                        return Err(mismatch(format!(
                            "compiled, so it must address every one of {} experts by identity \
                             — any route, including one the baseline never took, must resolve \
                             — and it addresses {experts}",
                            g.experts
                        )));
                    }
                }
                (ProjectionAddressing::Table(_), false) => {}
                (a, c) => return Err(mismatch(format!("compiled={c} but addressed by {a:?}"))),
            }
        }

        let shared = probe_c
            .bank
            .shared
            .as_ref()
            .ok_or_else(|| unexpected("candidate", "shared", SOURCE_DECODER_STACK, "absent"))?;
        if shared.gate.store_id() != SOURCE_DECODER_STACK {
            return Err(unexpected(
                "candidate",
                "shared",
                SOURCE_DECODER_STACK,
                shared.gate.store_id(),
            ));
        }
        if shared.gate.encoding != ExpertEncoding::Bf16 {
            return Err(unexpected(
                "candidate",
                "shared",
                "BF16",
                &format!("{:?}", shared.gate.encoding),
            ));
        }

        // A layer the overlay does NOT compile is the same physical
        // bytes in both arms — pointer-identical, not merely same-named.
        let neighbour = (g.dense_prefix_layers..g.num_layers)
            .find(|l| !compiled_set.contains(l))
            .ok_or(MeasurementRefusal::Inadmissible(
                Inadmissible::CandidateCompilesNothing,
            ))?;
        let nb = model.device_layer(&metal, neighbour, None).map_err(|e| {
            ExecutionFailure::LayerIncomplete {
                layer: neighbour,
                detail: e.to_string(),
            }
        })?;
        let nc = model
            .device_layer(&metal, neighbour, Some(&overlay))
            .map_err(|e| ExecutionFailure::LayerIncomplete {
                layer: neighbour,
                detail: e.to_string(),
            })?;
        if nc.bank.gate.store_id() != SOURCE_EXPERT_BANK {
            return Err(unexpected(
                "candidate",
                "neighbour gate",
                SOURCE_EXPERT_BANK,
                nc.bank.gate.store_id(),
            ));
        }
        if nb.bank.gate.region.region.bytes().as_ptr()
            != nc.bank.gate.region.region.bytes().as_ptr()
        {
            return Err(Inadmissible::ProtectedOperandChanged {
                layer: neighbour,
                projection: "neighbour gate".into(),
            }
            .into());
        }
        verified.invariant_neighbour_layer = Some(neighbour);

        // The bytes the loader reads must be the bytes the compiler
        // sealed. Without this, "the candidate" is whatever is on disk
        // under a path the overlay names.
        let checked = overlay
            .verify_reads_match_seals(probe_layer as u32, 16)
            .map_err(|e| Inadmissible::SealMismatch {
                layer: probe_layer as u32,
                detail: e.to_string(),
            })?;
        verified.seal_checked_operands += checked;
        verified.attribution_checked_layers.push(probe_layer);
    }

    // ── The null arm: BF16 against itself must be EXACTLY zero. ──
    let t2 = Instant::now();
    {
        let mut null_partner = build_stack(&metal, &model, None)?;
        let mut builder = BankBuilder::new();
        for seq in 0..NULL_SEQUENCES {
            let rows = sequence_embeddings(&bank_dir, seq, positions_per_seq, g.hidden)?;
            let a = run_sequence(&metal, &mut baseline, &rows, g.hidden)?;
            let b = run_sequence(&metal, &mut null_partner, &rows, g.hidden)?;
            for (pos, ((la, ta), (lb, tb))) in a.into_iter().zip(b).enumerate() {
                builder.observe(&observation(seq, pos, &la, &ta, &lb, &tb));
            }
        }
        let null_bank = builder.finish();
        let want = (NULL_SEQUENCES * positions_per_seq) as u64;
        if null_bank.positions != want {
            return Err(Inadmissible::PositionCountMismatch {
                expected: want,
                measured: null_bank.positions,
            }
            .into());
        }
        // **The determinism control.** BF16 against itself must be
        // exactly zero. If it is not, the device is injecting
        // nondeterminism and every downstream KL, flip and route
        // statistic is artifact — while attribution, seals, pointer
        // identity and position counts all still pass.
        for (statistic, observed) in [
            ("kl_p99", null_bank.logits.kl_p99),
            ("max_logit_delta", null_bank.logits.max_logit_delta),
            ("top1_flips", null_bank.logits.top1_flips as f64),
            ("top10_changes", null_bank.logits.top10_changes as f64),
            ("route_flips", null_bank.routing.route_flips as f64),
            (
                "positions_with_route_change",
                null_bank.routing.positions_with_route_change as f64,
            ),
        ] {
            if observed != 0.0 {
                return Err(Inadmissible::NullArmNotZero {
                    statistic: statistic.into(),
                    observed,
                }
                .into());
            }
        }
        eprintln!(
            "[q2a] null arm: {} positions, everything exactly zero ({:.1}s)",
            null_bank.positions,
            t2.elapsed().as_secs_f64()
        );
    }

    // A recording failure ABORTS the run. A partially recorded stream
    // would carry a manifest built from the writer's own counter, so it
    // would be self-consistent and pass `read_stream` while describing
    // less than the run measured. Failing loudly is the only way that
    // cannot happen silently.
    fn recording_failed(
        e: crate::format::vindex3::represent::observation_stream::StreamError,
        path: &std::path::Path,
    ) -> MeasurementRefusal {
        MeasurementRefusal::Execution(ExecutionFailure::ArtifactUnreadable {
            what: "observation stream".into(),
            path: path.display().to_string(),
            detail: e.to_string(),
        })
    }

    // ── The measurement: 32 sequences x 32 teacher-forced positions. ──
    let t3 = Instant::now();
    let mut builder = BankBuilder::new();
    // REAL-EVIDENCE-1. Opt-in: unset, nothing below changes. The
    // recorder is handed the SAME observation the bank sees, never a
    // reconstruction of one.
    let record_to = env_dir(RECORD_ENV).map(|root| stream_dir(&root, &request.label));
    let mut recorder = match record_to.as_ref() {
        Some(dir) => Some(
            StreamWriter::create(
                dir,
                &StreamIdentity {
                    source_identity: source_dir.display().to_string(),
                    candidate_identity: candidate_dir.display().to_string(),
                    // Overlay-only: this path applies no runtime requant.
                    // It is NOT REAL-EVIDENCE-1's historical arm.
                    scope: Default::default(),
                    producer: "measure_teacher_forced (overlay-only)".into(),
                    protocol_identity: TEACHER_FORCED_TWO_ARM.to_string(),
                    code_identity: option_env!("VERGEN_GIT_SHA")
                        .unwrap_or("unknown")
                        .to_string(),
                    bank_identity: bank_dir.display().to_string(),
                    draw_identity: request.label.clone(),
                    sequences: sequences() as u32,
                    positions_per_sequence: positions_per_seq as u32,
                },
            )
            .map_err(|e| recording_failed(e, dir))?,
        ),
        None => None,
    };
    let mut candidate_l1_routed: std::collections::BTreeSet<u32> =
        std::collections::BTreeSet::new();
    let mut baseline_l1_routed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for seq in 0..sequences() {
        let rows = sequence_embeddings(&bank_dir, seq, positions_per_seq, g.hidden)?;
        let base = run_sequence(&metal, &mut baseline, &rows, g.hidden)?;
        let cand = run_sequence(&metal, &mut candidate, &rows, g.hidden)?;
        for (pos, ((lb, tb), (lc, tc))) in base.into_iter().zip(cand).enumerate() {
            baseline_l1_routed.extend(tb.routes[overlay_layer].iter().copied());
            candidate_l1_routed.extend(tc.routes[overlay_layer].iter().copied());
            let o = observation(seq, pos, &lb, &tb, &lc, &tc);
            builder.observe(&o);
            if let Some(r) = recorder.as_mut() {
                r.record(&o)
                    .map_err(|e| recording_failed(e, record_to.as_ref().unwrap()))?;
            }
        }
    }
    let recorded = match recorder.take() {
        Some(r) => Some(
            r.finish()
                .map_err(|e| recording_failed(e, record_to.as_ref().unwrap()))?,
        ),
        None => None,
    };
    let min_covered = builder.min_covered_mass();
    let bank = builder.finish();
    // **The stream must describe the whole run.** The manifest's count
    // comes from the writer's own counter, so only the BANK can say
    // whether recording stopped early.
    if let Some(m) = &recorded {
        if m.observations != bank.positions {
            return Err(MeasurementRefusal::Execution(
                ExecutionFailure::ArtifactUnreadable {
                    what: "observation stream".into(),
                    path: record_to.as_ref().unwrap().display().to_string(),
                    detail: format!(
                        "recorded {} observations but the bank measured {} positions; \
                         a partial stream is not evidence about this run",
                        m.observations, bank.positions
                    ),
                },
            ));
        }
        eprintln!(
            "recorded {} observations over {} sequences -> {}",
            m.observations, m.sequences, m.stream_sha256
        );
    }
    let run_s = t3.elapsed().as_secs_f64();
    let want_positions = (sequences() * positions_per_seq) as u64;
    if bank.positions != want_positions {
        return Err(Inadmissible::PositionCountMismatch {
            expected: want_positions,
            measured: bank.positions,
        }
        .into());
    }
    verified.positions = bank.positions;

    // ── The verdicts. v3 is the frozen authority; v1 and v2 are
    // reported because every earlier claim in this programme cited
    // them and a reader needs to compare like with like. ──
    let evidence = QualityEvidence {
        gate: requested_gate.clone(),
        bank: bank.clone(),
    };
    if evidence.gate.id != request.gate {
        return Err(Inadmissible::GateMismatch {
            requested: request.gate.clone(),
            evaluated: evidence.gate.id.clone(),
        }
        .into());
    }
    verified.gate_evaluated = evidence.gate.id.clone();
    let verdict = evidence.verdict();
    let v1_verdict = kimi_logit_v1().evaluate(&bank);
    let v2_verdict = kimi_logit_v2().evaluate(&bank);

    // ── The closure report — written BEFORE any verdict assertion, so
    // a refused run still leaves its evidence on disk. An 8192-position
    // measurement that panics after the numbers exist and before the
    // write has destroyed twenty minutes of instrument time; that
    // happened once and must not happen again. ──
    //
    // The bank's identity travels with the report: /tmp has proven
    // ephemeral, and a verdict whose evidence names no bank cannot be
    // distinguished from a verdict on a different one.
    let bank_manifest_sha256 = {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| unreadable("quality bank manifest", &manifest_path, e.to_string()))?;
        format!("{:x}", Sha256::digest(&bytes))
    };
    let label = request.label.clone();
    let report = serde_json::json!({
        "run": label,
        "gate": evidence.gate,
        "authority_report": evidence.report(),
        "verdict_passed": verdict.passed(),
        "verdict_failures": verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "verdict_failures_v2": v2_verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "verdict_failures_v1": v1_verdict.failures.iter()
            .map(|(c, d)| format!("{}: {d}", c.name())).collect::<Vec<_>>(),
        "candidate_map": overlay.index.map.name,
        "candidate_layers": overlay.compiled_layers(),
        "bank_manifest_sha256": bank_manifest_sha256,
        "bank": bank,
        "positions": bank.positions,
        "sequences": sequences(),
        "top_n": TOP_N,
        "min_covered_mass": min_covered,
        "stores": {
            "baseline_layer1_routed": "kimi-source-expert-bank (BF16, Table)",
            "candidate_layer1_routed": format!(
                "kimi-candidate-bank ({}, Identity)",
                overlay
                    .encoding_of(overlay_layer as u32)
                    .map(|e| e.name())
                    .unwrap_or("unknown")
            ),
            "shared_experts": "kimi-source-decoder-stack (BF16, both arms)",
        },
        "layer1_routed_experts": {
            "baseline_distinct": baseline_l1_routed.len(),
            "candidate_distinct": candidate_l1_routed.len(),
            "candidate_only": candidate_l1_routed.difference(&baseline_l1_routed).count(),
        },
        "wall_seconds": run_s,
    });
    let path = report_path(&label);
    let rendered = serde_json::to_vec_pretty(&report)
        .map_err(|e| unreadable("closure report", Path::new(&path), e.to_string()))?;
    std::fs::write(&path, rendered)
        .map_err(|e| unreadable("closure report", Path::new(&path), e.to_string()))?;

    Ok(MeasurementReceipt {
        request: request.clone(),
        gate: evidence.gate.clone(),
        verified,
        bank,
        min_covered_mass: min_covered,
        wall_seconds: run_s,
        report_path: path,
        verdict_passed: verdict.passed(),
        verdict_failures: verdict
            .failures
            .iter()
            .map(|(c, d)| format!("{}: {d}", c.name()))
            .collect(),
    })
}
