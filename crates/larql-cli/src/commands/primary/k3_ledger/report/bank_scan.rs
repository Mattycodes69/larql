//! Bank scans: transcode scan and bandwidth ceilings.

use super::super::fetch;
use super::super::geometry::K3Geometry;

#[allow(unused_imports)]
use super::*;

pub fn transcode_scan(
    repo: &fetch::Repo,
    shard: &str,
    a: &super::super::args::TranscodeScanArgs,
    as_json: bool,
) -> R {
    use super::super::transcode;

    let (header, header_len) = repo.shard_header_with_len(shard)?;
    let tensors = fetch::scale_tensors(&header);
    if tensors.is_empty() {
        return Err("no weight_scale tensors in this shard".into());
    }

    println!("scanning e8m0 exponent spread in {shard}");
    println!(
        "  {} scale tensors present; sampling {}",
        tensors.len(),
        a.tensors.min(tensors.len())
    );

    let mut scans = Vec::new();
    for (name, shape, lo, hi) in tensors.iter().take(a.tensors) {
        let groups_per_row = *shape.last().unwrap_or(&1) as usize;
        let bytes = repo.tensor_bytes_at(shard, header_len, *lo, *hi)?;
        let s = transcode::scan_scales(&bytes, groups_per_row);
        if scans.is_empty() {
            println!(
                "  e.g. {name}: shape {shape:?}, {} groups/row, {} superblocks",
                groups_per_row, s.superblocks
            );
        }
        scans.push(s);
    }
    let m = transcode::merge(&scans);

    if as_json {
        println!("{}", serde_json::to_string_pretty(&m)?);
        return Ok(());
    }

    println!();
    println!("--- exponent field ---");
    println!(
        "  scanned {} tensors, {} superblocks, {} scale bytes",
        m.tensors_scanned, m.superblocks, m.groups_scanned
    );
    println!(
        "  e8m0 range {}..{}  (d exponent {}..{})",
        m.e_min, m.e_max, m.d_exponent_min, m.d_exponent_max
    );
    println!("  sentinels: {} zero, {} NaN", m.zero_scales, m.nan_scales);
    println!();
    println!("--- spread per 256-weight superblock ---");
    for (spread, count) in &m.spread_histogram {
        let pct = 100.0 * *count as f64 / m.superblocks.max(1) as f64;
        let flag = if *spread > transcode::MAX_EXPONENT_SPREAD {
            "  <- EXCEEDS int8 sub-scale"
        } else {
            ""
        };
        println!("  spread {spread:>2}: {count:>9} ({pct:>5.1}%){flag}");
    }
    println!(
        "  max spread {} (limit {})",
        m.max_spread,
        transcode::MAX_EXPONENT_SPREAD
    );
    println!();
    println!("--- constraints ---");
    println!(
        "  C1 spread <= {}   : {}  ({} superblocks over)",
        transcode::MAX_EXPONENT_SPREAD,
        if m.c1_spread_ok { "PASS" } else { "FAIL" },
        m.superblocks_over_spread
    );
    println!(
        "  C2 d in f16 range : {}  ({} superblocks over)",
        if m.c2_f16_ok { "PASS" } else { "FAIL" },
        m.superblocks_over_f16_range
    );
    println!();
    println!("VERDICT: {}", m.verdict());
    Ok(())
}

pub fn ceilings(geom: &K3Geometry, a: &super::super::args::CeilingsArgs, as_json: bool) -> R {
    use super::super::classes::{self, KernelClass, Scenario};

    let base = classes::census(geom, a.dense_bits, a.routed_bits);
    let composed = classes::compose(base.clone(), classes::BW_GB_S, KernelClass::Down.eta());

    if as_json {
        println!("{}", serde_json::to_string_pretty(&composed)?);
        return Ok(());
    }

    header(geom);
    println!();
    println!(
        "--- per-class census (dense {:.2} bits, routed {:.2} bits) ---",
        a.dense_bits, a.routed_bits
    );
    println!(
        "{:<28} {:>9} {:>9} {:>6} {:>9}",
        "class", "params B", "GB/token", "eta", "ms/token"
    );
    for r in &composed.rows {
        println!(
            "{:<28} {:>9.2} {:>9.2} {:>6.2} {:>9.2}",
            r.name,
            r.params as f64 / 1e9,
            r.bytes() / 1e9,
            r.eta(),
            r.seconds(classes::BW_GB_S) * 1e3
        );
    }
    println!(
        "{:<28} {:>9} {:>9.2} {:>6} {:>9.2}",
        "TOTAL",
        "",
        composed.total_bytes / 1e9,
        "",
        composed.seconds_per_token * 1e3
    );
    println!();
    let (lo, hi) = classes::observed_range(&composed.rows, classes::BW_GB_S);
    println!("  COMPOSED ceiling {:.2} tok/s", composed.tok_s);
    println!(
        "  observed composition range {lo:.2}-{hi:.2}  (min-max of {} PAIRED per-run \
         compositions, not a component-extrema envelope)",
        classes::ACCEPTED_RUNS
    );
    println!(
        "  scalar-eta quote {:.2} tok/s  -> overstates by {:.2}x",
        composed.scalar_tok_s, composed.scalar_overstates_by
    );
    println!();

    println!("  ETA PROVENANCE: post-shape-census, true cold rotation, 3 repeats.");
    println!("  A class whose range exceeds 5% is NOT decision-grade — do not let it");
    println!("  select a bit-width without more repeats (R0).");
    println!("  eta provenance (R0 — none of these may be quoted bare):");
    for c in [
        KernelClass::AttnProjection,
        KernelClass::GateUp,
        KernelClass::Down,
        KernelClass::RoutedExpert,
    ] {
        let m = c.efficiency();
        println!(
            "    {:.3} +/-{:.4} [{:.2}-{:.2} = {:.0}%] n={}  {:<28} {}{}{}",
            m.central,
            m.std_error,
            m.observed_low,
            m.observed_high,
            100.0 * m.spread_fraction(),
            m.repeats,
            c.label(),
            c.provenance(),
            if c.is_provisional() {
                "   [PROVISIONAL]"
            } else {
                ""
            },
            if m.is_decision_grade() {
                ""
            } else {
                "   [NOT DECISION-GRADE]"
            }
        );
    }
    println!();

    if classes::density_only_bound(a.dense_bits, a.routed_bits) {
        println!(
            "  !! DENSITY-ONLY UPPER BOUND — these etas were measured under {:.4} bpw",
            classes::ETA_MEASURED_UNDER_BITS
        );
        println!(
            "     (Q6_K), and this composition reuses them at {:.4}/{:.4}. R7 forbids",
            a.dense_bits, a.routed_bits
        );
        println!("     carrying eta across a container change: the crossover measured MXFP4");
        println!("     arm D at 0.724 against Q6_K's 0.89 on the routed class. Apply that");
        println!("     penalty before quoting this as reachable — it is an upper bound.");
        println!();
    }

    let ungraded = classes::not_decision_grade(&composed.rows);
    println!("--- R4 best case per proposed lever (ceiling if it FULLY succeeds) ---");
    if !ungraded.is_empty() {
        println!("  REFUSED — this ordering selects the programme, and these classes are");
        println!(
            "  too noisy to select anything (need n>={} and relative SE <= {:.0}%):",
            classes::MIN_REPEATS,
            100.0 * classes::MAX_RELATIVE_SE
        );
        for (c, m) in &ungraded {
            println!(
                "    {:<28} {:.3} +/-{:.4} SE ({:.1}% rel) [{:.2}-{:.2}] n={}",
                c.label(),
                m.central,
                m.std_error,
                100.0 * m.relative_std_error(),
                m.observed_low,
                m.observed_high,
                m.repeats
            );
        }
        println!("  Run `diag_eta_repeats` on those shapes and re-bank before ordering.");
        println!();
        println!(
            "  baseline (measured per-class eta, today's formats) {:.2} tok/s",
            composed.tok_s
        );
        return Ok(());
    }
    let mut scen: Vec<(&str, Vec<classes::ClassRow>)> = vec![
        (
            "exp1 exact MXFP4 layout: all classes -> best eta",
            classes::apply(&base, Scenario::AllClassesAtBestEta),
        ),
        (
            "exp3 3-bit KDA attention",
            classes::apply(&base, Scenario::Rebits("KDA attention projections", 3.0)),
        ),
        (
            "exp3+ 2.5-bit KDA attention",
            classes::apply(&base, Scenario::Rebits("KDA attention projections", 2.5)),
        ),
        (
            "K3a grouped experts: routed 0.64 -> 0.89 [MEASURED]",
            classes::apply(
                &base,
                Scenario::LiftClass(KernelClass::RoutedExpert, classes::GROUPED_ROUTED_ETA),
            ),
        ),
        (
            "exp3+exp4 stacked: 2.5-bit KDA + expert shape fixed",
            classes::apply_all(
                &base,
                &[
                    Scenario::Rebits("KDA attention projections", 2.5),
                    Scenario::LiftClass(KernelClass::RoutedExpert, classes::GROUPED_ROUTED_ETA),
                ],
            ),
        ),
        (
            "K3a+ grouped experts to the roofline (0.95)",
            classes::apply(&base, Scenario::LiftClass(KernelClass::RoutedExpert, 0.95)),
        ),
        (
            "R4 floor: routed experts ZEROED",
            classes::apply(&base, Scenario::Zero("routed experts (top-k)")),
        ),
        (
            "STACKED: 2.5-bit KDA + every class at best eta",
            classes::apply_all(
                &base,
                &[
                    Scenario::Rebits("KDA attention projections", 2.5),
                    Scenario::AllClassesAtBestEta,
                ],
            ),
        ),
        (
            "STACKED + routed experts also zeroed (absolute ceiling)",
            classes::apply_all(
                &base,
                &[
                    Scenario::Rebits("KDA attention projections", 2.5),
                    Scenario::AllClassesAtBestEta,
                    Scenario::Zero("routed experts (top-k)"),
                ],
            ),
        ),
    ];
    // Each scenario carries its OWN band. Quoting the baseline's range beside a
    // scenario's central value is the R0 failure in miniature — the band would
    // not even contain the number it sits next to.
    let mut scored: Vec<(String, f64, f64, f64)> = scen
        .drain(..)
        .map(|(n, rows)| {
            let c = classes::compose(rows.clone(), classes::BW_GB_S, KernelClass::Down.eta());
            let (lo, hi) = classes::observed_range(&rows, classes::BW_GB_S);
            (n.to_string(), c.tok_s, lo, hi)
        })
        .collect();
    scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    for (name, tok, lo, hi) in &scored {
        let gain = tok / composed.tok_s;
        println!(
            "  {:<48} {:>6.2} [{:.2}-{:.2}]  ({:+.0}%)",
            name,
            tok,
            lo,
            hi,
            100.0 * (gain - 1.0)
        );
    }
    println!();
    println!(
        "  baseline (measured per-class eta, today's formats) {:.2} tok/s",
        composed.tok_s
    );
    println!("  Order the programme by these, not by how interesting each sounds.");
    Ok(())
}
