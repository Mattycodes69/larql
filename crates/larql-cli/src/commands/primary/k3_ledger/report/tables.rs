//! Frequency mass, formats, homogeneity and actions tables.

use super::super::geometry::K3Geometry;
use serde_json::json;

#[allow(unused_imports)]
use super::*;

/// Frequency-mass coverage from a local capture pool. Takes no geometry: the
/// instrument is model-agnostic and must keep working on non-K3 traces.
pub fn freqmass(a: &super::super::args::FreqMassArgs, as_json: bool) -> R {
    use super::super::freqmass::{self, MassCurveConfig};
    use super::super::selection_trace::SelectionTrace;
    use crate::commands::primary::dec_bench::capture_format::CapturePool;

    let pool = CapturePool::open(&a.pool)?;
    let trace = SelectionTrace::from_routing_pool(&pool)?;
    let curve = freqmass::mass_curve(
        &trace,
        &MassCurveConfig {
            cache_sizes: a.cache_sizes.clone(),
            lambdas: a.lambdas.clone(),
            fit_fraction: a.fit_fraction,
            seed: a.seed,
        },
    )?;
    let support = freqmass::support_curve(&trace, &a.support_steps);

    // The operating point is a residency fraction, not a count: it is what
    // transfers across banks of different size.
    let op = curve
        .residency_frac
        .iter()
        .enumerate()
        .min_by(|x, y| {
            (x.1 - a.operating_residency)
                .abs()
                .total_cmp(&(y.1 - a.operating_residency).abs())
        })
        .map(|(i, _)| i);

    // Always censused: the cold-tail counts are a measurement DEC-8.4 wants and
    // cost one pass. Only the per-symbol vector is opt-in, since at K3's 92x896
    // it would dwarf everything else in the document.
    let census = super::super::symbol_mass::census(&trace);
    let census = if a.per_symbol {
        census
    } else {
        census.without_rows()
    };

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "pool": a.pool.display().to_string(),
                "model": pool.manifest.model,
                "curve": curve,
                "support": support,
                "operating_index": op,
                "symbol_mass": census,
            }))?
        );
        return Ok(());
    }
    if a.per_symbol {
        eprintln!("note: --per-symbol emits the row vector into JSON only; add --json");
    }

    println!("=== frequency-mass coverage — {} ===", a.pool.display());
    println!("model    {}", pool.manifest.model);
    println!(
        "trace    {} sessions x {} steps x {} routing strata, width {} of {} ({:.3}% activation)",
        trace.sessions(),
        trace.steps(),
        trace.active_strata().len(),
        trace.width(),
        trace.alphabet(),
        trace.activation_fraction() * 100.0,
    );
    println!("         bank size INFERRED from the trace (max observed symbol + 1)");
    println!(
        "         fit {} steps -> eval {} steps, {} (session, stratum) cells scored",
        curve.fit_steps, curve.eval_steps, curve.cells_scored,
    );
    println!("R2       every number below is conditional on that activation fraction");
    println!();

    if !support.is_empty() {
        println!("support growth — measured against the memoryless-uniform null");
        println!(
            "  {:>6} {:>10} {:>8} {:>10} {:>15}",
            "steps", "measured", "%bank", "uniform", "overstatement"
        );
        for p in &support {
            println!(
                "  {:>6} {:>10.1} {:>7.1}% {:>10.1} {:>14.2}x",
                p.steps,
                p.measured,
                p.measured_frac * 100.0,
                p.uniform,
                p.overstatement,
            );
        }
        println!();
    }

    print!("{:>5} {:>7}", "C", "%bank");
    for arm in &curve.shrinkage {
        print!("{:>9}", format!("λ={:.2}", arm.lambda));
    }
    println!("{:>9}{:>9}{:>9}", "oracle", "null", "mass/res");
    let stat = curve.static_arm().map(|s| s.coverage.clone());
    for (i, &c) in curve.cache_sizes.iter().enumerate() {
        let marker = if Some(i) == op { "*" } else { " " };
        print!("{marker}{:>4} {:>6.1}%", c, curve.residency_frac[i] * 100.0);
        for arm in &curve.shrinkage {
            print!("{:>9.4}", arm.coverage[i]);
        }
        let per = stat
            .as_ref()
            .map(|s| curve.mass_per_residency(s)[i])
            .unwrap_or(f64::NAN);
        println!(
            "{:>9.4}{:>9.4}{:>8.2}x",
            curve.oracle[i], curve.null[i], per
        );
    }
    println!();
    let unobs: Vec<usize> = census
        .coverage_by_stratum
        .iter()
        .map(|c| c.unobserved)
        .collect();
    if let (Some(&lo), Some(&hi)) = (unobs.iter().min(), unobs.iter().max()) {
        println!(
            "cold tail — {} of {} symbol-stratum pairs UNOBSERVED in this capture ({:.1}%),",
            census.unobserved_symbols,
            census.routing_strata * census.alphabet,
            100.0 * census.unobserved_symbols as f64
                / (census.routing_strata * census.alphabet) as f64,
        );
        println!(
            "  spread {lo}-{hi} per stratum — tier per stratum, not globally. Unobserved is NOT"
        );
        println!("  zero probability: it bounds the rate, it does not measure it.");
        println!("  This is STATIC structure varying by layer — not the per-layer ADAPTATION");
        println!("  the causal arm above refutes. Both hold; do not collapse them.");
        println!("  R2 WARNING: quantile balancing exists to flatten exactly this, so of");
        println!("  everything here it is the number least likely to survive to K3.");
        println!();
        println!("  domain-mixture caveat, BY STATISTIC — the sign differs, so it cannot be");
        println!("  stated once for the whole capture:");
        println!("    coverage curve      ANTI-CONSERVATIVE — a mixture is the anti-case for");
        println!("                        locality; a domain slice would beat these numbers.");
        println!("    unobserved count    CONSERVATIVE — a domain-selective layer would show a");
        println!("                        WIDE union across unrelated prompts. Narrow anyway");
        println!("                        is stronger evidence, not weaker.");
        println!();
    }
    println!("  λ=1.00 static slice (pooled prior, leave-one-out) · λ=0.00 causal adaptive cache");
    println!("  oracle ranks by the scored events themselves · null destroys session identity");
    println!("  mass/res = coverage per unit residency — the axis a cold-tier lever moves along");
    println!();

    if let Some(i) = op {
        let (realisable, oracle) = curve.miss_ratios(i).unwrap_or((f64::NAN, f64::NAN));
        let (best_lambda, best) = curve.best_realisable(i).unwrap_or((f64::NAN, f64::NAN));
        let stat_cov = stat.as_ref().map(|s| s[i]).unwrap_or(f64::NAN);
        println!(
            "operating point — C={} ({:.1}% of bank)",
            curve.cache_sizes[i],
            curve.residency_frac[i] * 100.0
        );
        println!(
            "  static coverage        {stat_cov:.4}  (miss {:.4})",
            1.0 - stat_cov
        );
        println!(
            "  best realisable        {best:.4}  at λ={best_lambda:.2}  (miss {:.4})",
            1.0 - best
        );
        println!("  REALISABLE PRIZE       {realisable:.2}x on the miss stream — what adaptation actually buys");
        println!("  oracle bound           {oracle:.2}x — with every benefit of the doubt granted");
        println!(
            "  locality present       {:.4}  (oracle - null; the rest of the oracle gap is counting noise)",
            curve.locality_signal(i)
        );
    }
    Ok(())
}

pub fn formats(as_json: bool) -> R {
    use super::super::serving_format::{self as sf, Container, ScaleEncoding};

    if as_json {
        println!("{}", serde_json::to_string_pretty(&sf::candidates())?);
        return Ok(());
    }

    println!("=== exact serving containers for a K3 MXFP4 expert bank ===");
    println!("no checkpoint read: this is decided by the source alphabet");
    println!();
    println!("  fp4 magnitudes           +/-{:?}", sf::FP4_MAGNITUDES);
    println!("  alphabet (x2)            {:?}", sf::FP4_INTEGER_ALPHABET);
    println!("  distinct codepoints      {}", sf::distinct_codepoints());
    println!(
        "  PAYLOAD FLOOR            {} bits — no exact fixed-width code beats this",
        sf::min_payload_bits()
    );
    println!(
        "  affine levels required   {}  (gcd of gaps = 1, span -12..12)",
        sf::affine_levels_required()
    );
    println!();

    println!(
        "{:<22} {:>7} {:>8} {:>7} {:>12}",
        "container", "levels", "exact", "bpw", "kernel reach"
    );
    for c in [
        Container::Mxfp4,
        Container::Q4K,
        Container::Q5K,
        Container::Q6K,
        Container::Q8_0,
    ] {
        println!(
            "{:<22} {:>7} {:>8} {:>7.4} {:>12}",
            c.label(),
            c.grid_levels().map_or("LUT".into(), |l| l.to_string()),
            if c.holds_fp4_exactly() { "yes" } else { "NO" },
            c.all_in_bits(),
            format!(
                "{:?}{}",
                c.kernel_maturity(),
                if c.kernel_maturity().is_servable() {
                    ""
                } else {
                    "*"
                }
            ),
        );
    }
    println!();
    println!(
        "  Q4_K is {} levels short — no scan or scale trick rescues it.",
        sf::affine_levels_required() - Container::Q4K.grid_levels().unwrap()
    );
    println!("  Q6_K is the cheapest exact container that can SERVE today.");
    println!("  MXFP4 has Metal kernels (K1 standalone, K2 grouped) but is absent");
    println!("  from QuantMatVec dispatch, so kernels != servable. (* = cannot serve)");
    println!("  Q5_K is exact but needs a kernel written AND ships more bytes");
    println!("  than native MXFP4 would — strictly dominated, never build it.");
    println!();

    println!("--- scale stream: the only remaining exact lever ---");
    println!(
        "{:<38} {:>9} {:>10} {:>11}",
        "encoding", "bpw", "max spread", "exceptions"
    );
    for s in [
        ScaleEncoding::RawE8M0,
        ScaleEncoding::SuperblockBaseDelta2,
        ScaleEncoding::GlobalTable2Bit,
    ] {
        println!(
            "{:<38} {:>9.5} {:>10} {:>11}",
            s.label(),
            s.bits_per_weight(),
            s.max_spread().map_or("any".into(), |m| m.to_string()),
            if s.needs_exception_path() {
                "REQUIRED"
            } else {
                "none"
            },
        );
    }
    println!();
    println!(
        "  measured max spread {} over 1,032,192 superblocks; both compact",
        sf::MEASURED_MAX_SPREAD
    );
    println!(
        "  encodings clear it. The global table fits the observed {}-exponent",
        sf::MEASURED_DISTINCT_EXPONENTS
    );
    println!("  range with ZERO headroom, so its exception path is mandatory.");
    println!();

    println!("--- candidate ladder, cheapest first ---");
    for f in sf::candidates() {
        println!(
            "  {:>7.4} bpw  {:<22} {:<38} {}",
            f.all_in_bits(),
            f.container.label(),
            if f.container == Container::Mxfp4 {
                f.scale.label()
            } else {
                "—"
            },
            if f.is_exact_for_k3() {
                "exact"
            } else {
                "NOT EXACT"
            },
        );
    }
    println!();

    let floor = sf::exact_floor();
    let cut = 1.0 - floor.all_in_bits() / Container::Mxfp4.all_in_bits();
    println!(
        "  EXACT FLOOR {:.5} bpw — {:.1}% under MXFP4, {:.3}x under Q6_K.",
        floor.all_in_bits(),
        100.0 * cut,
        Container::Q6K.all_in_bits() / floor.all_in_bits()
    );
    println!("  Everything below this bar requires approximation, not encoding.");
    Ok(())
}

/// `k3-ledger homogeneity` — read every layer's header and witness that
/// one measured layer represents its family.
pub fn homogeneity(
    repo: &super::super::fetch::Repo,
    geom: &K3Geometry,
    a: &super::super::args::HomogeneityArgs,
    as_json: bool,
) -> R {
    use super::super::homogeneity::HomogeneityWitness;

    let n = if a.limit == 0 {
        geom.n_layers
    } else {
        a.limit.min(geom.n_layers)
    };
    eprintln!("reading {n} layer headers (headers only, no weights)");
    let mut profiles = Vec::with_capacity(n);
    for layer in 0..n {
        profiles.push(super::super::fetch::layer_profile(
            repo,
            &a.shard_template,
            layer,
        )?);
        if (layer + 1) % 10 == 0 {
            eprintln!("  {} / {n}", layer + 1);
        }
    }
    let w = HomogeneityWitness::build(&profiles, &geom.kda_layer_indices, geom.topology);

    if as_json {
        println!("{}", serde_json::to_string_pretty(&w)?);
        return Ok(());
    }
    println!();
    println!("--- family homogeneity, from {n} layer headers ---");
    for f in &w.families {
        println!(
            "  {:<16} {:>3} layers, ref layer {:>2}  {}",
            f.family,
            f.members,
            f.reference,
            if f.is_homogeneous() {
                "WITNESSED HOMOGENEOUS".to_string()
            } else {
                format!("{} DIVERGENCES", f.divergences.len())
            }
        );
        for d in f.divergences.iter().take(5) {
            println!("      {d:?}");
        }
        if let Some(r) = profiles.iter().find(|p| p.index == f.reference) {
            println!(
                "      reference layer total {:.2} MB",
                r.total_bytes() as f64 / 1e6
            );
        }
    }
    println!();
    if w.all_homogeneous() {
        println!(
            "  VERDICT: the census may multiply ONE measured layer by its family \
             count — that is now a WITNESSED FACT, not an assumption."
        );
    } else {
        println!(
            "  VERDICT: {} divergences — the census's per-family multiplication \
             is NOT sound as stated.",
            w.divergence_count()
        );
    }
    Ok(())
}

/// `k3-ledger actions` — K3-ACTIONS-1, the physical action catalogue.
///
/// Physical opportunity only. Nothing here is a behavioural claim, and a
/// `family (CEILING)` row is not something one authority run can decide.
pub fn actions(geom: &K3Geometry, a: &super::super::args::ActionsArgs, as_json: bool) -> R {
    use super::super::actions::{catalogue, whole_side_ceiling, ActionScope, DENSE_CODECS};

    let cat = catalogue(&geom.dense_census, geom.dense_stored_bits);
    if as_json {
        println!("{}", serde_json::to_string_pretty(&cat)?);
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "--- K3-ACTIONS-1: physical opportunity from {} ---",
        geom.dense_stored_label
    );
    println!("  NO behavioural claim. A CEILING row is not an atomic candidate.");
    println!();
    println!(
        "  {:<18} {:<7} {:<16} {:>10} {:>10}  execution role",
        "family", "codec", "scope", "activated", "resident"
    );
    for act in &cat {
        if !a.ceilings && act.scope != ActionScope::Layer {
            continue;
        }
        let gb = act.activated_saving() as f64 / 1e9;
        if gb < a.min_gb && act.scope != ActionScope::Layer {
            continue;
        }
        println!(
            "  {:<18} {:<7} {:<16} {:>7.2} GB {:>7.2} GB  {}",
            act.family,
            act.codec,
            act.scope.label(),
            gb,
            act.resident_saving() as f64 / 1e9,
            act.role.label()
        );
    }
    let candidates = cat.iter().filter(|a| a.scope.is_atomic_candidate()).count();
    println!();
    println!(
        "  {candidates} atomic candidates (one authority run each); the rest are \
         CEILINGS that aggregate members which must each earn their own admission"
    );
    println!();
    println!("  --- whole-dense-side ceilings (every family moves AND earns it) ---");
    for codec in DENSE_CODECS {
        println!(
            "  {:<7} {:>7.2} GB/token removed of {:.2} GB dense",
            codec.name,
            whole_side_ceiling(&geom.dense_census, geom.dense_stored_bits, codec) as f64 / 1e9,
            geom.dense_census.activated_bytes() as f64 / 1e9
        );
    }
    Ok(())
}
