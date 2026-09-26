//! Serving reports: budget, touch, frontier and block.

use super::super::args::{BlockArgs, BudgetArgs, FrontierArgs, TouchArgs};
use super::super::block::{self, DraftProfile, PhysicalReuse, StateTraffic};
use super::super::budget::{self, LinkPremises};
use super::super::frontier::{self, ServingPremises};
use super::super::geometry::{BitConvention, K3Geometry};
use super::super::scenario::{self, ScenarioPremises};
use super::super::touch::{self, SliceTier};
use serde_json::json;

#[allow(unused_imports)]
use super::*;

pub fn budget(geom: &K3Geometry, a: &BudgetArgs, as_json: bool) -> R {
    let link = LinkPremises {
        gb_s: a.link_gb_s,
        target_tok_s: a.target_tok_s,
        ports: a.ports,
    };
    let m = budget::miss_budget(geom, &link);
    let g = budget::granularity(geom, &link, a.feature_fraction, 2.0, a.locality);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"budget": m, "granularity": g}))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "miss budget   {:.1} MB/token ({} GB/s x {} port(s) / {} tok/s)",
        m.miss_budget_bytes / 1e6,
        a.link_gb_s,
        a.ports,
        a.target_tok_s
    );
    println!("required      {:.2} GB/token", m.required_fetch_bytes / 1e9);
    println!("GAP           {:.1}x", m.gap);
    println!(
        "  {} expert-visits/token, {:.1} KB allowed at each = {:.3}% of an expert",
        m.expert_visits,
        m.bytes_allowed_per_visit / 1e3,
        100.0 * m.fraction_of_expert_allowed,
    );
    println!();
    println!(
        "--- read granularity at {:.1}% of features ---",
        100.0 * a.feature_fraction
    );
    println!(
        "  feature row {:.0} B, {:.0} rows/visit",
        g.feature_row_bytes, g.features_selected
    );
    println!(
        "  ideal {:.0} KB/visit -> paged {:.2} MB/visit ({:.2}x amplification)",
        g.ideal_bytes_per_visit / 1e3,
        g.paged_bytes_per_visit / 1e6,
        g.read_amplification
    );
    println!(
        "  {:.2}M IOPS required vs {:.2}M bandwidth-equivalent -> {:.1}x short",
        g.iops_required / 1e6,
        g.iops_bandwidth_equivalent / 1e6,
        g.iops_shortfall
    );
    println!("  (request rate binds before bandwidth does)");
    Ok(())
}

pub fn touch(geom: &K3Geometry, p: &ServingPremises, a: &TouchArgs, as_json: bool) -> R {
    let l = touch::touch_ledger(geom, p, a.target_tok_s);
    let slices: Vec<_> = a
        .slice_gb
        .iter()
        .map(|gb| {
            touch::slice_composition(
                geom,
                p,
                &SliceTier {
                    size_bytes: gb * 1e9,
                },
            )
        })
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({"ledger": l, "slices": slices}))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "--- per-token touch: dense {:.2} all-in bits [{}] ---",
        p.dense_all_in_bits(geom),
        p.dense.label(geom)
    );
    println!(
        // NOT "irreducible": this 111 GB is exactly what REPRESENT exists to
        // reduce. What is irreducible is that these operations are ACTIVATED
        // every token under the current execution graph — a statement about
        // the graph, never about the representation's byte width.
        "  dense (always-on, full-read) {:>7.2} GB -> {:>5.1} tok/s",
        l.dense_bytes / 1e9,
        l.tok_s_dense_only
    );
    println!("  routed experts      {:>7.2} GB", l.routed_bytes / 1e9);
    println!(
        "  TOTAL               {:>7.2} GB -> {:>5.1} tok/s",
        l.total_bytes / 1e9,
        l.tok_s_total
    );
    println!();
    // Two different shares, printed together on purpose. The param share is
    // representation-independent; the BYTE share is what a bandwidth lever
    // actually acts on, and they diverge by 28 points on today's K3 because
    // dense is BF16 and routed is MXFP4. Printing only the first next to a
    // table of bytes is how a 53.6% reads as though it described traffic.
    // K3-DENSE-1. The census is the ledger's own answer to "what IS the
    // dense side", and it is ordered by ACTIVATED bytes — `embed_tokens` is
    // the joint-largest RESIDENT family and nearly the smallest activated
    // one, so ordering by footprint would put it at the top of a traffic
    // report.
    println!();
    println!(
        "--- dense families, by activated bytes/token [{}] ---",
        p.dense.label(geom)
    );
    println!(
        "  {:<22} {:>9} {:>7} {:>10}  access",
        "family", "activated", "share", "resident"
    );
    for f in &geom.dense_census.families {
        println!(
            "  {:<22} {:>7.2} GB {:>6.1}% {:>8.2} GB  {}",
            f.name,
            f.activated_bytes() as f64 / 1e9,
            100.0 * geom.dense_census.activated_share(&f.name),
            f.resident_bytes() as f64 / 1e9,
            if f.access.is_full_read() {
                "full read"
            } else {
                "gather"
            }
        );
    }
    println!(
        "  {:<22} {:>7.2} GB {:>6} {:>8.2} GB",
        "TOTAL",
        geom.dense_census.activated_bytes() as f64 / 1e9,
        "100.0%",
        geom.dense_census.resident_bytes() as f64 / 1e9
    );
    println!(
        "  resident exceeds activated by {:.2} GB — the embed table that is \
         never read in full",
        (geom.dense_census.resident_bytes() - geom.dense_census.activated_bytes()) as f64 / 1e9
    );
    println!(
        "  layer topology: {} dense + {} MoE = {} (first_k_dense_replace)",
        geom.topology.n_dense(),
        geom.topology.n_moe(),
        geom.topology.n_layers
    );
    println!();
    println!(
        "  dense is {:.1}% of activated params, and {:.1}% of TODAY'S BYTES",
        100.0 * l.dense_share,
        100.0 * l.dense_bytes / l.total_bytes
    );
    println!(
        "  expert-side levers cap at {:.2}x; {:.2}x needed for {} tok/s",
        l.expert_side_ceiling,
        l.reduction_needed.unwrap_or(f64::NAN),
        a.target_tok_s
    );
    println!(
        "  >> even a perfect expert lever leaves {:.1} tok/s",
        l.tok_s_dense_only
    );
    println!(
        "  at CPU-attainable {:.0} GB/s the same read gives {:.1} tok/s",
        frontier::BW_CPU_GB_S,
        frontier::BW_CPU_GB_S * 1e9 * p.dequant_efficiency / l.total_bytes,
    );
    println!();
    println!("--- slice composition (down-row payloads, up folded) ---");
    for s in &slices {
        println!(
            "  {:>3.0} GB: {:>6.0} experts ({:.1}% of bank) -> {:.2} of {} hit/layer uniform",
            s.slice_bytes / 1e9,
            s.resident_experts,
            100.0 * s.resident_fraction,
            s.uniform_hits_per_layer,
            geom.top_k
        );
    }
    println!();
    println!("  the uniform hit count is a NULL, not a forecast — it assumes no routing skew,");
    println!("  and real routing departs from it in BOTH directions. Measured on the DEC-0");
    println!("  Gemma pool (8-of-128, 6.25% activation): a 6.25% resident set covered 42% of");
    println!("  routing events, not 6.25% — uniform understated coverage ~6.8x. It also");
    println!("  OVERstated per-session support 3.4x at 16 steps. Neither magnitude transfers");
    println!("  to K3 (R2: different activation fraction, and quantile balancing flattens);");
    println!("  the sign of the coverage error does, since uniform understates under any skew.");
    println!("  Measure the real curve with `larql k3-ledger freq-mass --pool <capture>`.");
    Ok(())
}

/// The composed ceiling at the cheapest exact format, grouped experts included.
///
/// Derived rather than quoted: the frontier's own rows sit either side of this
/// bar, and a stale literal here would reintroduce exactly the mislabelling the
/// scenario banner exists to prevent.
pub(super) fn exact_floor_tok_s(geom: &K3Geometry) -> f64 {
    use super::super::classes::{self, KernelClass, Scenario};

    let bits = super::super::serving_format::exact_floor().all_in_bits();
    let rows = classes::apply(
        &classes::census(geom, bits, bits),
        Scenario::LiftClass(KernelClass::RoutedExpert, classes::GROUPED_ROUTED_ETA),
    );
    classes::compose(rows, classes::BW_GB_S, KernelClass::Down.eta()).tok_s
}

pub fn frontier(geom: &K3Geometry, p: &ServingPremises, a: &FrontierArgs, as_json: bool) -> R {
    let p = ServingPremises {
        routed_retention: a.routed_retention.unwrap_or(geom.up_fold_retention()),
        ..*p
    };
    let rows = frontier::frontier(geom, &p, &a.targets);
    let ceiling = frontier::expert_side_ceiling_tok_s(geom, &p);

    let scenario = ScenarioPremises::describe(geom, &p);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": rows, "expert_side_ceiling_tok_s": ceiling,
                "routed_retention": p.routed_retention,
                // Carried so a row read out of this JSON arrives with its
                // conditions attached rather than as a bare number.
                "scenario": scenario,
                "scenario_banner": scenario::SCENARIO_BANNER,
                "bits_columns_mean": scenario::BITS_COLUMN_MEANING,
                "warnings": scenario.warnings(),
            }))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!("=== {} ===", scenario::SCENARIO_BANNER);
    println!(
        "  routed_retention   {:.3}  ({:.2}x credited reduction)",
        scenario.routed_retention, scenario.routed_reduction
    );
    if a.routed_retention.is_none() {
        println!("                     banked up-fold; anything lower is row sparsity, UNMEASURED");
    }
    println!(
        "  routed_bits        {:.2} all-in  (checkpoint MXFP4 bytes, uncompressed)",
        scenario.routed_all_in_bits
    );
    println!(
        "  scalar_efficiency  {:.2}{}",
        scenario.scalar_efficiency,
        if scenario.efficiency_is_ideal() {
            "  <- IDEAL"
        } else {
            ""
        }
    );
    println!(
        "  bandwidth          {:.0} GB/s attainable",
        scenario.bandwidth_gb_s
    );
    println!("  bit convention     rows report BOTH payload and all-in");
    for w in scenario.warnings() {
        println!("  ! {w}");
    }
    println!("  For a whole-model ceiling use `k3-ledger ceilings` (per-class eta).");
    println!(
        "  The exact-format floor is {:.2} tok/s (`k3-ledger formats`) — targets",
        exact_floor_tok_s(geom)
    );
    println!("  above it are fidelity-bounded, not exact-serving, results.");
    println!();
    println!("{:>58}", scenario::BITS_COLUMN_MEANING);
    println!(
        "{:>12} {:>12} {:>13} {:>8} {:>8}  verdict",
        "target tok/s", "budget", "dense budget", "payload", "all-in"
    );
    for r in &rows {
        println!(
            "{:>12.1} {:>11.2}G {:>12.2}G {:>8.2} {:>8.2}  {:?}",
            r.target_tok_s,
            r.budget_bytes / 1e9,
            r.dense_budget_bytes / 1e9,
            r.payload_bits,
            r.all_in_bits,
            r.verdict
        );
    }
    println!(
        "  (bits are per weight; {} excludes block scales, {} includes them)",
        BitConvention::Payload.label(),
        BitConvention::AllIn.label(),
    );
    let undecidable: Vec<_> = rows
        .iter()
        .filter(|r| !frontier::expert_side_can_decide(geom, &p, r.target_tok_s))
        .map(|r| format!("{:.1}", r.target_tok_s))
        .collect();
    if !undecidable.is_empty() {
        println!(
            "  R4: no expert-side experiment can decide {} tok/s",
            undecidable.join(", ")
        );
    }
    if a.zero_out_check {
        println!();
        println!("--- R4 zero-out: routed traffic deleted entirely ---");
        println!(
            "  dense at {:.2} bits [{}] caps decode at {:.1} tok/s",
            p.dense_all_in_bits(geom),
            p.dense.label(geom),
            ceiling
        );
        println!("  >> no expert-side experiment can decide any target above {ceiling:.1}");
    }
    Ok(())
}

pub fn block(geom: &K3Geometry, p: &ServingPremises, a: &BlockArgs, as_json: bool) -> R {
    let p = ServingPremises {
        routed_retention: a.routed_retention.unwrap_or(geom.up_fold_retention()),
        ..*p
    };
    let draft = DraftProfile::new(a.proposal_width, a.mean_accepted);
    let state = match a.state_prefix_ladder_bytes {
        Some(w) => StateTraffic {
            bytes_per_pass: a.state_bytes_per_pass,
            ..StateTraffic::prefix_state_ladder(geom.n_kda_layers, w)
        },
        None => StateTraffic {
            bytes_per_pass: a.state_bytes_per_pass,
            bytes_per_position: a.state_bytes_per_position,
            measured: a.state_bytes_per_pass > 0.0 || a.state_bytes_per_position > 0.0,
        },
    };

    if draft.assumes_perfect() {
        eprintln!(
            "WARNING: proposal width == mean accepted. That is the R5 error \
             (perfect acceptance assumed). Costs key on positions EVALUATED, \
             throughput on positions COMMITTED."
        );
    }

    let rows: Vec<_> = if a.token_loop {
        a.widths
            .iter()
            .map(|&t| {
                let reuse = PhysicalReuse::token_loop(t);
                block::evaluate(geom, &p, &draft, t, reuse, state, a.target_tok_s)
            })
            .collect()
    } else {
        block::sweep(geom, &p, &draft, &a.widths, state, a.target_tok_s)
    };

    let refused: Vec<_> = rows
        .iter()
        .filter(|r| {
            let reuse = PhysicalReuse::assumed_ideal(r.union_equivalents);
            block::dense_alone_refuses(geom, &p, r.accepted, r.width, reuse, state, a.target_tok_s)
        })
        .map(|r| r.width.to_string())
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "rows": rows, "fitted_alpha": draft.fitted_alpha(),
                "provisional": "alpha fitted from ONE observed (width, accepted) pair \
                                under a constant-alpha geometric model; optima are \
                                design points, not performance bars",
            }))?
        );
        return Ok(());
    }
    header(geom);
    println!();
    println!(
        "drafter: {} proposed / {:.2} committed, fitted alpha {:.4}",
        a.proposal_width,
        a.mean_accepted,
        draft.fitted_alpha()
    );
    println!(
        "execution: {}",
        if a.token_loop {
            "TOKEN LOOP (no grouping)"
        } else {
            "ideal grouping ASSUMED (R6, unmeasured)"
        }
    );
    println!(
        "state traffic: {}",
        match (state.bytes_per_pass, state.bytes_per_position) {
            (0.0, 0.0) => "OMITTED — no state term in this budget (owner M1-B)".to_string(),
            (pass, pos) => format!(
                "{:.2} GB/pass + {:.2} GB/position [{}]",
                pass / 1e9,
                pos / 1e9,
                if state.measured {
                    "measured"
                } else {
                    "DERIVED, not measured"
                }
            ),
        }
    );
    println!();
    println!(
        "{:>3} {:>8} {:>8} {:>10} {:>9} {:>9}",
        "T", "A(T)", "u(T)", "GB/token", "tok/s", "G_r need"
    );
    for r in &rows {
        let need = r
            .routed_reduction_needed
            .map(|v| format!("{v:>9.2}"))
            .unwrap_or_else(|| "      inf".into());
        println!(
            "{:>3} {:>8.2} {:>8.2} {:>10.2} {:>9.2} {}",
            r.width,
            r.accepted,
            r.union_equivalents,
            r.bytes_per_committed_token / 1e9,
            r.tok_s,
            need
        );
    }
    if let Some(best) = rows
        .iter()
        .filter(|r| r.rho_max.is_some())
        .max_by(|x, y| x.rho_max.partial_cmp(&y.rho_max).unwrap())
    {
        println!();
        println!(
            "  interior optimum at T={} (needs routed reduction {:.2}x)",
            best.width,
            best.routed_reduction_needed.unwrap_or(f64::NAN)
        );
        println!("  >> a drafter's SHIPPED width is not its best width");
    }
    if !refused.is_empty() {
        println!();
        println!(
            "  R4: dense traffic alone refuses width(s) {} regardless of the routed side",
            refused.join(", ")
        );
    }
    println!();
    println!("  PROVISIONAL: alpha fitted from one observed pair under a constant-alpha");
    println!("  geometric model. These optima are design points, not performance bars.");
    if rows.iter().any(|r| r.rests_on_assumptions) {
        println!("  RESTS ON ASSUMPTIONS: physical reuse and/or state traffic unmeasured.");
    }
    Ok(())
}
