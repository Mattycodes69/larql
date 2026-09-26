//! Symbol census and KDA graph reports.

use super::super::fetch;
use serde_json::json;

#[allow(unused_imports)]
use super::*;

pub fn symbol_census(
    repo: &fetch::Repo,
    shard: &str,
    a: &super::super::args::SymbolCensusArgs,
    as_json: bool,
) -> R {
    use super::super::symbol_census::{self as sc, SymbolCounts};

    let (hdr, hdr_len) = repo.shard_header_with_len(shard)?;
    let all = fetch::packed_expert_tensors(&hdr);
    let wanted: Vec<_> = all
        .into_iter()
        .filter(|(name, ..)| {
            a.experts.iter().any(|e| {
                name.split(".experts.")
                    .nth(1)
                    .and_then(|r| r.split('.').next())
                    .is_some_and(|idx| idx == e.to_string())
            })
        })
        .collect();
    if wanted.is_empty() {
        return Err(format!("no packed expert tensors for {:?} in {shard}", a.experts).into());
    }

    let mut global = SymbolCounts::default();
    let mut nibbles: Vec<(usize, Vec<u8>)> = Vec::new();
    for (name, shape, lo, _hi) in &wanted {
        let row_bytes = *shape.get(1).unwrap_or(&0) as usize;
        let take = (a.rows * row_bytes) as u64;
        let packed = repo.tensor_bytes_at(shard, hdr_len, *lo, lo + take)?;
        global.add_packed(&packed);
        let mut n = Vec::with_capacity(packed.len() * 2);
        for b in &packed {
            n.push(b & 0xF);
            n.push(b >> 4);
        }
        nibbles.push((row_bytes * 2, n));
        eprintln!("  read {:.0} KB of {name}", take as f64 / 1e3);
    }

    let tiles: Vec<_> = a
        .blocks
        .iter()
        .map(|&b| {
            let mut merged = sc::TileStats {
                block: b,
                ..Default::default()
            };
            let mut ent_weighted = 0.0;
            for (row_len, n) in &nibbles {
                let t = sc::tile_stats(n, *row_len, b);
                ent_weighted += t.mean_local_entropy * t.tiles as f64;
                merged.tiles += t.tiles;
                merged.fits_two_bit += t.fits_two_bit;
                merged.fits_three_bit += t.fits_three_bit;
                merged.max_cardinality = merged.max_cardinality.max(t.max_cardinality);
                merged.min_cardinality = if merged.min_cardinality == 0 {
                    t.min_cardinality
                } else {
                    merged.min_cardinality.min(t.min_cardinality)
                };
                merged.median_cardinality = merged.median_cardinality.max(t.median_cardinality);
            }
            merged.mean_local_entropy = ent_weighted / merged.tiles.max(1) as f64;
            merged
        })
        .collect();

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "tensors": wanted.iter().map(|(n, ..)| n).collect::<Vec<_>>(),
                "counts": global, "entropy": global.entropy(),
                "entropy_zero_merged": global.entropy_zero_merged(),
                "tiles": tiles,
            }))?
        );
        return Ok(());
    }

    let n = global.total();
    println!();
    println!(
        "=== MXFP4 symbol census: {:.2} M weights, {} tensors ===",
        n as f64 / 1e6,
        wanted.len()
    );
    println!("  distinct symbols     {}", global.distinct());
    println!(
        "  Shannon entropy      {:.4} bits   (fixed-width floor {:.1})",
        global.entropy(),
        sc::FIXED_PAYLOAD_BITS
    );
    println!(
        "  with +/-0 merged     {:.4} bits   <- free, value-exact",
        global.entropy_zero_merged()
    );
    for k in [1usize, 3, 7] {
        println!("  top-{k} coverage       {:.4}", global.top_k_coverage(k));
    }
    println!();
    println!(
        "--- exact variable-rate candidates (payload bits, floor {:.1}) ---",
        sc::FIXED_PAYLOAD_BITS
    );
    let verdict = |b: f64| {
        if sc::beats_fixed_floor(b) {
            "WINS"
        } else {
            "loses"
        }
    };
    let e3 = sc::escape_bpw(3, global.top_k_coverage(7), 256);
    let e2 = sc::escape_bpw(2, global.top_k_coverage(3), 256);
    let rp = sc::residual_plane_bpw(global.optimal_pairing_rare_mass(), 256);
    println!("  3-bit + escape (7 hot)        {e3:.4}  {}", verdict(e3));
    println!("  2-bit + escape (3 hot)        {e2:.4}  {}", verdict(e2));
    println!(
        "  3-bit primary + residual      {rp:.4}  {}   (rare mass {:.3}, optimal pairing)",
        verdict(rp),
        global.optimal_pairing_rare_mass()
    );
    println!(
        "  ideal arithmetic coder        {:.4}  {}   (serial decode, no random access)",
        global.entropy(),
        verdict(global.entropy())
    );
    println!();
    println!("--- per-tile cardinality and the entropy-bias check ---");
    println!(
        "{:>6} {:>9} {:>7} {:>7} {:>7} {:>9} {:>9} {:>8}",
        "block", "tiles", "min", "med", "max", "<=8 syms", "local H", "bias?"
    );
    for t in &tiles {
        let gap = global.entropy() - t.mean_local_entropy;
        println!(
            "{:>6} {:>9} {:>7} {:>7} {:>7} {:>8.4}% {:>9.4} {:>8}",
            t.block,
            t.tiles,
            t.min_cardinality,
            t.median_cardinality,
            t.max_cardinality,
            100.0 * t.fits_three_bit as f64 / t.tiles.max(1) as f64,
            t.mean_local_entropy,
            if sc::local_skew_is_bias_shaped(gap, sc::FP4_CODES, t.block as u64, 0.06) {
                "YES"
            } else {
                "no"
            },
        );
    }
    println!("  'bias? YES' = the local-vs-global gap is explained by plug-in");
    println!("  entropy bias (K-1)/(2B ln2), NOT by real local concentration.");
    println!();

    println!("--- palette mode at each block's BEST-CASE tile ---");
    for t in &tiles {
        match sc::palette_bpw(t.min_cardinality, t.block as u64) {
            Some(b) => println!(
                "  B={:<4} min cardinality {:>2} -> {b:.4} bpw  {}",
                t.block,
                t.min_cardinality,
                if sc::beats_fixed_floor(b) {
                    "WINS"
                } else {
                    "loses even here"
                }
            ),
            None => println!(
                "  B={:<4} min cardinality {:>2} -> mode cannot fire on ANY tile",
                t.block, t.min_cardinality
            ),
        }
    }
    println!();

    println!("--- decode economics: the score is eta/bpw, not bpw ---");
    // Variable-rate candidates change the payload only; they still owe the same
    // compact scale stream the fixed exact floor pays for.
    let floor_fmt = super::super::serving_format::exact_floor();
    let floor = floor_fmt.all_in_bits();
    let scale_bits = floor_fmt.scale.bits_per_weight();
    for (label, bpw) in [
        ("ideal arithmetic coder", global.entropy() + scale_bits),
        ("3-bit + escape", e3 + scale_bits),
    ] {
        println!(
            "  {label:<24} {bpw:.4} all-in -> needs eta >= {:.3} to beat the {floor:.4} bpw \
             fixed floor at eta {:.2}",
            sc::break_even_eta(bpw, floor, super::super::classes::GROUPED_ROUTED_ETA),
            super::super::classes::GROUPED_ROUTED_ETA,
        );
    }
    println!("  A serial-decode format holding eta 0.81 is the bar to clear.");
    Ok(())
}

pub fn kda_graph(
    repo: &fetch::Repo,
    shard: &str,
    a: &super::super::args::KdaGraphArgs,
    as_json: bool,
) -> R {
    use super::super::kda_graph::{self as kg, KdaTensor};

    let hdr = repo.shard_header(shard)?;
    let prefix = format!(".layers.{}.self_attn.", a.layer);
    let obj = hdr.as_object().ok_or("shard header is not an object")?;
    let mut tensors: Vec<KdaTensor> = obj
        .iter()
        .filter_map(|(k, v)| {
            let suffix = k.split(&prefix).nth(1)?;
            let shape: Vec<u64> = v
                .get("shape")?
                .as_array()?
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .collect();
            Some(KdaTensor {
                name: suffix.to_string(),
                shape,
                role: kg::role(suffix),
            })
        })
        .collect();
    if tensors.is_empty() {
        return Err(format!(
            "no self_attn tensors for layer {} in {shard} — is it a KDA layer?",
            a.layer
        )
        .into());
    }
    tensors.sort_by(|x, y| x.name.cmp(&y.name));
    let cov = kg::coverage(&tensors);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "layer": a.layer, "tensors": tensors, "coverage": cov,
                "byte_coverage": cov.byte_coverage(),
            }))?
        );
        return Ok(());
    }

    println!(
        "=== KDA layer {} — shared-input rotation bundle ===",
        a.layer
    );
    println!("traced from the real tensor table, not from the modeling source");
    println!();
    println!("{:<22} {:>18} {:>12}  role", "tensor", "shape", "params");
    for t in &tensors {
        println!(
            "{:<22} {:>18} {:>12}  {:?}",
            t.name,
            format!("{:?}", t.shape),
            t.params(),
            t.role
        );
    }
    println!();
    println!(
        "  bundle covers {} of {} full-rank projections = {:.1}% of projection bytes",
        cov.bundled_full_rank,
        cov.total_full_rank,
        100.0 * cov.byte_coverage()
    );
    println!("  o_proj reads the gated-normed kernel output, so it CANNOT join:");
    println!("  a rotation does not commute with o_norm's per-channel gain.");
    println!();
    println!("  ! MUST also be folded, and easy to miss:");
    for n in &cov.easy_to_miss {
        println!("      {n}");
    }
    println!("    Rotating h without rewriting these silently corrupts the decay");
    println!("    gate and beta — a plausible wrong number, not a crash.");
    println!();
    println!("  ! output-side folding unavailable: q/k/v pass through a depthwise");
    println!("    ShortConvolution + silu, which no per-channel transform commutes with.");
    if let Some(al) = tensors.iter().find(|t| t.name.starts_with("A_log")) {
        use super::super::kda_a_log::{classify, ALogAxis, ALogVerdict};
        let numel: u64 = al.shape.iter().product();
        let v = classify(numel, kg::NUM_HEADS, kg::HEAD_DIM);
        println!();
        println!(
            "  A_log [{numel}] against num_heads {} / head_dim {}:",
            kg::NUM_HEADS,
            kg::HEAD_DIM
        );
        match v {
            ALogVerdict::Determined(ALogAxis::PerHead) => {
                println!("    per-head — matches fla's documented [HV] contract.");
            }
            ALogVerdict::Determined(ALogAxis::PerChannel) => {
                println!("    ! per-CHANNEL — contradicts fla, whose gate does A_log.view(H,1).");
                println!("      Two readings survive the big tensors and each breaks one small");
                println!("      one; shapes cannot decide. BLOCKER for KDA numerics — resolve");
                println!("      against an oracle using kda_a_log::asymmetric_fixture.");
            }
            ALogVerdict::AmbiguousBySymmetry => {
                println!("    ! ambiguous: num_heads == head_dim, so the length names both.");
            }
            ALogVerdict::Unexplained => {
                println!("    ! length matches neither axis.");
            }
        }
        if v.axis() != Some(ALogAxis::PerHead) {
            println!("      Sibling Kimi-Linear-48B has A_log numel == num_heads; K3 deviates.");
            // Hand over a fixture the two readings provably disagree on, so an
            // oracle run that "passes" cannot do so vacuously.
            let (fh, fk) = (3usize, 4usize);
            let (a, g, dt) = super::super::kda_a_log::asymmetric_fixture(fh, fk);
            let ph = super::super::kda_a_log::decay(ALogAxis::PerHead, &a, &g, &dt, fh, fk);
            let pc = super::super::kda_a_log::decay(ALogAxis::PerChannel, &a, &g, &dt, fh, fk);
            let differing = ph
                .iter()
                .zip(&pc)
                .filter(|(x, y)| (*x - *y).abs() > 1e-12)
                .count();
            println!();
            println!("      Discriminating fixture ({fh}x{fk}, rectangular on purpose):");
            println!(
                "        separates the readings in {differing} of {} cells",
                fh * fk
            );
            println!("        per-head   [0..3] {:.6?}", &ph[..3]);
            println!("        per-channel[0..3] {:.6?}", &pc[..3]);
            println!("        a SQUARE fixture would agree on the diagonal and prove nothing.");
        }
    }
    Ok(())
}
