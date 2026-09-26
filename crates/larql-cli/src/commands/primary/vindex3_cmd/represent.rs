//! `vindex3 represent`.

#[allow(unused_imports)]
use super::*;

/// `larql vindex3 represent` — compile a physical representation.
///
/// Prints what each object cost before and after, because the whole point
/// of the operation is a number: the pack is only worth persisting if it is
/// materially smaller than the bytes it was compiled from.
pub(super) fn run_represent(args: RepresentArgs) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::represent::{
        compile_representation_weighted, compile_representation_with, RepresentSpec,
    };

    let mut roles = larql_vindex::format::vindex3::represent::policy::RolePolicy::default();
    for name in &args.include_roles {
        let role = larql_vindex::format::vindex3::represent::policy::Role::parse(name).ok_or_else(
            || {
                // Derived, never restated: a hand-written list is wrong
                // one commit after a role is added, and this message is
                // the only place the caller learns what is valid.
                let known: Vec<&str> = larql_vindex::format::vindex3::represent::policy::Role::ALL
                    .iter()
                    .map(|r| r.name())
                    .collect();
                format!(
                    "unknown role `{name}` — the roles are: {}",
                    known.join(", ")
                )
            },
        )?;
        roles = roles.including(role);
    }
    let mut protect = larql_vindex::format::vindex3::represent::policy::Protections::default();
    for p in &args.protect {
        protect = match p.split_once('@') {
            Some((name, range)) => {
                let (lo, hi) = range
                    .split_once('-')
                    .ok_or_else(|| format!("--protect {p}: expected PROJ@LO-HI"))?;
                protect.projection_in(
                    name,
                    lo.trim().parse::<u32>().map_err(|e| format!("{p}: {e}"))?,
                    hi.trim().parse::<u32>().map_err(|e| format!("{p}: {e}"))?,
                )
            }
            None => protect.projection(p),
        };
    }
    for r in &args.protect_layers {
        let (lo, hi) = r
            .split_once('-')
            .ok_or_else(|| format!("--protect-layers expects LO-HI, got `{r}`"))?;
        protect = protect.layers(
            lo.trim().parse::<u32>().map_err(|e| format!("{r}: {e}"))?,
            hi.trim().parse::<u32>().map_err(|e| format!("{r}: {e}"))?,
        );
    }
    if !protect.is_empty() {
        println!("  protect: {}", protect.describe());
    }
    let spec = RepresentSpec {
        encoding: args.encoding.clone(),
        objects: args.objects.clone(),
        roles,
        deployment: args.deployment,
        protect,
    };
    println!(
        "== represent {} ({}) ==",
        args.encoding,
        if args.deployment {
            "deployment image"
        } else {
            "archival container"
        }
    );
    println!("  in     : {}", args.container.display());
    println!("  out    : {}", args.output.display());
    if !args.objects.is_empty() {
        println!("  objects: {}", args.objects.join(", "));
    }

    let started = std::time::Instant::now();
    let plugins = plugins::Plugins::load(&plugins::PluginArgs {
        plugins: args.plugins.clone(),
        lowering: None,
        representation: None,
    })?;
    let report = match &args.moments {
        Some(path) => {
            let (weights, sources) = input_moments::input_weights(&args.container, path)?;
            println!(
                "  moments: {} (sha256 {})",
                path.display(),
                &weights.digest[..16]
            );
            for (source, n) in &sources {
                println!("    {source:<36} {n} tensor(s)");
            }
            compile_representation_weighted(
                &args.container,
                &args.output,
                &spec,
                plugins.encoders,
                &weights,
            )?
        }
        None => {
            compile_representation_with(&args.container, &args.output, &spec, plugins.encoders)?
        }
    };

    println!("\n── compiled ──");
    println!(
        "  {:<34} {:>12} {:>12} {:>8} {:>9}",
        "object", "source", "compiled", "ratio", "tensors"
    );
    println!("  {}", "-".repeat(80));
    let mut src_total = 0u64;
    let mut out_total = 0u64;
    for c in &report.compiled_objects {
        src_total += c.source_bytes;
        out_total += c.compiled_bytes;
        println!(
            "  {:<34} {:>12} {:>12} {:>7.2}x {:>4} +{:<4}",
            c.object,
            human_bytes(c.source_bytes),
            human_bytes(c.compiled_bytes),
            c.compression(),
            c.compiled_tensors,
            c.carried_tensors,
        );
    }
    println!("  {}", "-".repeat(80));
    let ratio = if out_total == 0 {
        0.0
    } else {
        src_total as f64 / out_total as f64
    };
    println!(
        "  {:<34} {:>12} {:>12} {:>7.2}x",
        "TOTAL",
        human_bytes(src_total),
        human_bytes(out_total),
        ratio
    );
    let weighted: usize = report
        .compiled_objects
        .iter()
        .map(|c| c.weighted_tensors)
        .sum();
    if args.moments.is_some() {
        let compiled: usize = report
            .compiled_objects
            .iter()
            .map(|c| c.compiled_tensors)
            .sum();
        println!(
            "  encoded under input-feature weights: {weighted} of {compiled} compiled tensor(s)"
        );
    }
    let mut protected: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for c in &report.compiled_objects {
        for (role, n) in &c.preserved {
            *protected.entry(role.to_string()).or_insert(0) += n;
        }
    }
    for po in &report.preserved_objects {
        let roles: Vec<String> = po.roles.iter().map(|(r, n)| format!("{r} x{n}")).collect();
        println!(
            "  {:<34} {:>12} {:>12}   preserved whole [{}]",
            po.object,
            human_bytes(po.bytes),
            po.encoding,
            roles.join(", ")
        );
    }
    if !protected.is_empty() {
        println!("\n  preserved at source precision (conservative default):");
        for (role, n) in &protected {
            println!("    {role:<16} {n} tensor(s)");
        }
    }
    println!(
        "\n  {} segment(s) carried unchanged; canonical bytes are untouched.",
        report.linked_segments
    );
    println!("  wall time: {:.1}s", started.elapsed().as_secs_f64());
    println!("\n→ {}", args.output.display());
    Ok(())
}

pub(super) fn human_bytes(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}
