//! `vindex3 references` and `vindex3 inspect`.

use larql_models::inventory::{build_inventory, ArchitectureInventory};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

pub(super) fn run_references(args: ReferencesArgs) -> Result<(), Box<dyn std::error::Error>> {
    use larql_vindex::format::vindex3::auxiliary_references::AuxiliaryReferences;
    if args.declare {
        let declared = larql_vindex::format::vindex3::encode::declare_references(&args.container)?;
        if declared == 0 {
            println!("no dependency to declare; nothing written");
        } else {
            println!("declared {declared} reference(s)");
        }
        return Ok(());
    }
    let inspection =
        larql_vindex::format::vindex3::inspect::inspect_container(&args.container, false)?;
    let Some(name) = &inspection.index.auxiliary_references else {
        println!("the container declares no dependencies");
        return Ok(());
    };
    let table = AuxiliaryReferences::read(&args.container, name)?;
    println!("{} reference(s) in {name}:", table.len());
    for row in table.stored().references {
        println!(
            "  {}/{}  --{}-->  {}/{}",
            row.owner.object, row.owner.tensor, row.auxiliary, row.target.object, row.target.tensor
        );
    }
    Ok(())
}

pub(super) fn run_inspect(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let inspection =
        larql_vindex::format::vindex3::inspect::inspect_container(&args.container, args.verify)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&inspection)?);
    } else {
        println!("components:");
        for c in &inspection.components {
            let policy = match (c.sliding_layers, c.full_layers, c.nope_layers) {
                (Some(s), Some(f), Some(n)) => {
                    format!(
                        ", {s} sliding / {f} full{}, {n} NoPE, window {:?}",
                        match c.recurrent_layers {
                            Some(r) if r > 0 => format!(" / {r} gated-delta recurrent"),
                            _ => String::new(),
                        },
                        c.window
                    )
                }
                _ => ", (no per-layer table)".to_string(),
            };
            println!(
                "  {:8} {:12} {} layers, hidden {}{policy}",
                c.id, c.role, c.num_layers, c.hidden_size
            );
        }
        println!("objects:");
        for (id, entry) in &inspection.index.representations {
            println!(
                "  {:40} {:8.3} GB  {} tensors  sha256 {}…",
                id,
                entry.payload_bytes as f64 / 1e9,
                entry.tensor_count,
                &entry.payload_sha256[..12],
            );
        }
        println!("edges:");
        for e in &inspection.graph.edges {
            println!(
                "  {}.hidden{:?} -> {} via {} (block {:?})",
                e.producer_component,
                e.producer_layers,
                e.consumer_component,
                e.consumer_object,
                e.block_size,
            );
        }
    }
    let surface_defects = if args.execution_complete {
        let defects = inspection.execution_completeness();
        if !args.json {
            println!("execution surfaces:");
            for component in &inspection.graph.components {
                match &component.execution {
                    Some(surface) => {
                        // Presence follows the program (schema 6): each
                        // group prints only when the component runs it,
                        // and absence is said in words — a pure-SSM
                        // component has no attention/FFN to describe.
                        let attention = match &surface.attention {
                            Some(a) => format!(
                                "attention {}q/{}kv head {} q-scale {:.4} s-scale {:.4}{}",
                                a.num_q_heads,
                                a.num_kv_heads,
                                a.head_dim,
                                optional_op::scalar(a.query_scale),
                                a.score_scale,
                                if a.output_gate.is_some() {
                                    " gated"
                                } else {
                                    ""
                                },
                            ),
                            None => "attention absent".to_string(),
                        };
                        let ffn = match &surface.ffn {
                            Some(f) => format!(
                                "ffn {:?} {:?} {:?}",
                                f.activation, f.ffn_type, f.intermediate_size
                            ),
                            None => "ffn absent".to_string(),
                        };
                        let mixer = match &surface.mamba2 {
                            Some(m) => format!(
                                ", mamba2 mixer {}h×{}×{} conv {}",
                                m.geometry.num_heads,
                                m.geometry.head_dim,
                                m.geometry.state_size,
                                m.geometry.conv_kernel
                            ),
                            None => String::new(),
                        };
                        println!(
                            "  {:8} {attention}, {ffn}{mixer}, norm {:?} eps {:e}{}",
                            component.id,
                            surface.norm.pre.kind,
                            surface.norm.pre.eps,
                            match &surface.head {
                                Some(head) => format!(", head vocab {}", head.vocab_size),
                                None => String::new(),
                            },
                        )
                    }
                    None => println!("  {:8} (no execution surface)", component.id),
                }
            }
            for defect in &defects {
                println!("  defect: {defect}");
            }
            println!(
                "executable: {}",
                if defects.is_empty() { "yes" } else { "NO" }
            );
        }
        defects
    } else {
        Vec::new()
    };

    if !surface_defects.is_empty() {
        return Err(format!(
            "container not executable: {} execution-completeness defect(s)",
            surface_defects.len()
        )
        .into());
    }
    if inspection.is_coherent() {
        eprintln!(
            "container coherent{}",
            if args.verify {
                " (payloads verified)"
            } else {
                ""
            }
        );
        Ok(())
    } else {
        for defect in &inspection.defects {
            eprintln!("defect: {defect:?}");
        }
        Err(format!(
            "container incoherent: {} defect(s)",
            inspection.defects.len()
        )
        .into())
    }
}

/// Artifact display name: the file/directory stem.
pub(super) fn artifact_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Load one artifact: a `.json` file deserialises as a saved inventory;
/// anything else is inspected as a checkpoint directory.
pub(super) fn load_artifact(
    path: &Path,
) -> Result<ArchitectureInventory, Box<dyn std::error::Error>> {
    if path.extension().is_some_and(|ext| ext == INVENTORY_EXT) {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    } else {
        Ok(build_inventory(path)?)
    }
}
